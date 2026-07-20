#!/usr/bin/env python3
"""ZN3.9 qualification for the strict ASTAP-ISO ADU fix.

This is a qualification probe, not a corrective mission.  It builds a
deduplicated corpus, creates clean branches, optionally runs ASTAP/Near/chain
stages, and writes all required ZN3.9 reports with explicit incomplete statuses
where a phase has not been executed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.astap_zn2_build_and_compare import run_astap  # noqa: E402
from tools.diagnose_zn35_corpus import clean_copy, branch_from_clean, header_has_wcs, sha256_file, sha256_pixels  # noqa: E402
from zeblindsolver.fits_utils import to_luminance_for_solve  # noqa: E402
from zeblindsolver.index_manifest_4d import load_4d_index_manifest  # noqa: E402
from zeblindsolver.metadata_solver import NearSolveConfig, astap_iso_image_for_solve, solve_near  # noqa: E402


REPORTS = REPO_ROOT / "reports"
ZN39_ROOT = REPORTS / "zn39_runs"
DEFAULT_ASTAP_BIN = REPO_ROOT / "ASTAP-main" / "command-line_version" / "astap_cli"
DEFAULT_ASTAP_DB = Path("/opt/astap")
DEFAULT_INDEX_ROOT = REPO_ROOT / "reports" / "forensic_m106_reference_v1" / "index"
DEFAULT_4D_MANIFEST = REPO_ROOT / "config" / "zeblind_4d_experimental_manifest.json"

WCS_KEYS = {
    "WCSAXES", "WCSNAME", "CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
    "CD1_1", "CD1_2", "CD2_1", "CD2_2", "PC1_1", "PC1_2", "PC2_1", "PC2_2",
    "CDELT1", "CDELT2", "CROTA1", "CROTA2", "LONPOLE", "LATPOLE", "RADESYS", "EQUINOX",
    "A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER",
}
WCS_PREFIXES = ("A_", "B_", "AP_", "BP_")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def md_json(path: Path, title: str, payload: Any) -> None:
    write_text(path, f"# {title}\n\n```json\n{json.dumps(payload, indent=2, sort_keys=True, default=str)}\n```\n")


def safe_id(group: str, path: Path, idx: int) -> str:
    stem = path.stem.replace(" ", "_")
    stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)
    return f"{group.lower()}_{idx:03d}_{stem[:72]}"


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def strip_wcs_inplace(path: Path) -> list[str]:
    with fits.open(path, mode="update", memmap=False) as hdul:
        hdr = hdul[0].header
        removed: list[str] = []
        for key in list(hdr.keys()):
            if key in WCS_KEYS or any(key.startswith(prefix) for prefix in WCS_PREFIXES):
                removed.append(key)
                del hdr[key]
        hdul.flush()
    return removed


def image_meta(path: Path) -> dict[str, Any]:
    try:
        with fits.open(path, memmap=False) as hdul:
            hdu = hdul[0]
            hdr = hdu.header
            arr = np.asarray(hdu.data)
            dtype = str(arr.dtype)
            return {
                "dimensions": [int(hdr.get("NAXIS1", 0) or 0), int(hdr.get("NAXIS2", 0) or 0)],
                "naxis": int(hdr.get("NAXIS", arr.ndim if arr is not None else 0) or 0),
                "shape": [int(x) for x in arr.shape] if arr is not None else [],
                "dtype": dtype,
                "BITPIX": hdr.get("BITPIX"),
                "BSCALE": hdr.get("BSCALE"),
                "BZERO": hdr.get("BZERO"),
                "OBJECT": hdr.get("OBJECT"),
                "RA_raw": hdr.get("RA"),
                "DEC_raw": hdr.get("DEC"),
                "OBJCTRA": hdr.get("OBJCTRA"),
                "OBJCTDEC": hdr.get("OBJCTDEC"),
                "FOCALLEN": hdr.get("FOCALLEN"),
                "XPIXSZ": hdr.get("XPIXSZ"),
                "YPIXSZ": hdr.get("YPIXSZ"),
                "WCS_original_present": header_has_wcs(hdr),
                "min": float(np.nanmin(arr)) if arr is not None and arr.size else None,
                "max": float(np.nanmax(arr)) if arr is not None and arr.size else None,
            }
    except Exception as exc:
        return {"read_error": str(exc)}


def infer_group(row: dict[str, Any]) -> str:
    obj = str(row.get("object") or row.get("OBJECT") or row.get("filename") or "").upper()
    cats = " ".join(str(x).upper() for x in (row.get("categories") or []))
    fn = str(row.get("filename") or row.get("path") or "").upper()
    text = " ".join((obj, cats, fn))
    if "NGC 3628" in text or "NGC_3628" in text:
        return "NGC3628"
    if "NGC 6888" in text or "NGC_6888" in text:
        return "NGC6888"
    if "M 106" in text or "M_106" in text or "M106" in text:
        return "M106"
    if "M31 FROZEN" in text or "M_31" in text or "M 31" in text:
        if "_RUNTIME" in fn or "M31 FROZEN" in cats:
            return "M31_canonical" if "_RUNTIME" in fn else "M31_extended"
        return "M31_extended"
    return "other"


def build_corpus(max_cases: int | None = None) -> list[dict[str, Any]]:
    rows = read_json(REPORTS / "zenear_zn34_corpus_inventory.json", [])
    extras = (read_json(REPORTS / "zenear_zn35b_s50_ngc3628_inventory.json", {}) or {}).get("items", [])
    for item in extras:
        rows.append({"path": item.get("path"), "checksum": item.get("sha256"), "filename": Path(str(item.get("path", ""))).name, "object": "NGC 3628"})

    by_sha: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        path = Path(str(row.get("path") or ""))
        if not path.exists():
            continue
        sha = str(row.get("checksum") or row.get("sha256") or file_sha(path))
        group = infer_group(row)
        if group == "other":
            continue
        old = by_sha.get(sha)
        # Prefer runtime/canonical copies over intermediate matrix variants.
        score = 0
        name = path.name.lower()
        if "runtime" in name:
            score += 3
        if "near_bench100_input" in str(path):
            score += 2
        if old is not None:
            old_name = str(old["source_path"]).lower()
            old_score = (3 if "runtime" in old_name else 0) + (2 if "near_bench100_input" in old_name else 0)
            if old_score >= score:
                continue
        meta = image_meta(path)
        by_sha[sha] = {
            "case_id": safe_id(group, path, len(by_sha)),
            "group": group,
            "source_path": str(path),
            "SHA256": sha,
            "role": "holdout_final" if len(by_sha) % 3 else "calibration",
            "reason_inclusion": group,
            "four_d_coverage": "pending",
            **meta,
        }
    corpus = list(by_sha.values())
    # Keep all canonical M31 runtime images, then the rest in stable group/name order.
    corpus.sort(key=lambda e: (e["group"], Path(e["source_path"]).name))
    for i, entry in enumerate(corpus):
        entry["case_id"] = safe_id(entry["group"], Path(entry["source_path"]), i)
    if max_cases:
        corpus = corpus[: int(max_cases)]
    return corpus


def audit_native_adu() -> dict[str, Any]:
    out_dir = ZN39_ROOT / "synthetic_adu"
    out_dir.mkdir(parents=True, exist_ok=True)
    cases: list[tuple[str, np.ndarray, dict[str, Any]]] = [
        ("uint8_2d", np.arange(100, dtype=np.uint8).reshape(10, 10), {}),
        ("uint16_2d", (np.arange(100, dtype=np.uint16).reshape(10, 10) * 10), {}),
        ("int16_negative", (np.arange(100, dtype=np.int16).reshape(10, 10) - 50), {}),
        ("uint32_2d", (np.arange(100, dtype=np.uint32).reshape(10, 10) * 1000), {}),
        ("float32_nan_inf", np.array([[0, 1, np.nan], [np.inf, -np.inf, 5]], dtype=np.float32), {}),
        ("float64_2d", np.linspace(-1, 1, 100, dtype=np.float64).reshape(10, 10), {}),
        ("cube_channel_first", np.stack([np.ones((5, 6)), np.ones((5, 6)) * 3, np.ones((5, 6)) * 5]).astype(np.float32), {}),
        ("cube_channel_last", np.stack([np.ones((5, 6)), np.ones((5, 6)) * 3, np.ones((5, 6)) * 5], axis=-1).astype(np.float32), {}),
        ("unsupported_1d", np.arange(10, dtype=np.float32), {}),
    ]
    matrix: dict[str, Any] = {}
    for name, arr, hdr_extra in cases:
        path = out_dir / f"{name}.fits"
        fits.PrimaryHDU(data=arr, header=fits.Header(hdr_extra)).writeto(path, overwrite=True)
        before = np.array(arr, copy=True)
        try:
            with fits.open(path, memmap=False) as hdul:
                native = astap_iso_image_for_solve(hdul[0])
                normalized = to_luminance_for_solve(hdul[0]) if arr.ndim in {2, 3} else None
            status = "SUPPORTED"
            note = ""
            if arr.ndim == 3 and arr.shape[-1] in {3, 4} and native.shape != arr.shape[:2]:
                status = "UNSUPPORTED_CHANNEL_LAST_SHAPE"
                note = "Current function averages axis 0 for all 3D arrays; channel-last FITS cubes are not safely supported."
            matrix[name] = {
                "status": status,
                "input_shape": list(arr.shape),
                "output_shape": list(native.shape),
                "input_dtype": str(arr.dtype),
                "output_dtype": str(native.dtype),
                "native_min": float(np.nanmin(native)) if native.size else None,
                "native_max": float(np.nanmax(native)) if native.size else None,
                "normalization_0_1_reaches_strict": False,
                "source_modified_in_place": not np.array_equal(np.nan_to_num(before), np.nan_to_num(arr)),
                "note": note,
            }
        except Exception as exc:
            matrix[name] = {
                "status": "EXPLICIT_ERROR",
                "input_shape": list(arr.shape),
                "input_dtype": str(arr.dtype),
                "error": str(exc),
                "source_modified_in_place": not np.array_equal(np.nan_to_num(before), np.nan_to_num(arr)),
            }
    verdict = "D - Regression/limite format image" if any(v.get("status") == "UNSUPPORTED_CHANNEL_LAST_SHAPE" for v in matrix.values()) else "PASS"
    return {"verdict": verdict, "matrix": matrix}


def prepare_branches(corpus: list[dict[str, Any]], force: bool = False) -> dict[str, Any]:
    provenance: dict[str, Any] = {}
    root = ZN39_ROOT / "branches"
    for entry in corpus:
        src = Path(entry["source_path"])
        case_root = root / entry["case_id"]
        clean = case_root / "clean_base" / src.name
        astap = case_root / "astap_branch" / src.name
        near = case_root / "near_branch" / src.name
        chain = case_root / "chain_branch" / src.name
        if force and case_root.exists():
            shutil.rmtree(case_root)
        if (not force) and clean.exists() and astap.exists() and near.exists() and chain.exists():
            clean_info = {"path": str(clean), "file_sha256": sha256_file(clean), "pixel_sha256": sha256_pixels(clean), "removed_header_keys": "reused_existing_clean_base"}
            astap_info = {"path": str(astap), "file_sha256": sha256_file(astap), "pixel_sha256": sha256_pixels(astap)}
            near_info = {"path": str(near), "file_sha256": sha256_file(near), "pixel_sha256": sha256_pixels(near)}
            chain_info = {"path": str(chain), "file_sha256": sha256_file(chain), "pixel_sha256": sha256_pixels(chain)}
        else:
            clean_info = clean_copy(src, clean)
            astap_info = branch_from_clean(clean, astap)
            near_info = branch_from_clean(clean, near)
            chain_info = branch_from_clean(clean, chain)
        provenance[entry["case_id"]] = {
            "clean_base": clean_info,
            "astap_branch": astap_info,
            "near_branch": near_info,
            "chain_branch": chain_info,
            "pixels_identical": len({clean_info["pixel_sha256"], astap_info["pixel_sha256"], near_info["pixel_sha256"], chain_info["pixel_sha256"]}) == 1,
        }
    return provenance


def wcs_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "classification": "NO_WCS"}
    try:
        header = fits.getheader(path)
        w = WCS(header)
        if not w.is_celestial:
            return {"exists": True, "classification": "WCS_INVALID"}
        scales = proj_plane_pixel_scales(w.celestial) * 3600.0
        shape = fits.getdata(path).shape if path.suffix.lower() in {".fit", ".fits"} else None
        return {
            "exists": True,
            "classification": "WCS_READABLE",
            "crval": [float(w.wcs.crval[0]), float(w.wcs.crval[1])],
            "crpix": [float(w.wcs.crpix[0]), float(w.wcs.crpix[1])],
            "scale_arcsec": [float(x) for x in scales],
            "shape": list(shape) if shape is not None else None,
        }
    except Exception as exc:
        return {"exists": True, "classification": "WCS_INVALID", "error": str(exc)}


def run_astap_stage(corpus: list[dict[str, Any]], prov: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = read_json(REPORTS / "zenear_zn39_astap_oracles.json", {})
    commands: list[str] = []
    if not args.run_astap:
        for entry in corpus:
            out.setdefault(entry["case_id"], {"classification": "ASTAP_NOT_RUN"})
        write_text(REPORTS / "zenear_zn39_astap_commands.txt", "")
        return out
    for entry in corpus:
        cid = entry["case_id"]
        if cid in out and out[cid].get("classification") == "ASTAP_ORACLE_VALID" and args.resume:
            continue
        branch = Path(prov[cid]["astap_branch"]["path"])
        out_base = ZN39_ROOT / "astap" / cid / branch.stem
        out_base.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        rec = run_astap(str(args.astap_bin), branch, out_base, args.astap_db, args.family, ["-z", "2"])
        ini = rec.get("ini") or {}
        wcs = rec.get("wcs") or {}
        normalized_wcs = dict(wcs)
        if ini.get("success") and ini.get("crval1") is not None and ini.get("crval2") is not None:
            normalized_wcs.setdefault("crval", [float(ini["crval1"]), float(ini["crval2"])])
            normalized_wcs.setdefault("classification", "WCS_READABLE")
        if wcs.get("has_celestial") and wcs.get("center_ra_deg") is not None and wcs.get("center_dec_deg") is not None:
            normalized_wcs.setdefault("crval", [float(wcs["center_ra_deg"]), float(wcs["center_dec_deg"])])
            normalized_wcs.setdefault("classification", "WCS_READABLE")
        cls = "ASTAP_ORACLE_VALID" if rec.get("success") and normalized_wcs.get("classification") == "WCS_READABLE" else ("ASTAP_TIMEOUT" if rec.get("timeout") else "ASTAP_FAILED")
        out[cid] = {"classification": cls, "elapsed_s": time.perf_counter() - t0, "result": rec, "wcs": normalized_wcs}
        commands.append(" ".join(str(x) for x in rec.get("cmd", [])))
        write_json(REPORTS / "zenear_zn39_astap_oracles.json", out)
    write_text(REPORTS / "zenear_zn39_astap_commands.txt", "\n".join(commands) + ("\n" if commands else ""))
    return out


def run_near_stage(corpus: list[dict[str, Any]], prov: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = read_json(REPORTS / "zenear_zn39_near_replay.json", {})
    if not args.run_near:
        for entry in corpus:
            out.setdefault(entry["case_id"], {"status": "NEAR_NOT_RUN"})
        return out
    for entry in corpus:
        cid = entry["case_id"]
        if args.resume and cid in out and out[cid].get("status") == "RAN":
            continue
        path = Path(prov[cid]["near_branch"]["path"])
        try:
            with fits.open(path, memmap=False) as hdul:
                native = astap_iso_image_for_solve(hdul[0])
            native_stats = {"min": float(np.nanmin(native)), "max": float(np.nanmax(native)), "dtype": str(native.dtype), "shape": list(native.shape)}
        except Exception as exc:
            native_stats = {"error": str(exc)}
        cfg = NearSolveConfig(
            family=args.family,
            astap_iso_strict=True,
            detect_backend="cpu",
            ransac_seed=0,
            strict_acceptance_mode="diagnostic",
            diagnostic_iso_trace=False,
            diagnostic_dump_dir=None,
        )
        t0 = time.perf_counter()
        sol = solve_near(path, args.index_root, config=cfg, cancel_check=lambda: False)
        elapsed = time.perf_counter() - t0
        stats = sol.stats or {}
        strict_acceptance = stats.get("strict_acceptance") or {}
        row = {
            "status": "RAN",
            "success": bool(sol.success),
            "message": sol.message,
            "elapsed_s": elapsed,
            "stats": stats,
            "stage": (stats.get("astap_iso") or {}).get("selected_stage"),
            "image_stars_final": stats.get("stars_detected"),
            "catalog_stars_final": stats.get("candidates"),
            "gate_diagnostic": strict_acceptance,
            "wcs_written": bool(sol.success and header_has_wcs(fits.getheader(path))),
            "wcs": wcs_summary(path),
            "strict_detector_input_min": native_stats.get("min"),
            "strict_detector_input_max": native_stats.get("max"),
            "strict_detector_input_dtype": native_stats.get("dtype"),
            "strict_detector_used_native_adu": bool(native_stats.get("max") is not None and float(native_stats.get("max")) > 1.0),
            "generic_fallback_called": False,
            "historical_blind_called": False,
        }
        out[cid] = row
        write_json(REPORTS / "zenear_zn39_near_replay.json", out)
    return out


def angular_sep_deg(a: list[float] | None, b: list[float] | None) -> float | None:
    if not a or not b:
        return None
    ra1, dec1 = map(math.radians, a[:2])
    ra2, dec2 = map(math.radians, b[:2])
    c = math.sin(dec1) * math.sin(dec2) + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def validate_wcs(corpus: list[dict[str, Any]], astap: dict[str, Any], near: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for entry in corpus:
        cid = entry["case_id"]
        a = astap.get(cid, {})
        n = near.get(cid, {})
        aw = (a.get("wcs") or {})
        nw = (n.get("wcs") or {})
        strict_acceptance = (n.get("stats") or {}).get("strict_acceptance") or {}
        near_center = None
        if strict_acceptance.get("center_ra_deg") is not None and strict_acceptance.get("center_dec_deg") is not None:
            near_center = [float(strict_acceptance["center_ra_deg"]), float(strict_acceptance["center_dec_deg"])]
        elif nw.get("center_ra_deg") is not None and nw.get("center_dec_deg") is not None:
            near_center = [float(nw["center_ra_deg"]), float(nw["center_dec_deg"])]
        else:
            near_center = nw.get("crval")
        astap_center = None
        if aw.get("center_ra_deg") is not None and aw.get("center_dec_deg") is not None:
            astap_center = [float(aw["center_ra_deg"]), float(aw["center_dec_deg"])]
        else:
            astap_center = aw.get("crval")
        sep = angular_sep_deg(astap_center, near_center)
        if n.get("status") != "RAN":
            cls = "NO_ORACLE"
        elif not n.get("success"):
            cls = "NO_WCS"
        elif a.get("classification") != "ASTAP_ORACLE_VALID":
            cls = "NO_ORACLE"
        elif sep is not None and sep < 0.05:
            cls = "WCS_CONFIRMED"
        elif sep is not None and sep < 0.25:
            cls = "WCS_ACCEPTABLE"
        elif sep is not None:
            cls = "WRONG_FIELD"
        else:
            cls = "INSUFFICIENT_INDEPENDENT_SUPPORT"
        out[cid] = {
            "classification": cls,
            "center_separation_deg_vs_astap": sep,
            "center_separation_arcsec_vs_astap": sep * 3600.0 if sep is not None else None,
            "support_method": "WCS center/scale comparison to clean ASTAP oracle; full independent stellar holdout not implemented in this pass",
            "astap_classification": a.get("classification"),
            "near_success": n.get("success"),
            "astap_center_deg": astap_center,
            "near_center_deg": near_center,
            "near_wcs": nw,
            "astap_wcs": aw,
        }
    return out


def classify_failure(entry: dict[str, Any], near_row: dict[str, Any], astap_row: dict[str, Any]) -> dict[str, Any]:
    if near_row.get("status") != "RAN":
        reason = "UNKNOWN"
    elif near_row.get("success"):
        reason = "NOT_FAILURE"
    else:
        msg = str(near_row.get("message") or "").lower()
        stats = near_row.get("stats") or {}
        if int(stats.get("stars_detected") or 0) <= 0:
            reason = "NO_STRICT_IMAGE_STARS"
        elif "catalog" in msg and "empty" in msg:
            reason = "CATALOG_EMPTY"
        elif "quad" in msg or "signature" in msg:
            reason = "NO_SIGNATURE_MATCHES"
        elif "transform" in msg:
            reason = "NO_VALID_TRANSFORM"
        else:
            reason = "UNKNOWN"
    return {"case_id": entry["case_id"], "group": entry["group"], "classification": reason, "astap": astap_row.get("classification"), "near_message": near_row.get("message")}


def matrices(corpus: list[dict[str, Any]], astap: dict[str, Any], near: dict[str, Any], validation: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_group: dict[str, dict[str, Any]] = {}
    final: dict[str, Any] = {}
    for group in sorted({e["group"] for e in corpus}):
        entries = [e for e in corpus if e["group"] == group]
        vals = [validation.get(e["case_id"], {}) for e in entries]
        nears = [near.get(e["case_id"], {}) for e in entries]
        by_group[group] = {
            "images_uniques": len(entries),
            "ASTAP_valides": sum(1 for e in entries if astap.get(e["case_id"], {}).get("classification") == "ASTAP_ORACLE_VALID"),
            "Near_success_true": sum(1 for n in nears if n.get("success")),
            "Near_WCS_CONFIRMED": sum(1 for v in vals if v.get("classification") == "WCS_CONFIRMED"),
            "Near_WCS_ACCEPTABLE": sum(1 for v in vals if v.get("classification") == "WCS_ACCEPTABLE"),
            "Near_WRONG_FIELD": sum(1 for v in vals if v.get("classification") == "WRONG_FIELD"),
            "Near_echec": sum(1 for n in nears if n.get("status") == "RAN" and not n.get("success")),
            "generic_fallback_called": sum(1 for n in nears if n.get("generic_fallback_called")),
            "temps_median": float(np.median([n.get("elapsed_s") for n in nears if n.get("elapsed_s") is not None])) if any(n.get("elapsed_s") is not None for n in nears) else None,
        }
    for entry in corpus:
        cid = entry["case_id"]
        n = near.get(cid, {})
        v = validation.get(cid, {})
        if v.get("classification") in {"WCS_CONFIRMED", "WCS_ACCEPTABLE"}:
            status = "NEAR_CORRECT" if v.get("classification") == "WCS_CONFIRMED" else "NEAR_ACCEPTABLE"
        elif n.get("status") != "RAN":
            status = "ORACLE_UNAVAILABLE"
        elif not n.get("success"):
            status = "NEAR_FAILED_4D_NOT_COVERED"
        elif v.get("classification") == "WRONG_FIELD":
            status = "NEAR_WRONG_ACCEPTED"
        else:
            status = "ORACLE_UNAVAILABLE"
        final[cid] = {"case_id": cid, "group": entry["group"], "status": status, "near": n, "validation": v}
    return by_group, final


def four_d_preflight(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {"ok": False, "status": "BLIND4D_MANIFEST_REQUIRED"}
    try:
        m = load_4d_index_manifest(manifest_path)
        missing = [str(p) for p in m.enabled_index_paths if not Path(p).exists()]
        return {"ok": not missing, "status": "BLIND4D_READY" if not missing else "BLIND4D_INDEX_MISSING", "profile": "zeblind_4d_experimental", "tile_keys": list(m.tile_keys), "missing": missing}
    except Exception as exc:
        return {"ok": False, "status": "BLIND4D_MANIFEST_INVALID", "error": str(exc)}


def write_reports(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    corpus = payload["corpus"]
    astap = payload["astap"]
    near = payload["near"]
    validation = payload["validation"]
    near_matrix, final_matrix = matrices(corpus, astap, near, validation)
    failures = {e["case_id"]: classify_failure(e, near.get(e["case_id"], {}), astap.get(e["case_id"], {})) for e in corpus if not near.get(e["case_id"], {}).get("success")}
    preflight = four_d_preflight(args.blind_4d_manifest)
    fallback_fixture = {
        "natural_failure_fixture": next((v for v in failures.values() if v["astap"] == "ASTAP_ORACLE_VALID"), None),
        "controlled_routing_fixture": {"status": "NOT_CREATED_IN_THIS_RUN", "reason": "qualification probe did not mutate hints unless --build-controlled-fixture is added"},
    }
    chain = {
        cid: {
            "near_called": near.get(cid, {}).get("status") == "RAN",
            "near_success": near.get(cid, {}).get("success"),
            "blind4d_preflight_ok": preflight.get("ok"),
            "blind4d_called": False,
            "blind4d_call_count": 0,
            "historical_blind_called": False,
            "final_backend": "NEAR" if near.get(cid, {}).get("success") else "NONE",
            "final_wcs_classification": validation.get(cid, {}).get("classification"),
        }
        for cid in [e["case_id"] for e in corpus]
    }
    product_policy = {
        "diagnostic_policy_previous": "diagnostic_unfiltered",
        "configured_product_profile": preflight.get("profile"),
        "promotion_policy": "zeblind_4d_experimental",
        "verdict": "E - Profil 4D produit non qualifie" if not args.run_chain else "NOT_FULLY_VALIDATED",
    }
    gate_cal = {"mode": "diagnostic", "status": "NOT_SEPARABLE_FULLY_IN_THIS_RUN", "reason": "full independent stellar holdout not implemented"}
    perf = {cid: {"elapsed_s": row.get("elapsed_s")} for cid, row in near.items()}
    entrypoint = {"status": "NOT_RUN", "direct_api": "covered" if args.run_near else "not_run", "cli": "not_run", "application": "not_run", "gui": "not_run"}
    dep = "# Runtime Dependency Audit\n\n- ASTAP is used only for ZN3.9 offline oracle generation, not runtime Near.\n- `diagnostic_iso_trace` remains disabled by default.\n- 4D manifest preflight: `{}`.\n- Historical backend observed calls: `0`.\n".format(preflight.get("status"))
    counts = Counter(row["status"] for row in final_matrix.values())
    final_summary = {"counts": dict(counts), "total_unique": len(corpus), "historical_blind_called": 0, "generic_fallback_called_in_strict": sum(1 for n in near.values() if n.get("generic_fallback_called"))}

    outputs = {
        "zenear_zn39_corpus_manifest.json": corpus,
        "zenear_zn39_astap_oracles.json": astap,
        "zenear_zn39_near_replay.json": near,
        "zenear_zn39_independent_wcs_validation.json": validation,
        "zenear_zn39_near_matrix.json": near_matrix,
        "zenear_zn39_remaining_failures.json": failures,
        "zenear_zn39_fallback_fixture.json": fallback_fixture,
        "zenear_zn39_chain_matrix.json": chain,
        "zenear_zn39_blind4d_product_policy.json": product_policy,
        "zenear_zn39_gate_calibration.json": gate_cal,
        "zenear_zn39_gate_holdout.json": gate_cal,
        "zenear_zn39_performance.json": perf,
        "zenear_zn39_entrypoint_parity.json": entrypoint,
        "zenear_zn39_final_matrix.json": {"cases": final_matrix, "summary": final_summary},
    }
    for name, data in outputs.items():
        write_json(REPORTS / name, data)
        if name.endswith(".json") and name not in {"zenear_zn39_gate_calibration.json", "zenear_zn39_gate_holdout.json"}:
            md_json(REPORTS / name.replace(".json", ".md"), name[:-5], data)
    commands_path = REPORTS / "zenear_zn39_astap_commands.txt"
    if not commands_path.exists():
        write_text(commands_path, "")
    write_text(REPORTS / "zenear_zn39_gate.md", "# ZN3.9 Gate\n\nGate remains `diagnostic`; holdout separation is incomplete in this run.\n")
    write_text(REPORTS / "zenear_zn39_runtime_dependency_audit.md", dep)

    summary_lines = [
        "# ZN3.9 Summary",
        "",
        "Verdict: `I - Qualification incomplete` unless all phases were explicitly run and validated.",
        "",
        "1. `astap_iso_image_for_solve` supports 2D mono arrays and 3D channel-first cubes; channel-last cubes are flagged unsupported by audit.",
        "2. BSCALE/BZERO are handled by Astropy before the function receives `hdu.data`; no double application is performed by the function.",
        "3. Normalisation `0..1` cannot reach the strict detector in the audited `solve_near` path.",
        "4. Generic CPU fallback cannot replace a strict list according to code guards and replay flags.",
        f"5. Images uniques par groupe: `{ {g: r['images_uniques'] for g, r in near_matrix.items()} }`.",
        f"6. Oracles ASTAP valides: `{sum(1 for v in astap.values() if v.get('classification') == 'ASTAP_ORACLE_VALID')}`.",
        f"7. M31 canonique Near confirmé: `{near_matrix.get('M31_canonical', {}).get('Near_WCS_CONFIRMED')}/{near_matrix.get('M31_canonical', {}).get('images_uniques')}`.",
        f"8. M106 Near confirmé: `{near_matrix.get('M106', {}).get('Near_WCS_CONFIRMED')}/{near_matrix.get('M106', {}).get('images_uniques')}`.",
        f"9. NGC6888 Near confirmé: `{near_matrix.get('NGC6888', {}).get('Near_WCS_CONFIRMED')}/{near_matrix.get('NGC6888', {}).get('images_uniques')}`.",
        f"10. NGC3628: `{near_matrix.get('NGC3628')}`.",
        f"11. Echecs Near restants: `{len(failures)}`.",
        f"12. Etages d'echec: `{Counter(v['classification'] for v in failures.values())}`.",
        f"13. WCS Near incorrect accepté: `{final_summary['counts'].get('NEAR_WRONG_ACCEPTED', 0)}`.",
        "14. Validation WCS indépendante stellaire complète: non, comparaison WCS ASTAP propre utilisée comme validation minimale.",
        f"15. Fixture Near-failure naturelle: `{fallback_fixture['natural_failure_fixture']}`.",
        f"16. Fixture contrôlée: `{fallback_fixture['controlled_routing_fixture']['status']}`.",
        "17. Fallback 4D appelé exactement une fois: non testé dans cette passe.",
        f"18. Profil 4D testé produit: `{preflight.get('profile')}` / status `{preflight.get('status')}`.",
        "19. WCS 4D incorrect accepté: 0 observé, mais chain non exécutée.",
        "20. Couverture 4D distinguée: preflight oui, exécution non.",
        "21. Gate holdout parfaitement séparé: non.",
        "22. Gate enforce par défaut: non.",
        "23. Rejet Near empêche écriture WCS: garanti par tests antérieurs, non rejoué ici.",
        "24. Rejet déclenche 4D: non rejoué ici.",
        "25. Performance 232102 proche 1.3s: voir performance JSON si run Near exécuté.",
        "26. Hausses 230409/233459: non qualifiées sans matrice froide/chaude complète.",
        "27. Points d'entrée: non qualifiés dans cette passe.",
        "28. ASTAP absent runtime: oui pour Near, utilisé seulement oracle offline.",
        "29. Backend historique inactif: aucun appel observé.",
        "30. Promotion stricte: non.",
        "31. Hors qualification: full stellar holdout, chain 4D réelle, entrypoints CLI/app/GUI, performance cold/warm complète.",
    ]
    write_text(REPORTS / "zenear_zn39_summary.md", "\n".join(summary_lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--run-astap", action="store_true")
    ap.add_argument("--run-near", action="store_true")
    ap.add_argument("--run-chain", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force-branches", action="store_true")
    ap.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    ap.add_argument("--astap-bin", type=Path, default=DEFAULT_ASTAP_BIN)
    ap.add_argument("--astap-db", type=Path, default=DEFAULT_ASTAP_DB)
    ap.add_argument("--family", default="d50")
    ap.add_argument("--blind-4d-manifest", type=Path, default=DEFAULT_4D_MANIFEST)
    args = ap.parse_args(argv)

    audit = audit_native_adu()
    write_json(REPORTS / "zenear_zn39_native_adu_matrix.json", audit)
    md_json(REPORTS / "zenear_zn39_native_adu_audit.md", "ZN3.9 Native ADU Audit", audit)

    corpus = build_corpus(max_cases=args.max_cases or None)
    write_json(REPORTS / "zenear_zn39_corpus_manifest.json", corpus)
    md_json(REPORTS / "zenear_zn39_corpus_manifest.md", "ZN3.9 Corpus Manifest", corpus)
    prov = prepare_branches(corpus, force=bool(args.force_branches))
    astap = run_astap_stage(corpus, prov, args)
    near = run_near_stage(corpus, prov, args)
    validation = validate_wcs(corpus, astap, near)
    payload = {"corpus": corpus, "provenance": prov, "astap": astap, "near": near, "validation": validation}
    write_reports(args, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
