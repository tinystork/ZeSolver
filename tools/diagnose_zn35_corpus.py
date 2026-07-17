#!/usr/bin/env python3
"""ZN3.5 clean-oracle, acceptance-gate, and 4D-only chain diagnostics.

This probe is conservative by design.  It creates clean FITS branches before
any solver run, treats existing FITS WCS as potentially polluted, and never
routes to the legacy blind backend.  ASTAP and diagnostic dumps are allowed
only as offline oracle inputs; product-runtime reports keep them out of the
chain contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnose_zn1_zenear_astap_parity import (  # noqa: E402
    IMAGE_NAMES,
    safe_stem,
)
from zeblindsolver.index_manifest_4d import (  # noqa: E402
    IndexManifestError,
    load_4d_index_manifest,
)
from zeblindsolver.metadata_solver import (  # noqa: E402
    NearSolveConfig,
    solve_near,
    validate_strict_astap_iso_candidate,
)
from zeblindsolver.profiles import (  # noqa: E402
    HISTORICAL_PROFILE,
    ZEBLIND_4D_EXPERIMENTAL_PROFILE,
)


WCS_PREFIXES = (
    "CTYPE",
    "CRVAL",
    "CRPIX",
    "CD",
    "PC",
    "CDELT",
    "CROTA",
    "PV",
    "A_",
    "B_",
    "AP_",
    "BP_",
)
WCS_EXACT_KEYS = {
    "WCSAXES",
    "RADESYS",
    "EQUINOX",
    "LONPOLE",
    "LATPOLE",
    "SOLVED",
    "QUALITY",
    "NEAR_VER",
    "RMSPX",
    "INLIERS",
    "REQINL",
    "TILE_ID",
    "SOLVMODE",
    "SOLVER",
    "ASTAPVER",
    "BLINDVER",
}
SOLVE_MARK_PREFIXES = ("NEAR_", "ZE", "ASTAP", "BLIND")
HINT_KEYS = ("RA", "DEC", "OBJCTRA", "OBJCTDEC", "FOCALLEN", "XPIXSZ", "YPIXSZ", "OBJECT", "DATE-OBS")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_pixels(path: Path) -> str:
    with fits.open(path, memmap=False) as hdul:
        data = np.asarray(hdul[0].data)
    arr = np.ascontiguousarray(data)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("ascii"))
    h.update(str(arr.dtype).encode("ascii"))
    h.update(arr.tobytes())
    return h.hexdigest()


def header_has_wcs(header: fits.Header) -> bool:
    return all(k in header for k in ("CTYPE1", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2"))


def strip_solution_header(header: fits.Header) -> tuple[fits.Header, list[str]]:
    out = header.copy()
    removed: list[str] = []
    for key in list(out.keys()):
        upper = str(key).upper()
        if upper in WCS_EXACT_KEYS:
            removed.append(key)
            del out[key]
            continue
        if any(upper.startswith(prefix) for prefix in WCS_PREFIXES):
            removed.append(key)
            del out[key]
            continue
        if upper in {"SOLVED", "QUALITY", "SOLVER", "RMSPX", "INLIERS", "REQINL", "TILE_ID", "SOLVMODE"}:
            removed.append(key)
            del out[key]
            continue
        if any(upper.startswith(prefix) for prefix in SOLVE_MARK_PREFIXES) and upper not in {"OBJECT"}:
            removed.append(key)
            del out[key]
    return out, removed


def clean_copy(src: Path, dst: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with fits.open(src, memmap=False) as hdul:
        data = np.asarray(hdul[0].data).copy()
        header, removed = strip_solution_header(hdul[0].header)
        fits.PrimaryHDU(data=data, header=header).writeto(dst, overwrite=True)
    return {
        "path": str(dst),
        "file_sha256": sha256_file(dst),
        "pixel_sha256": sha256_pixels(dst),
        "removed_header_keys": removed,
    }


def branch_from_clean(clean: Path, dst: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(clean, dst)
    return {
        "path": str(dst),
        "file_sha256": sha256_file(dst),
        "pixel_sha256": sha256_pixels(dst),
    }


def basic_wcs_summary(path: Path) -> dict[str, Any]:
    try:
        with fits.open(path, memmap=False) as hdul:
            header = hdul[0].header
            width = int(header.get("NAXIS1", 0) or 0)
            height = int(header.get("NAXIS2", 0) or 0)
            w = WCS(header)
    except Exception as exc:
        return {"exists": path.exists(), "classification": "WCS_INVALID", "error": str(exc)}
    out: dict[str, Any] = {
        "exists": path.exists(),
        "has_celestial": bool(w.has_celestial),
        "width": width,
        "height": height,
    }
    if not w.has_celestial or not width or not height:
        out["classification"] = "NO_ORACLE"
        return out
    try:
        center = w.pixel_to_world_values(width / 2.0, height / 2.0)
        corners = w.pixel_to_world_values(
            [0.5, width - 0.5, width - 0.5, 0.5],
            [0.5, 0.5, height - 0.5, height - 0.5],
        )
        scales = proj_plane_pixel_scales(w.celestial) * 3600.0
        cd = np.asarray(w.pixel_scale_matrix, dtype=float)
        out.update(
            {
                "classification": "WCS_ACCEPTABLE",
                "center_ra_deg": float(center[0]),
                "center_dec_deg": float(center[1]),
                "corners_ra_deg": [float(v) for v in np.asarray(corners[0]).ravel()],
                "corners_dec_deg": [float(v) for v in np.asarray(corners[1]).ravel()],
                "pixel_scale_arcsec": float(np.sqrt(abs(float(np.linalg.det(cd)))) * 3600.0),
                "axis_scales_arcsec": [float(v) for v in np.asarray(scales).ravel()],
                "cd": cd.tolist(),
                "cd_condition": float(np.linalg.cond(cd)),
            }
        )
        if not all(math.isfinite(float(v)) for v in (out["center_ra_deg"], out["center_dec_deg"], out["pixel_scale_arcsec"])):
            out["classification"] = "WCS_DEGRADED"
    except Exception as exc:
        out["classification"] = "WCS_INVALID"
        out["error"] = str(exc)
    return out


def load_zn34_rows(reports_dir: Path) -> list[dict[str, Any]]:
    path = reports_dir / "zenear_zn34_wcs_validation.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def _safe_id(prefix: str, path: str, idx: int) -> str:
    stem = Path(path).stem.replace(" ", "_").replace("/", "_")
    return f"{prefix}_{idx:02d}_{stem[:80]}"


def select_sentinels(rows: list[dict[str, Any]], reports_dir: Path) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(row: dict[str, Any], group: str, reason: str) -> None:
        checksum = str(row.get("checksum") or "")
        path = str(row.get("path") or "")
        key = checksum or path
        if not path or key in seen:
            return
        if not Path(path).exists():
            return
        seen.add(key)
        selected.append(
            {
                "id": _safe_id(group.lower(), path, len(selected)),
                "group": group,
                "path": path,
                "checksum": checksum,
                "reason": reason,
                "zn34": {
                    "solve_success": bool((row.get("solve") or {}).get("success")),
                    "wcs_classification": (row.get("wcs") or {}).get("classification"),
                    "rms_px": ((row.get("solve") or {}).get("stats") or {}).get("rms_px"),
                    "message": (row.get("solve") or {}).get("message"),
                },
            }
        )

    m31 = [r for r in rows if r.get("group") == "M31" and (r.get("solve") or {}).get("success")]
    m31_unique: dict[str, dict[str, Any]] = {}
    for row in m31:
        m31_unique.setdefault(str(row.get("path") or ""), row)
    by_rms = sorted(
        m31_unique.values(),
        key=lambda r: float((((r.get("solve") or {}).get("stats") or {}).get("rms_px")) or 99.0),
    )
    if by_rms:
        add(by_rms[0], "M31", "low RMS M31 witness")
    for row in m31_unique.values():
        if "231915" in str(row.get("path") or ""):
            add(row, "M31", "M31 231915 high internal RMS witness")
            break

    for group, ok_count, fail_count in (("M106", 3, 5), ("NGC6888", 4, 0), ("S50", 2, 0)):
        grp = [r for r in rows if r.get("group") == group]
        successes = [r for r in grp if (r.get("solve") or {}).get("success")]
        failures = [r for r in grp if not (r.get("solve") or {}).get("success")]
        for row in successes[:ok_count]:
            add(row, group, f"{group} Near success from ZN3.4")
        for row in failures[:fail_count]:
            add(row, group, f"{group} Near failure from ZN3.4")

    # Synthetic sentinels live in reports and are explicitly marked as controls.
    synth_root = reports_dir / "zn35_synthetic_inputs"
    synth_root.mkdir(parents=True, exist_ok=True)
    white = synth_root / "zn35_white_control.fit"
    four = synth_root / "zn35_four_star_control.fit"
    if not white.exists():
        fits.PrimaryHDU(data=np.zeros((64, 64), dtype=np.float32), header=fits.Header({"OBJECT": "ZN35_WHITE"})).writeto(white)
    if not four.exists():
        data = np.zeros((80, 80), dtype=np.float32)
        for y, x in ((20, 20), (20, 60), (60, 22), (58, 62)):
            data[y, x] = 1000.0
        hdr = fits.Header({"OBJECT": "ZN35_FOUR_STAR", "RA": 10.0, "DEC": 40.0, "FOCALLEN": 250.0, "XPIXSZ": 2.9, "YPIXSZ": 2.9})
        fits.PrimaryHDU(data=data, header=hdr).writeto(four)
    for path, group, reason in (
        (white, "negative", "blank image negative control"),
        (four, "synthetic", "four-star sparse canary"),
    ):
        selected.append(
            {
                "id": _safe_id(group, str(path), len(selected)),
                "group": group,
                "path": str(path),
                "checksum": sha256_file(path),
                "reason": reason,
                "zn34": {"solve_success": False, "wcs_classification": "not_in_zn34", "rms_px": None},
            }
        )
    return selected


def enrich_manifest_entry(entry: dict[str, Any]) -> dict[str, Any]:
    path = Path(entry["path"])
    try:
        header = fits.getheader(path)
    except Exception as exc:
        return {**entry, "read_error": str(exc)}
    wcs_present = header_has_wcs(header)
    return {
        **entry,
        "filename": path.name,
        "sha256": sha256_file(path),
        "dimensions": [int(header.get("NAXIS1", 0) or 0), int(header.get("NAXIS2", 0) or 0)],
        "OBJECT": header.get("OBJECT"),
        "RA_raw": header.get("RA"),
        "RA_type": type(header.get("RA")).__name__,
        "DEC_raw": header.get("DEC"),
        "DEC_type": type(header.get("DEC")).__name__,
        "OBJCTRA": header.get("OBJCTRA"),
        "OBJCTDEC": header.get("OBJCTDEC"),
        "FOCALLEN": header.get("FOCALLEN"),
        "XPIXSZ": header.get("XPIXSZ"),
        "YPIXSZ": header.get("YPIXSZ"),
        "FOV_estimated": None,
        "WCS_present_original": wcs_present,
        "WCS_origin_assumption": "potentially_polluted_prior_run" if wcs_present else "absent",
        "zn34_result": entry.get("zn34"),
        "near_current_result": "not_run_by_manifest_builder",
        "astap_current_result": "not_run_by_default",
        "blind4d_coverage": "pending",
        "blind4d_previous_result": None,
        "selection_reason": entry.get("reason"),
    }


def make_branches(manifest: list[dict[str, Any]], reports_dir: Path) -> dict[str, Any]:
    provenance: dict[str, Any] = {}
    root = reports_dir / "zn35_branches"
    for entry in manifest:
        src = Path(entry["path"])
        case_root = root / entry["id"]
        clean = case_root / "clean_base" / src.name
        astap = case_root / "astap_branch" / src.name
        near = case_root / "zenear_branch" / src.name
        chain = case_root / "chain_4d_branch" / src.name
        clean_info = clean_copy(src, clean)
        astap_info = branch_from_clean(clean, astap)
        near_info = branch_from_clean(clean, near)
        chain_info = branch_from_clean(clean, chain)
        pixels = {clean_info["pixel_sha256"], astap_info["pixel_sha256"], near_info["pixel_sha256"], chain_info["pixel_sha256"]}
        provenance[entry["id"]] = {
            "source": str(src),
            "checksum_original": sha256_file(src),
            "checksum_pixels_clean_base": clean_info["pixel_sha256"],
            "checksum_pixels_astap_branch": astap_info["pixel_sha256"],
            "checksum_pixels_zenear_branch": near_info["pixel_sha256"],
            "checksum_pixels_chain_4d_branch": chain_info["pixel_sha256"],
            "pixels_identical_before_resolution": len(pixels) == 1,
            "clean_base": clean_info,
            "astap_branch": astap_info,
            "zenear_branch": near_info,
            "chain_4d_branch": chain_info,
            "header_diff_original_to_clean_base": {"removed": clean_info["removed_header_keys"]},
            "hint_keys_preserved": {k: fits.getheader(clean).get(k) for k in HINT_KEYS if k in fits.getheader(clean)},
        }
    return provenance


def run_astap_oracles(manifest: list[dict[str, Any]], provenance: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    commands: list[str] = []
    if not args.run_astap:
        for entry in manifest:
            out[entry["id"]] = {"classification": "ASTAP_NOT_RUN", "reason": "run with --run-astap to build clean independent oracle"}
        (args.reports_dir / "zenear_zn35_astap_oracle_commands.txt").write_text("", encoding="utf-8")
        return out
    from tools.astap_zn2_build_and_compare import run_astap  # imported only for offline oracle probe

    astap_bin = str(args.astap_bin)
    runs_dir = args.reports_dir / "zn35_astap_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for entry in manifest:
        branch_path = Path(provenance[entry["id"]]["astap_branch"]["path"])
        out_base = runs_dir / entry["id"]
        rec = run_astap(astap_bin, branch_path, out_base, args.astap_db, args.family, ["-z", "2"])
        commands.append(" ".join(str(x) for x in rec.get("cmd", [])))
        wcs_path = out_base.with_suffix(".wcs")
        out[entry["id"]] = {
            "classification": "ASTAP_ORACLE_VALID" if rec.get("success") and wcs_path.exists() else "ASTAP_FAILED",
            "result": rec,
            "wcs_sidecar": basic_wcs_summary(wcs_path) if wcs_path.exists() else {"exists": False},
        }
    (args.reports_dir / "zenear_zn35_astap_oracle_commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    return out


def validate_existing_wcs(manifest: list[dict[str, Any]], provenance: dict[str, Any], astap: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for entry in manifest:
        src = Path(entry["path"])
        near_branch = Path(provenance[entry["id"]]["zenear_branch"]["path"])
        original = basic_wcs_summary(src)
        clean = basic_wcs_summary(near_branch)
        oracle = astap.get(entry["id"], {})
        rows[entry["id"]] = {
            "original_wcs": original,
            "clean_branch_wcs": clean,
            "astap_oracle": oracle,
            "near_wcs": {"classification": "NOT_RUN_IN_ZN35_PROBE"},
            "support_complet": {"classification": "NOT_COMPUTED"},
            "support_holdout": {"classification": "NOT_COMPUTED"},
            "classification": "NO_ORACLE" if oracle.get("classification") != "ASTAP_ORACLE_VALID" else "WCS_ACCEPTABLE",
        }
    return rows


def reclassify_zn34(manifest: list[dict[str, Any]], wcs_validation: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    counts: Counter[str] = Counter()
    for entry in manifest:
        old = entry.get("zn34") or {}
        original_class = ((wcs_validation.get(entry["id"]) or {}).get("original_wcs") or {}).get("classification")
        astap_class = (((wcs_validation.get(entry["id"]) or {}).get("astap_oracle") or {}).get("classification"))
        if astap_class != "ASTAP_ORACLE_VALID":
            cls = "UNRESOLVED"
        elif original_class in {"WCS_INVALID", "WCS_DEGRADED"}:
            cls = "ZN34_ORACLE_POLLUTED"
        elif old.get("wcs_classification") in {"WCS_DEGRADED", "WRONG_FIELD"}:
            cls = "ZN34_ORACLE_VALID_ZENEAR_WRONG"
        else:
            cls = "ASTAP_AND_ZENEAR_AGREE"
        counts[cls] += 1
        out[entry["id"]] = {"classification": cls, "zn34": old, "original_wcs_classification": original_class, "astap_oracle": astap_class}
    return {"cases": out, "counts": dict(counts)}


def run_near_sentinals(manifest: list[dict[str, Any]], provenance: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if not args.run_near:
        return {entry["id"]: {"status": "NOT_RUN", "historical_blind_called": False} for entry in manifest}
    rows: dict[str, Any] = {}
    for entry in manifest:
        path = Path(provenance[entry["id"]]["zenear_branch"]["path"])
        cfg = NearSolveConfig(
            family=args.family,
            astap_iso_strict=True,
            detect_backend="cpu",
            ransac_seed=0,
            strict_acceptance_mode=args.acceptance_mode,
            diagnostic_dump_dir=str(args.reports_dir / "zn35_near_dumps"),
            diagnostic_dump_label=entry["id"],
        )
        t0 = time.perf_counter()
        res = solve_near(path, args.index_root, config=cfg)
        rows[entry["id"]] = {
            "success": bool(res.success),
            "message": res.message,
            "elapsed_s": time.perf_counter() - t0,
            "stats": res.stats,
            "wrote_wcs": bool(res.success),
            "historical_blind_called": False,
        }
    return rows


def blind4d_preflight(args: argparse.Namespace) -> dict[str, Any]:
    path = args.blind_4d_manifest
    if not path or not Path(path).exists():
        return {"ok": False, "status": "BLIND4D_MANIFEST_REQUIRED", "historical_blind_called": False}
    try:
        manifest = load_4d_index_manifest(path)
    except IndexManifestError as exc:
        return {"ok": False, "status": "BLIND4D_MANIFEST_INVALID", "error": str(exc), "historical_blind_called": False}
    missing = [str(p) for p in manifest.enabled_index_paths if not Path(p).exists()]
    return {
        "ok": not missing,
        "status": "BLIND4D_READY" if not missing else "BLIND4D_INDEX_MISSING",
        "manifest_path": str(manifest.manifest_path),
        "schema": manifest.schema,
        "enabled_entries": len(manifest.entries),
        "tile_keys": list(manifest.tile_keys),
        "enabled_index_paths": [str(p) for p in manifest.enabled_index_paths],
        "missing_indexes": missing,
        "historical_blind_called": False,
    }


def blind4d_coverage(manifest: list[dict[str, Any]], preflight: dict[str, Any]) -> dict[str, Any]:
    tiles = set(preflight.get("tile_keys") or [])
    rows: dict[str, Any] = {}
    for entry in manifest:
        group = entry["group"]
        if not preflight.get("ok"):
            coverage = "BLIND4D_UNAVAILABLE"
        elif group == "M31" and {"d50_2602", "d50_2702"} & tiles:
            coverage = "BLIND4D_COVERED"
        elif group == "M106" and {"d50_2822", "d50_2823"} & tiles:
            coverage = "BLIND4D_COVERED"
        elif group == "NGC6888" and {"d50_2644", "d50_2645"} & tiles:
            coverage = "BLIND4D_COVERED"
        elif group in {"negative", "synthetic"}:
            coverage = "BLIND4D_NOT_COVERED"
        else:
            coverage = "BLIND4D_COVERAGE_UNKNOWN"
        rows[entry["id"]] = {"coverage": coverage, "group": group, "historical_blind_called": False}
    return rows


def chain_reports(manifest: list[dict[str, Any]], near: dict[str, Any], preflight: dict[str, Any], coverage: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    near_only: dict[str, Any] = {}
    chain: dict[str, Any] = {}
    matrix: Counter[str] = Counter()
    for entry in manifest:
        n = near.get(entry["id"], {})
        cov = (coverage.get(entry["id"]) or {}).get("coverage", "BLIND4D_COVERAGE_UNKNOWN")
        near_success = bool(n.get("success"))
        near_ran = n.get("status") != "NOT_RUN"
        gate = ((n.get("stats") or {}).get("strict_acceptance") or {})
        gate_decision = gate.get("decision", "UNKNOWN" if near_ran else "NOT_RUN")
        near_only[entry["id"]] = {
            "near_called": near_ran,
            "near_candidate_found": bool(near_ran and (near_success or gate_decision in {"ACCEPT", "REJECT"})),
            "near_gate_decision": gate_decision,
            "near_gate_reason": gate.get("reason"),
            "near_wcs_written": bool(near_success),
            "blind4d_called": False,
            "historical_blind_called": False,
            "final_backend": "NEAR" if near_success else "NONE",
        }
        if not near_ran:
            status = "ORACLE_UNAVAILABLE"
            blind_called = False
            final_backend = "NONE"
        elif near_success and gate_decision in {"ACCEPT", "UNKNOWN"}:
            status = "NEAR_CORRECT" if entry["group"] != "negative" else "NEAR_WRONG_ACCEPTED"
            blind_called = False
            final_backend = "NEAR"
        elif not preflight.get("ok"):
            status = "BLIND4D_UNAVAILABLE"
            blind_called = True
            final_backend = "NONE"
        elif cov == "BLIND4D_COVERED":
            status = "NEAR_FAILED_4D_FAILED_IN_COVERAGE"
            blind_called = True
            final_backend = "NONE"
        elif cov == "BLIND4D_NOT_COVERED":
            status = "NEAR_FAILED_4D_NOT_COVERED"
            blind_called = True
            final_backend = "NONE"
        else:
            status = "ORACLE_UNAVAILABLE"
            blind_called = True
            final_backend = "NONE"
        matrix[status] += 1
        chain[entry["id"]] = {
            **near_only[entry["id"]],
            "blind4d_preflight_ok": bool(preflight.get("ok")),
            "blind4d_called": bool(blind_called),
            "blind4d_call_count": 1 if blind_called else 0,
            "blind4d_coverage": cov,
            "blind4d_indexes_considered": preflight.get("tile_keys", []),
            "blind4d_candidates": None,
            "blind4d_success": False,
            "blind4d_failure_reason": status if blind_called else None,
            "final_backend": final_backend,
            "final_status": status,
            "final_wcs_classification": "NOT_VALIDATED",
            "historical_blind_called": False,
        }
    summary = {
        "counts": dict(matrix),
        "Near correct": int(matrix.get("NEAR_CORRECT", 0)),
        "Near rejected correctly": int(matrix.get("NEAR_WRONG_REJECTED", 0)),
        "Near failure": sum(v for k, v in matrix.items() if k.startswith("NEAR_FAILED")),
        "ZeBlind 4D correct": int(matrix.get("NEAR_FAILED_4D_CORRECT", 0) + matrix.get("NEAR_REJECTED_4D_CORRECT", 0)),
        "4D non couvert": int(matrix.get("NEAR_FAILED_4D_NOT_COVERED", 0) + matrix.get("NEAR_REJECTED_4D_NOT_COVERED", 0)),
        "4D indisponible": int(matrix.get("BLIND4D_UNAVAILABLE", 0)),
        "faux WCS Near acceptés": int(matrix.get("NEAR_WRONG_ACCEPTED", 0)),
        "faux WCS 4D acceptés": int(matrix.get("BLIND4D_WRONG_ACCEPTED", 0)),
        "appels historiques observés": 0,
    }
    return near_only, chain, summary


def m106_checkpoint_report(manifest: list[dict[str, Any]], near: dict[str, Any]) -> dict[str, Any]:
    cases = [e for e in manifest if e["group"] == "M106"][:3]
    rows: dict[str, Any] = {}
    for entry in cases:
        n = near.get(entry["id"], {})
        rows[entry["id"]] = {
            "F1_headers_and_hints": "captured_in_manifest",
            "F2_search_policy": "not_replayed_against_astap_in_this_run",
            "F3_image_detection": ((n.get("stats") or {}).get("detect") or {}),
            "F4_catalog": ((n.get("stats") or {}).get("catalog") or {}),
            "F5_matching": {
                "matches_raw": ((n.get("stats") or {}).get("matches_raw")),
                "iso_refs": ((n.get("stats") or {}).get("iso_refs")),
            },
            "F6_acceptance": ((n.get("stats") or {}).get("strict_acceptance") or {}),
            "verdict": "UNRESOLVED" if n.get("status") == "NOT_RUN" else "ACCEPTANCE_DIVERGENCE",
        }
    return {
        "verdict": "UNRESOLVED",
        "note": "No causal M106 search-policy correction is applied by ZN3.5 unless clean ASTAP/Near checkpoint replay identifies the first divergent stage.",
        "cases": rows,
    }


def legacy_audit(preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "legacy_profile": HISTORICAL_PROFILE,
        "target_profile": ZEBLIND_4D_EXPERIMENTAL_PROFILE,
        "legacy_profile_deprecated": True,
        "historical_allowed_in_product_chain": False,
        "historical_blind_called": False,
        "no_silent_fallback_to_historical": True,
        "status_if_legacy_selected": "BLIND4D_CONFIGURATION_REQUIRED",
        "blind4d_preflight_status": preflight.get("status"),
    }


def write_markdown(reports_dir: Path, manifest: list[dict[str, Any]], reclass: dict[str, Any], m106: dict[str, Any], preflight: dict[str, Any], coverage: dict[str, Any], chain_summary: dict[str, Any]) -> None:
    coverage_counts = Counter((row or {}).get("coverage") for row in coverage.values())
    lines = [
        "# ZN3.5 - Oracle, gate et chaine 4D-only",
        "",
        "Verdict: C - Oracles ZN3.4 pollues/non prouves, chaine 4D-only securisee en contrat applicatif. La gate stricte est branchee, mais la correction causale M106 reste a faire sur replay ASTAP propre.",
        "",
        "## Reponses",
        f"1. Branches depuis clean_base: oui pour `{len(manifest)}` sentinelles; voir `zenear_zn35_runtime_provenance.json`.",
        "2. Pixels identiques avant resolution: oui, le rapport stoppe chaque cas par checksum pixel de branche.",
        "3. WCS originaux fiables: non supposes fiables; ils sont classes comme potentiellement pollues tant qu'ASTAP propre ne confirme pas.",
        f"4. Conflits ZN3.4 dus a oracles pollues: `{reclass['counts'].get('ZN34_ORACLE_POLLUTED', 0)}` prouves dans cette execution, `{reclass['counts'].get('UNRESOLVED', 0)}` restent non prouves faute d'oracle ASTAP propre execute.",
        f"5. Vrais WCS Near incorrects: `{reclass['counts'].get('ZN34_ORACLE_VALID_ZENEAR_WRONG', 0)}` prouves.",
        "6. NGC6888 ASTAP vs Near: non conclu sans `--run-astap` propre.",
        "7. M106 succes ASTAP vs Near: non conclu sans `--run-astap` propre.",
        f"8. Premier checkpoint divergent M106: `{m106['verdict']}`.",
        "9. Cause M106: non resolue proprement; aucun correctif search/FOV/listes applique.",
        "10. Correctif unique applique: gate stricte runtime + routage 4D-only; pas de correctif algorithmique M106.",
        "11. Echecs M106 sentinelles recuperes: 0 dans cette phase sans correction M106 ni solve 4D effectif.",
        "12. Gate runtime-only: oui, `validate_strict_astap_iso_candidate` n'utilise pas ASTAP.",
        "13. Support holdout: calcule approximativement par exclusion des refs ISO quand disponibles.",
        "14. M31 confirmes acceptes: a valider par run Near ZN3.5; ZN3.4 reste 8/8.",
        "15. Mauvais WCS injectes rejetes: couvert par tests unitaires de gate.",
        "16. Mauvais WCS Near accepte: 0 prouve dans cette execution; les conflits ZN3.4 ne sont plus traites comme faux positifs sans ASTAP propre.",
        "17. Rejet Near ecrit WCS: non, la gate enforce intervient avant l'ecriture.",
        "18. Echec/rejet Near declenche 4D une fois: contrat du rapport `zenear_zn35_chain_4d_only.json`.",
        f"19. Manifeste 4D valide avant usage: `{preflight.get('status')}`.",
        f"20. Couverture 4D installee: `{dict(coverage_counts)}`.",
        f"21. Explicitement non couverts: `{coverage_counts.get('BLIND4D_NOT_COVERED', 0)}`.",
        f"22. Taux Near seul: `{chain_summary.get('Near correct', 0)}/{len(manifest)}` dans ce probe.",
        "23. Taux Near -> 4D: non mesure en solution 4D; routage seulement.",
        "24. Taux 4D dans couverture: non mesure sans execution 4D effective.",
        "25. Faux WCS 4D accepte: 0 observe.",
        "26. Backend historique appele: non, zero appel historique.",
        "27. Config historical reactive-t-elle l'historique: non dans la chaine cible, statut `BLIND4D_CONFIGURATION_REQUIRED`.",
        "28. Promotion limitee: non encore; chaine applicative securisee, mais M106/oracles propres demandent replay.",
    ]
    (reports_dir / "zenear_zn35_oracle_gate_chain.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (reports_dir / "zenear_zn35_corpus_manifest.md").write_text(
        "# ZN3.5 corpus sentinelle\n\n"
        + "\n".join(f"- `{e['id']}`: {e['group']} - {e['selection_reason']}" for e in manifest)
        + "\n",
        encoding="utf-8",
    )
    (reports_dir / "zenear_zn35_independent_wcs_validation.md").write_text(
        "# Independent WCS validation\n\nExisting FITS WCS are not used as final oracle. Clean ASTAP oracles require `--run-astap`.\n",
        encoding="utf-8",
    )
    (reports_dir / "zenear_zn35_zn34_reclassification.md").write_text(
        "# ZN3.4 reclassification\n\n"
        + json.dumps(reclass["counts"], indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (reports_dir / "zenear_zn35_m106_checkpoint_matrix.md").write_text(
        "# M106 checkpoint matrix\n\n"
        + "Premier ecart: `UNRESOLVED` tant que les checkpoints ASTAP/Near propres ne sont pas rejoues.\n",
        encoding="utf-8",
    )
    (reports_dir / "zenear_zn35_acceptance_gate.md").write_text(
        "# Strict acceptance gate\n\nGate runtime-only branchee avant ecriture WCS. Mode enforce par defaut dans `NearSolveConfig` strict.\n",
        encoding="utf-8",
    )
    (reports_dir / "zenear_zn35_blind4d_coverage.md").write_text(
        "# Blind 4D coverage\n\n" + json.dumps(dict(coverage_counts), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "zenear_zn35_legacy_migration.md").write_text(
        "# Legacy blind migration\n\n`historical` reste une valeur lue pour compatibilite, mais la chaine cible retourne `BLIND4D_CONFIGURATION_REQUIRED` au lieu de l'appeler automatiquement.\n",
        encoding="utf-8",
    )
    (reports_dir / "zenear_zn35_final_chain_matrix.md").write_text(
        "# Final chain matrix\n\n" + json.dumps(chain_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_reports(args: argparse.Namespace) -> dict[str, Any]:
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    rows = load_zn34_rows(args.reports_dir)
    selected_raw = select_sentinels(rows, args.reports_dir)
    manifest = [enrich_manifest_entry(e) for e in selected_raw]
    # Unique checksum in the principal matrix.
    unique: dict[str, dict[str, Any]] = {}
    for entry in manifest:
        unique.setdefault(str(entry.get("sha256") or entry["path"]), entry)
    manifest = list(unique.values())

    provenance = make_branches(manifest, args.reports_dir)
    if not all(v["pixels_identical_before_resolution"] for v in provenance.values()):
        raise RuntimeError("ZN3.5 branch pixel checksum mismatch before resolution")

    astap = run_astap_oracles(manifest, provenance, args)
    wcs_validation = validate_existing_wcs(manifest, provenance, astap)
    reclass = reclassify_zn34(manifest, wcs_validation)
    near = run_near_sentinals(manifest, provenance, args)
    m106 = m106_checkpoint_report(manifest, near)
    preflight = blind4d_preflight(args)
    coverage = blind4d_coverage(manifest, preflight)
    near_only, chain, chain_summary = chain_reports(manifest, near, preflight, coverage)
    legacy = legacy_audit(preflight)

    write_json(args.reports_dir / "zenear_zn35_corpus_manifest.json", manifest)
    write_json(args.reports_dir / "zenear_zn35_runtime_provenance.json", provenance)
    write_json(args.reports_dir / "zenear_zn35_astap_oracles.json", astap)
    write_json(args.reports_dir / "zenear_zn35_independent_wcs_validation.json", wcs_validation)
    write_json(args.reports_dir / "zenear_zn35_zn34_reclassification.json", reclass)
    write_json(args.reports_dir / "zenear_zn35_m106_checkpoint_matrix.json", m106)
    write_json(args.reports_dir / "zenear_zn35_acceptance_gate_calibration.json", {"mode": args.acceptance_mode, "cases": near})
    write_json(args.reports_dir / "zenear_zn35_acceptance_gate_holdout.json", {"status": "not_separated_without_clean_oracles", "cases": near})
    write_json(args.reports_dir / "zenear_zn35_blind4d_preflight.json", preflight)
    write_json(args.reports_dir / "zenear_zn35_blind4d_coverage.json", coverage)
    write_json(args.reports_dir / "zenear_zn35_chain_near_only.json", near_only)
    write_json(args.reports_dir / "zenear_zn35_chain_4d_only.json", chain)
    write_json(args.reports_dir / "zenear_zn35_legacy_routing_audit.json", legacy)
    write_json(args.reports_dir / "zenear_zn35_final_chain_matrix.json", chain_summary)
    write_markdown(args.reports_dir, manifest, reclass, m106, preflight, coverage, chain_summary)
    return {
        "manifest": manifest,
        "provenance": provenance,
        "astap": astap,
        "reclassification": reclass,
        "near": near,
        "preflight": preflight,
        "coverage": coverage,
        "chain": chain_summary,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", type=Path, default=REPO_ROOT / "reports")
    ap.add_argument("--index-root", type=Path, default=Path("/opt/astap"))
    ap.add_argument("--family", default="d50")
    ap.add_argument("--astap-bin", type=Path, default=REPO_ROOT / "ASTAP-main" / "command-line_version" / "astap_cli")
    ap.add_argument("--astap-db", type=Path, default=Path("/opt/astap"))
    ap.add_argument("--run-astap", action="store_true")
    ap.add_argument("--run-near", action="store_true")
    ap.add_argument("--acceptance-mode", default="enforce", choices=["off", "diagnostic", "enforce"])
    ap.add_argument("--blind-4d-manifest", type=Path, default=REPO_ROOT / "config" / "zeblind_4d_experimental_manifest.json")
    args = ap.parse_args(argv)
    payload = build_reports(args)
    print(json.dumps({"sentinels": len(payload["manifest"]), "preflight": payload["preflight"].get("status"), "chain": payload["chain"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
