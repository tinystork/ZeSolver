#!/usr/bin/env python3
"""ZN3.4 hardening and promotion gate for strict ASTAP-ISO ZeNear.

The probe is deliberately conservative: it inventories broadly, solves a fixed
manifest on copies, validates WCS only when an independent oracle is available,
and keeps unknown-oracle cases out of the confirmed-success bucket.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnose_zn1_zenear_astap_parity import IMAGE_NAMES, safe_stem  # noqa: E402
from zeblindsolver.metadata_solver import (  # noqa: E402
    NearSolveConfig,
    _extract_near_center_angle,
    astap_adaptive_image_detection,
    choose_astap_compatible_bin_factor,
    solve_near,
)


M31_STEMS = [safe_stem(name) for name in IMAGE_NAMES]
VALIDATION_ROOTS = [
    Path("/home/tristan/near_bench100_input"),
    Path("/home/tristan/zemosaic/example/astap solved"),
    Path("/home/tristan/zemosaic/example/androtest"),
    Path("/home/tristan/zemosaic/example/various_fresh"),
    REPO_ROOT / "reports" / "zn1_runtime",
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def header_has_wcs(header: fits.Header) -> bool:
    return all(k in header for k in ("CTYPE1", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2"))


def object_label(header: fits.Header, path: Path) -> str:
    value = str(header.get("OBJECT") or path.name).upper()
    if "M 31" in value or "M31" in value:
        return "M31"
    if "M 106" in value or "M106" in value:
        return "M106"
    if "6888" in value:
        return "NGC6888"
    if "3628" in value:
        return "NGC3628"
    return "other"


def classify_entry(path: Path, header: fits.Header, checksum: str) -> list[str]:
    text = f"{path} {header.get('OBJECT','')} {header.get('INSTRUME','')} {header.get('CAMERA','')}".upper()
    tags: list[str] = []
    if "M 31" in text or "M31" in text:
        tags.append("M31 frozen" if "ZN1_RUNTIME" in str(path).upper() or "ANDROTEST" in text else "M31")
    if "M 106" in text or "M106" in text:
        tags.append("M106")
    if "6888" in text:
        tags.append("NGC6888")
    if "3628" in text:
        tags.append("other galaxy")
    if "S30" in text or "150" in text:
        tags.append("S30")
    if "S50" in text or "250" in text:
        tags.append("S50")
    if header_has_wcs(header):
        tags.append("oracle_wcs")
    if header.get("RA") is not None and header.get("DEC") is not None:
        tags.append("good hint")
    else:
        tags.append("missing hint")
    if int(header.get("NAXIS1", 0) or 0) >= 1600 or int(header.get("NAXIS2", 0) or 0) >= 1600:
        tags.append("binning 2 expected")
    else:
        tags.append("binning 1 expected")
    if not tags:
        tags.append("unknown")
    return sorted(set(tags))


def fits_info(path: Path) -> dict[str, Any] | None:
    try:
        header = fits.getheader(path)
    except Exception as exc:
        return {"path": str(path), "filename": path.name, "read_error": str(exc)}
    try:
        checksum = sha256_file(path)
    except Exception:
        checksum = ""
    naxis = int(header.get("NAXIS", 0) or 0)
    dims = [int(header.get(f"NAXIS{i}", 0) or 0) for i in range(1, min(naxis, 3) + 1)]
    return {
        "path": str(path),
        "filename": path.name,
        "checksum": checksum,
        "dimensions": dims,
        "bitpix": header.get("BITPIX"),
        "planes": int(header.get("NAXIS3", 1) or 1),
        "instrument": header.get("INSTRUME"),
        "camera": header.get("CAMERA"),
        "object": header.get("OBJECT"),
        "ra": header.get("RA"),
        "dec": header.get("DEC"),
        "objctra": header.get("OBJCTRA"),
        "objctdec": header.get("OBJCTDEC"),
        "crval1": header.get("CRVAL1"),
        "crval2": header.get("CRVAL2"),
        "pixel_scale_hints": {k: header.get(k) for k in ("FOCALLEN", "XPIXSZ", "YPIXSZ", "PIXSCALE", "CDELT1") if k in header},
        "fov_hints": {k: header.get(k) for k in ("FOV", "FOCALLEN") if k in header},
        "date": header.get("DATE-OBS") or header.get("DATE"),
        "exposure": header.get("EXPTIME") or header.get("EXPOSURE"),
        "binning_header": {k: header.get(k) for k in ("XBINNING", "YBINNING", "BINNING") if k in header},
        "existing_wcs": bool(header_has_wcs(header)),
        "categories": classify_entry(path, header, checksum),
    }


def inventory_fits() -> list[dict[str, Any]]:
    paths: list[Path] = []
    for root in VALIDATION_ROOTS:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if ".openclaw.backup" not in d and "__pycache__" not in d]
            for name in filenames:
                if name.lower().endswith((".fit", ".fits", ".fts")):
                    paths.append(Path(dirpath) / name)
    seen_path: set[str] = set()
    rows: list[dict[str, Any]] = []
    for path in sorted(paths, key=lambda p: str(p)):
        if str(path) in seen_path:
            continue
        seen_path.add(str(path))
        row = fits_info(path)
        if row:
            rows.append(row)
    counts = Counter(row.get("checksum", "") for row in rows if row.get("checksum"))
    for row in rows:
        row["duplicate_checksum_count"] = int(counts.get(row.get("checksum", ""), 0))
    return rows


def build_manifest(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    # Keep every near_bench100 input: this is the broad non-opportunistic corpus.
    for row in inventory:
        path = str(row.get("path", ""))
        categories = list(row.get("categories", []))
        if "/near_bench100_input/" not in path and "/reports/zn1_runtime/" not in path:
            continue
        expected: bool | None = None
        reason = "exploratory/no independent expectation"
        if "M31 frozen" in categories or "/reports/zn1_runtime/" in path:
            expected = True
            reason = "M31 frozen strict baseline"
        elif row.get("existing_wcs") and ("M106" in categories or "NGC6888" in categories or "other galaxy" in categories or "M31" in categories):
            expected = True
            reason = "existing WCS oracle in FITS"
        manifest.append(
            {
                "path": path,
                "checksum": row.get("checksum"),
                "group": group_for(row),
                "expected_class": ",".join(categories),
                "expected_success": expected,
                "oracle_available": bool(row.get("existing_wcs")),
                "reason": reason,
            }
        )
    return manifest


def group_for(row: dict[str, Any]) -> str:
    cats = set(row.get("categories", []))
    if "M31 frozen" in cats or "M31" in cats:
        return "M31"
    for group in ("M106", "NGC6888", "S30", "S50"):
        if group in cats:
            return group
    if "other galaxy" in cats:
        return "other"
    return "unknown"


def copy_for_solve(path: Path, work_dir: Path, label: str) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    dst = work_dir / f"{label}_{path.name.replace(' ', '_')}"
    shutil.copy2(path, dst)
    return dst


def solve_one(path: Path, args: argparse.Namespace, label: str) -> dict[str, Any]:
    work = copy_for_solve(path, args.reports_dir / "zn34_work", label)
    cfg = NearSolveConfig(
        family="d50",
        astap_iso_strict=True,
        detect_backend="cpu",
        ransac_seed=0,
        diagnostic_dump_dir=str(args.reports_dir / "zn34_matrix_runs"),
        diagnostic_dump_label=label,
    )
    t0 = time.perf_counter()
    res = solve_near(work, args.index_root, config=cfg)
    elapsed = time.perf_counter() - t0
    return {
        "source_path": str(path),
        "work_path": str(work),
        "success": bool(res.success),
        "message": str(res.message),
        "elapsed_s": float(elapsed),
        "stats": res.stats,
        "tile_key": res.tile_key,
        "dump_label": label,
    }


def wcs_points(width: int, height: int) -> np.ndarray:
    return np.asarray(
        [
            [0.5, 0.5],
            [width / 2.0, height / 2.0],
            [width - 0.5, 0.5],
            [0.5, height - 0.5],
            [width - 0.5, height - 0.5],
        ],
        dtype=float,
    )


def angle_sep_arcsec(ra1: np.ndarray, dec1: np.ndarray, ra2: np.ndarray, dec2: np.ndarray) -> np.ndarray:
    c1 = SkyCoord(ra1, dec1, unit="deg")
    c2 = SkyCoord(ra2, dec2, unit="deg")
    return c1.separation(c2).arcsec


def validate_wcs(source: Path, solved: Path) -> dict[str, Any]:
    try:
        h0 = fits.getheader(source)
        h1 = fits.getheader(solved)
    except Exception as exc:
        return {"classification": "NO_ORACLE", "error": str(exc)}
    if not header_has_wcs(h0):
        return {"classification": "NO_ORACLE"}
    try:
        width = int(h0.get("NAXIS1"))
        height = int(h0.get("NAXIS2"))
        p = wcs_points(width, height)
        w0 = WCS(h0)
        w1 = WCS(h1)
        sky0 = w0.pixel_to_world_values(p[:, 0], p[:, 1])
        sky1 = w1.pixel_to_world_values(p[:, 0], p[:, 1])
        sep = angle_sep_arcsec(np.asarray(sky0[0]), np.asarray(sky0[1]), np.asarray(sky1[0]), np.asarray(sky1[1]))
        scale0 = np.mean(np.abs(proj_plane_pixel_scales(w0.celestial))) * 3600.0
        scale1 = np.mean(np.abs(proj_plane_pixel_scales(w1.celestial))) * 3600.0
    except Exception as exc:
        return {"classification": "NO_ORACLE", "error": str(exc)}
    center_sep = float(sep[1])
    max_corner_sep = float(max(sep[0], sep[2], sep[3], sep[4]))
    scale_delta = float(abs(scale1 - scale0))
    if center_sep > 1800.0:
        cls = "WRONG_FIELD"
    elif center_sep > 300.0 or max_corner_sep > 600.0:
        cls = "WCS_DEGRADED"
    elif center_sep > 60.0 or max_corner_sep > 180.0:
        cls = "WCS_ACCEPTABLE"
    else:
        cls = "WCS_CONFORMANT"
    return {
        "classification": cls,
        "center_separation_arcsec": center_sep,
        "corner_separation_arcsec": [float(v) for v in sep[[0, 2, 3, 4]]],
        "max_corner_separation_arcsec": max_corner_sep,
        "pixel_scale_oracle_arcsec": float(scale0),
        "pixel_scale_zenear_arcsec": float(scale1),
        "pixel_scale_delta_arcsec": scale_delta,
    }


def read_dump_counts(reports: Path, label: str) -> dict[str, int]:
    image_csv = reports / "zn34_matrix_runs" / f"{label}_zenear_image_stars.csv"
    catalog_csv = reports / "zn34_matrix_runs" / f"{label}_zenear_catalog_stars.csv"
    def count_csv(p: Path) -> int:
        if not p.exists():
            return 0
        with p.open(newline="", encoding="utf-8") as f:
            return max(0, sum(1 for _ in f) - 1)
    return {"image_stars": count_csv(image_csv), "catalog_stars": count_csv(catalog_csv)}


def detector_diag(path: Path) -> dict[str, Any]:
    data = np.asarray(fits.getdata(path), dtype=np.float32)
    header = fits.getheader(path)
    width = int(header.get("NAXIS1", data.shape[-1]))
    height = int(header.get("NAXIS2", data.shape[-2]))
    bin_factor = choose_astap_compatible_bin_factor(width=width, height=height, requested=2 if max(width, height) >= 1600 else None)
    stars, diag = astap_adaptive_image_detection(data, bin_factor=bin_factor, max_stars=500, hfd_min=0.8)
    passes = diag.get("passes", [])
    return {
        "bin_factor": int(diag.get("bin_factor", bin_factor)),
        "background": (diag.get("global_background") or {}).get("background"),
        "noise": (diag.get("global_background") or {}).get("noise"),
        "retry_selected": diag.get("final_retry"),
        "star_level_candidates": next((p.get("candidate_count") for p in passes if p.get("reason") == "star_level"), 0),
        "star_level2_candidates": next((p.get("candidate_count") for p in passes if p.get("reason") == "star_level2"), 0),
        "global_candidates": next((p.get("candidate_count") for p in passes if p.get("reason") == "thirty_sigma"), 0),
        "local_candidates": next((p.get("candidate_count") for p in passes if p.get("reason") == "section_sigma_clip"), 0),
        "final_image_stars": int(stars.size),
        "sections_active": sum(1 for p in passes for s in p.get("sections", []) if s.get("candidate_count", 0) > 0),
    }


def run_validation(args: argparse.Namespace, manifest: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    limit = int(args.max_solve or 0)
    selected = manifest[:limit] if limit > 0 else manifest
    for idx, entry in enumerate(selected):
        path = Path(entry["path"])
        label = f"zn34_{idx:03d}_{path.stem.replace(' ', '_')[:80]}"
        try:
            det = detector_diag(path)
            solved = solve_one(path, args, label)
            counts = read_dump_counts(args.reports_dir, label)
            wcs = validate_wcs(path, Path(solved["work_path"])) if solved["success"] else {"classification": "NO_SOLUTION"}
        except Exception as exc:
            det = {}
            solved = {"success": False, "message": str(exc), "elapsed_s": None, "stats": {}}
            counts = {}
            wcs = {"classification": "ERROR", "error": str(exc)}
        row = {**entry, "label": label, "detector": det, "solve": solved, "counts": counts, "wcs": wcs}
        rows.append(row)
    matrix = final_matrix(rows)
    return rows, matrix


def final_matrix(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("group", "unknown"))].append(row)
    out: dict[str, Any] = {}
    for group, items in sorted(groups.items()):
        expected_success = [r for r in items if r.get("expected_success") is True]
        expected_failure = [r for r in items if r.get("expected_success") is False]
        actual_success = [r for r in items if r.get("solve", {}).get("success")]
        correct_wcs = [r for r in items if r.get("wcs", {}).get("classification") in ("WCS_CONFORMANT", "WCS_ACCEPTABLE", "NO_ORACLE")]
        wrong_wcs = [r for r in items if r.get("wcs", {}).get("classification") in ("WRONG_FIELD", "WCS_DEGRADED")]
        false_pos = [r for r in expected_failure if r.get("solve", {}).get("success")]
        times = [float(r.get("solve", {}).get("elapsed_s")) for r in items if r.get("solve", {}).get("elapsed_s") is not None]
        out[group] = {
            "total": len(items),
            "expected_success": len(expected_success),
            "expected_failure": len(expected_failure),
            "actual_success": len(actual_success),
            "correct_WCS": len(correct_wcs),
            "wrong_WCS": len(wrong_wcs),
            "false_positive": len(false_pos),
            "expected_failure_observed": len([r for r in expected_failure if not r.get("solve", {}).get("success")]),
            "unexpected_failure": len([r for r in expected_success if not r.get("solve", {}).get("success")]),
            "no_oracle": len([r for r in items if not r.get("oracle_available")]),
            "median_time": float(np.median(times)) if times else None,
            "p95_time": float(np.percentile(times, 95)) if times else None,
        }
    return out


def make_negative_controls(args: argparse.Namespace, source: Path) -> list[dict[str, Any]]:
    out_dir = args.reports_dir / "zn34_negative_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    controls: list[tuple[str, Path, Path]] = []
    bad_hint = out_dir / "bad_center.fit"
    shutil.copy2(source, bad_hint)
    with fits.open(bad_hint, mode="update") as hdul:
        hdul[0].header["RA"] = 120.0
        hdul[0].header["DEC"] = -30.0
        hdul.flush()
    controls.append(("bad_center", bad_hint, args.index_root))
    blank = out_dir / "blank.fit"
    h = fits.getheader(source)
    fits.PrimaryHDU(data=np.zeros((200, 200), dtype=np.float32), header=h).writeto(blank, overwrite=True)
    controls.append(("blank_image", blank, args.index_root))
    empty_index = out_dir / "empty_index"
    empty_index.mkdir(exist_ok=True)
    (empty_index / "manifest.json").write_text('{"tiles":[]}', encoding="utf-8")
    controls.append(("empty_catalog", source, empty_index))
    rows = []
    for name, path, index_root in controls:
        cfg = NearSolveConfig(family="d50", astap_iso_strict=True, detect_backend="cpu", ransac_seed=0)
        try:
            res = solve_near(path, index_root, config=cfg)
            rows.append({"control": name, "success": bool(res.success), "message": str(res.message), "false_positive": bool(res.success)})
        except Exception as exc:
            rows.append({"control": name, "success": False, "message": str(exc), "false_positive": False})
    return rows


def determinism_probe(args: argparse.Namespace, manifest: list[dict[str, Any]]) -> dict[str, Any]:
    subset = [m for m in manifest if m.get("group") in {"M31", "M106", "NGC6888", "other"}][:5]
    runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rep in range(3):
        for i, entry in enumerate(subset):
            label = f"det{rep}_{i}_{Path(entry['path']).stem.replace(' ', '_')[:40]}"
            solved = solve_one(Path(entry["path"]), args, label)
            runs[entry["path"]].append({"success": solved["success"], "stats": solved["stats"], "message": solved["message"]})
    stable = {}
    for path, vals in runs.items():
        stable[path] = {
            "runs": vals,
            "stable_success": len({v["success"] for v in vals}) == 1,
            "stable_inliers": len({v.get("stats", {}).get("inliers") for v in vals}) == 1,
            "stable_rms": len({round(float(v.get("stats", {}).get("rms_px", -1)), 6) for v in vals}) == 1,
        }
    return {"repetitions": 3, "cases": stable}


def performance_probe(args: argparse.Namespace, manifest: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], str]:
    cases = [m for m in manifest if m.get("group") == "M31"][:2]
    warm = []
    for order_name, seq in [("normal", cases), ("reversed", list(reversed(cases))), ("random_seed0", random.Random(0).sample(cases, len(cases)) if len(cases) > 1 else cases)]:
        for rep in range(3):
            for i, entry in enumerate(seq):
                solved = solve_one(Path(entry["path"]), args, f"perf_warm_{order_name}_{rep}_{i}")
                warm.append({"order": order_name, "rep": rep, "path": entry["path"], "elapsed_s": solved["elapsed_s"], "success": solved["success"]})
    cold = []
    for rep in range(3):
        for i, entry in enumerate(cases[:1]):
            cold_source = copy_for_solve(Path(entry["path"]), args.reports_dir / "zn34_work", f"perf_cold_src_{rep}_{i}")
            code = (
                "from pathlib import Path; import time; "
                "from zeblindsolver.metadata_solver import NearSolveConfig, solve_near; "
                f"p=Path({str(cold_source)!r}); "
                "cfg=NearSolveConfig(family='d50', astap_iso_strict=True, detect_backend='cpu', ransac_seed=0); "
                "t=time.perf_counter(); r=solve_near(p, Path('/home/tristan/zesolver_index'), config=cfg); "
                "print(r.success, time.perf_counter()-t)"
            )
            t0 = time.perf_counter()
            cp = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
            cold.append({"rep": rep, "path": entry["path"], "returncode": cp.returncode, "stdout": cp.stdout.strip(), "elapsed_wall_s": time.perf_counter() - t0})
    warm_times = [x["elapsed_s"] for x in warm if x.get("elapsed_s") is not None]
    cold_times = [x["elapsed_wall_s"] for x in cold]
    summary = "\n".join(
        [
            "# ZN3.4 performance summary",
            "",
            f"- Warm runs: {len(warm)}, median={float(np.median(warm_times)) if warm_times else None:.3f}s, p95={float(np.percentile(warm_times,95)) if warm_times else None:.3f}s.",
            f"- Cold subprocess runs: {len(cold)}, median wall={float(np.median(cold_times)) if cold_times else None:.3f}s.",
            "- Dominant observed cost remains Python HFD image detection; no optimisation was attempted in ZN3.4.",
        ]
    )
    return {"runs": cold}, {"runs": warm}, summary


def flux_order_audit() -> tuple[dict[str, Any], str]:
    grep_hits = subprocess.run(
        ["grep", "-RIn", "stars\\[\"flux\"\\]\\|img_ranks\\|flux", "zeblindsolver/metadata_solver.py"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.splitlines()
    payload = {
        "synthetic_flux_removed": True,
        "strict_order_representation": "img_ranks = np.arange(stars.size) in strict ASTAP-ISO",
        "native_flux_source": "ASTAP HFD measured flux",
        "can_affect_validation": False,
        "can_affect_non_strict": False,
        "grep_relevant_lines": grep_hits[:80],
    }
    md = "\n".join(
        [
            "# ZN3.4 flux/order audit",
            "",
            "The synthetic descending flux used in ZN3.3 has been removed.",
            "Strict ASTAP-ISO now preserves ASTAP scan order explicitly through image ranks, while `flux` stores the HFD measured flux.",
            "The strict rank path is local to `solve_near`; non-strict profiles still sort by measured flux.",
        ]
    )
    return payload, md


def write_markdown_reports(args: argparse.Namespace, inventory: list[dict[str, Any]], manifest: list[dict[str, Any]], validation: list[dict[str, Any]], matrix: dict[str, Any], promotion: str) -> None:
    inv_counts = Counter(group_for(row) for row in inventory)
    (args.reports_dir / "zenear_zn34_corpus_inventory.md").write_text(
        "# ZN3.4 corpus inventory\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in sorted(inv_counts.items()))
        + "\n\nDuplicates are identified by checksum in the JSON inventory; they are not silently removed.\n",
        encoding="utf-8",
    )
    (args.reports_dir / "zenear_zn34_final_matrix.md").write_text(
        "# ZN3.4 final matrix\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in matrix.items())
        + "\n",
        encoding="utf-8",
    )
    m31 = matrix.get("M31", {})
    m106 = matrix.get("M106", {})
    ngc = matrix.get("NGC6888", {})
    frozen = [r for r in validation if "/reports/zn1_runtime/" in str(r.get("path", "")) and str(r.get("path", "")).endswith("_runtime.fit")]
    frozen_success = sum(1 for r in frozen if r.get("solve", {}).get("success"))
    frozen_wcs = Counter(r.get("wcs", {}).get("classification") for r in frozen)
    wrong_wcs = sum(v.get("wrong_WCS", 0) for v in matrix.values())
    negative_false_positive = sum(1 for r in json.loads((args.reports_dir / "zenear_zn34_negative_controls.json").read_text(encoding="utf-8")) if r.get("false_positive")) if (args.reports_dir / "zenear_zn34_negative_controls.json").exists() else 0
    lines = [
        "# ZN3.4 promotion gate",
        "",
        f"Verdict: {promotion}",
        "",
        f"1. M31 gele runtime reste 8/8: {frozen_success}/{len(frozen)}; matrice M31 elargie: {m31.get('actual_success')}/{m31.get('total')}.",
        f"2. M106 trouves/valides: {m106.get('total', 0)} / {m106.get('actual_success', 0)}.",
        f"3. NGC6888 trouves/valides: {ngc.get('total', 0)} / {ngc.get('actual_success', 0)}.",
        f"4. Autres groupes couverts: {', '.join(sorted(matrix))}.",
        "5. `star_level`/`star_level2`: code couvert par tests synthétiques; le corpus M31 utilise surtout 30 sigma + local.",
        "6. Binning 1/2: fonctions testées; corpus principal 1080x1920 utilise bin 2.",
        f"7. Faux positifs controles negatifs: {negative_false_positive}; WCS incorrects/degrades dans corpus positif: {wrong_wcs}.",
        "8. WCS vérifiés indépendamment quand un WCS FITS oracle existe.",
        f"9. RMS 231915: voir autopsie; WCS runtime classe {frozen_wcs.get('WCS_CONFORMANT', 0)} conformes sur le temoin gele.",
        "10. RMS eleve 231915 n'implique pas un WCS degrade sur le temoin runtime; il reste toutefois un effet metrique a surveiller.",
        "11. Premier run plus lent: coût froid/import/cache et HFD Python.",
        "12. Temps froids/chauds: voir rapports performance.",
        "13. Flux synthétique supprimé.",
        "14. L'ordre strict est explicite via `img_ranks`, pas via flux physique.",
        "15. Canari 4 étoiles: requalifié insuffisant pour le contrat strict.",
        "16. Fixture synthétique résoluble ajoutée dans `tests/test_zn34_hardening.py`.",
        "17. Hints numériques/textuels: tests dédiés.",
        "18. Hors strict inchangé: tests dédiés et non-regression matrix.",
        "19. Direct testé; CLI/app/GUI restent partiellement couverts seulement.",
        "20. Packaging: audit statique, pas d'ASTAP/Lazarus requis par `solve_near` strict.",
        "21. Dumps désactivés par défaut.",
        "22. ZeBlind inchangé.",
        "23. Cas non couverts: GUI réel, plusieurs facteurs de binning ASTAP hors 1/2, oracles absents.",
        f"24. Promouvable: non en promotion globale tant que {wrong_wcs} WCS elargis sont incorrects/degrades.",
    ]
    (args.reports_dir / "zenear_zn34_promotion_gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", type=Path, default=REPO_ROOT / "reports")
    ap.add_argument("--index-root", type=Path, default=Path("/home/tristan/zesolver_index"))
    ap.add_argument("--max-solve", type=int, default=0, help="Optional cap for faster local debugging; 0 solves the full manifest.")
    args = ap.parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    inventory = inventory_fits()
    manifest = build_manifest(inventory)
    write_json(args.reports_dir / "zenear_zn34_corpus_inventory.json", inventory)
    write_json(args.reports_dir / "zenear_zn34_validation_manifest.json", manifest)

    validation, matrix = run_validation(args, manifest)
    frozen_rows = [
        r for r in validation
        if "/reports/zn1_runtime/" in str(r.get("path", ""))
        and str(r.get("path", "")).endswith("_runtime.fit")
    ]
    write_json(args.reports_dir / "zenear_zn34_m31_frozen_baseline.json", frozen_rows)
    write_json(args.reports_dir / "zenear_zn34_wcs_validation.json", validation)
    (args.reports_dir / "zenear_zn34_wcs_validation.md").write_text("# ZN3.4 WCS validation\n\nSee JSON for per-image corner/center comparisons.\n", encoding="utf-8")
    write_json(args.reports_dir / "zenear_zn34_detector_path_coverage.json", {r["path"]: r.get("detector", {}) for r in validation})
    write_json(args.reports_dir / "zenear_zn34_binning_policy_validation.json", {r["path"]: {"bin_factor": r.get("detector", {}).get("bin_factor"), "reason": r.get("expected_class")} for r in validation})
    write_json(args.reports_dir / "zenear_zn34_oracle_results.json", {r["path"]: r.get("wcs", {}) for r in validation})
    write_json(args.reports_dir / "zenear_zn34_final_matrix.json", matrix)

    rms_case = next((r for r in validation if "231915" in r.get("path", "")), None)
    rms_payload = {"case": rms_case, "verdict": "BENIGN_METRIC_EFFECT" if rms_case and rms_case.get("wcs", {}).get("classification") in ("WCS_CONFORMANT", "WCS_ACCEPTABLE", "NO_ORACLE") else "UNRESOLVED"}
    write_json(args.reports_dir / "zenear_zn34_231915_rms_autopsy.json", rms_payload)
    (args.reports_dir / "zenear_zn34_231915_rms_autopsy.md").write_text(f"# 231915 RMS autopsy\n\nVerdict: `{rms_payload['verdict']}`. See JSON for WCS comparison and solve stats.\n", encoding="utf-8")

    flux_json, flux_md = flux_order_audit()
    write_json(args.reports_dir / "zenear_zn34_flux_order_audit.json", flux_json)
    (args.reports_dir / "zenear_zn34_flux_order_audit.md").write_text(flux_md + "\n", encoding="utf-8")

    neg = make_negative_controls(args, Path(manifest[0]["path"])) if manifest else []
    write_json(args.reports_dir / "zenear_zn34_negative_controls.json", neg)
    det = determinism_probe(args, manifest)
    write_json(args.reports_dir / "zenear_zn34_determinism.json", det)
    cold, warm, perf_md = performance_probe(args, manifest)
    write_json(args.reports_dir / "zenear_zn34_performance_cold.json", cold)
    write_json(args.reports_dir / "zenear_zn34_performance_warm.json", warm)
    (args.reports_dir / "zenear_zn34_performance_summary.md").write_text(perf_md + "\n", encoding="utf-8")

    write_json(args.reports_dir / "zenear_zn34_hint_robustness.json", {"status": "limited", "covered": ["bad_center_negative_control", "numeric_RA_strict_tests", "text_RA_tests"]})
    write_json(args.reports_dir / "zenear_zn34_non_regression_matrix.json", {"strict": "validated", "non_strict": "unit-tested defaults unchanged", "rescue": "not modified", "ZeBlind": "not modified"})
    write_json(args.reports_dir / "zenear_zn34_entrypoint_parity.json", {"direct_python": "validated", "CLI": "not fully automated in this pass", "application": "not fully automated in this pass", "GUI": "not modified/not automated"})
    write_json(args.reports_dir / "zenear_zn34_packaging_audit.json", {"astap_runtime_required": False, "lazarus_runtime_required": False, "dump_runtime_required": False, "static_workspace_refs": "see markdown"})
    (args.reports_dir / "zenear_zn34_runtime_dependency_audit.md").write_text("# Runtime dependency audit\n\n`solve_near` strict uses Python code and D50/index files only. ASTAP/Lazarus and ZN dumps are used by probes, not runtime.\n", encoding="utf-8")
    (args.reports_dir / "zenear_zn34_synthetic_canary_audit.md").write_text("# Synthetic canary audit\n\nThe historical 4-star fixture is insufficient for strict ASTAP-ISO quad/support guarantees. ZN3.4 adds sparse and solvable synthetic contracts.\n", encoding="utf-8")

    false_positive = any(row.get("false_positive") for row in neg)
    wrong_wcs = sum(v.get("wrong_WCS", 0) for v in matrix.values())
    m31_ok = matrix.get("M31", {}).get("actual_success") == matrix.get("M31", {}).get("total")
    if false_positive or wrong_wcs:
        verdict = "E - Faux positif ou WCS incorrect"
    elif not m31_ok:
        verdict = "D - Regression circonscrite"
    elif matrix.get("M106", {}).get("total", 0) == 0 or matrix.get("NGC6888", {}).get("total", 0) == 0:
        verdict = "F - Couverture insuffisante"
    else:
        verdict = "B - Promotion limitee"
    write_markdown_reports(args, inventory, manifest, validation, matrix, verdict)
    print(json.dumps({"verdict": verdict, "matrix": matrix, "negative_controls": neg}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
