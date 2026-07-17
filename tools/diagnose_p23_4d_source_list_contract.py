#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zeblindsolver.asterisms import sample_quads
from zeblindsolver.image_prep import build_pyramid, downsample_image, read_fits_as_luma, remove_background
from zeblindsolver.quad_code_diagnostic import build_astrometry_quad_records
from zeblindsolver.quad_index_4d import ASTROMETRY_AB_CODE_4D_SCHEMA, Quad4DIndex
from zeblindsolver.star_detect import detect_stars
from zeblindsolver.verify import validate_solution
from zeblindsolver.zeblindsolver import (
    SolveConfig,
    _astrometry_4d_build_matches,
    _astrometry_4d_final_thresholds,
    _astrometry_source_list_gate,
    _blind_geometric_guardrails,
    _filter_blind_input_stars,
    _fit_astrometry_4d_quad_wcs,
)


DEFAULT_DATA_DIR = ROOT / "reports/eq_ircut_cleanbench_20260518_230249/data"
DEFAULT_INDEX_ROOT = ROOT / "reports/s3_focused_index_20260701_p16_multicase_v3/index"
DEFAULT_4D_INDEX = ROOT / "reports/p21_astrometry_ab_code_4d_v1_d50_2823_S_stars2000_q40000.npz"
DEFAULT_REPORT = ROOT / "reports/zeblind_p23_4d_source_list_contract.md"
DEFAULT_JSON = ROOT / "reports/zeblind_p23_4d_source_list_contract.json"
DEFAULT_CASES = {
    "232350": ("Light_mosaic_M 106_20.0s_IRCUT_20250518-232350.fit", "d50_2823"),
    "232102": ("Light_mosaic_M 106_20.0s_IRCUT_20250518-232102.fit", "d50_2823"),
}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _strip_wcs(path: Path) -> None:
    wcs_prefixes = ("CD", "PC", "PV", "A_", "B_", "AP_", "BP_")
    wcs_keys = {
        "WCSAXES",
        "CTYPE1",
        "CTYPE2",
        "CRVAL1",
        "CRVAL2",
        "CRPIX1",
        "CRPIX2",
        "CUNIT1",
        "CUNIT2",
        "CDELT1",
        "CDELT2",
        "CROTA1",
        "CROTA2",
        "LONPOLE",
        "LATPOLE",
        "RADESYS",
        "EQUINOX",
        "SOLVED",
        "RMSPX",
        "INLIERS",
        "PIXSCAL",
        "SIPORD",
        "QUALITY",
        "USED_DB",
        "SOLVER",
        "SOLVMODE",
        "TILE_ID",
        "DBSET",
    }
    with fits.open(path, mode="update", memmap=False) as hdul:
        header = hdul[0].header
        for key in list(header.keys()):
            if key in wcs_keys or any(str(key).startswith(prefix) for prefix in wcs_prefixes):
                header.remove(key, ignore_missing=True, remove_all=True)
        hdul.flush()


def _positions(stars: np.ndarray) -> np.ndarray:
    if stars.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.column_stack((stars["x"].astype(np.float64), stars["y"].astype(np.float64)))


def _detect_runtime_stars(path: Path, args: argparse.Namespace) -> tuple[np.ndarray, tuple[int, int], dict[str, Any]]:
    image = read_fits_as_luma(path)
    downsample_factor = max(1, min(4, int(args.downsample or 1)))
    work = image
    if downsample_factor > 1:
        work = downsample_image(work, downsample_factor)
    kernel = max(5, int(round(15 / downsample_factor)))
    work = remove_background(work, kernel_size=kernel)
    detection = build_pyramid(work)[-1]
    stars = detect_stars(
        detection,
        min_fwhm_px=max(1.0, 1.5 / downsample_factor),
        max_fwhm_px=max(2.5, 8.0 / downsample_factor),
        k_sigma=max(0.5, float(args.detect_k_sigma)),
        min_area=max(1, int(args.detect_min_area)),
        backend="cpu",
    )
    if downsample_factor > 1 and stars.size:
        stars = stars.copy()
        stars["x"] *= downsample_factor
        stars["y"] *= downsample_factor
    return stars, (int(image.shape[0]), int(image.shape[1])), {
        "raw_detected": int(stars.shape[0]),
        "downsample": int(downsample_factor),
        "detect_k_sigma": float(args.detect_k_sigma),
        "detect_min_area": int(args.detect_min_area),
    }


def _cap(stars: np.ndarray, max_stars: int) -> np.ndarray:
    if max_stars > 0 and stars.shape[0] > max_stars:
        return stars[: int(max_stars)]
    return stars


def _raw_ranks(raw: np.ndarray, stars: np.ndarray) -> list[int | None]:
    if raw.size == 0 or stars.size == 0:
        return []
    tree = cKDTree(_positions(raw))
    used: set[int] = set()
    ranks: list[int | None] = []
    for xy in _positions(stars):
        dist, idx = tree.query(xy, k=1, distance_upper_bound=1.0e-3)
        if np.isfinite(float(dist)) and int(idx) < raw.shape[0] and int(idx) not in used:
            used.add(int(idx))
            ranks.append(int(idx))
        else:
            ranks.append(None)
    return ranks


def _standard_qscore(raw: np.ndarray) -> np.ndarray:
    if raw.size == 0:
        return np.zeros(0, dtype=np.float64)
    fwhm = np.asarray(raw["fwhm"], dtype=np.float64)
    flux = np.asarray(raw["flux"], dtype=np.float64)
    med_f = float(np.median(fwhm)) if fwhm.size else 2.0
    mad_f = float(np.median(np.abs(fwhm - med_f))) if fwhm.size else 0.2
    xs = np.asarray(raw["x"], dtype=np.float64)
    ys = np.asarray(raw["y"], dtype=np.float64)
    dens_cell = max(1.0, 1.8 * max(0.8, med_f))
    cx = np.floor(xs / dens_cell).astype(np.int64)
    cy = np.floor(ys / dens_cell).astype(np.int64)
    cell_occ: dict[tuple[int, int], int] = {}
    for ii in range(raw.shape[0]):
        key = (int(cx[ii]), int(cy[ii]))
        cell_occ[key] = int(cell_occ.get(key, 0)) + 1
    density = np.asarray([cell_occ.get((int(cx[ii]), int(cy[ii])), 1) for ii in range(raw.shape[0])], dtype=np.float64)
    return (
        np.log1p(np.maximum(0.0, flux))
        - 0.35 * (np.abs(fwhm - med_f) / max(0.2, mad_f))
        - 0.22 * np.log1p(np.maximum(0.0, density - 1.0))
    )


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "min": None, "p10": None, "median": None, "p90": None, "max": None}
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def _spatial_summary(stars: np.ndarray, image_shape: tuple[int, int]) -> dict[str, Any]:
    if stars.size == 0:
        return {"n": 0}
    pos = _positions(stars)
    h, w = image_shape
    x = pos[:, 0]
    y = pos[:, 1]
    hist, _ye, _xe = np.histogram2d(
        y,
        x,
        bins=(4, 4),
        range=((0.0, float(h)), (0.0, float(w))),
    )
    if pos.shape[0] >= 2:
        tree = cKDTree(pos)
        dists, _idx = tree.query(pos, k=min(2, pos.shape[0]))
        nn = dists[:, 1] if dists.ndim == 2 and dists.shape[1] > 1 else np.zeros(0)
    else:
        nn = np.zeros(0)
    return {
        "n": int(stars.shape[0]),
        "bbox": [float(np.min(x)), float(np.min(y)), float(np.max(x)), float(np.max(y))],
        "centroid": [float(np.mean(x)), float(np.mean(y))],
        "grid4x4": hist.astype(int).tolist(),
        "nearest_neighbor_px": _summary(nn),
    }


def _list_summary(name: str, stars: np.ndarray, raw: np.ndarray, image_shape: tuple[int, int]) -> dict[str, Any]:
    ranks = _raw_ranks(raw, stars)
    rank_arr = np.asarray([r for r in ranks if r is not None], dtype=np.int64)
    qscore = _standard_qscore(raw)
    qvals = np.asarray([qscore[int(r)] for r in ranks if r is not None and int(r) < qscore.shape[0]], dtype=np.float64)
    return {
        "name": name,
        "kept": int(stars.shape[0]),
        "raw_ranks": ranks,
        "raw_rank_summary": _summary(rank_arr),
        "spatial": _spatial_summary(stars, image_shape),
        "flux": _summary(stars["flux"] if stars.size else np.zeros(0)),
        "fwhm": _summary(stars["fwhm"] if stars.size else np.zeros(0)),
        "snr": {"available": False, "note": "detect_stars output has no SNR field; flux and standard qscore are reported instead"},
        "standard_quality_proxy": _summary(qvals),
    }


def _make_lists(raw: np.ndarray, image_shape: tuple[int, int], args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    capped = _cap(raw, int(args.max_stars))
    standard, standard_stats = _filter_blind_input_stars(
        capped,
        image_shape=image_shape,
        min_sep_px=float(args.blind_star_min_sep_px),
    )
    standard = _cap(standard, int(args.max_stars))
    ast_like, ast_like_stats = _astrometry_source_list_gate(
        raw,
        image_shape=image_shape,
        approx_boxes=int(args.astrometry_like_boxes),
        max_sources=int(args.max_stars),
        min_keep_ratio=float(args.astrometry_like_min_keep_ratio),
    )
    return {
        "diagnostic_unfiltered": capped,
        "standard_runtime": standard,
        "astrometry_like_candidate": ast_like,
    }, {
        "initial_cap": int(capped.shape[0]),
        "standard_filter": standard_stats,
        "astrometry_like_candidate": ast_like_stats,
    }


def _quad_raw_key(local_quad: Any, ranks: list[int | None]) -> tuple[int, int, int, int] | None:
    vals: list[int] = []
    for idx in list(local_quad):
        if int(idx) < 0 or int(idx) >= len(ranks):
            return None
        rank = ranks[int(idx)]
        if rank is None:
            return None
        vals.append(int(rank))
    return tuple(vals)  # ordered AB/C/D quad, not sorted


def _evaluate_policy(
    name: str,
    stars: np.ndarray,
    raw: np.ndarray,
    image_shape: tuple[int, int],
    disk_index: Quad4DIndex,
    args: argparse.Namespace,
) -> dict[str, Any]:
    t_policy = time.perf_counter()
    ranks = _raw_ranks(raw, stars)
    image_positions = _positions(stars)
    obs_stars = np.zeros(stars.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("mag", "f4")])
    if stars.size:
        obs_stars["x"] = stars["x"]
        obs_stars["y"] = stars["y"]
        obs_stars["mag"] = -stars["flux"]

    t_quad = time.perf_counter()
    image_quads = sample_quads(obs_stars, max_quads=int(args.max_quads), strategy=str(args.image_strategy))
    image_records = build_astrometry_quad_records(image_quads, image_positions)
    quad_build_s = time.perf_counter() - t_quad
    t_lookup = time.perf_counter()
    hits = disk_index.search_records(
        image_records,
        code_tol=float(args.code_tol),
        max_hits=int(args.max_hits_4d),
        max_hits_per_image_quad=int(args.max_hits_per_image_quad),
    )
    lookup_s = time.perf_counter() - t_lookup
    thresholds = _astrometry_4d_final_thresholds(
        SolveConfig(
            quality_rms=float(args.quality_rms),
            quality_inliers=int(args.quality_inliers),
            pixel_scale_min_arcsec=1.79 if args.m106_hints else None,
            pixel_scale_max_arcsec=2.99 if args.m106_hints else None,
        ),
        scale_bounds_arcsec=(1.79, 2.99) if args.m106_hints else None,
    )
    reason_counts: Counter[str] = Counter()
    first_plausible: dict[str, Any] | None = None
    first_accepted: dict[str, Any] | None = None
    best_reject: dict[str, Any] | None = None
    useful_rows: list[dict[str, Any]] = []
    useful_quad_raw_keys: set[tuple[int, int, int, int]] = set()
    all_record_raw_keys: set[tuple[int, int, int, int]] = set()
    tested = 0
    t_val = time.perf_counter()
    for rank, hit in enumerate(hits, start=1):
        if tested >= int(args.max_hypotheses):
            reason_counts["max_hypotheses"] += 1
            break
        tested += 1
        image_record = image_records[hit.image_record_index]
        raw_key = _quad_raw_key(hit.image_quad_indices, ranks)
        if raw_key is not None:
            all_record_raw_keys.add(raw_key)
        try:
            image_quad_points = image_positions[np.asarray(hit.image_quad_indices, dtype=np.int64)]
            catalog_quad_world = disk_index.catalog_ra_dec[np.asarray(hit.catalog_quad_indices, dtype=np.int64)]
            wcs = _fit_astrometry_4d_quad_wcs(image_quad_points, catalog_quad_world)
            matches_array, matched_image_points = _astrometry_4d_build_matches(
                wcs,
                image_positions,
                disk_index.catalog_ra_dec,
                radius_px=float(args.match_radius_px),
            )
            validation = dict(validate_solution(wcs, matches_array, thresholds))
            geo_ok, geo = _blind_geometric_guardrails(
                matched_image_points,
                image_shape,
                sparse_min_span=0.08,
                sparse_min_area=0.003,
                dense_min_span=0.10,
                dense_min_area=0.005,
                dense_max_cond=2.0e4,
            )
            if validation.get("quality") == "GOOD" and first_plausible is None:
                first_plausible = {"rank": int(rank), "raw_quad": raw_key, "local_quad": list(hit.image_quad_indices), **validation}
            accepted = bool(validation.get("quality") == "GOOD" and geo_ok)
            if validation.get("quality") == "GOOD" and not geo_ok:
                validation["quality"] = "FAIL"
                validation["success"] = False
                validation["reason"] = f"geometric_guard_failed[{geo.get('reason', 'unknown')}]"
            reason = str(validation.get("reason", None) or ("accepted" if accepted else "validation_failed"))
            row = {
                "rank": int(rank),
                "image_record_index": int(hit.image_record_index),
                "catalog_record_index": int(hit.catalog_record_index),
                "code_distance": float(hit.code_distance),
                "local_quad": list(hit.image_quad_indices),
                "raw_quad": list(raw_key) if raw_key is not None else None,
                "catalog_quad": list(hit.catalog_quad_indices),
                "inliers": int(validation.get("inliers", 0) or 0),
                "rms_px": float(validation.get("rms_px", float("nan"))),
                "pix_scale_arcsec": float(validation.get("pix_scale_arcsec", float("nan"))),
                "quality": validation.get("quality"),
                "reason": reason,
                "geo_ok": bool(geo_ok),
            }
            if validation.get("quality") == "GOOD" or accepted:
                useful_rows.append(dict(row))
                if raw_key is not None:
                    useful_quad_raw_keys.add(raw_key)
            if accepted:
                first_accepted = dict(row)
                reason_counts["accepted"] += 1
                break
            reason_counts[reason] += 1
            if best_reject is None or (
                int(row.get("inliers", 0)),
                -float(row.get("rms_px", float("inf")) if np.isfinite(float(row.get("rms_px", float("inf")))) else 1e9),
            ) > (
                int(best_reject.get("inliers", 0) or 0),
                -float(best_reject.get("rms_px", float("inf")) if np.isfinite(float(best_reject.get("rms_px", float("inf")))) else 1e9),
            ):
                best_reject = dict(row)
        except Exception as exc:
            reason_counts["fit_or_validate_failed"] += 1
            if best_reject is None:
                best_reject = {"rank": int(rank), "reason": str(exc), "inliers": 0, "rms_px": float("inf")}
    validation_s = time.perf_counter() - t_val
    t_summary = time.perf_counter()
    source_summary = _list_summary(name, stars, raw, image_shape)
    source_summary_s = time.perf_counter() - t_summary
    total_s = time.perf_counter() - t_policy
    return {
        "name": name,
        "source": source_summary,
        "quads_4d_generated": int(image_quads.shape[0]),
        "records_4d_generated": int(len(image_records)),
        "hits_4d": int(len(hits)),
        "hits_tested": int(tested),
        "first_plausible": first_plausible,
        "first_accepted": first_accepted,
        "reason_counts": dict(reason_counts),
        "best_reject": best_reject,
        "useful_hits": useful_rows[: int(args.max_useful_examples)],
        "useful_raw_quads": [list(v) for v in sorted(useful_quad_raw_keys)],
        "all_record_raw_quads": [list(v) for v in sorted(all_record_raw_keys)],
        "timing": {
            "total_s": float(total_s),
            "source_summary_s": float(source_summary_s),
            "quad_build_s": float(quad_build_s),
            "lookup_s": float(lookup_s),
            "validation_s": float(validation_s),
        },
    }


def _diff_lists(raw: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    ar = _raw_ranks(raw, a)
    br = _raw_ranks(raw, b)
    aset = {int(v) for v in ar if v is not None}
    bset = {int(v) for v in br if v is not None}
    common = sorted(aset & bset)
    removed = sorted(aset - bset)
    added = sorted(bset - aset)
    moved = []
    bpos = {int(v): i for i, v in enumerate(br) if v is not None}
    for i, rank in enumerate(ar):
        if rank is not None and int(rank) in bpos and bpos[int(rank)] != i:
            moved.append({"raw_rank": int(rank), "a_rank": int(i), "b_rank": int(bpos[int(rank)])})
    return {
        "common_count": int(len(common)),
        "common_raw_ranks": common,
        "removed_by_standard_raw_ranks": removed,
        "added_or_replacement_raw_ranks": added,
        "moved_common_count": int(len(moved)),
        "moved_common_examples": moved[:40],
    }


def _case_cause(case: dict[str, Any]) -> dict[str, Any]:
    diag = case["policies"]["diagnostic_unfiltered"]
    standard = case["policies"]["standard_runtime"]
    diff = case["diffs"]["diagnostic_vs_standard"]
    removed = set(int(v) for v in diff["removed_by_standard_raw_ranks"])
    standard_quads = {tuple(int(x) for x in row) for row in standard.get("all_record_raw_quads", [])}
    useful_missing = []
    useful_using_removed = []
    for hit in diag.get("useful_hits", []):
        raw_quad = hit.get("raw_quad")
        if not raw_quad:
            continue
        key = tuple(int(v) for v in raw_quad)
        row = {"rank": int(hit["rank"]), "raw_quad": list(key), "inliers": hit.get("inliers"), "rms_px": hit.get("rms_px")}
        if key not in standard_quads:
            useful_missing.append(row)
        if any(int(v) in removed for v in key):
            useful_using_removed.append(row)
    accepted = diag.get("first_accepted")
    accepted_uses_removed = False
    accepted_missing = False
    if accepted and accepted.get("raw_quad"):
        key = tuple(int(v) for v in accepted["raw_quad"])
        accepted_uses_removed = any(int(v) in removed for v in key)
        accepted_missing = key not in standard_quads
    return {
        "standard_success": standard.get("first_accepted") is not None,
        "diagnostic_success": diag.get("first_accepted") is not None,
        "diagnostic_accepted_uses_removed_star": bool(accepted_uses_removed),
        "diagnostic_accepted_quad_absent_from_standard_quads": bool(accepted_missing),
        "useful_hits_missing_from_standard_quads": useful_missing[:20],
        "useful_hits_using_removed_stars": useful_using_removed[:20],
        "clear": bool(
            diag.get("first_accepted") is not None
            and standard.get("first_accepted") is None
            and (accepted_uses_removed or accepted_missing or len(useful_using_removed) > 0)
        ),
    }


def _run_case(label: str, filename: str, tile_key: str, disk_index: Quad4DIndex, args: argparse.Namespace) -> dict[str, Any]:
    source = args.data_dir.expanduser().resolve() / filename
    raw, image_shape, detect_meta = _detect_runtime_stars(source, args)
    lists, list_build_stats = _make_lists(raw, image_shape, args)
    policies: dict[str, Any] = {}
    for name, stars in lists.items():
        policies[name] = _evaluate_policy(name, stars, raw, image_shape, disk_index, args)
    diffs = {
        "diagnostic_vs_standard": _diff_lists(raw, lists["diagnostic_unfiltered"], lists["standard_runtime"]),
        "diagnostic_vs_astrometry_like_candidate": _diff_lists(raw, lists["diagnostic_unfiltered"], lists["astrometry_like_candidate"]),
    }
    case = {
        "label": label,
        "filename": filename,
        "tile_key": tile_key,
        "source_fits": str(source),
        "image_shape": list(image_shape),
        "detect_meta": detect_meta,
        "list_build_stats": list_build_stats,
        "policies": policies,
        "diffs": diffs,
    }
    case["cause"] = _case_cause(case)
    return case


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# ZeBlind P2.3 - contrat source-list backend 4D",
        "",
        "## Verdict",
        "",
        f"- Verdict global: **{payload['global_verdict']}**",
        f"- Index 4D audite: `{payload['params']['quad4d_index']}`",
        "- Aucun elargissement d'index, aucun all30, aucun rebuild complet, aucun changement de seuil.",
        "- `astrometry_like_candidate` est un audit C seulement; la politique implementee/proposee reste separee.",
        "",
        "## Synthese A/B",
        "",
        "| cas | A gardees | B gardees | communes | supprimees par B | hits A/B | accepte A/B | cause courte |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for case in payload["cases"]:
        diff = case["diffs"]["diagnostic_vs_standard"]
        a = case["policies"]["diagnostic_unfiltered"]
        b = case["policies"]["standard_runtime"]
        cause = case["cause"]
        short = "quad accepte A absent/retire par B" if cause["clear"] else "cause non fermee"
        lines.append(
            "| `{}` | {} | {} | {} | {} | {}/{} | {}/{} | {} |".format(
                case["label"],
                a["source"]["kept"],
                b["source"]["kept"],
                diff["common_count"],
                len(diff["removed_by_standard_raw_ranks"]),
                a["hits_4d"],
                b["hits_4d"],
                a["first_accepted"]["rank"] if a["first_accepted"] else "",
                b["first_accepted"]["rank"] if b["first_accepted"] else "",
                short,
            )
        )
    lines.extend(["", "## Details par cas", ""])
    for case in payload["cases"]:
        lines.extend([f"### {case['label']} / {case['tile_key']}", ""])
        lines.append(f"- etoiles brutes detectees: `{case['detect_meta']['raw_detected']}` ; cap initial: `{case['list_build_stats']['initial_cap']}`")
        diff = case["diffs"]["diagnostic_vs_standard"]
        lines.append(
            "- A/B sources: communes `{}` ; supprimees par B `{}` ; ajoutees/remplacees `{}` ; etoiles communes deplacees `{}`".format(
                diff["common_count"],
                diff["removed_by_standard_raw_ranks"],
                diff["added_or_replacement_raw_ranks"],
                diff["moved_common_count"],
            )
        )
        for name in ("diagnostic_unfiltered", "standard_runtime", "astrometry_like_candidate"):
            pol = case["policies"][name]
            src = pol["source"]
            fp = pol.get("first_plausible")
            fa = pol.get("first_accepted")
            lines.extend(
                [
                    f"- `{name}`: gardees `{src['kept']}` ; quads/records `{pol['quads_4d_generated']}/{pol['records_4d_generated']}` ; hits/testes `{pol['hits_4d']}/{pol['hits_tested']}` ; premier plausible `{None if fp is None else fp.get('rank')}` ; premier accepte `{None if fa is None else fa.get('rank')}`",
                    f"  - rangs gardes: `{src['raw_ranks']}`",
                    f"  - spatial: bbox `{src['spatial'].get('bbox')}` ; grid4x4 `{src['spatial'].get('grid4x4')}`",
                    f"  - flux: `{src['flux']}` ; fwhm: `{src['fwhm']}` ; qualite standard proxy: `{src['standard_quality_proxy']}`",
                    f"  - rejets principaux: `{dict(sorted(pol['reason_counts'].items(), key=lambda kv: kv[1], reverse=True)[:8])}`",
                ]
            )
            if fa:
                lines.append(f"  - hit accepte: rang `{fa['rank']}` ; raw_quad `{fa.get('raw_quad')}` ; inliers `{fa.get('inliers')}` ; RMS `{fa.get('rms_px')}`")
            elif pol.get("best_reject"):
                br = pol["best_reject"]
                lines.append(f"  - meilleur rejet: rang `{br.get('rank')}` ; reason `{br.get('reason')}` ; inliers `{br.get('inliers')}` ; RMS `{br.get('rms_px')}` ; raw_quad `{br.get('raw_quad')}`")
        cause = case["cause"]
        lines.extend(
            [
                "",
                "**Cause identifiee**",
                "",
                f"- diagnostic reussit: `{cause['diagnostic_success']}` ; standard reussit: `{cause['standard_success']}`",
                f"- le quad accepte A utilise une etoile supprimee par B: `{cause['diagnostic_accepted_uses_removed_star']}`",
                f"- le quad accepte A est absent des quads generes par B: `{cause['diagnostic_accepted_quad_absent_from_standard_quads']}`",
                f"- hits utiles A absents de B: `{cause['useful_hits_missing_from_standard_quads']}`",
                f"- hits utiles A utilisant des etoiles supprimees: `{cause['useful_hits_using_removed_stars']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Reponses",
            "",
            f"- Difference exacte qui casse `232350`: {payload['answers']['232350']}",
            f"- Difference exacte qui casse `232102`: {payload['answers']['232102']}",
            f"- Les bons quads P2.2 utilisent des etoiles supprimees par le filtre standard: `{payload['answers']['good_quads_use_removed_stars']}`",
            f"- Politique source-list 4D stable sans desactiver le filtre global: `{payload['answers']['recommended_policy']}`",
            "",
            "## Politique proposee",
            "",
            "- `standard_runtime`: filtre actuel, conserve pour le backend historique et le defaut global.",
            "- `diagnostic_unfiltered`: comportement P2.2, limite au backend 4D experimental quand explicitement demande.",
            "- `astrometry_like`: candidat dedie observe dans l'audit, a construire plus tard si necessaire; non active ici.",
            "",
            "## Parametres",
            "",
            "```json",
            json.dumps(payload["params"], indent=2),
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _answers(cases: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    any_removed = False
    for case in cases:
        cause = case["cause"]
        label = str(case["label"])
        if cause["clear"]:
            removed_hits = cause["useful_hits_using_removed_stars"]
            if removed_hits:
                any_removed = True
                first = removed_hits[0]
                out[label] = (
                    f"le filtre standard retire/reordonne des rangs source necessaires; "
                    f"le premier hit utile concerne raw_quad={first['raw_quad']} au rang {first['rank']}."
                )
            else:
                out[label] = "le filtre standard change la source-list et le quad accepte diagnostic n'est plus genere par B."
        else:
            out[label] = "non ferme par cet audit."
    out["good_quads_use_removed_stars"] = bool(any_removed)
    out["recommended_policy"] = "diagnostic_unfiltered" if all(case["cause"]["clear"] for case in cases) else "no_patch"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="P2.3 audit source-list contract for the experimental 4D backend.")
    ap.add_argument("--case", action="append", choices=sorted(DEFAULT_CASES), default=None)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    ap.add_argument("--quad4d-index", type=Path, default=DEFAULT_4D_INDEX)
    ap.add_argument("--max-stars", type=int, default=120)
    ap.add_argument("--max-quads", type=int, default=2500)
    ap.add_argument("--quality-rms", type=float, default=1.2)
    ap.add_argument("--quality-inliers", type=int, default=40)
    ap.add_argument("--code-tol", type=float, default=0.015)
    ap.add_argument("--max-hits-4d", type=int, default=2000)
    ap.add_argument("--max-hits-per-image-quad", type=int, default=8)
    ap.add_argument("--max-hypotheses", type=int, default=2000)
    ap.add_argument("--max-useful-examples", type=int, default=16)
    ap.add_argument("--image-strategy", default="log_spaced")
    ap.add_argument("--match-radius-px", type=float, default=3.0)
    ap.add_argument("--detect-k-sigma", type=float, default=3.0)
    ap.add_argument("--detect-min-area", type=int, default=5)
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--blind-star-min-sep-px", type=float, default=0.0)
    ap.add_argument("--astrometry-like-boxes", type=int, default=10)
    ap.add_argument("--astrometry-like-min-keep-ratio", type=float, default=0.05)
    ap.add_argument("--m106-hints", action="store_true", default=True)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    args = ap.parse_args()

    disk_index = Quad4DIndex.load(args.quad4d_index.expanduser().resolve())
    cases = []
    for label in list(args.case or sorted(DEFAULT_CASES)):
        filename, tile_key = DEFAULT_CASES[label]
        cases.append(_run_case(label, filename, tile_key, disk_index, args))
    answers = _answers(cases)
    if answers["recommended_policy"] == "diagnostic_unfiltered":
        verdict = "P2.3 positif: le filtre standard casse le backend 4D en retirant des etoiles utiles; politique 4D explicite justifiee"
    else:
        verdict = "P2.3 audit incomplet: cause A/B non suffisamment claire, pas de patch recommande"
    payload = {
        "schema": "zeblind.p23_4d_source_list_contract.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "global_verdict": verdict,
        "cases": cases,
        "answers": answers,
        "params": {
            "quad_hash_schema": ASTROMETRY_AB_CODE_4D_SCHEMA,
            "quad4d_index": str(args.quad4d_index.expanduser().resolve()),
            "index_root": str(args.index_root.expanduser().resolve()),
            "max_stars": int(args.max_stars),
            "max_quads": int(args.max_quads),
            "quality_rms": float(args.quality_rms),
            "quality_inliers": int(args.quality_inliers),
            "code_tol": float(args.code_tol),
            "max_hits_4d": int(args.max_hits_4d),
            "max_hits_per_image_quad": int(args.max_hits_per_image_quad),
            "max_hypotheses": int(args.max_hypotheses),
            "image_strategy": str(args.image_strategy),
            "match_radius_px": float(args.match_radius_px),
        },
    }
    args.json_out.expanduser().resolve().write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    _write_report(args.report.expanduser().resolve(), payload)
    print(json.dumps({"global_verdict": verdict, "report": str(args.report), "json": str(args.json_out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
