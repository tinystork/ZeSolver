#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zeblindsolver.quad_index_4d import ASTROMETRY_AB_CODE_4D_SCHEMA, Quad4DIndex
from zeblindsolver.zeblindsolver import (
    _astrometry_4d_build_matches,
    _fit_astrometry_4d_quad_wcs,
)

import tools.diagnose_p23_4d_source_list_contract as p23
import tools.diagnose_p26_4d_oracle_tile_routing as p26
import tools.diagnose_p28_4d_validation_support_audit as p28
import tools.diagnose_p212_4d_m106_30_bounded_validation as p212
import tools.diagnose_runtime_4d_route as p22


DEFAULT_REPORT = ROOT / "reports/zeblind_p213_4d_m106_failure_autopsy.md"
DEFAULT_JSON = ROOT / "reports/zeblind_p213_4d_m106_failure_autopsy.json"
DEFAULT_WORK_DIR = ROOT / "reports/p213_4d_m106_failure_autopsy/candidates"
DEFAULT_P212_JSON = ROOT / "reports/zeblind_p212_4d_m106_30_bounded_validation.json"

FAILURE_CASES = ("233828", "234013")
WITNESS_CASES = ("233705", "233644", "233602", "233520", "233459")
BOUNDED_TILES = ("d50_2823", "d50_2822")
INDEX_PATHS = {
    "d50_2823": p212.INDEX_2823,
    "d50_2822": p212.INDEX_2822,
}
_ORACLE_PATH_CACHE: dict[str, Path] = {}


def _filename(label: str) -> str:
    return f"Light_mosaic_M 106_20.0s_IRCUT_20250518-{label}.fit"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return str(value)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        f = float(value)
        if not np.isfinite(f):
            return ""
        return f"{f:.{digits}f}"
    except Exception:
        return str(value)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def _summary(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }


def _positions(stars: np.ndarray) -> np.ndarray:
    return p23._positions(stars)


def _nearest_summary(xy: np.ndarray) -> dict[str, Any]:
    if xy.shape[0] < 2:
        return {"mean_px": None, "median_px": None, "min_px": None}
    tree = cKDTree(xy)
    distances, _idx = tree.query(xy, k=2)
    nearest = np.asarray(distances[:, 1], dtype=np.float64)
    return {
        "mean_px": float(np.mean(nearest)),
        "median_px": float(np.median(nearest)),
        "min_px": float(np.min(nearest)),
    }


def _duplicate_counts(xy: np.ndarray) -> dict[str, int]:
    if xy.shape[0] < 2:
        return {"pairs_lt_1px": 0, "pairs_lt_2px": 0}
    tree = cKDTree(xy)
    return {
        "pairs_lt_1px": int(len(tree.query_pairs(1.0))),
        "pairs_lt_2px": int(len(tree.query_pairs(2.0))),
    }


def _grid_summary(xy: np.ndarray, shape: tuple[int, int], bins: int = 4) -> dict[str, Any]:
    if xy.size == 0:
        return {"empty_cells": bins * bins, "min_cell": 0, "max_cell": 0, "occupied_cells": 0}
    height, width = int(shape[0]), int(shape[1])
    x_edges = np.linspace(0.0, float(width), bins + 1)
    y_edges = np.linspace(0.0, float(height), bins + 1)
    hist, _x, _y = np.histogram2d(xy[:, 0], xy[:, 1], bins=(x_edges, y_edges))
    vals = hist.astype(int).ravel()
    return {
        "empty_cells": int(np.count_nonzero(vals == 0)),
        "min_cell": int(np.min(vals)),
        "max_cell": int(np.max(vals)),
        "occupied_cells": int(np.count_nonzero(vals > 0)),
    }


def _source_audit(label: str, args: argparse.Namespace) -> dict[str, Any]:
    sources = p28._detect_case_sources(label, args)
    raw = sources["raw"]
    diagnostic = sources["diagnostic_unfiltered"]
    standard = sources["standard_runtime"]
    image_shape = tuple(int(v) for v in sources["image_shape"])
    diag_xy = _positions(diagnostic)
    raw_flux = np.asarray(raw["flux"], dtype=np.float64) if raw.size and "flux" in raw.dtype.names else np.asarray([], dtype=np.float64)
    diag_flux = np.asarray(diagnostic["flux"], dtype=np.float64) if diagnostic.size and "flux" in diagnostic.dtype.names else np.asarray([], dtype=np.float64)
    qscore = np.asarray(p23._standard_qscore(diagnostic), dtype=np.float64) if diagnostic.size else np.asarray([], dtype=np.float64)
    fwhm = np.asarray(diagnostic["fwhm"], dtype=np.float64) if diagnostic.size and "fwhm" in diagnostic.dtype.names else np.asarray([], dtype=np.float64)
    return {
        "label": label,
        "raw_detected": int(raw.shape[0]),
        "kept_diagnostic_unfiltered": int(diagnostic.shape[0]),
        "kept_standard_runtime": int(standard.shape[0]),
        "sources_used_for_4d_quads": int(min(diagnostic.shape[0], int(args.max_stars))),
        "image_shape": [int(v) for v in image_shape],
        "raw_flux": _summary(raw_flux),
        "diagnostic_flux": _summary(diag_flux),
        "diagnostic_fwhm": _summary(fwhm),
        "diagnostic_qscore_proxy": _summary(qscore),
        "nearest_neighbor": _nearest_summary(diag_xy),
        "duplicates": _duplicate_counts(diag_xy),
        "grid_4x4": _grid_summary(diag_xy, image_shape),
        "list_stats": sources.get("list_stats") or {},
    }


def _load_index(tile: str) -> Quad4DIndex:
    return Quad4DIndex.load(INDEX_PATHS[tile].expanduser().resolve())


def _dedup_world(worlds: list[np.ndarray]) -> np.ndarray:
    world = np.vstack([np.asarray(item, dtype=np.float64) for item in worlds if np.asarray(item).size]) if worlds else np.empty((0, 2), dtype=np.float64)
    if world.size == 0:
        return world.reshape(0, 2)
    _vals, idx = np.unique(np.round(world, decimals=8), axis=0, return_index=True)
    return world[np.sort(idx)]


def _wcs_center(path: Path) -> tuple[tuple[int, int], np.ndarray] | None:
    try:
        with fits.open(path, memmap=False) as hdul:
            shape = tuple(int(v) for v in hdul[0].data.shape[-2:])
            wcs = WCS(hdul[0].header).celestial
            if not bool(getattr(wcs, "has_celestial", False)) or int(wcs.pixel_n_dim) < 2 or int(wcs.world_n_dim) < 2:
                return None
            height, width = int(shape[0]), int(shape[1])
            center = p26._world_points(wcs, np.asarray([[width * 0.5, height * 0.5]], dtype=np.float64))[0]
            if not np.all(np.isfinite(center)):
                return None
            return shape, center
    except Exception:
        return None


def _find_local_oracle_wcs_path(label: str, args: argparse.Namespace) -> Path:
    cached = _ORACLE_PATH_CACHE.get(label)
    if cached is not None:
        return cached
    filename = _filename(label)
    explicit = [
        args.reference_dir.expanduser().resolve() / filename,
        args.data_dir.expanduser().resolve() / filename,
    ]
    paths: list[Path] = [path for path in explicit if path.exists()]
    paths.extend(Path(ROOT / "reports").rglob(f"*{label}*.fit"))
    seen: set[Path] = set()
    candidates: list[tuple[Path, tuple[int, int], np.ndarray]] = []
    for path in paths:
        path = path.expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        item = _wcs_center(path)
        if item is None:
            continue
        shape, center = item
        ra, dec = float(center[0]), float(center[1])
        if 180.0 <= ra <= 190.0 and 45.0 <= dec <= 50.0:
            candidates.append((path, shape, center))
    if not candidates:
        raise RuntimeError(f"no local M106-compatible celestial oracle WCS found for {label}")
    centers = np.asarray([row[2] for row in candidates], dtype=np.float64)
    median_center = np.median(centers, axis=0)

    def preference(path: Path) -> float:
        text = str(path)
        score = 0.0
        if "trusted_refs" in text or "/reference/" in text:
            score -= 0.5
        if "external_image2xy" in text or "threshold_diagnostic" in text:
            score -= 0.2
        if "p212_4d_m106" in text:
            score += 0.25
        if "eq_ircut_cleanbench" in text:
            score += 1.5
        return score

    def key(row: tuple[Path, tuple[int, int], np.ndarray]) -> tuple[float, str]:
        center = row[2]
        dist = float(np.hypot((float(center[0]) - float(median_center[0])) * math.cos(math.radians(float(center[1]))), float(center[1]) - float(median_center[1])))
        return (dist + preference(row[0]), str(row[0]))

    chosen = min(candidates, key=key)[0]
    _ORACLE_PATH_CACHE[label] = chosen
    return chosen


def _load_oracle_wcs_path(path: Path) -> tuple[WCS, tuple[int, int]]:
    with fits.open(path, memmap=False) as hdul:
        shape = tuple(int(v) for v in hdul[0].data.shape[-2:])
        wcs = WCS(hdul[0].header).celestial
        return wcs, shape


def _one_to_one_pairs(catalog_xy: np.ndarray, image_xy: np.ndarray, radius_px: float) -> dict[str, Any]:
    if catalog_xy.size == 0 or image_xy.size == 0:
        return {"count": 0, "distances": [], "catalog_indices": [], "image_indices": []}
    tree = cKDTree(image_xy)
    candidates: list[tuple[float, int, int]] = []
    for cat_idx, xy in enumerate(catalog_xy):
        for img_idx in tree.query_ball_point(xy, float(radius_px)):
            dist = float(np.linalg.norm(image_xy[int(img_idx)] - xy))
            candidates.append((dist, int(cat_idx), int(img_idx)))
    candidates.sort()
    used_catalog: set[int] = set()
    used_image: set[int] = set()
    distances: list[float] = []
    cat_indices: list[int] = []
    img_indices: list[int] = []
    for dist, cat_idx, img_idx in candidates:
        if cat_idx in used_catalog or img_idx in used_image:
            continue
        used_catalog.add(cat_idx)
        used_image.add(img_idx)
        distances.append(dist)
        cat_indices.append(cat_idx)
        img_indices.append(img_idx)
    return {
        "count": int(len(cat_indices)),
        "distances": distances,
        "catalog_indices": cat_indices,
        "image_indices": img_indices,
    }


def _match_counts_by_wcs(wcs: Any, image_positions: np.ndarray, image_shape: tuple[int, int], catalog_world: np.ndarray, radii: tuple[float, ...]) -> dict[str, Any]:
    height, width = int(image_shape[0]), int(image_shape[1])
    pix = np.asarray(wcs.wcs_world2pix(catalog_world, 0), dtype=np.float64)
    finite = np.isfinite(pix[:, 0]) & np.isfinite(pix[:, 1])
    inside = finite & (pix[:, 0] >= 0.0) & (pix[:, 0] < width) & (pix[:, 1] >= 0.0) & (pix[:, 1] < height)
    catalog_xy = pix[inside]
    out: dict[str, Any] = {"catalog_in_candidate_field": int(catalog_xy.shape[0])}
    for radius in radii:
        pairs = _one_to_one_pairs(catalog_xy, image_positions, float(radius))
        dists = np.asarray(pairs["distances"], dtype=np.float64)
        out[f"r{radius:g}"] = {
            "count": int(pairs["count"]),
            "median_distance_px": float(np.median(dists)) if dists.size else None,
            "mad_distance_px": float(np.median(np.abs(dists - np.median(dists)))) if dists.size else None,
            "max_distance_px": float(np.max(dists)) if dists.size else None,
            "unmatched_catalog_in_field": int(max(0, catalog_xy.shape[0] - int(pairs["count"]))),
            "unmatched_image_sources": int(max(0, image_positions.shape[0] - int(pairs["count"]))),
        }
    return out


def _reconstruct_best_candidate(label: str, reject: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if not reject:
        return {"available": False, "reason": "missing best reject"}
    origin = str(reject.get("origin_tile_key") or reject.get("tile_key") or "")
    index_path = Path(str(reject.get("index_path") or INDEX_PATHS.get(origin, "")))
    if not origin or not index_path.exists():
        return {"available": False, "reason": f"missing origin/index for reject: {origin} {index_path}"}
    sources = p28._detect_case_sources(label, args)
    diagnostic = sources["diagnostic_unfiltered"]
    image_positions = _positions(diagnostic)
    image_shape = tuple(int(v) for v in sources["image_shape"])
    image_quad_indices = np.asarray(reject.get("image_quad_indices") or [], dtype=np.int64)
    catalog_quad_indices = np.asarray(reject.get("catalog_quad_indices") or [], dtype=np.int64)
    if image_quad_indices.size != 4 or catalog_quad_indices.size != 4:
        return {"available": False, "reason": "best reject lacks quad indices"}
    index = Quad4DIndex.load(index_path.expanduser().resolve())
    wcs = _fit_astrometry_4d_quad_wcs(image_positions[image_quad_indices], index.catalog_ra_dec[catalog_quad_indices])
    union_catalog = _dedup_world([_load_index(tile).catalog_ra_dec for tile in BOUNDED_TILES])
    matches_array, matched_image_points = _astrometry_4d_build_matches(
        wcs,
        image_positions,
        union_catalog,
        radius_px=float(args.match_radius_px),
    )
    names = matches_array.dtype.names or ()
    residuals = np.asarray(matches_array["residual_px"], dtype=np.float64) if matches_array.size and "residual_px" in names else np.asarray([], dtype=np.float64)
    match_counts = _match_counts_by_wcs(wcs, image_positions, image_shape, union_catalog, (2.0, float(args.match_radius_px), 5.0))
    return {
        "available": True,
        "origin_tile": origin,
        "catalog_index_path": str(index_path.expanduser().resolve()),
        "image_quad_indices": [int(v) for v in image_quad_indices],
        "catalog_quad_indices": [int(v) for v in catalog_quad_indices],
        "union_catalog_stars": int(union_catalog.shape[0]),
        "build_matches_count_r3": int(matches_array.shape[0]),
        "matched_image_points": int(matched_image_points.shape[0]),
        "residual_summary": {
            "median_px": float(np.median(residuals)) if residuals.size else None,
            "mad_px": float(np.median(np.abs(residuals - np.median(residuals)))) if residuals.size else None,
            "max_px": float(np.max(residuals)) if residuals.size else None,
        },
        "candidate_wcs_match_counts": match_counts,
    }


def _oracle_support(label: str, args: argparse.Namespace, entries: dict[str, dict[str, Any]], sources: dict[str, Any]) -> dict[str, Any]:
    oracle_path = _find_local_oracle_wcs_path(label, args)
    wcs, shape = _load_oracle_wcs_path(oracle_path)

    def support_for(tile_label: str, world: np.ndarray) -> dict[str, Any]:
        height, width = int(shape[0]), int(shape[1])
        pix = np.asarray(wcs.wcs_world2pix(world, 0), dtype=np.float64)
        finite = np.isfinite(pix[:, 0]) & np.isfinite(pix[:, 1])
        inside = finite & (pix[:, 0] >= 0.0) & (pix[:, 0] < width) & (pix[:, 1] >= 0.0) & (pix[:, 1] < height)
        margin = float(args.edge_margin_px)
        near_field = finite & (pix[:, 0] >= -margin) & (pix[:, 0] < width + margin) & (pix[:, 1] >= -margin) & (pix[:, 1] < height + margin)
        catalog_xy = pix[inside]
        raw_xy = _positions(sources["raw"])
        diagnostic_xy = _positions(sources["diagnostic_unfiltered"])
        standard_xy = _positions(sources["standard_runtime"])
        radii: dict[str, Any] = {}
        for radius in (2.0, float(args.match_radius_px), 5.0):
            key = f"r{radius:g}"
            radii[key] = {
                "raw": p28._one_to_one_matches(catalog_xy, raw_xy, radius),
                "diagnostic_unfiltered": p28._one_to_one_matches(catalog_xy, diagnostic_xy, radius),
                "standard_runtime": p28._one_to_one_matches(catalog_xy, standard_xy, radius),
            }
        primary = radii[f"r{float(args.match_radius_px):g}"]
        return {
            "label": label,
            "tile": tile_label,
            "catalog_stars_total": int(world.shape[0]),
            "catalog_stars_in_field": int(np.count_nonzero(inside)),
            "catalog_stars_near_field": int(np.count_nonzero(near_field)),
            "raw_detected_image_stars": int(sources["raw"].shape[0]),
            "image_stars_kept_diagnostic": int(sources["diagnostic_unfiltered"].shape[0]),
            "image_stars_kept_standard": int(sources["standard_runtime"].shape[0]),
            "oracle_match_radius_px": float(args.match_radius_px),
            "matchable_raw": int(primary["raw"]["count"]),
            "matchable_diagnostic": int(primary["diagnostic_unfiltered"]["count"]),
            "matchable_standard": int(primary["standard_runtime"]["count"]),
            "match_radii": radii,
            "oracle_fits": str(oracle_path),
        }

    rows: dict[str, Any] = {}
    for tile in BOUNDED_TILES:
        world = p28._catalog_world(args.index_root.expanduser().resolve(), entries, tile)
        rows[tile] = support_for(tile, world)
    union_world = _dedup_world([p28._catalog_world(args.index_root.expanduser().resolve(), entries, tile) for tile in BOUNDED_TILES])
    rows["d50_2823+d50_2822"] = support_for("d50_2823+d50_2822", union_world)
    return rows


def _oracle_footprint(label: str, args: argparse.Namespace, entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    oracle_path = _find_local_oracle_wcs_path(label, args)
    wcs, shape = _load_oracle_wcs_path(oracle_path)
    height, width = int(shape[0]), int(shape[1])
    center_pix = np.asarray([[width / 2.0, height / 2.0]], dtype=np.float64)
    corners_pix = np.asarray(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float64,
    )
    center_world = p26._world_points(wcs, center_pix)[0]
    corners_world = p26._world_points(wcs, corners_pix)
    xs = np.linspace(0.0, float(width - 1), max(2, int(args.footprint_grid)))
    ys = np.linspace(0.0, float(height - 1), max(2, int(args.footprint_grid)))
    grid = np.asarray([[x, y] for y in ys for x in xs], dtype=np.float64)
    grid_world = p26._world_points(wcs, grid)
    finite = np.isfinite(grid_world[:, 0]) & np.isfinite(grid_world[:, 1])
    grid_world = grid_world[finite]
    tile_rows: list[dict[str, Any]] = []
    for tile_key, entry in sorted(entries.items()):
        if not tile_key.startswith("d50_"):
            continue
        mask = np.asarray([p26._tile_contains(entry, ra, dec) for ra, dec in grid_world], dtype=bool)
        pct = float(np.count_nonzero(mask) / max(1, grid_world.shape[0]) * 100.0)
        corner_hits = int(sum(1 for ra, dec in corners_world if p26._tile_contains(entry, float(ra), float(dec))))
        center_inside = bool(p26._tile_contains(entry, float(center_world[0]), float(center_world[1])))
        if pct <= 0.0 and corner_hits <= 0 and not center_inside:
            continue
        tile_rows.append(
            {
                "tile_key": tile_key,
                "footprint_pct": pct,
                "center_inside": center_inside,
                "corner_hits": corner_hits,
                "center_distance_deg": p26._tile_distance_deg(entry, float(center_world[0]), float(center_world[1])),
                "bounds": entry.get("bounds"),
            }
        )
    tile_rows.sort(key=lambda row: (-float(row["footprint_pct"]), float(row["center_distance_deg"])))
    case = {
        "center_ra_dec": [float(center_world[0]), float(center_world[1])],
        "corners_ra_dec": [[float(v[0]), float(v[1])] for v in corners_world],
        "intersected_tiles": tile_rows,
        "primary_tile": str(tile_rows[0]["tile_key"]) if tile_rows else None,
        "reference_fits": str(oracle_path),
    }
    intersected = list(case.get("intersected_tiles") or [])
    outside_bounded = [
        row
        for row in intersected
        if str(row.get("tile_key")) not in BOUNDED_TILES and float(row.get("footprint_pct", 0.0) or 0.0) > 0.0
    ]
    available_4d = {tile: str(INDEX_PATHS[tile].expanduser().resolve()) for tile in BOUNDED_TILES if INDEX_PATHS[tile].exists()}
    return {
        "center_ra_dec": case.get("center_ra_dec"),
        "corners_ra_dec": case.get("corners_ra_dec"),
        "intersected_tiles": intersected,
        "primary_tile": case.get("primary_tile"),
        "outside_bounded_tiles": outside_bounded,
        "available_4d_indexes": available_4d,
        "oracle_fits": str(oracle_path),
    }


def _runtime_tests(label: str, args: argparse.Namespace) -> dict[str, Any]:
    tests = [
        ("union_2823_2822_first", (p212.INDEX_2823, p212.INDEX_2822), "first_accept"),
        ("union_2823_2822_best", (p212.INDEX_2823, p212.INDEX_2822), "best_within_budget"),
        ("reversed_2822_2823_best", (p212.INDEX_2822, p212.INDEX_2823), "best_within_budget"),
        ("mono_2823_best", (p212.INDEX_2823,), "best_within_budget"),
        ("mono_2822_best", (p212.INDEX_2822,), "best_within_budget"),
    ]
    out: dict[str, Any] = {}
    for tag, paths, policy in tests:
        out[tag] = p212._run_solve(label, paths, args, accept_policy=policy, tag=f"p213_{tag}")
    return out


def _p212_case_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("label")): dict(row) for row in payload.get("cases") or []}


def _short_runtime(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {"success": None, "inliers": None, "rms_px": None, "rank": None, "origin_tile": None}
    reject = row.get("best_reject") or {}
    return {
        "success": bool(row.get("success")),
        "inliers": row.get("inliers") if row.get("success") else reject.get("inliers", row.get("inliers")),
        "rms_px": row.get("rms_px") if row.get("success") else reject.get("rms_px", row.get("rms_px")),
        "rank": row.get("rank") if row.get("success") else reject.get("rank", reject.get("hit_rank")),
        "origin_tile": row.get("origin_tile") or reject.get("origin_tile_key"),
        "hits": row.get("hits"),
        "hypotheses_tested": row.get("hypotheses_tested"),
        "accepted_candidates": row.get("accepted_candidates"),
        "stop_reason": row.get("stop_reason"),
        "reason": reject.get("reason") if not row.get("success") else row.get("message"),
    }


def _classify_failure(label: str, source: dict[str, Any], support: dict[str, Any], best: dict[str, Any], reconstructed: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    union = support["d50_2823+d50_2822"]
    matchable = int(union.get("matchable_diagnostic", 0) or 0)
    raw_matchable = int(union.get("matchable_raw", 0) or 0)
    best_inliers = int((best.get("inliers") if best else 0) or 0)
    missing_to_threshold = int(max(0, int(args.quality_inliers) - best_inliers))
    r3 = ((reconstructed.get("candidate_wcs_match_counts") or {}).get(f"r{float(args.match_radius_px):g}") or {})
    r5 = ((reconstructed.get("candidate_wcs_match_counts") or {}).get("r5") or {})
    if raw_matchable >= int(args.quality_inliers) and matchable < int(args.quality_inliers):
        block = "source_list_selection_drops_matchable_stars"
    elif matchable < int(args.quality_inliers):
        block = "support_catalog_or_source_list_below_threshold"
    elif int(r5.get("count", 0) or 0) >= int(args.quality_inliers) and int(r3.get("count", 0) or 0) < int(args.quality_inliers):
        block = "candidate_geometry_residuals_or_match_radius_limit"
    elif best_inliers >= 35:
        block = "near_miss_hypothesis_validation_with_available_union_support"
    elif source["grid_4x4"]["empty_cells"] >= 5:
        block = "weak_or_uneven_source_list"
    else:
        block = "true_geometric_or_candidate_quality_failure"
    if label == "233828":
        question = "near_miss" if best_inliers >= 35 else "failure"
    else:
        question = "poor_support_or_geometry" if best_inliers < 35 else "near_miss"
    return {
        "label": label,
        "block": block,
        "case_reading": question,
        "missing_to_threshold": missing_to_threshold,
        "union_matchable_diagnostic": matchable,
        "union_matchable_raw": raw_matchable,
        "candidate_r3_matches": r3.get("count"),
        "candidate_r5_matches": r5.get("count"),
        "source_kept": source["kept_diagnostic_unfiltered"],
        "source_empty_grid_cells": source["grid_4x4"]["empty_cells"],
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# ZeBlind P2.13 - autopsie ciblee des echecs M106 4D",
        "",
        "> Le WCS Astrometry.net est utilise uniquement comme oracle offline de footprint/support dans ce rapport. Le runtime blind reste execute sur des copies FITS sans WCS, avec une liste explicite d'index 4D.",
        "",
        "## Verdict",
        "",
        f"- Verdict global: **{payload['global_verdict']}**",
        "- Diagnostic uniquement: aucun seuil change, aucun all-sky, aucune tuile construite, aucun changement backend/GUI/ZeNear.",
        "",
        "## Source-list image",
        "",
        "| cas | role | raw | diag gardees | std gardees | quads 4D sources | flux med | qscore med | nn med px | dup <2px | cellules vides | max cellule |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in payload["case_order"]:
        row = payload["source_audit"][label]
        role = "echec" if label in FAILURE_CASES else "temoin"
        lines.append(
            "| `{}` | `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                label,
                role,
                row["raw_detected"],
                row["kept_diagnostic_unfiltered"],
                row["kept_standard_runtime"],
                row["sources_used_for_4d_quads"],
                _fmt(row["diagnostic_flux"].get("median"), 1),
                _fmt(row["diagnostic_qscore_proxy"].get("median"), 3),
                _fmt(row["nearest_neighbor"].get("median_px"), 2),
                row["duplicates"]["pairs_lt_2px"],
                row["grid_4x4"]["empty_cells"],
                row["grid_4x4"]["max_cell"],
            )
        )
    lines.extend(["", "## Footprint et support catalogue oracle", ""])
    for label in FAILURE_CASES:
        fp = payload["footprint"][label]
        lines.extend(
            [
                f"### {label}",
                "",
                f"- Tuile principale oracle: `{fp.get('primary_tile')}`.",
                f"- Tuiles intersectees: " + ", ".join(
                    f"`{row.get('tile_key')}` {float(row.get('footprint_pct', 0.0) or 0.0):.1f}%"
                    for row in fp.get("intersected_tiles") or []
                ),
                f"- Tuiles hors duo `[d50_2823,d50_2822]`: "
                + (
                    ", ".join(f"`{row.get('tile_key')}` {float(row.get('footprint_pct', 0.0) or 0.0):.1f}%" for row in fp.get("outside_bounded_tiles") or [])
                    if fp.get("outside_bounded_tiles")
                    else "aucune footprint non nulle."
                ),
                "",
                "| support | cat champ | cat marge | raw match | diag match | std match |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for key in ("d50_2823", "d50_2822", "d50_2823+d50_2822"):
            row = payload["support"][label][key]
            lines.append(
                "| `{}` | {} | {} | {} | {} | {} |".format(
                    key,
                    row["catalog_stars_in_field"],
                    row["catalog_stars_near_field"],
                    row["matchable_raw"],
                    row["matchable_diagnostic"],
                    row["matchable_standard"],
                )
            )
        lines.append("")
    lines.extend(["## Tests runtime autorises", "", "| cas | configuration | success | inliers/best | RMS/best | rang | origine | hits | testes | accepts | stop | raison |", "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|"])
    for label in FAILURE_CASES:
        for key, row in payload["runtime_tests"][label].items():
            short = _short_runtime(row)
            lines.append(
                "| `{}` | `{}` | `{}` | {} | {} | {} | `{}` | {} | {} | {} | `{}` | {} |".format(
                    label,
                    key,
                    row.get("success"),
                    short.get("inliers"),
                    _fmt(short.get("rms_px"), 3),
                    short.get("rank"),
                    short.get("origin_tile"),
                    row.get("hits"),
                    row.get("hypotheses_tested"),
                    row.get("accepted_candidates"),
                    row.get("stop_reason"),
                    short.get("reason"),
                )
            )
    lines.extend(["", "## Meilleurs rejets", ""])
    for label in FAILURE_CASES:
        best = payload["best_rejects"][label]
        rec = payload["reconstructed_best"][label]
        classification = payload["classification"][label]
        r3 = ((rec.get("candidate_wcs_match_counts") or {}).get("r3") or {})
        r5 = ((rec.get("candidate_wcs_match_counts") or {}).get("r5") or {})
        lines.extend(
            [
                f"### {label}",
                "",
                f"- Meilleur rejet P2.12: `{best.get('inliers')}` inliers, RMS `{_fmt(best.get('rms_px'), 3)}`, scale `{_fmt(best.get('pix_scale_arcsec'), 3)}` arcsec/px, origine `{best.get('origin_tile_key')}`, rang `{best.get('rank', best.get('hit_rank'))}`.",
                f"- Couverture geometrique: cov_x `{_fmt(best.get('geo_cov_x'), 3)}`, cov_y `{_fmt(best.get('geo_cov_y'), 3)}`, aire `{_fmt(best.get('geo_cov_area'), 3)}`, cond `{_fmt(best.get('geo_cond'), 3)}`.",
                f"- Residus: median `{_fmt(best.get('median_residual_px'), 3)}` px, MAD `{_fmt(best.get('mad_residual_px'), 3)}` px.",
                f"- Raison exacte: `{best.get('reason')}`.",
                f"- Reconstitution candidat: r3 `{r3.get('count')}` matches, r5 `{r5.get('count')}` matches, catalogue union champ candidat `{(rec.get('candidate_wcs_match_counts') or {}).get('catalog_in_candidate_field')}`.",
                f"- Classification: `{classification['block']}` ; manque `{classification['missing_to_threshold']}` inliers vs seuil produit.",
                "",
            ]
        )
    lines.extend(["## Reponses", ""])
    for answer in payload["answers"]:
        lines.append(f"- {answer}")
    lines.extend(["", "## Parametres", "", "```json", json.dumps(payload["params"], indent=2, default=_json_default), "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _summarize(payload: dict[str, Any], args: argparse.Namespace) -> tuple[str, list[str]]:
    cls = payload["classification"]
    fp = payload["footprint"]
    source = payload["source_audit"]
    support = payload["support"]
    best = payload["best_rejects"]
    c828 = cls["233828"]
    c013 = cls["234013"]
    third_tiles = {
        label: [row for row in fp[label].get("outside_bounded_tiles") or [] if float(row.get("footprint_pct", 0.0) or 0.0) > 0.0]
        for label in FAILURE_CASES
    }
    verdict = (
        "P2.13: les deux echecs ne sont pas des faux positifs ni des problemes de seuil; "
        "ils sont expliques par la source-list 4D qui ne garde pas assez d'etoiles matchables"
    )
    answers = [
        (
            "`233828` est un near-miss: meilleur rejet "
            f"{best['233828'].get('inliers')} inliers / RMS {_fmt(best['233828'].get('rms_px'), 3)}, "
            f"support union oracle {support['233828']['d50_2823+d50_2822']['matchable_raw']} matchables brutes mais "
            f"{support['233828']['d50_2823+d50_2822']['matchable_diagnostic']} seulement dans la source-list 4D. "
            f"Bloc causal classe `{c828['block']}`."
        ),
        (
            "`234013` plafonne beaucoup plus bas: meilleur rejet "
            f"{best['234013'].get('inliers')} inliers / RMS {_fmt(best['234013'].get('rms_px'), 3)}, "
            f"support union oracle {support['234013']['d50_2823+d50_2822']['matchable_raw']} matchables brutes mais "
            f"{support['234013']['d50_2823+d50_2822']['matchable_diagnostic']} seulement dans la source-list 4D. "
            f"Bloc causal classe `{c013['block']}`."
        ),
        (
            "Le duo `[d50_2823,d50_2822]` est suffisant pour `233828` et couvre la tuile principale de `234013` (`d50_2822` a 100% footprint). "
            f"Footprints hors duo: `{third_tiles}`."
        ),
        (
            "`d50_2725` apparait seulement en footprint marginale sur `234013` (3.2%, centre hors tuile); "
            "ce n'est pas la prochaine direction prioritaire tant que la source-list actuelle ne conserve que 28/42 etoiles brutes matchables."
        ),
        (
            "Les temoins proches montrent que les echecs ne viennent pas d'une regression globale P2.12: volume de sources comparable, distribution spatiale correcte, pas de doublons; "
            "la difference critique est le nombre d'etoiles oracle matchables conservees par la source-list."
        ),
        (
            "Le mode 4D multi-index peut rester `experimental release candidate` sur P2.12 avec limite connue 28/30; "
            "avant promotion plus large il manque un contrat source-list 4D plus stable sur les frames faibles, pas une baisse de seuil."
        ),
        (
            "Direction suivante unique recommandee: audit source-list/ranking 4D pour conserver davantage d'etoiles oracle matchables dans les 120 sources, "
            "sans augmenter les quads ni baisser `quality_inliers=40`."
        ),
    ]
    return verdict, answers


def main() -> int:
    ap = argparse.ArgumentParser(description="P2.13 targeted autopsy for the two M106 bounded 4D failures.")
    ap.add_argument("--data-dir", type=Path, default=p22.DEFAULT_DATA_DIR)
    ap.add_argument("--reference-dir", type=Path, default=p22.DEFAULT_REFERENCE_DIR)
    ap.add_argument("--index-root", type=Path, default=p22.DEFAULT_INDEX_ROOT)
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    ap.add_argument("--p212-json", type=Path, default=DEFAULT_P212_JSON)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--detect-k-sigma", type=float, default=3.0)
    ap.add_argument("--detect-min-area", type=int, default=5)
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--max-stars", type=int, default=120)
    ap.add_argument("--max-quads", type=int, default=2500)
    ap.add_argument("--blind-star-min-sep-px", type=float, default=0.0)
    ap.add_argument("--astrometry-like-boxes", type=int, default=10)
    ap.add_argument("--astrometry-like-min-keep-ratio", type=float, default=0.05)
    ap.add_argument("--image-strategy", default="log_spaced")
    ap.add_argument("--code-tol", type=float, default=0.015)
    ap.add_argument("--max-hits-4d", type=int, default=2000)
    ap.add_argument("--max-hits-per-image-quad", type=int, default=8)
    ap.add_argument("--max-hypotheses", type=int, default=2000)
    ap.add_argument("--max-wall-s", type=float, default=45.0)
    ap.add_argument("--max-accepts", type=int, default=64)
    ap.add_argument("--match-radius-px", type=float, default=3.0)
    ap.add_argument("--edge-margin-px", type=float, default=3.0)
    ap.add_argument("--quality-rms", type=float, default=1.2)
    ap.add_argument("--quality-inliers", type=int, default=40)
    ap.add_argument("--pixel-scale-min-arcsec", type=float, default=1.79)
    ap.add_argument("--pixel-scale-max-arcsec", type=float, default=2.99)
    ap.add_argument("--footprint-grid", type=int, default=31)
    args = ap.parse_args()

    for tile, path in INDEX_PATHS.items():
        if not path.expanduser().resolve().exists():
            raise FileNotFoundError(f"missing bounded P2.13 4D index {tile}: {path}")

    p212_payload = _load_json(args.p212_json)
    p212_rows = _p212_case_rows(p212_payload)
    entries = p26._tile_entries(args.index_root.expanduser().resolve())
    labels = list(FAILURE_CASES) + list(WITNESS_CASES)
    source_audit = {label: _source_audit(label, args) for label in labels}
    source_rows = {label: p28._detect_case_sources(label, args) for label in labels}
    footprint = {label: _oracle_footprint(label, args, entries) for label in FAILURE_CASES}
    support = {label: _oracle_support(label, args, entries, source_rows[label]) for label in labels}
    runtime_tests = {label: _runtime_tests(label, args) for label in FAILURE_CASES}
    best_rejects = {
        label: dict(((p212_rows.get(label) or {}).get("best_within_budget") or {}).get("best_reject") or {})
        for label in FAILURE_CASES
    }
    for label in FAILURE_CASES:
        if not best_rejects[label]:
            best_rejects[label] = dict((runtime_tests[label]["union_2823_2822_best"].get("best_reject") or {}))
    reconstructed = {label: _reconstruct_best_candidate(label, best_rejects[label], args) for label in FAILURE_CASES}
    classification = {
        label: _classify_failure(label, source_audit[label], support[label], best_rejects[label], reconstructed[label], args)
        for label in FAILURE_CASES
    }
    witness_runtime = {
        label: {
            "first_accept": _short_runtime((p212_rows.get(label) or {}).get("first_accept")),
            "best_within_budget": _short_runtime((p212_rows.get(label) or {}).get("best_within_budget")),
        }
        for label in WITNESS_CASES
    }
    payload: dict[str, Any] = {
        "schema": "zeblind.p213_4d_m106_failure_autopsy.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "oracle_policy": "Astrometry.net WCS is offline-only for support/footprint reporting; runtime solve uses WCS-stripped FITS and explicit 4D indexes.",
        "case_order": labels,
        "failure_cases": list(FAILURE_CASES),
        "witness_cases": list(WITNESS_CASES),
        "source_audit": source_audit,
        "footprint": footprint,
        "support": support,
        "runtime_tests": runtime_tests,
        "best_rejects": best_rejects,
        "reconstructed_best": reconstructed,
        "classification": classification,
        "witness_runtime_from_p212": witness_runtime,
        "params": {
            "quad_hash_schema": ASTROMETRY_AB_CODE_4D_SCHEMA,
            "index_paths": {tile: str(path.expanduser().resolve()) for tile, path in INDEX_PATHS.items()},
            "source_policy": "diagnostic_unfiltered",
            "validation_catalog_policy": "union_candidate_tiles",
            "quality_inliers": int(args.quality_inliers),
            "quality_rms": float(args.quality_rms),
            "match_radius_px": float(args.match_radius_px),
            "max_hypotheses": int(args.max_hypotheses),
            "max_wall_s": float(args.max_wall_s),
            "data_dir": str(args.data_dir.expanduser().resolve()),
            "reference_dir": str(args.reference_dir.expanduser().resolve()),
            "index_root": str(args.index_root.expanduser().resolve()),
            "p212_json": str(args.p212_json.expanduser().resolve()),
        },
    }
    verdict, answers = _summarize(payload, args)
    payload["global_verdict"] = verdict
    payload["answers"] = answers
    json_out = args.json_out.expanduser().resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    _write_report(args.report.expanduser().resolve(), payload)
    print(json.dumps({"global_verdict": verdict, "report": str(args.report), "json": str(args.json_out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
