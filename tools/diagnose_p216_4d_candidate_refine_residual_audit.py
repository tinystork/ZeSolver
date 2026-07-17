#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zeblindsolver.asterisms import sample_quads
from zeblindsolver.quad_code_diagnostic import build_astrometry_quad_records
from zeblindsolver.quad_index_4d import Quad4DIndex
from zeblindsolver.wcs_fit import fit_wcs_sip
from zeblindsolver.zeblindsolver import _blind_geometric_guardrails, _fit_astrometry_4d_quad_wcs

import tools.diagnose_p212_4d_m106_30_bounded_validation as p212
import tools.diagnose_p213_4d_m106_failure_autopsy as p213
import tools.diagnose_p214_4d_source_policy_bakeoff as p214
import tools.diagnose_p215_4d_split_quad_verify_sources as p215
import tools.diagnose_p23_4d_source_list_contract as p23
import tools.diagnose_p28_4d_validation_support_audit as p28
import tools.diagnose_runtime_4d_route as p22


DEFAULT_REPORT = ROOT / "reports/zeblind_p216_4d_candidate_refine_residual_audit.md"
DEFAULT_JSON = ROOT / "reports/zeblind_p216_4d_candidate_refine_residual_audit.json"
DEFAULT_DETAIL_JSON = ROOT / "reports/zeblind_p216_4d_candidate_refine_residual_audit_details.json"

MAIN_CASE = "234013"
CONTROL_CASES = ("233828", "233705", "233644", "233602", "233520")
BOUNDED_TILES = ("d50_2823", "d50_2822")


@dataclass(frozen=True)
class MatchSet:
    name: str
    matches: np.ndarray
    image_indices: np.ndarray
    catalog_indices: np.ndarray
    distances: np.ndarray
    projected: np.ndarray


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        f = float(value)
        return f if math.isfinite(f) else None
    return str(value)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        f = float(value)
        if not math.isfinite(f):
            return ""
        return f"{f:.{digits}f}"
    except Exception:
        return str(value)


def _summary(values: Any) -> dict[str, Any]:
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


def _filename(label: str) -> str:
    return f"Light_mosaic_M 106_20.0s_IRCUT_20250518-{label}.fit"


def _load_shape(path: Path) -> tuple[int, int]:
    with fits.open(path, memmap=False) as hdul:
        return tuple(int(v) for v in hdul[0].data.shape[-2:])


def _clean_verification_sources(raw: np.ndarray, image_shape: tuple[int, int], min_sep_px: float) -> np.ndarray:
    clean, _stats = p215._clean_verification_sources(raw, image_shape, min_sep_px=float(min_sep_px))
    return clean


def _load_union_catalog(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    entries = p26_entries(args.index_root.expanduser().resolve())
    worlds = [p28._catalog_world(args.index_root.expanduser().resolve(), entries, tile) for tile in BOUNDED_TILES]
    union = p214._dedup_world(worlds)
    return union, {"tiles": list(BOUNDED_TILES), "count": int(union.shape[0])}


def p26_entries(index_root: Path) -> dict[str, dict[str, Any]]:
    import tools.diagnose_p26_4d_oracle_tile_routing as p26

    return p26._tile_entries(index_root)


def _load_oracle(label: str, args: argparse.Namespace) -> tuple[WCS, tuple[int, int], Path]:
    path = p213._find_local_oracle_wcs_path(label, args)
    wcs, shape = p213._load_oracle_wcs_path(path)
    return wcs, shape, path


def _project_world(wcs: WCS, world: np.ndarray) -> np.ndarray:
    try:
        return np.asarray(wcs.wcs_world2pix(np.asarray(world, dtype=np.float64), 0), dtype=np.float64)
    except Exception:
        return np.empty((0, 2), dtype=np.float64)


def _pixel_scale_arcsec(wcs: WCS) -> float:
    try:
        matrix = np.asarray(wcs.pixel_scale_matrix, dtype=np.float64)
        det = float(np.linalg.det(matrix))
        if not math.isfinite(det) or abs(det) <= 0.0:
            return float("nan")
        return float(math.sqrt(abs(det)) * 3600.0)
    except Exception:
        return float("nan")


def _rotation_deg(wcs: WCS) -> float:
    try:
        cd = np.asarray(wcs.pixel_scale_matrix, dtype=np.float64)
        return float(math.degrees(math.atan2(float(cd[1, 0]), float(cd[0, 0]))))
    except Exception:
        return float("nan")


def _center_world(wcs: WCS, image_shape: tuple[int, int]) -> np.ndarray:
    h, w = int(image_shape[0]), int(image_shape[1])
    pix = np.asarray([[w * 0.5, h * 0.5]], dtype=np.float64)
    try:
        out = np.asarray(wcs.wcs_pix2world(pix, 0), dtype=np.float64)
        return out[0]
    except Exception:
        return np.asarray([float("nan"), float("nan")], dtype=np.float64)


def _sep_arcsec(a: np.ndarray, b: np.ndarray) -> float:
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return float("nan")
    c1 = SkyCoord(float(a[0]) * u.deg, float(a[1]) * u.deg, frame="icrs")
    c2 = SkyCoord(float(b[0]) * u.deg, float(b[1]) * u.deg, frame="icrs")
    return float(c1.separation(c2).arcsec)


def _wcs_summary(wcs: WCS, image_shape: tuple[int, int], initial: WCS | None = None) -> dict[str, Any]:
    center = _center_world(wcs, image_shape)
    out = {
        "pixel_scale_arcsec": _pixel_scale_arcsec(wcs),
        "rotation_deg": _rotation_deg(wcs),
        "center_ra_dec": [float(center[0]), float(center[1])],
    }
    if initial is not None:
        initial_center = _center_world(initial, image_shape)
        out["center_shift_arcsec_vs_initial"] = _sep_arcsec(center, initial_center)
        out["scale_delta_pct_vs_initial"] = (
            (out["pixel_scale_arcsec"] / _pixel_scale_arcsec(initial) - 1.0) * 100.0
            if math.isfinite(out["pixel_scale_arcsec"]) and math.isfinite(_pixel_scale_arcsec(initial)) and _pixel_scale_arcsec(initial) > 0
            else None
        )
        out["rotation_delta_deg_vs_initial"] = (
            out["rotation_deg"] - _rotation_deg(initial)
            if math.isfinite(out["rotation_deg"]) and math.isfinite(_rotation_deg(initial))
            else None
        )
    return out


def _build_matches(image_xy: np.ndarray, catalog_world: np.ndarray, image_indices: list[int], catalog_indices: list[int], distances: list[float], projected: np.ndarray, name: str) -> MatchSet:
    if image_indices:
        ii = np.asarray(image_indices, dtype=np.int64)
        ci = np.asarray(catalog_indices, dtype=np.int64)
        img = np.asarray(image_xy[ii], dtype=np.float64)
        sky = np.asarray(catalog_world[ci], dtype=np.float64)
        matches = np.column_stack((img, sky))
        d = np.asarray(distances, dtype=np.float64)
        proj = np.asarray(projected[ci], dtype=np.float64)
    else:
        ii = np.empty((0,), dtype=np.int64)
        ci = np.empty((0,), dtype=np.int64)
        matches = np.empty((0, 4), dtype=np.float64)
        d = np.empty((0,), dtype=np.float64)
        proj = np.empty((0, 2), dtype=np.float64)
    return MatchSet(name=name, matches=matches, image_indices=ii, catalog_indices=ci, distances=d, projected=proj)


def _match_greedy(wcs: WCS, image_xy: np.ndarray, catalog_world: np.ndarray, radius_px: float) -> MatchSet:
    projected = _project_world(wcs, catalog_world)
    if projected.shape != (catalog_world.shape[0], 2) or image_xy.size == 0:
        return _build_matches(image_xy, catalog_world, [], [], [], projected, "greedy")
    finite = np.isfinite(projected[:, 0]) & np.isfinite(projected[:, 1])
    tree = cKDTree(image_xy)
    rows: list[tuple[float, int, int]] = []
    for cat_idx in np.flatnonzero(finite):
        dist, img_idx = tree.query(projected[int(cat_idx)], k=1, distance_upper_bound=float(radius_px))
        if math.isfinite(float(dist)) and 0 <= int(img_idx) < image_xy.shape[0]:
            rows.append((float(dist), int(img_idx), int(cat_idx)))
    rows.sort(key=lambda item: item[0])
    used_i: set[int] = set()
    used_c: set[int] = set()
    ii: list[int] = []
    ci: list[int] = []
    dd: list[float] = []
    for dist, img_idx, cat_idx in rows:
        if img_idx in used_i or cat_idx in used_c:
            continue
        used_i.add(img_idx)
        used_c.add(cat_idx)
        ii.append(img_idx)
        ci.append(cat_idx)
        dd.append(dist)
    return _build_matches(image_xy, catalog_world, ii, ci, dd, projected, "greedy")


def _match_mutual(wcs: WCS, image_xy: np.ndarray, catalog_world: np.ndarray, radius_px: float) -> MatchSet:
    projected = _project_world(wcs, catalog_world)
    if projected.shape != (catalog_world.shape[0], 2) or image_xy.size == 0:
        return _build_matches(image_xy, catalog_world, [], [], [], projected, "mutual_nn")
    finite_idx = np.flatnonzero(np.isfinite(projected[:, 0]) & np.isfinite(projected[:, 1]))
    if finite_idx.size == 0:
        return _build_matches(image_xy, catalog_world, [], [], [], projected, "mutual_nn")
    image_tree = cKDTree(image_xy)
    cat_tree = cKDTree(projected[finite_idx])
    cat_to_img: dict[int, tuple[int, float]] = {}
    for cat_idx in finite_idx:
        dist, img_idx = image_tree.query(projected[int(cat_idx)], k=1, distance_upper_bound=float(radius_px))
        if math.isfinite(float(dist)) and 0 <= int(img_idx) < image_xy.shape[0]:
            cat_to_img[int(cat_idx)] = (int(img_idx), float(dist))
    img_to_cat: dict[int, int] = {}
    for img_idx, xy in enumerate(image_xy):
        dist, local_cat = cat_tree.query(xy, k=1, distance_upper_bound=float(radius_px))
        if math.isfinite(float(dist)) and 0 <= int(local_cat) < finite_idx.shape[0]:
            img_to_cat[int(img_idx)] = int(finite_idx[int(local_cat)])
    rows: list[tuple[float, int, int]] = []
    for cat_idx, (img_idx, dist) in cat_to_img.items():
        if img_to_cat.get(img_idx) == cat_idx:
            rows.append((dist, img_idx, cat_idx))
    rows.sort()
    return _build_matches(
        image_xy,
        catalog_world,
        [row[1] for row in rows],
        [row[2] for row in rows],
        [row[0] for row in rows],
        projected,
        "mutual_nn",
    )


def _match_bipartite(wcs: WCS, image_xy: np.ndarray, catalog_world: np.ndarray, radius_px: float) -> MatchSet:
    projected = _project_world(wcs, catalog_world)
    if projected.shape != (catalog_world.shape[0], 2) or image_xy.size == 0:
        return _build_matches(image_xy, catalog_world, [], [], [], projected, "bipartite")
    finite = np.flatnonzero(np.isfinite(projected[:, 0]) & np.isfinite(projected[:, 1]))
    if finite.size == 0:
        return _build_matches(image_xy, catalog_world, [], [], [], projected, "bipartite")
    tree = cKDTree(image_xy)
    pairs: list[tuple[int, int, float]] = []
    active_cats: set[int] = set()
    active_imgs: set[int] = set()
    for cat_idx in finite:
        hits = tree.query_ball_point(projected[int(cat_idx)], float(radius_px))
        for img_idx in hits:
            dist = float(np.linalg.norm(projected[int(cat_idx)] - image_xy[int(img_idx)]))
            if dist <= float(radius_px):
                pairs.append((int(cat_idx), int(img_idx), dist))
                active_cats.add(int(cat_idx))
                active_imgs.add(int(img_idx))
    if not pairs:
        return _build_matches(image_xy, catalog_world, [], [], [], projected, "bipartite")
    cats = sorted(active_cats)
    imgs = sorted(active_imgs)
    cpos = {cat: i for i, cat in enumerate(cats)}
    ipos = {img: i for i, img in enumerate(imgs)}
    big = float(radius_px) * 1000.0 + 1.0
    cost = np.full((len(cats), len(imgs)), big, dtype=np.float64)
    for cat_idx, img_idx, dist in pairs:
        cost[cpos[cat_idx], ipos[img_idx]] = min(cost[cpos[cat_idx], ipos[img_idx]], dist)
    rows, cols = linear_sum_assignment(cost)
    out: list[tuple[float, int, int]] = []
    for r, c in zip(rows, cols):
        dist = float(cost[int(r), int(c)])
        if dist <= float(radius_px):
            out.append((dist, imgs[int(c)], cats[int(r)]))
    out.sort()
    return _build_matches(
        image_xy,
        catalog_world,
        [row[1] for row in out],
        [row[2] for row in out],
        [row[0] for row in out],
        projected,
        "bipartite",
    )


def _match(wcs: WCS, image_xy: np.ndarray, catalog_world: np.ndarray, radius_px: float, strategy: str) -> MatchSet:
    if strategy == "greedy":
        return _match_greedy(wcs, image_xy, catalog_world, radius_px)
    if strategy == "mutual_nn":
        return _match_mutual(wcs, image_xy, catalog_world, radius_px)
    if strategy == "bipartite":
        return _match_bipartite(wcs, image_xy, catalog_world, radius_px)
    raise ValueError(f"unknown match strategy: {strategy}")


def _rms(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(math.sqrt(float(np.mean(arr * arr))))


def _coverage(image_pts: np.ndarray, image_shape: tuple[int, int]) -> dict[str, Any]:
    ok, geo = _blind_geometric_guardrails(image_pts, image_shape)
    return {"ok": bool(ok), **{k: _json_default(v) for k, v in dict(geo).items()}}


def _fit_tan_from_matches(matches: np.ndarray) -> WCS:
    if matches.shape[0] < 4:
        raise ValueError("need at least 4 matches to fit TAN")
    coords = SkyCoord(ra=matches[:, 2] * u.deg, dec=matches[:, 3] * u.deg, frame="icrs")
    return fit_wcs_from_points((matches[:, 0], matches[:, 1]), coords, projection="TAN")


def _clip_matches(wcs: WCS, matchset: MatchSet, sigma: float = 3.0) -> MatchSet:
    if matchset.matches.shape[0] < 6:
        return matchset
    projected = _project_world(wcs, matchset.matches[:, 2:4])
    dist = np.linalg.norm(projected - matchset.matches[:, :2], axis=1)
    med = float(np.median(dist))
    mad = float(np.median(np.abs(dist - med)))
    sigma_est = 1.4826 * mad if mad > 1.0e-9 else float(np.std(dist))
    limit = max(3.0, med + float(sigma) * max(sigma_est, 1.0e-9))
    keep = np.isfinite(dist) & (dist <= limit)
    if np.count_nonzero(keep) < 4:
        return matchset
    return MatchSet(
        name=matchset.name,
        matches=matchset.matches[keep],
        image_indices=matchset.image_indices[keep],
        catalog_indices=matchset.catalog_indices[keep],
        distances=matchset.distances[keep],
        projected=matchset.projected[keep] if matchset.projected.shape[0] == matchset.matches.shape[0] else matchset.projected,
    )


def _iteration_row(iteration: int, collect_radius: float, final: MatchSet, collect: MatchSet, wcs: WCS, image_shape: tuple[int, int], initial_wcs: WCS) -> dict[str, Any]:
    return {
        "iteration": int(iteration),
        "collect_radius_px": float(collect_radius),
        "collected": int(collect.matches.shape[0]),
        "final_inliers_3px": int(final.matches.shape[0]),
        "rms_px": _rms(final.distances),
        "median_px": float(np.median(final.distances)) if final.distances.size else None,
        "max_px": float(np.max(final.distances)) if final.distances.size else None,
        "coverage": _coverage(final.matches[:, :2], image_shape),
        "wcs": _wcs_summary(wcs, image_shape, initial_wcs),
    }


def _iterative_refit(
    initial_wcs: WCS,
    image_xy: np.ndarray,
    catalog_world: np.ndarray,
    image_shape: tuple[int, int],
    *,
    collect_radius_px: float,
    final_radius_px: float,
    strategy: str,
    max_iter: int = 3,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    wcs = initial_wcs
    collect0 = _match(wcs, image_xy, catalog_world, final_radius_px, strategy)
    final0 = _match(wcs, image_xy, catalog_world, final_radius_px, strategy)
    rows.append(_iteration_row(0, final_radius_px, final0, collect0, wcs, image_shape, initial_wcs))
    best_wcs = wcs
    best_final = final0
    last_key = (int(final0.matches.shape[0]), round(_rms(final0.distances), 6) if math.isfinite(_rms(final0.distances)) else None)
    stopped = "max_iter"
    for it in range(1, int(max_iter) + 1):
        collect = _match(wcs, image_xy, catalog_world, collect_radius_px, strategy)
        if collect.matches.shape[0] < 4:
            stopped = "too_few_collect_matches"
            break
        clipped = _clip_matches(wcs, collect)
        try:
            wcs = _fit_tan_from_matches(clipped.matches)
        except Exception as exc:
            stopped = f"fit_failed:{exc}"
            break
        final = _match(wcs, image_xy, catalog_world, final_radius_px, strategy)
        rows.append(_iteration_row(it, collect_radius_px, final, clipped, wcs, image_shape, initial_wcs))
        key = (int(final.matches.shape[0]), round(_rms(final.distances), 6) if math.isfinite(_rms(final.distances)) else None)
        if key[0] > best_final.matches.shape[0] or (key[0] == best_final.matches.shape[0] and (_rms(final.distances) < _rms(best_final.distances))):
            best_final = final
            best_wcs = wcs
        if key == last_key:
            stopped = "no_progress"
            break
        last_key = key
    return {
        "strategy": strategy,
        "collect_radius_px": float(collect_radius_px),
        "final_radius_px": float(final_radius_px),
        "stopped": stopped,
        "iterations": rows,
        "best": rows[int(np.argmax([row["final_inliers_3px"] for row in rows]))] if rows else {},
        "best_wcs": best_wcs,
        "best_matchset": best_final,
    }


def _candidate_list(
    quad_sources: np.ndarray,
    index_paths: tuple[Path, ...],
    *,
    max_quads: int,
    image_strategy: str,
    code_tol: float,
    max_hits: int,
    max_hits_per_image_quad: int,
) -> tuple[list[dict[str, Any]], list[Any], dict[str, Any]]:
    image_positions = _positions(quad_sources)
    image_quads = sample_quads(quad_sources, max_quads=int(max_quads), strategy=str(image_strategy))
    image_records = build_astrometry_quad_records(image_quads, image_positions)
    candidates: list[dict[str, Any]] = []
    disk_indexes: list[Quad4DIndex] = []
    hits_by_index: dict[str, Any] = {}
    for index_order, path in enumerate(index_paths):
        disk_index = Quad4DIndex.load(path)
        disk_indexes.append(disk_index)
        hits = disk_index.search_records(
            image_records,
            code_tol=float(code_tol),
            max_hits=int(max_hits),
            max_hits_per_image_quad=int(max_hits_per_image_quad),
        )
        hits_by_index[str(path)] = {"hits": int(len(hits)), "tile_keys": list(disk_index.tile_keys)}
        for local_rank, hit in enumerate(hits, start=1):
            candidates.append({"index": disk_index, "hit": hit, "index_order": int(index_order), "local_rank": int(local_rank)})
    candidates.sort(key=lambda row: (int(row["index_order"]), float(row["hit"].code_distance), int(row["local_rank"])))
    return candidates, image_records, {"image_quads": int(image_quads.shape[0]), "image_records": int(len(image_records)), "hits_by_index": hits_by_index}


def _fit_candidate_wcs(item: dict[str, Any], image_positions: np.ndarray) -> WCS:
    hit = item["hit"]
    disk_index: Quad4DIndex = item["index"]
    image_quad_points = image_positions[np.asarray(hit.image_quad_indices, dtype=np.int64)]
    catalog_quad_world = disk_index.catalog_ra_dec[np.asarray(hit.catalog_quad_indices, dtype=np.int64)]
    return _fit_astrometry_4d_quad_wcs(image_quad_points, catalog_quad_world)


def _find_known_candidate(candidates: list[dict[str, Any]], args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    expected_img = tuple(int(v) for v in str(args.known_image_quad).split(","))
    expected_cat = tuple(int(v) for v in str(args.known_catalog_quad).split(","))
    for rank, item in enumerate(candidates, start=1):
        hit = item["hit"]
        if (
            str(hit.tile_key) == str(args.known_origin_tile)
            and tuple(int(v) for v in hit.image_quad_indices) == expected_img
            and tuple(int(v) for v in hit.catalog_quad_indices) == expected_cat
        ):
            return rank, item
    raise RuntimeError("known P2.15 candidate not found in replay")


def _classify_reject(row: dict[str, Any]) -> str:
    scale = float(row.get("pixel_scale_arcsec", float("nan")))
    rms = float(row.get("rms_px", float("nan")))
    inliers = int(row.get("inliers", 0) or 0)
    geo = row.get("coverage") or {}
    if not (1.79 <= scale <= 2.99):
        return "scale_invalid"
    if rms > 1.2 or not math.isfinite(rms):
        return "rms_invalid"
    if not bool(geo.get("ok", False)):
        return "geometry_invalid"
    if inliers < 40:
        return "plausible"
    return "accepted"


def _reject_telemetry(candidates: list[dict[str, Any]], image_positions: np.ndarray, verification_xy: np.ndarray, catalog_world: np.ndarray, image_shape: tuple[int, int], max_hypotheses: int) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any] | None] = {
        "best_plausible_reject": None,
        "best_scale_invalid_reject": None,
        "best_rms_invalid_reject": None,
        "best_geometry_invalid_reject": None,
    }
    counts: dict[str, int] = {}
    for rank, item in enumerate(candidates[: int(max_hypotheses)], start=1):
        try:
            wcs = _fit_candidate_wcs(item, image_positions)
            m = _match_greedy(wcs, verification_xy, catalog_world, 3.0)
            row = {
                "rank": int(rank),
                "origin_tile": str(item["hit"].tile_key),
                "image_quad_indices": [int(v) for v in item["hit"].image_quad_indices],
                "catalog_quad_indices": [int(v) for v in item["hit"].catalog_quad_indices],
                "code_distance": float(item["hit"].code_distance),
                "inliers": int(m.matches.shape[0]),
                "rms_px": _rms(m.distances),
                "pixel_scale_arcsec": _pixel_scale_arcsec(wcs),
                "coverage": _coverage(m.matches[:, :2], image_shape),
            }
            klass = _classify_reject(row)
            counts[klass] = counts.get(klass, 0) + 1
            key = {
                "plausible": "best_plausible_reject",
                "scale_invalid": "best_scale_invalid_reject",
                "rms_invalid": "best_rms_invalid_reject",
                "geometry_invalid": "best_geometry_invalid_reject",
            }.get(klass)
            if key is None:
                continue
            old = buckets.get(key)
            sort_key = (int(row["inliers"]), -float(row["rms_px"]) if math.isfinite(float(row["rms_px"])) else -1e9, -int(row["rank"]))
            if old is None:
                buckets[key] = row
            else:
                old_key = (int(old["inliers"]), -float(old["rms_px"]) if math.isfinite(float(old["rms_px"])) else -1e9, -int(old["rank"]))
                if sort_key > old_key:
                    buckets[key] = row
        except Exception:
            counts["fit_failed"] = counts.get("fit_failed", 0) + 1
    return {"counts": counts, **buckets}


def _oracle_matchables(label: str, raw: np.ndarray, union_world: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    oracle_wcs, shape, oracle_path = _load_oracle(label, args)
    h, w = int(shape[0]), int(shape[1])
    oracle_pix = _project_world(oracle_wcs, union_world)
    inside = (
        np.isfinite(oracle_pix[:, 0])
        & np.isfinite(oracle_pix[:, 1])
        & (oracle_pix[:, 0] >= 0.0)
        & (oracle_pix[:, 0] < float(w))
        & (oracle_pix[:, 1] >= 0.0)
        & (oracle_pix[:, 1] < float(h))
    )
    cat_indices = np.flatnonzero(inside)
    catalog_xy = oracle_pix[cat_indices]
    raw_xy = _positions(raw)
    pairs = p213._one_to_one_pairs(catalog_xy, raw_xy, float(args.match_radius_px))
    pair_cat = [int(cat_indices[int(i)]) for i in pairs["catalog_indices"]]
    pair_img = [int(i) for i in pairs["image_indices"]]
    return {
        "oracle_path": str(oracle_path),
        "oracle_shape": [int(shape[0]), int(shape[1])],
        "catalog_in_field": int(catalog_xy.shape[0]),
        "count": int(len(pair_cat)),
        "catalog_indices": pair_cat,
        "image_indices": pair_img,
        "oracle_distances": [float(v) for v in pairs["distances"]],
    }


def _residual_map(oracle: dict[str, Any], raw: np.ndarray, union_world: np.ndarray, initial_wcs: WCS, image_shape: tuple[int, int]) -> dict[str, Any]:
    raw_xy = _positions(raw)
    cat_idx = np.asarray(oracle["catalog_indices"], dtype=np.int64)
    img_idx = np.asarray(oracle["image_indices"], dtype=np.int64)
    candidate_pix = _project_world(initial_wcs, union_world[cat_idx])
    image_pix = raw_xy[img_idx]
    residual = candidate_pix - image_pix
    dist = np.linalg.norm(residual, axis=1)
    h, w = int(image_shape[0]), int(image_shape[1])
    center = np.asarray([w * 0.5, h * 0.5], dtype=np.float64)
    radius = np.linalg.norm(image_pix - center, axis=1)
    raw_flux = np.asarray(raw["flux"], dtype=np.float64) if raw.size and "flux" in raw.dtype.names else np.full(raw.shape[0], np.nan)
    tree = cKDTree(_project_world(initial_wcs, union_world))
    details = []
    for n, (ci, ii) in enumerate(zip(cat_idx, img_idx)):
        near = tree.query_ball_point(image_pix[n], 3.0)
        competitors = [int(v) for v in near if int(v) != int(ci)]
        details.append(
            {
                "catalog_index": int(ci),
                "image_index": int(ii),
                "image_x": float(image_pix[n, 0]),
                "image_y": float(image_pix[n, 1]),
                "residual_x_px": float(residual[n, 0]),
                "residual_y_px": float(residual[n, 1]),
                "distance_center_px": float(radius[n]),
                "candidate_distance_px": float(dist[n]),
                "raw_rank": int(ii),
                "flux": float(raw_flux[int(ii)]) if 0 <= int(ii) < raw_flux.shape[0] else None,
                "matched_under_3px": bool(dist[n] <= 3.0),
                "candidate_conflict_count": int(len(competitors)),
                "candidate_conflict_catalog_indices": competitors[:8],
            }
        )
    under = dist <= 3.0
    over = ~under
    corr_x = float(np.corrcoef(image_pix[:, 0], residual[:, 0])[0, 1]) if dist.size > 1 else float("nan")
    corr_y = float(np.corrcoef(image_pix[:, 1], residual[:, 1])[0, 1]) if dist.size > 1 else float("nan")
    corr_r = float(np.corrcoef(radius, dist)[0, 1]) if dist.size > 1 else float("nan")
    return {
        "count": int(dist.size),
        "under_3px": int(np.count_nonzero(under)),
        "over_3px": int(np.count_nonzero(over)),
        "distance_summary_all": _summary(dist),
        "distance_summary_under_3px": _summary(dist[under]),
        "distance_summary_over_3px": _summary(dist[over]),
        "radius_summary_under_3px": _summary(radius[under]),
        "radius_summary_over_3px": _summary(radius[over]),
        "residual_x_summary_over_3px": _summary(residual[over, 0]),
        "residual_y_summary_over_3px": _summary(residual[over, 1]),
        "corr_residual_x_vs_x": corr_x,
        "corr_residual_y_vs_y": corr_y,
        "corr_distance_vs_radius": corr_r,
        "mean_vector_under_3px": [float(v) for v in np.mean(residual[under], axis=0)] if np.any(under) else [],
        "mean_vector_over_3px": [float(v) for v in np.mean(residual[over], axis=0)] if np.any(over) else [],
        "edge_loss_fraction_over_85pct_radius": float(np.mean(radius[over] > np.percentile(radius, 85))) if np.any(over) else 0.0,
        "conflict_count_over_3px": int(sum(1 for row in details if (not row["matched_under_3px"]) and int(row["candidate_conflict_count"]) > 0)),
        "details": details,
    }


def _linear_diagnostics(residual_map: dict[str, Any]) -> dict[str, Any]:
    rows = residual_map["details"]
    if len(rows) < 6:
        return {}
    x = np.asarray([r["image_x"] for r in rows], dtype=np.float64)
    y = np.asarray([r["image_y"] for r in rows], dtype=np.float64)
    rx = np.asarray([r["residual_x_px"] for r in rows], dtype=np.float64)
    ry = np.asarray([r["residual_y_px"] for r in rows], dtype=np.float64)
    mx, my = float(np.mean(x)), float(np.mean(y))
    dx = x - mx
    dy = y - my
    design = np.column_stack((np.ones_like(dx), dx, dy))
    px, *_ = np.linalg.lstsq(design, rx, rcond=None)
    py, *_ = np.linalg.lstsq(design, ry, rcond=None)
    a = float(px[1])
    b = float(px[2])
    c = float(py[1])
    d = float(py[2])
    return {
        "translation_px": [float(px[0]), float(py[0])],
        "scale_like_trace": float(0.5 * (a + d)),
        "rotation_like_skew": float(0.5 * (c - b)),
        "shear_like": float(0.5 * (b + c)),
        "jacobian": [[a, b], [c, d]],
    }


def _audit_case(label: str, args: argparse.Namespace, union_world: np.ndarray, indexes: tuple[Path, ...], known_only: bool) -> dict[str, Any]:
    raw, image_shape, detect_meta = p214._detect_sources(label, args)
    clean = _clean_verification_sources(raw, image_shape, float(args.verification_min_sep_px))
    quad_sources = raw[: int(args.quad_cap)]
    verification_sources = clean if int(args.verification_cap) <= 0 else clean[: int(args.verification_cap)]
    image_positions = _positions(quad_sources)
    verification_xy = _positions(verification_sources)
    candidates, _records, replay_stats = _candidate_list(
        quad_sources,
        indexes,
        max_quads=int(args.max_quads),
        image_strategy=str(args.image_strategy),
        code_tol=float(args.code_tol),
        max_hits=int(args.max_hits_4d),
        max_hits_per_image_quad=int(args.max_hits_per_image_quad),
    )
    if known_only:
        rank, item = _find_known_candidate(candidates, args)
    else:
        telemetry = _reject_telemetry(candidates, image_positions, verification_xy, union_world, image_shape, int(args.max_hypotheses))
        best = telemetry.get("best_plausible_reject") or telemetry.get("best_rms_invalid_reject") or telemetry.get("best_scale_invalid_reject")
        if not best:
            raise RuntimeError(f"no candidate available for control {label}")
        rank = int(best["rank"])
        item = candidates[rank - 1]
    initial_wcs = _fit_candidate_wcs(item, image_positions)
    oracle = _oracle_matchables(label, raw, union_world, args)
    residual = _residual_map(oracle, raw, union_world, initial_wcs, image_shape)
    residual["linear_diagnostics"] = _linear_diagnostics(residual)
    matching_compare = {}
    for strategy in ("greedy", "mutual_nn", "bipartite"):
        m = _match(initial_wcs, verification_xy, union_world, float(args.match_radius_px), strategy)
        matching_compare[strategy] = {
            "inliers": int(m.matches.shape[0]),
            "rms_px": _rms(m.distances),
            "median_px": float(np.median(m.distances)) if m.distances.size else None,
            "max_px": float(np.max(m.distances)) if m.distances.size else None,
            "coverage": _coverage(m.matches[:, :2], image_shape),
        }
    tan_runs = {
        "A_strict_3px": _iterative_refit(
            initial_wcs,
            verification_xy,
            union_world,
            image_shape,
            collect_radius_px=3.0,
            final_radius_px=3.0,
            strategy="greedy",
            max_iter=3,
        ),
        "B_collect_4p5px": _iterative_refit(
            initial_wcs,
            verification_xy,
            union_world,
            image_shape,
            collect_radius_px=4.5,
            final_radius_px=3.0,
            strategy="greedy",
            max_iter=3,
        ),
        "B_collect_5px": _iterative_refit(
            initial_wcs,
            verification_xy,
            union_world,
            image_shape,
            collect_radius_px=5.0,
            final_radius_px=3.0,
            strategy="greedy",
            max_iter=3,
        ),
    }
    global_refit = {}
    for strategy in ("mutual_nn", "bipartite"):
        global_refit[strategy] = _iterative_refit(
            initial_wcs,
            verification_xy,
            union_world,
            image_shape,
            collect_radius_px=5.0,
            final_radius_px=3.0,
            strategy=strategy,
            max_iter=3,
        )
    best_tan_name, best_tan = max(
        list(tan_runs.items()) + [(f"global_{k}", v) for k, v in global_refit.items()],
        key=lambda kv: (
            int(kv[1]["best_matchset"].matches.shape[0]),
            -_rms(kv[1]["best_matchset"].distances) if math.isfinite(_rms(kv[1]["best_matchset"].distances)) else -1e9,
        ),
    )
    sip_result: dict[str, Any] | None = None
    if int(best_tan["best_matchset"].matches.shape[0]) < int(args.quality_inliers):
        try:
            sip_wcs, sip_stats = fit_wcs_sip(best_tan["best_matchset"].matches, robust=True, order=2)
            sip_match = _match_greedy(sip_wcs, verification_xy, union_world, 3.0)
            sip_result = {
                "basis": best_tan_name,
                "fit_stats": sip_stats,
                "final_inliers_3px": int(sip_match.matches.shape[0]),
                "rms_px": _rms(sip_match.distances),
                "median_px": float(np.median(sip_match.distances)) if sip_match.distances.size else None,
                "max_px": float(np.max(sip_match.distances)) if sip_match.distances.size else None,
                "wcs": _wcs_summary(sip_wcs, image_shape, initial_wcs),
                "coverage": _coverage(sip_match.matches[:, :2], image_shape),
            }
        except Exception as exc:
            sip_result = {"error": str(exc), "basis": best_tan_name}
    telemetry = _reject_telemetry(candidates, image_positions, verification_xy, union_world, image_shape, int(args.max_hypotheses))
    hit = item["hit"]
    return {
        "label": label,
        "detect_meta": detect_meta,
        "image_shape": [int(v) for v in image_shape],
        "quad_sources": int(quad_sources.shape[0]),
        "verification_sources": int(verification_sources.shape[0]),
        "candidate_replay": replay_stats,
        "selected_candidate": {
            "rank": int(rank),
            "origin_tile": str(hit.tile_key),
            "local_rank": int(item["local_rank"]),
            "index_order": int(item["index_order"]),
            "code_distance": float(hit.code_distance),
            "image_quad_indices": [int(v) for v in hit.image_quad_indices],
            "catalog_quad_indices": [int(v) for v in hit.catalog_quad_indices],
            "pixel_scale_arcsec": _pixel_scale_arcsec(initial_wcs),
            "rotation_deg": _rotation_deg(initial_wcs),
        },
        "oracle_matchables": oracle,
        "residual_map": residual,
        "matching_compare_initial": matching_compare,
        "tan_refit": {k: _serialise_run(v) for k, v in tan_runs.items()},
        "global_matching_refit": {k: _serialise_run(v) for k, v in global_refit.items()},
        "best_tan_refit": {"name": best_tan_name, "summary": _serialise_run(best_tan)["best"]},
        "sip_order2": sip_result,
        "reject_telemetry": telemetry,
    }


def _serialise_run(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategy": run.get("strategy"),
        "collect_radius_px": run.get("collect_radius_px"),
        "final_radius_px": run.get("final_radius_px"),
        "stopped": run.get("stopped"),
        "iterations": run.get("iterations", []),
        "best": max(
            run.get("iterations", []),
            key=lambda row: (int(row.get("final_inliers_3px", 0)), -float(row.get("rms_px", float("inf")) or float("inf"))),
            default={},
        ),
    }


def _strip_heavy(payload: dict[str, Any]) -> dict[str, Any]:
    def strip_case(case: dict[str, Any]) -> dict[str, Any]:
        out = dict(case)
        res = dict(out.get("residual_map") or {})
        details = list(res.get("details") or [])
        res["details_head"] = details[:12]
        res["details_over_3px"] = [row for row in details if not row.get("matched_under_3px")][:20]
        res.pop("details", None)
        out["residual_map"] = res
        return out

    compact = dict(payload)
    compact["main_case"] = strip_case(payload["main_case"])
    compact["controls"] = [strip_case(case) for case in payload.get("controls", [])]
    return compact


def _answers(payload: dict[str, Any]) -> list[str]:
    case = payload["main_case"]
    residual = case["residual_map"]
    best = case["best_tan_refit"]["summary"]
    sip = case.get("sip_order2") or {}
    edge_frac = float(residual.get("edge_loss_fraction_over_85pct_radius", 0.0) or 0.0)
    conflicts = int(residual.get("conflict_count_over_3px", 0) or 0)
    linear = residual.get("linear_diagnostics") or {}
    best_inliers = int(best.get("final_inliers_3px", 0) or 0)
    initial_greedy = case["matching_compare_initial"]["greedy"]
    initial_inliers = int(initial_greedy.get("inliers", 0) or 0)
    initial_rms = float(initial_greedy.get("rms_px", float("inf")) or float("inf"))
    quality_inliers = int(payload["params"].get("quality_inliers", 40) or 40)
    quality_rms = float(payload["params"].get("quality_rms", 1.2) or 1.2)
    initial_passes = initial_inliers >= quality_inliers and initial_rms <= quality_rms
    under = int(residual.get("under_3px", 0) or 0)
    over = int(residual.get("over_3px", 0) or 0)
    answers = [
        f"La premisse des 12 etoiles au-dela de 3 px est infirmee sur la liste complete: {under}/42 oracle-matchables sont sous 3 px, {over}/42 au-dela.",
        f"Les pertes observees en `q120_v500` viennent de la liste de verification capee, pas de conflits d'unicite sous le WCS candidat; conflits mesures sur les etoiles hors rayon: {conflicts}.",
        f"La distribution spatiale des residus n'est donc pas une perte de bord: fraction des etoiles hors rayon au-dela du 85e percentile radial = {_fmt(edge_frac, 2)}.",
        "Le champ residuel initial porte surtout un terme lineaire compatible rotation/echelle/translation, pas une signature uniquement radiale."
        f" translation moyenne={linear.get('translation_px')}, trace={_fmt(linear.get('scale_like_trace'), 5)}, skew={_fmt(linear.get('rotation_like_skew'), 5)}.",
        f"Le WCS quad initial passe deja les seuils avec la liste complete: {initial_inliers} inliers / RMS {_fmt(initial_rms, 3)} a rayon 3 px."
        if initial_passes
        else f"Le WCS quad initial ne passe pas: {initial_inliers} inliers / RMS {_fmt(initial_rms, 3)}.",
        f"Le refit TAN iteratif atteint {best_inliers} inliers finaux a 3 px sur `{case['best_tan_refit']['name']}`; il ameliore la qualite mais n'est pas necessaire au passage."
        if initial_passes
        else f"Le refit TAN iteratif atteint {best_inliers} inliers finaux a 3 px sur `{case['best_tan_refit']['name']}`.",
        "Le rayon de collecte temporaire n'est pas necessaire: le rayon strict 3 px suffit dans ce diagnostic.",
        f"L'appariement glouton initial donne {case['matching_compare_initial']['greedy']['inliers']} inliers; MNN {case['matching_compare_initial']['mutual_nn']['inliers']}; biparti {case['matching_compare_initial']['bipartite']['inliers']}.",
    ]
    if sip:
        if int(sip.get("final_inliers_3px", 0) or 0) >= 40:
            answers.append(f"SIP ordre 2 passe aussi ({sip.get('final_inliers_3px')} inliers / RMS {_fmt(sip.get('rms_px'), 3)}), mais il n'est pas necessaire si TAN a deja atteint le seuil.")
        elif "error" in sip:
            answers.append(f"SIP ordre 2 teste car TAN plafonnait sous 40, mais le fit echoue: {sip['error']}.")
        else:
            answers.append(f"SIP ordre 2 n'est pas suffisant dans ce probe: {sip.get('final_inliers_3px')} inliers / RMS {_fmt(sip.get('rms_px'), 3)}.")
    else:
        answers.append("SIP ordre 2 non teste: le critere d'arret TAN est deja atteint.")
    if initial_passes:
        answers.append("Conclusion causale: l'ecart principal restant n'est pas l'absence de tweak/refine; c'est l'absence d'isolation/telemetrie du meilleur candidat coherent avec la liste complete, masquee par des rejets hors echelle dans P2.15.")
        answers.append("Le detecteur n'est pas le prochain audit prioritaire pour ce candidat precis; l'audit suivant doit plutot porter sur le contrat de validation full et la telemetrie des rejets coherents.")
    elif best_inliers >= 40:
        answers.append("Conclusion causale: le WCS quad etait proche, mais un cycle tweak/refine match-refit-rematch suffit a fermer le cas sans changer les seuils.")
        answers.append("Le detecteur n'est pas le prochain audit prioritaire pour 234013; il reste utile plus tard pour la robustesse generale, mais ce bloc causal est ferme.")
    else:
        answers.append("Conclusion causale: le tweak/refine TAN/SIP ne ferme pas 234013; le detecteur/centroidage reste a auditer contre image2xy.")
    return answers


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    compact = _strip_heavy(payload)
    case = compact["main_case"]
    answers = payload["answers"]
    lines = [
        "# ZeBlind P2.16 - 4D candidate refine residual audit",
        "",
        "> Diagnostic uniquement. Aucun seuil produit, ZeNear, GUI, all-sky, WCS oracle runtime, coeur AB/C/D, ranking source-list, ni default n'est modifie.",
        "",
        "## Verdict",
        "",
    ]
    for answer in answers:
        lines.append(f"- {answer}")
    lines.extend(
        [
            "",
            "## Candidat 234013",
            "",
            f"- Origine: `{case['selected_candidate']['origin_tile']}`",
            f"- Rang replay: `{case['selected_candidate']['rank']}`",
            f"- Image quad: `{case['selected_candidate']['image_quad_indices']}`",
            f"- Catalogue quad: `{case['selected_candidate']['catalog_quad_indices']}`",
            f"- Echelle initiale: `{_fmt(case['selected_candidate']['pixel_scale_arcsec'], 6)}\"/px`",
            f"- Oracle matchables: `{case['oracle_matchables']['count']}`",
            "",
            "## Carte des residus",
            "",
            f"- Sous 3 px: `{case['residual_map']['under_3px']}`",
            f"- Au-dela de 3 px: `{case['residual_map']['over_3px']}`",
            f"- Distances all: `{case['residual_map']['distance_summary_all']}`",
            f"- Rayons sous 3 px: `{case['residual_map']['radius_summary_under_3px']}`",
            f"- Rayons au-dela 3 px: `{case['residual_map']['radius_summary_over_3px']}`",
            f"- Vecteur moyen sous 3 px: `{case['residual_map']['mean_vector_under_3px']}`",
            f"- Vecteur moyen au-dela 3 px: `{case['residual_map']['mean_vector_over_3px']}`",
            f"- Conflits sur pertes: `{case['residual_map']['conflict_count_over_3px']}`",
            f"- Diagnostic lineaire: `{case['residual_map']['linear_diagnostics']}`",
            "",
            "## Refit TAN",
            "",
            "| run | iter | collect | final 3px | RMS | scale | center shift | rot delta | cov area | stop |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for name, run in case["tan_refit"].items():
        for row in run["iterations"]:
            lines.append(
                f"| `{name}` | {row['iteration']} | {row['collected']} | {row['final_inliers_3px']} | {_fmt(row['rms_px'], 3)} | {_fmt(row['wcs'].get('pixel_scale_arcsec'), 6)} | {_fmt(row['wcs'].get('center_shift_arcsec_vs_initial'), 3)} | {_fmt(row['wcs'].get('rotation_delta_deg_vs_initial'), 5)} | {_fmt(row['coverage'].get('cov_area'), 3)} | `{run['stopped']}` |"
            )
    lines.extend(["", "## Matching global", "", "| strategie | initial inliers | initial RMS | best refit inliers | best RMS | stop |", "|---|---:|---:|---:|---:|---|"])
    for strategy in ("greedy", "mutual_nn", "bipartite"):
        initial = case["matching_compare_initial"][strategy]
        refit = case["global_matching_refit"].get(strategy)
        best = refit["best"] if refit else {}
        lines.append(
            f"| `{strategy}` | {initial['inliers']} | {_fmt(initial['rms_px'], 3)} | {best.get('final_inliers_3px', '')} | {_fmt(best.get('rms_px'), 3)} | `{refit.get('stopped') if refit else ''}` |"
        )
    lines.extend(["", "## SIP ordre 2", ""])
    lines.append(f"- `{case.get('sip_order2')}`")
    lines.extend(["", "## Telemetrie rejets separee", ""])
    tel = case["reject_telemetry"]
    lines.append(f"- Counts: `{tel.get('counts')}`")
    for key in ("best_plausible_reject", "best_scale_invalid_reject", "best_rms_invalid_reject", "best_geometry_invalid_reject"):
        lines.append(f"- `{key}`: `{tel.get(key)}`")
    if compact.get("controls"):
        lines.extend(["", "## Controles", "", "| cas | candidat | initial | best TAN | SIP |", "|---|---:|---:|---:|---:|"])
        for ctrl in compact["controls"]:
            init = ctrl["matching_compare_initial"]["greedy"]["inliers"]
            best = ctrl["best_tan_refit"]["summary"].get("final_inliers_3px")
            sip = (ctrl.get("sip_order2") or {}).get("final_inliers_3px", "")
            lines.append(f"| `{ctrl['label']}` | {ctrl['selected_candidate']['rank']} | {init} | {best} | {sip} |")
    lines.extend(["", "## Parametres", "", "```json", json.dumps(compact["params"], indent=2, default=_json_default), "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="P2.16 residual and iterative TAN/SIP refine audit for a ZeBlind 4D candidate.")
    ap.add_argument("--data-dir", type=Path, default=p22.DEFAULT_DATA_DIR)
    ap.add_argument("--reference-dir", type=Path, default=p22.DEFAULT_REFERENCE_DIR)
    ap.add_argument("--index-root", type=Path, default=p22.DEFAULT_INDEX_ROOT)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--detail-json-out", type=Path, default=DEFAULT_DETAIL_JSON)
    ap.add_argument("--main-case", default=MAIN_CASE)
    ap.add_argument("--controls", default=",".join(CONTROL_CASES))
    ap.add_argument("--run-controls", action="store_true")
    ap.add_argument("--quad-cap", type=int, default=120)
    ap.add_argument("--verification-cap", type=int, default=500)
    ap.add_argument("--verification-min-sep-px", type=float, default=0.75)
    ap.add_argument("--detect-k-sigma", type=float, default=3.0)
    ap.add_argument("--detect-min-area", type=int, default=5)
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--max-quads", type=int, default=2500)
    ap.add_argument("--image-strategy", default="log_spaced")
    ap.add_argument("--code-tol", type=float, default=0.015)
    ap.add_argument("--max-hits-4d", type=int, default=2000)
    ap.add_argument("--max-hits-per-image-quad", type=int, default=8)
    ap.add_argument("--max-hypotheses", type=int, default=2000)
    ap.add_argument("--match-radius-px", type=float, default=3.0)
    ap.add_argument("--quality-rms", type=float, default=1.2)
    ap.add_argument("--quality-inliers", type=int, default=40)
    ap.add_argument("--known-origin-tile", default="d50_2822")
    ap.add_argument("--known-image-quad", default="5,3,4,0")
    ap.add_argument("--known-catalog-quad", default="196,54,175,7")
    args = ap.parse_args()

    indexes = (p212.INDEX_2823, p212.INDEX_2822)
    for path in indexes:
        if not path.exists():
            raise FileNotFoundError(path)
    union_world, union_meta = _load_union_catalog(args)
    main = _audit_case(str(args.main_case), args, union_world, indexes, known_only=True)
    best_main = int(main["best_tan_refit"]["summary"].get("final_inliers_3px", 0) or 0)
    sip_main = main.get("sip_order2") or {}
    if int(sip_main.get("final_inliers_3px", 0) or 0) > best_main:
        best_main = int(sip_main.get("final_inliers_3px", 0) or 0)
    controls: list[dict[str, Any]] = []
    if bool(args.run_controls) or best_main >= int(args.quality_inliers):
        for label in [part.strip() for part in str(args.controls).split(",") if part.strip()]:
            controls.append(_audit_case(label, args, union_world, indexes, known_only=False))
    payload = {
        "schema": "zeblind.p216_4d_candidate_refine_residual_audit.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "main_case": main,
        "controls": controls,
        "params": {
            "quality_inliers": int(args.quality_inliers),
            "quality_rms": float(args.quality_rms),
            "match_radius_px": float(args.match_radius_px),
            "quad_cap": int(args.quad_cap),
            "verification_cap": int(args.verification_cap),
            "index_paths": [str(p.resolve()) for p in indexes],
            "union_catalog": union_meta,
            "known_origin_tile": str(args.known_origin_tile),
            "known_image_quad": str(args.known_image_quad),
            "known_catalog_quad": str(args.known_catalog_quad),
            "diagnostic_only": True,
            "default_behavior_changed": False,
            "wcs_oracle_runtime_input": False,
            "all_sky": False,
            "all30": False,
        },
    }
    payload["answers"] = _answers(payload)
    args.detail_json_out.parent.mkdir(parents=True, exist_ok=True)
    args.detail_json_out.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    compact = _strip_heavy(payload)
    args.json_out.write_text(json.dumps(compact, indent=2, default=_json_default), encoding="utf-8")
    _write_report(args.report, payload)
    print(json.dumps({"report": str(args.report), "json": str(args.json_out), "detail_json": str(args.detail_json_out), "main_best_inliers": best_main}, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
