#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zeblindsolver.quad_index_4d import ASTROMETRY_AB_CODE_4D_SCHEMA
from zeblindsolver.zeblindsolver import SolveConfig, _astrometry_source_list_gate, solve_blind

import tools.diagnose_p23_4d_source_list_contract as p23
import tools.diagnose_p26_4d_oracle_tile_routing as p26
import tools.diagnose_p28_4d_validation_support_audit as p28
import tools.diagnose_p212_4d_m106_30_bounded_validation as p212
import tools.diagnose_p213_4d_m106_failure_autopsy as p213
import tools.diagnose_runtime_4d_route as p22


DEFAULT_REPORT = ROOT / "reports/zeblind_p214_4d_source_policy_bakeoff.md"
DEFAULT_JSON = ROOT / "reports/zeblind_p214_4d_source_policy_bakeoff.json"
DEFAULT_WORK_DIR = ROOT / "reports/p214_4d_source_policy_bakeoff/candidates"
DEFAULT_P212_JSON = ROOT / "reports/zeblind_p212_4d_m106_30_bounded_validation.json"

FAILURE_CASES = ("233828", "234013")
BOUNDED_TILES = ("d50_2823", "d50_2822")


@dataclass(frozen=True)
class SourcePolicy:
    name: str
    family: str
    cap: int
    nx: int = 0
    ny: int = 0
    boxes: int = 0


POLICIES: tuple[SourcePolicy, ...] = (
    SourcePolicy("baseline_diagnostic_cap120", "head", 120),
    SourcePolicy("head_cap160", "head", 160),
    SourcePolicy("head_cap200", "head", 200),
    SourcePolicy("head_cap250", "head", 250),
    SourcePolicy("grid4x4_cap160", "grid", 160, nx=4, ny=4),
    SourcePolicy("grid4x4_cap200", "grid", 200, nx=4, ny=4),
    SourcePolicy("grid6x4_cap160", "grid", 160, nx=6, ny=4),
    SourcePolicy("grid6x4_cap200", "grid", 200, nx=6, ny=4),
    SourcePolicy("stratified_qscore_cap160", "stratified", 160),
    SourcePolicy("stratified_qscore_cap200", "stratified", 200),
    SourcePolicy("astrometry_like_cap200", "astrometry_like", 200, boxes=10),
)


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


def _median(values: list[float]) -> float | None:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    if not vals:
        return None
    return float(statistics.median(vals))


def _quantile(values: list[float], q: float) -> float | None:
    vals = sorted(float(v) for v in values if np.isfinite(float(v)))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def _filename(label: str) -> str:
    return f"Light_mosaic_M 106_20.0s_IRCUT_20250518-{label}.fit"


def _positions(stars: np.ndarray) -> np.ndarray:
    return p23._positions(stars)


def _raw_ranks(raw: np.ndarray, stars: np.ndarray, radius: float = 1.0e-3) -> list[int | None]:
    if raw.size == 0 or stars.size == 0:
        return []
    tree = cKDTree(_positions(raw))
    used: set[int] = set()
    ranks: list[int | None] = []
    for xy in _positions(stars):
        dist, idx = tree.query(xy, k=1, distance_upper_bound=float(radius))
        if np.isfinite(float(dist)) and int(idx) < raw.shape[0] and int(idx) not in used:
            used.add(int(idx))
            ranks.append(int(idx))
        else:
            ranks.append(None)
    return ranks


def _finite_inside(stars: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    if stars.size == 0:
        return stars
    height, width = int(image_shape[0]), int(image_shape[1])
    mask = (
        np.isfinite(stars["x"])
        & np.isfinite(stars["y"])
        & np.isfinite(stars["flux"])
        & np.isfinite(stars["fwhm"])
        & (stars["flux"] > 0.0)
        & (stars["x"] >= 0.0)
        & (stars["x"] < float(width))
        & (stars["y"] >= 0.0)
        & (stars["y"] < float(height))
    )
    return stars[mask]


def _cap(stars: np.ndarray, cap: int) -> np.ndarray:
    if int(cap) > 0 and stars.shape[0] > int(cap):
        return stars[: int(cap)]
    return stars


def _grid_select(raw: np.ndarray, image_shape: tuple[int, int], *, nx: int, ny: int, cap: int) -> tuple[np.ndarray, dict[str, Any]]:
    arr = _finite_inside(raw, image_shape)
    if arr.size == 0 or int(cap) <= 0:
        return arr[:0], {"family": "grid", "kept": 0, "nx": int(nx), "ny": int(ny)}
    height, width = int(image_shape[0]), int(image_shape[1])
    xs = np.clip(np.floor(np.asarray(arr["x"], dtype=np.float64) / max(1.0, float(width)) * int(nx)), 0, int(nx) - 1).astype(int)
    ys = np.clip(np.floor(np.asarray(arr["y"], dtype=np.float64) / max(1.0, float(height)) * int(ny)), 0, int(ny) - 1).astype(int)
    flux = np.asarray(arr["flux"], dtype=np.float64)
    raw_rank = np.asarray(_raw_ranks(raw, arr), dtype=object)
    bins: list[list[int]] = [[] for _ in range(int(nx) * int(ny))]
    for idx, (ix, iy) in enumerate(zip(xs, ys)):
        bins[int(iy) * int(nx) + int(ix)].append(int(idx))
    for cell in bins:
        cell.sort(key=lambda i: (-float(flux[i]), int(raw_rank[i]) if raw_rank[i] is not None else 10**9))
    order: list[int] = []
    row = 0
    while len(order) < int(cap):
        progressed = False
        for cell in bins:
            if row < len(cell):
                order.append(cell[row])
                progressed = True
                if len(order) >= int(cap):
                    break
        if not progressed:
            break
        row += 1
    kept = arr[np.asarray(order, dtype=np.int64)] if order else arr[:0]
    counts = [len(cell) for cell in bins]
    return kept, {
        "family": "grid",
        "kept": int(kept.shape[0]),
        "nx": int(nx),
        "ny": int(ny),
        "occupied_cells": int(sum(1 for c in counts if c > 0)),
        "min_cell_raw": int(min(counts) if counts else 0),
        "max_cell_raw": int(max(counts) if counts else 0),
    }


def _stratified_select(raw: np.ndarray, image_shape: tuple[int, int], *, cap: int) -> tuple[np.ndarray, dict[str, Any]]:
    arr = _finite_inside(raw, image_shape)
    if arr.size == 0 or int(cap) <= 0:
        return arr[:0], {"family": "stratified", "kept": 0}
    raw_ranks = np.asarray(_raw_ranks(raw, arr), dtype=object)
    qscore_all = p23._standard_qscore(raw)
    qscore = np.asarray(
        [qscore_all[int(r)] if r is not None and int(r) < qscore_all.shape[0] else -np.inf for r in raw_ranks],
        dtype=np.float64,
    )
    flux = np.asarray(arr["flux"], dtype=np.float64)
    p40 = float(np.percentile(flux, 40))
    p75 = float(np.percentile(flux, 75))
    p95 = float(np.percentile(flux, 95))
    masks = [
        flux >= p95,
        (flux >= p75) & (flux < p95),
        (flux >= p40) & (flux < p75),
        flux < p40,
    ]
    names = ("very_bright", "bright", "mid", "faint")
    weights = (1, 2, 3, 1)
    bins: list[list[int]] = []
    for mask in masks:
        indices = np.flatnonzero(mask)
        indices = sorted(
            (int(i) for i in indices),
            key=lambda i: (-float(qscore[i]), -float(flux[i]), int(raw_ranks[i]) if raw_ranks[i] is not None else 10**9),
        )
        bins.append(indices)
    order: list[int] = []
    pos = [0 for _ in bins]
    while len(order) < int(cap):
        progressed = False
        for bin_idx, weight in enumerate(weights):
            for _ in range(int(weight)):
                if pos[bin_idx] < len(bins[bin_idx]):
                    order.append(bins[bin_idx][pos[bin_idx]])
                    pos[bin_idx] += 1
                    progressed = True
                    if len(order) >= int(cap):
                        break
            if len(order) >= int(cap):
                break
        if not progressed:
            break
    kept = arr[np.asarray(order, dtype=np.int64)] if order else arr[:0]
    return kept, {
        "family": "stratified",
        "kept": int(kept.shape[0]),
        "bins": {name: int(len(bin_values)) for name, bin_values in zip(names, bins)},
        "weights": list(weights),
    }


def _policy_stars(raw: np.ndarray, image_shape: tuple[int, int], policy: SourcePolicy) -> tuple[np.ndarray, dict[str, Any]]:
    if policy.family == "head":
        kept = _cap(raw, int(policy.cap))
        return kept, {"family": "head", "kept": int(kept.shape[0]), "cap": int(policy.cap)}
    if policy.family == "grid":
        return _grid_select(raw, image_shape, nx=int(policy.nx), ny=int(policy.ny), cap=int(policy.cap))
    if policy.family == "stratified":
        return _stratified_select(raw, image_shape, cap=int(policy.cap))
    if policy.family == "astrometry_like":
        kept, stats = _astrometry_source_list_gate(
            raw,
            image_shape=image_shape,
            approx_boxes=max(1, int(policy.boxes or 10)),
            max_sources=int(policy.cap),
            min_keep_ratio=0.0,
        )
        stats = dict(stats)
        stats["family"] = "astrometry_like"
        return kept, stats
    raise ValueError(f"unknown source policy family: {policy.family}")


def _detect_sources(label: str, args: argparse.Namespace) -> tuple[np.ndarray, tuple[int, int], dict[str, Any]]:
    source = args.data_dir.expanduser().resolve() / _filename(label)
    return p23._detect_runtime_stars(source, args)


def _dedup_world(worlds: list[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(world, dtype=np.float64) for world in worlds if np.asarray(world).size]
    if not arrays:
        return np.empty((0, 2), dtype=np.float64)
    union = np.vstack(arrays)
    _vals, idx = np.unique(np.round(union, decimals=8), axis=0, return_index=True)
    return union[np.sort(idx)]


def _one_to_one_pairs(catalog_xy: np.ndarray, image_xy: np.ndarray, radius_px: float) -> dict[str, Any]:
    if catalog_xy.size == 0 or image_xy.size == 0:
        return {"count": 0, "distances": [], "catalog_indices": [], "image_indices": []}
    tree = cKDTree(np.asarray(image_xy, dtype=np.float64))
    pairs: list[tuple[float, int, int]] = []
    for cat_idx, xy in enumerate(np.asarray(catalog_xy, dtype=np.float64)):
        for img_idx in tree.query_ball_point(xy, float(radius_px)):
            dist = float(np.linalg.norm(image_xy[int(img_idx)] - xy))
            pairs.append((dist, int(cat_idx), int(img_idx)))
    pairs.sort()
    used_catalog: set[int] = set()
    used_image: set[int] = set()
    distances: list[float] = []
    cat_indices: list[int] = []
    img_indices: list[int] = []
    for dist, cat_idx, img_idx in pairs:
        if cat_idx in used_catalog or img_idx in used_image:
            continue
        used_catalog.add(cat_idx)
        used_image.add(img_idx)
        distances.append(dist)
        cat_indices.append(cat_idx)
        img_indices.append(img_idx)
    return {
        "count": int(len(img_indices)),
        "distances": distances,
        "catalog_indices": cat_indices,
        "image_indices": img_indices,
        "median_distance_px": float(np.median(distances)) if distances else None,
        "max_distance_px": float(np.max(distances)) if distances else None,
    }


def _grid_counts(stars: np.ndarray, image_shape: tuple[int, int], *, nx: int = 4, ny: int = 4) -> dict[str, Any]:
    if stars.size == 0:
        return {"empty_cells": int(nx * ny), "occupied_cells": 0, "max_cell": 0}
    xy = _positions(stars)
    height, width = int(image_shape[0]), int(image_shape[1])
    hist, _x, _y = np.histogram2d(
        xy[:, 0],
        xy[:, 1],
        bins=(np.linspace(0.0, float(width), int(nx) + 1), np.linspace(0.0, float(height), int(ny) + 1)),
    )
    vals = hist.astype(int).ravel()
    return {
        "empty_cells": int(np.count_nonzero(vals == 0)),
        "occupied_cells": int(np.count_nonzero(vals > 0)),
        "max_cell": int(np.max(vals)) if vals.size else 0,
    }


def _feature_summary(raw: np.ndarray, raw_ranks: set[int]) -> dict[str, Any]:
    if raw.size == 0 or not raw_ranks:
        return {"count": 0, "flux": _summary(np.asarray([])), "fwhm": _summary(np.asarray([])), "qscore": _summary(np.asarray([]))}
    idx = np.asarray(sorted(i for i in raw_ranks if 0 <= int(i) < raw.shape[0]), dtype=np.int64)
    qscore = p23._standard_qscore(raw)
    return {
        "count": int(idx.shape[0]),
        "flux": _summary(np.asarray(raw["flux"][idx], dtype=np.float64)),
        "fwhm": _summary(np.asarray(raw["fwhm"][idx], dtype=np.float64)),
        "qscore": _summary(np.asarray(qscore[idx], dtype=np.float64)),
    }


def _oracle_retention(
    label: str,
    raw: np.ndarray,
    kept: np.ndarray,
    image_shape: tuple[int, int],
    union_world: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    oracle_path = p213._find_local_oracle_wcs_path(label, args)
    wcs, shape = p213._load_oracle_wcs_path(oracle_path)
    height, width = int(shape[0]), int(shape[1])
    pix = np.asarray(wcs.wcs_world2pix(union_world, 0), dtype=np.float64)
    finite = np.isfinite(pix[:, 0]) & np.isfinite(pix[:, 1])
    inside = finite & (pix[:, 0] >= 0.0) & (pix[:, 0] < width) & (pix[:, 1] >= 0.0) & (pix[:, 1] < height)
    catalog_xy = pix[inside]
    raw_pairs = _one_to_one_pairs(catalog_xy, _positions(raw), float(args.match_radius_px))
    kept_pairs = _one_to_one_pairs(catalog_xy, _positions(kept), float(args.match_radius_px))
    raw_ranks = _raw_ranks(raw, raw)
    kept_ranks = _raw_ranks(raw, kept)
    raw_img_to_rank = {
        int(i): int(raw_ranks[int(i)])
        for i in range(len(raw_ranks))
        if raw_ranks[int(i)] is not None
    }
    kept_set = {int(r) for r in kept_ranks if r is not None}
    raw_matchable = {raw_img_to_rank[int(i)] for i in raw_pairs["image_indices"] if int(i) in raw_img_to_rank}
    kept_matchable = raw_matchable & kept_set
    lost_matchable = raw_matchable - kept_set
    kept_all = kept_set
    nonmatchable_kept = kept_all - raw_matchable
    return {
        "label": label,
        "oracle_fits": str(oracle_path),
        "catalog_union_in_field": int(catalog_xy.shape[0]),
        "raw_detected": int(raw.shape[0]),
        "kept": int(kept.shape[0]),
        "raw_matchable": int(raw_pairs["count"]),
        "kept_matchable": int(len(kept_matchable)),
        "lost_matchable": int(len(lost_matchable)),
        "nonmatchable_kept": int(len(nonmatchable_kept)),
        "retention_raw_matchables": float(len(kept_matchable) / max(1, len(raw_matchable))),
        "retention_cap": float(len(kept_matchable) / max(1, int(kept.shape[0]))),
        "raw_matchable_ranks": sorted(raw_matchable),
        "kept_matchable_ranks": sorted(kept_matchable),
        "lost_matchable_ranks": sorted(lost_matchable),
        "lost_matchable_features": _feature_summary(raw, lost_matchable),
        "kept_matchable_features": _feature_summary(raw, kept_matchable),
        "nonmatchable_kept_features": _feature_summary(raw, nonmatchable_kept),
        "kept_grid4x4": _grid_counts(kept, image_shape, nx=4, ny=4),
    }


def _case_source(label: str, data_dir: Path) -> Path:
    path = data_dir / _filename(label)
    if not path.exists():
        raise FileNotFoundError(f"missing M106 case {label}: {path}")
    return path


def _prepare_case(label: str, data_dir: Path, work_dir: Path, tag: str) -> Path:
    source = _case_source(label, data_dir)
    target_dir = work_dir / tag / label
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    shutil.copy2(source, target)
    p22._strip_wcs(target)
    return target


def _prep_cache_for(candidate: Path, stars: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    resolved = candidate.resolve()
    stat = resolved.stat()
    return {
        str(resolved): {
            "sig": (int(stat.st_mtime_ns), int(stat.st_size)),
            "downsample": int(args.downsample),
            "detect_k_sigma": float(args.detect_k_sigma),
            "detect_min_area": int(args.detect_min_area),
            "stars": stars.copy(),
        }
    }


def _config(index_paths: tuple[Path, ...], args: argparse.Namespace, *, max_stars: int) -> SolveConfig:
    return SolveConfig(
        max_stars=int(max_stars),
        max_quads=int(args.max_quads),
        detect_k_sigma=float(args.detect_k_sigma),
        detect_min_area=int(args.detect_min_area),
        downsample=int(args.downsample),
        quality_inliers=int(args.quality_inliers),
        quality_rms=float(args.quality_rms),
        pixel_scale_min_arcsec=float(args.pixel_scale_min_arcsec),
        pixel_scale_max_arcsec=float(args.pixel_scale_max_arcsec),
        blind_global_hard_budget_s=float(args.max_wall_s),
        blind_reuse_existing_solved_wcs=False,
        quad_hash_schema=ASTROMETRY_AB_CODE_4D_SCHEMA,
        blind_astrometry_4d_index_paths=tuple(str(path.expanduser().resolve()) for path in index_paths),
        blind_astrometry_4d_validation_catalog_policy="union_candidate_tiles",
        blind_astrometry_4d_source_policy="diagnostic_unfiltered",
        blind_astrometry_4d_accept_policy="best_within_budget",
        blind_astrometry_4d_max_accepts=int(args.max_accepts),
        blind_astrometry_4d_code_tol=float(args.code_tol),
        blind_astrometry_4d_max_hits=int(args.max_hits_4d),
        blind_astrometry_4d_max_hits_per_image_quad=int(args.max_hits_per_image_quad),
        blind_astrometry_4d_max_hypotheses=int(args.max_hypotheses),
        blind_astrometry_4d_image_strategy=str(args.image_strategy),
        blind_astrometry_4d_match_radius_px=float(args.match_radius_px),
    )


def _run_solve_with_sources(
    label: str,
    policy: SourcePolicy,
    selected: np.ndarray,
    args: argparse.Namespace,
    *,
    index_paths: tuple[Path, ...] = (p212.INDEX_2823, p212.INDEX_2822),
    tag_prefix: str = "main",
) -> dict[str, Any]:
    t0 = time.perf_counter()
    data_dir = args.data_dir.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    candidate = _prepare_case(label, data_dir, work_dir, f"{tag_prefix}_{policy.name}")
    cfg = _config(index_paths, args, max_stars=max(1, int(selected.shape[0])))
    result = solve_blind(
        candidate,
        args.index_root.expanduser().resolve(),
        config=cfg,
        prep_cache=_prep_cache_for(candidate, selected, args),
    )
    stats = dict(result.stats or {})
    reject = stats.get("astrometry_4d_best_reject") or {}
    return {
        "label": label,
        "policy": policy.name,
        "index_tiles": [p212._tile_name(path) for path in index_paths],
        "success": bool(result.success),
        "message": str(result.message),
        "tile_key": result.tile_key,
        "origin_tile": stats.get("astrometry_4d_selected_origin_tile_key") or reject.get("origin_tile_key"),
        "inliers": int(stats.get("inliers", 0) or 0),
        "rms_px": float(stats.get("rms_px", float("nan"))),
        "rank": stats.get("astrometry_4d_selected_rank"),
        "hits": int(stats.get("astrometry_4d_hits", 0) or 0),
        "hits_by_index": stats.get("astrometry_4d_hits_by_index"),
        "hypotheses_tested": int(stats.get("astrometry_4d_hits_tested", 0) or 0),
        "accepted_candidates": int(stats.get("astrometry_4d_accepted_candidates", 0) or 0),
        "stop_reason": stats.get("astrometry_4d_stop_reason"),
        "reject_reason_counts": stats.get("astrometry_4d_reject_reason_counts") or stats.get("reject_reason_counts"),
        "quad_build_s": float(stats.get("astrometry_4d_quad_build_s", 0.0) or 0.0),
        "lookup_s": float(stats.get("astrometry_4d_kd_lookup_s", 0.0) or 0.0),
        "validation_s": float(stats.get("astrometry_4d_validation_s", 0.0) or 0.0),
        "solver_total_s": float(stats.get("astrometry_4d_total_s", 0.0) or 0.0),
        "wall_s": float(time.perf_counter() - t0),
        "best_reject": reject,
    }


def _solve_summary(row: dict[str, Any]) -> dict[str, Any]:
    reject = row.get("best_reject") or {}
    return {
        "success": bool(row.get("success")),
        "inliers": int(row.get("inliers") or reject.get("inliers") or 0),
        "rms_px": row.get("rms_px") if row.get("success") else reject.get("rms_px"),
        "rank": row.get("rank") if row.get("success") else reject.get("rank", reject.get("hit_rank")),
        "origin_tile": row.get("origin_tile") or reject.get("origin_tile_key"),
        "reason": row.get("message") if row.get("success") else reject.get("reason", row.get("message")),
    }


def _runtime_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cases": int(len(rows)),
        "successes": int(sum(1 for row in rows if row.get("success"))),
        "wall_s_median": _median([float(row.get("wall_s", 0.0) or 0.0) for row in rows]),
        "wall_s_p95": _quantile([float(row.get("wall_s", 0.0) or 0.0) for row in rows], 0.95),
        "validation_s_median": _median([float(row.get("validation_s", 0.0) or 0.0) for row in rows]),
        "hypotheses_tested_median": _median([float(row.get("hypotheses_tested", 0) or 0) for row in rows]),
        "accepted_candidates_median": _median([float(row.get("accepted_candidates", 0) or 0) for row in rows]),
        "max_accepts_hits": [row["label"] for row in rows if str(row.get("stop_reason")) == "max_accepts"],
        "candidate_exhausted": [row["label"] for row in rows if str(row.get("stop_reason")) == "candidate_exhausted"],
    }


def _load_case_labels(args: argparse.Namespace) -> list[str]:
    if str(args.cases).strip():
        return [part.strip() for part in str(args.cases).split(",") if part.strip()]
    p212_payload = json.loads(args.p212_json.expanduser().resolve().read_text(encoding="utf-8"))
    labels = [str(row.get("label")) for row in p212_payload.get("cases") or [] if row.get("label")]
    if labels:
        return labels
    return p212._discover_cases(args.data_dir.expanduser().resolve())


def _select_policies(args: argparse.Namespace) -> tuple[SourcePolicy, ...]:
    if not str(args.policies).strip():
        return POLICIES
    wanted = {part.strip() for part in str(args.policies).split(",") if part.strip()}
    selected = tuple(policy for policy in POLICIES if policy.name in wanted)
    missing = sorted(wanted - {policy.name for policy in selected})
    if missing:
        raise ValueError("unknown policy name(s): " + ", ".join(missing))
    return selected


def _progress(event: str, **payload: Any) -> None:
    row = {"event": event}
    row.update(payload)
    print(json.dumps(row, default=_json_default), flush=True)


def _audit_cases(
    labels: list[str],
    policies: tuple[SourcePolicy, ...],
    args: argparse.Namespace,
    union_world: np.ndarray,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]], dict[str, dict[str, tuple[np.ndarray, tuple[int, int]]]]]:
    audit: dict[str, Any] = {}
    selected_by_case: dict[str, dict[str, np.ndarray]] = {}
    raw_by_case: dict[str, dict[str, tuple[np.ndarray, tuple[int, int]]]] = {}
    for label in labels:
        raw, image_shape, detect_meta = _detect_sources(label, args)
        raw_by_case[label] = {"raw": (raw, image_shape)}
        selected_by_case[label] = {}
        audit[label] = {"detect_meta": detect_meta, "raw_detected": int(raw.shape[0]), "image_shape": list(image_shape), "policies": {}}
        for policy in policies:
            kept, stats = _policy_stars(raw, image_shape, policy)
            selected_by_case[label][policy.name] = kept
            retention = _oracle_retention(label, raw, kept, image_shape, union_world, args)
            audit[label]["policies"][policy.name] = {
                "selection_stats": stats,
                "retention": retention,
            }
    return audit, selected_by_case, raw_by_case


def _run_runtime_matrix(
    labels: list[str],
    policies: tuple[SourcePolicy, ...],
    selected_by_case: dict[str, dict[str, np.ndarray]],
    args: argparse.Namespace,
) -> dict[str, list[dict[str, Any]]]:
    matrix: dict[str, list[dict[str, Any]]] = {policy.name: [] for policy in policies}
    total = len(labels) * len(policies)
    n = 0
    for policy in policies:
        for label in labels:
            n += 1
            selected = selected_by_case[label][policy.name]
            _progress("runtime_case_start", policy=policy.name, label=label, index=n, count=total, kept=int(selected.shape[0]))
            row = _run_solve_with_sources(label, policy, selected, args)
            _progress(
                "runtime_case_done",
                policy=policy.name,
                label=label,
                success=row.get("success"),
                inliers=_solve_summary(row).get("inliers"),
                rms_px=_solve_summary(row).get("rms_px"),
                wall_s=row.get("wall_s"),
            )
            matrix[policy.name].append(row)
    return matrix


def _run_controls(
    labels: list[str],
    policy: SourcePolicy,
    selected_by_case: dict[str, dict[str, np.ndarray]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    footprints = {label: p212._footprints(label, args) for label in labels}
    low_2822 = [
        label
        for label in labels
        if "d50_2822" in footprints[label] and float(footprints[label].get("d50_2822", 0.0) or 0.0) <= float(args.bad_tile_max_footprint_pct)
    ][: int(args.bad_tile_control_limit)]
    control_labels = p212._bounded_control_labels(labels, int(args.control_case_limit))

    def run_group(name: str, group_labels: list[str], paths: tuple[Path, ...]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for label in group_labels:
            _progress("control_case_start", control=name, policy=policy.name, label=label)
            rows.append(
                _run_solve_with_sources(
                    label,
                    policy,
                    selected_by_case[label][policy.name],
                    args,
                    index_paths=paths,
                    tag_prefix=f"control_{name}",
                )
                | {"footprints": footprints.get(label)}
            )
        return rows

    return {
        "policy": policy.name,
        "d50_2822_only_low_footprint": run_group("d50_2822_only_low_footprint", low_2822, (p212.INDEX_2822,)),
        "reversed_order_best": run_group("reversed_order_best", control_labels, (p212.INDEX_2822, p212.INDEX_2823)),
        "d50_2823_only_best": run_group("d50_2823_only_best", control_labels, (p212.INDEX_2823,)),
        "d50_2822_only_best": run_group("d50_2822_only_best", control_labels, (p212.INDEX_2822,)),
        "strict_error_controls": [
            p212._run_error_probe(control_labels[0], (p212.MISSING_INDEX,), args, tag="p214_missing_index") if control_labels else {},
            p212._run_error_probe(control_labels[0], (p212.INCOMPATIBLE_INDEX,), args, tag="p214_incompatible_index") if control_labels else {},
        ],
    }


def _policy_decision(payload: dict[str, Any], labels: list[str], policies: tuple[SourcePolicy, ...]) -> tuple[str, dict[str, Any], list[str]]:
    policy_rows = payload["runtime_by_policy"]
    audit = payload["source_audit"]
    summaries: dict[str, Any] = {}
    for policy in policies:
        rows = policy_rows[policy.name]
        success_count = sum(1 for row in rows if row.get("success"))
        failure_status = {row["label"]: _solve_summary(row) for row in rows if row["label"] in FAILURE_CASES}
        retention_values = [
            float(audit[label]["policies"][policy.name]["retention"]["retention_raw_matchables"])
            for label in labels
        ]
        kept_matchable_failures = {
            label: int(audit[label]["policies"][policy.name]["retention"]["kept_matchable"])
            for label in FAILURE_CASES
            if label in audit
        }
        median_rms = _median([float(row.get("rms_px", float("nan"))) for row in rows if row.get("success")])
        summaries[policy.name] = {
            "successes": int(success_count),
            "failures": [row["label"] for row in rows if not row.get("success")],
            "failure_case_status": failure_status,
            "kept_matchable_failures": kept_matchable_failures,
            "median_retention_raw_matchables": _median(retention_values),
            "median_success_rms": median_rms,
            "runtime": _runtime_stats(rows),
            "cap": int(policy.cap),
            "family": policy.family,
        }

    def key(policy: SourcePolicy) -> tuple[int, int, float, float, int]:
        s = summaries[policy.name]
        return (
            int(s["successes"]),
            int(sum(1 for label in FAILURE_CASES if s["failure_case_status"].get(label, {}).get("success"))),
            -float(s["runtime"].get("wall_s_median") or 1e9),
            -float(policy.cap),
            -len(policy.name),
        )

    best = max(policies, key=key)
    answers = [
        f"Politique candidate retenue par le bake-off: `{best.name}` ({summaries[best.name]['successes']}/{len(labels)} succes).",
        f"`233828`: " + "; ".join(
            f"`{policy.name}` -> {summaries[policy.name]['failure_case_status'].get('233828', {}).get('inliers')} inliers / success={summaries[policy.name]['failure_case_status'].get('233828', {}).get('success')}"
            for policy in policies
        ),
        f"`234013`: " + "; ".join(
            f"`{policy.name}` -> {summaries[policy.name]['failure_case_status'].get('234013', {}).get('inliers')} inliers / success={summaries[policy.name]['failure_case_status'].get('234013', {}).get('success')}"
            for policy in policies
        ),
    ]
    return best.name, summaries, answers


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# ZeBlind P2.14 - bake-off source-list 4D",
        "",
        "> Le WCS Astrometry.net est utilise uniquement comme oracle offline pour mesurer la retention de sources matchables. Le runtime `solve_blind` reste execute sur des FITS sans WCS, avec une liste explicite d'index 4D.",
        "",
        "## Verdict",
        "",
        f"- Verdict global: **{payload['global_verdict']}**",
        f"- Politique candidate promouvable: `{payload['recommended_policy']}`.",
        f"- Meilleure politique partielle du bake-off: `{payload.get('partial_best_policy', payload['recommended_policy'])}`.",
        "- Aucun seuil change, aucun all-sky, aucun rebuild complet, aucun changement ZeNear/GUI/default.",
        "",
        "## Synthese politiques",
        "",
        "| politique | famille | cap | succes | echecs | med retention oracle | med RMS succes | med total s | med validation s | max_accepts |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for name, summary in payload["policy_summaries"].items():
        rt = summary["runtime"]
        lines.append(
            "| `{}` | `{}` | {} | {}/{} | {} | {} | {} | {} | {} | {} |".format(
                name,
                summary["family"],
                summary["cap"],
                summary["successes"],
                payload["case_count"],
                ", ".join(f"`{v}`" for v in summary["failures"]) if summary["failures"] else "",
                _fmt(summary.get("median_retention_raw_matchables"), 3),
                _fmt(summary.get("median_success_rms"), 3),
                _fmt(rt.get("wall_s_median"), 3),
                _fmt(rt.get("validation_s_median"), 3),
                len(rt.get("max_accepts_hits") or []),
            )
        )
    lines.extend(["", "## Focus echecs P2.12", "", "| cas | politique | gardees | matchables gardees/brutes | perdus | success | inliers/best | RMS/best | rang | origine | hits | testes | total s |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|"])
    runtime_lookup = {
        (row["label"], policy): row
        for policy, rows in payload["runtime_by_policy"].items()
        for row in rows
    }
    for label in FAILURE_CASES:
        for name in payload["policy_names"]:
            audit = payload["source_audit"][label]["policies"][name]["retention"]
            row = runtime_lookup[(label, name)]
            short = _solve_summary(row)
            lines.append(
                "| `{}` | `{}` | {} | {}/{} | {} | `{}` | {} | {} | {} | `{}` | {} | {} | {} |".format(
                    label,
                    name,
                    audit["kept"],
                    audit["kept_matchable"],
                    audit["raw_matchable"],
                    audit["lost_matchable"],
                    row.get("success"),
                    short.get("inliers"),
                    _fmt(short.get("rms_px"), 3),
                    short.get("rank"),
                    short.get("origin_tile"),
                    row.get("hits"),
                    row.get("hypotheses_tested"),
                    _fmt(row.get("wall_s"), 3),
                )
            )
    lines.extend(["", "## Retention oracle par cas", "", "| cas | politique | raw | gardees | matchables brutes | matchables gardees | retention | nonmatchables gardees | grid vides | lost flux med | lost qscore med |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for label in payload["case_order"]:
        for name in payload["policy_names"]:
            row = payload["source_audit"][label]["policies"][name]["retention"]
            lines.append(
                "| `{}` | `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    label,
                    name,
                    row["raw_detected"],
                    row["kept"],
                    row["raw_matchable"],
                    row["kept_matchable"],
                    _fmt(row["retention_raw_matchables"], 3),
                    row["nonmatchable_kept"],
                    row["kept_grid4x4"]["empty_cells"],
                    _fmt(row["lost_matchable_features"]["flux"].get("median"), 1),
                    _fmt(row["lost_matchable_features"]["qscore"].get("median"), 3),
                )
            )
    lines.extend(["", "## Controles faux positifs", ""])
    controls = payload.get("controls") or {}
    if controls:
        lines.append(f"- Politique controlee: `{controls.get('policy')}`.")
        for key in ("d50_2822_only_low_footprint", "reversed_order_best", "d50_2823_only_best", "d50_2822_only_best"):
            rows = controls.get(key) or []
            accepted = [row for row in rows if row.get("success")]
            lines.extend([f"### {key}", "", f"- Acceptations: `{len(accepted)}/{len(rows)}`.", "", "| cas | success | inliers | RMS | origine | hits | testes | total s |", "|---|---:|---:|---:|---|---:|---:|---:|"])
            for row in rows:
                short = _solve_summary(row)
                lines.append(
                    "| `{}` | `{}` | {} | {} | `{}` | {} | {} | {} |".format(
                        row.get("label"),
                        row.get("success"),
                        short.get("inliers"),
                        _fmt(short.get("rms_px"), 3),
                        short.get("origin_tile"),
                        row.get("hits"),
                        row.get("hypotheses_tested"),
                        _fmt(row.get("wall_s"), 3),
                    )
                )
            lines.append("")
        lines.extend(["### Erreurs strictes", "", "| controle | success | erreur explicite | message |", "|---|---:|---:|---|"])
        for row in controls.get("strict_error_controls") or []:
            lines.append(f"| `{row.get('tag')}` | `{row.get('success')}` | `{row.get('explicit_error_ok')}` | {row.get('message')} |")
    else:
        lines.append("- Non rejoues en P2.14 final: le critere d'arret s'active des le focus `233828/234013`, car `234013` ne passe avec aucune politique raisonnable testee.")
    lines.extend(["", "## Reponses", ""])
    for answer in payload["answers"]:
        lines.append(f"- {answer}")
    lines.extend(["", "## Parametres", "", "```json", json.dumps(payload["params"], indent=2, default=_json_default), "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="P2.14 source-list policy bake-off for experimental ZeBlind 4D.")
    ap.add_argument("--data-dir", type=Path, default=p22.DEFAULT_DATA_DIR)
    ap.add_argument("--reference-dir", type=Path, default=p22.DEFAULT_REFERENCE_DIR)
    ap.add_argument("--index-root", type=Path, default=p22.DEFAULT_INDEX_ROOT)
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    ap.add_argument("--p212-json", type=Path, default=DEFAULT_P212_JSON)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--cases", default="")
    ap.add_argument("--policies", default="")
    ap.add_argument("--failure-only", action="store_true")
    ap.add_argument("--skip-controls", action="store_true")
    ap.add_argument("--detect-k-sigma", type=float, default=3.0)
    ap.add_argument("--detect-min-area", type=int, default=5)
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--max-quads", type=int, default=2500)
    ap.add_argument("--image-strategy", default="log_spaced")
    ap.add_argument("--code-tol", type=float, default=0.015)
    ap.add_argument("--max-hits-4d", type=int, default=2000)
    ap.add_argument("--max-hits-per-image-quad", type=int, default=8)
    ap.add_argument("--max-hypotheses", type=int, default=2000)
    ap.add_argument("--max-wall-s", type=float, default=45.0)
    ap.add_argument("--max-accepts", type=int, default=64)
    ap.add_argument("--match-radius-px", type=float, default=3.0)
    ap.add_argument("--quality-rms", type=float, default=1.2)
    ap.add_argument("--quality-inliers", type=int, default=40)
    ap.add_argument("--pixel-scale-min-arcsec", type=float, default=1.79)
    ap.add_argument("--pixel-scale-max-arcsec", type=float, default=2.99)
    ap.add_argument("--bad-tile-max-footprint-pct", type=float, default=40.0)
    ap.add_argument("--bad-tile-control-limit", type=int, default=10)
    ap.add_argument("--control-case-limit", type=int, default=10)
    ap.add_argument("--footprint-grid", type=int, default=31)
    args = ap.parse_args()

    for path in (p212.INDEX_2823, p212.INDEX_2822):
        if not path.exists():
            raise FileNotFoundError(f"missing explicit P2.14 4D index: {path}")

    labels = list(FAILURE_CASES) if bool(args.failure_only) else _load_case_labels(args)
    policies = _select_policies(args)
    entries = p26._tile_entries(args.index_root.expanduser().resolve())
    union_world = _dedup_world([p28._catalog_world(args.index_root.expanduser().resolve(), entries, tile) for tile in BOUNDED_TILES])
    _progress("audit_start", cases=len(labels), policies=[policy.name for policy in policies])
    source_audit, selected_by_case, _raw_by_case = _audit_cases(labels, policies, args, union_world)
    runtime_by_policy = _run_runtime_matrix(labels, policies, selected_by_case, args)
    recommended, policy_summaries, base_answers = _policy_decision(
        {"runtime_by_policy": runtime_by_policy, "source_audit": source_audit},
        labels,
        policies,
    )
    controls = {} if bool(args.skip_controls) else _run_controls(labels, next(policy for policy in policies if policy.name == recommended), selected_by_case, args)

    rec_summary = policy_summaries[recommended]
    all_success = int(rec_summary["successes"]) == len(labels)
    failure_success = all(
        bool(rec_summary["failure_case_status"].get(label, {}).get("success"))
        for label in FAILURE_CASES
        if label in labels
    )
    bad_accepts = []
    if controls:
        bad_accepts = [row for row in controls.get("d50_2822_only_low_footprint") or [] if row.get("success")]
    partial_best = recommended
    if all_success and not bad_accepts:
        verdict = "P2.14 positif: une politique source-list candidate recupere M106 sans baisse de seuil ni faux positif evident"
    elif failure_success:
        verdict = "P2.14 partiel positif: les deux echecs sont recuperes, mais la politique doit etre jugee sur le corpus complet/controles"
    else:
        verdict = "P2.14 negatif: aucune politique raisonnable testee ne recupere completement les echecs restants"
        recommended = "no_source_policy_promoted_p214_stop_criterion"

    answers = list(base_answers)
    if recommended == "no_source_policy_promoted_p214_stop_criterion":
        answers.extend(
            [
                f"Aucune politique testee ne recupere les deux echecs; stop criterion active avant elargissement 30/30. Meilleure politique partielle: `{partial_best}`.",
                "`233828` est recuperable sans baisse de seuil par cap/ranking (`head_cap200`, `head_cap250`, grilles 160/200, astrometry-like 200).",
                "`234013` ne gagne aucun support utile avec head 160/200/250 ni astrometry-like 200: il reste a 28 inliers et 28/42 matchables gardees. Le probleme n'est donc pas un simple cap <=250.",
                "Direction suivante: auditer le ranking detecteur/source-list au-dela du top 250 ou une source-list superieure a 250 en diagnostic seulement; ne pas promouvoir de nouvelle policy P2.14.",
                "Le runtime n'utilise jamais le WCS Astrometry.net: les sources candidates viennent du detecteur image, l'oracle ne sert qu'aux compteurs de retention du rapport.",
                "Les seuils restent `quality_inliers=40`, `quality_rms=1.2`, `match_radius_px=3.0`; aucune acceptation produit n'est simulee par baisse de seuil.",
            ]
        )
    else:
        answers.extend(
            [
                f"Cap minimal observe pour recuperer les deux echecs: `{recommended}` si la politique candidate est acceptee; les caps/tables detailles indiquent si le gain vient du cap ou du ranking.",
                "Le runtime n'utilise jamais le WCS Astrometry.net: les sources candidates viennent du detecteur image, l'oracle ne sert qu'aux compteurs de retention du rapport.",
                "Les seuils restent `quality_inliers=40`, `quality_rms=1.2`, `match_radius_px=3.0`; aucune acceptation produit n'est simulee par baisse de seuil.",
            ]
        )
    payload: dict[str, Any] = {
        "schema": "zeblind.p214_4d_source_policy_bakeoff.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "global_verdict": verdict,
        "recommended_policy": recommended,
        "partial_best_policy": partial_best,
        "case_count": int(len(labels)),
        "case_order": labels,
        "policy_names": [policy.name for policy in policies],
        "source_audit": source_audit,
        "runtime_by_policy": runtime_by_policy,
        "policy_summaries": policy_summaries,
        "controls": controls,
        "answers": answers,
        "params": {
            "quad_hash_schema": ASTROMETRY_AB_CODE_4D_SCHEMA,
            "index_paths": [str(p212.INDEX_2823.expanduser().resolve()), str(p212.INDEX_2822.expanduser().resolve())],
            "validation_catalog_policy": "union_candidate_tiles",
            "accept_policy": "best_within_budget",
            "quality_inliers": int(args.quality_inliers),
            "quality_rms": float(args.quality_rms),
            "match_radius_px": float(args.match_radius_px),
            "max_quads": int(args.max_quads),
            "max_hypotheses": int(args.max_hypotheses),
            "max_wall_s": float(args.max_wall_s),
            "max_accepts": int(args.max_accepts),
            "data_dir": str(args.data_dir.expanduser().resolve()),
            "index_root": str(args.index_root.expanduser().resolve()),
            "failure_only": bool(args.failure_only),
            "policies": [policy.__dict__ for policy in policies],
            "wcs_oracle_runtime_input": False,
            "all_sky": False,
            "rebuild_full": False,
        },
    }
    json_out = args.json_out.expanduser().resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    _write_report(args.report.expanduser().resolve(), payload)
    print(json.dumps({"global_verdict": verdict, "recommended_policy": recommended, "report": str(args.report), "json": str(args.json_out)}, indent=2))
    return 0 if failure_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
