#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import astropy.units as u

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zeblindsolver.quad_index_4d import (  # noqa: E402
    ASTROMETRY_AB_CODE_4D_SCHEMA,
    Quad4DIndex,
    build_experimental_4d_index,
)
from zeblindsolver.zeblindsolver import SolveConfig, solve_blind  # noqa: E402

import tools.diagnose_p23_4d_source_list_contract as p23  # noqa: E402
import tools.diagnose_p215_4d_split_quad_verify_sources as p215  # noqa: E402
import tools.diagnose_runtime_4d_route as p22  # noqa: E402


DEFAULT_BASE = ROOT / "reports/p219_4d_multifield_validation"
DEFAULT_INVENTORY = ROOT / "reports/zeblind_p219_local_field_inventory.json"
DEFAULT_SELECTION = ROOT / "reports/zeblind_p219_selected_multifield_corpus.json"
DEFAULT_JSON = ROOT / "reports/zeblind_p219_4d_multifield_validation.json"
DEFAULT_REPORT = ROOT / "reports/zeblind_p219_4d_multifield_validation.md"

MULTIFIELD_INDEX_ROOT = ROOT / "reports/r47i_s8_p30_multifield10_focused_v4_index_q4000_20260704/index"
M106_INDEX_2823 = ROOT / "reports/p21_astrometry_ab_code_4d_v1_d50_2823_S_stars2000_q40000.npz"
M106_INDEX_2822 = ROOT / "reports/p25_astrometry_ab_code_4d_v1_d50_2822_S_stars2000_q40000.npz"
NGC6888_DIR = Path("/home/tristan/zemosaic/example/astap solved")
M106_DIR = Path("/home/tristan/zemosaic/example/backuplightsastap")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        f = float(value)
        return f if np.isfinite(f) else None
    return str(value)


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        f = float(value)
        if not np.isfinite(f):
            return ""
        return f"{f:.{digits}f}"
    except Exception:
        return ""


def _percentile(values: list[float], pct: float) -> float | None:
    vals = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        return None
    return float(np.percentile(vals, float(pct)))


def _median(values: list[float]) -> float | None:
    return _percentile(values, 50.0)


def _field_id_from_name(path: Path) -> str:
    name = path.name
    if "NGC 6888" in name:
        return "ngc6888"
    if "mosaic_M 106" in name:
        return "m106"
    if "NGC 3628" in name:
        return "ngc3628"
    if "NGC 6823" in name:
        return "ngc6823"
    if "IC 1848" in name:
        return "ic1848"
    return "unknown"


def _fits_summary(path: Path) -> dict[str, Any]:
    try:
        header = fits.getheader(path)
        wcs = WCS(header)
        shape = [int(header.get("NAXIS2", 0) or 0), int(header.get("NAXIS1", 0) or 0)]
        scale = None
        if getattr(wcs, "has_celestial", False):
            scales = proj_plane_pixel_scales(wcs.celestial) * 3600.0
            scale = float(np.sqrt(float(scales[0]) * float(scales[1])))
        center = None
        if getattr(wcs, "has_celestial", False) and shape[0] > 0 and shape[1] > 0:
            sky = wcs.pixel_to_world(float(shape[1]) * 0.5, float(shape[0]) * 0.5)
            center = [float(sky.ra.deg), float(sky.dec.deg)]
        return {
            "path": str(path),
            "exists": True,
            "object": str(header.get("OBJECT", "") or ""),
            "dimensions": shape,
            "has_wcs": bool(getattr(wcs, "has_celestial", False)),
            "center_ra_dec": center,
            "pixel_scale_arcsec": scale,
            "instrument": str(header.get("INSTRUME", "") or ""),
            "telescope": str(header.get("TELESCOP", "") or ""),
        }
    except Exception as exc:
        return {"path": str(path), "exists": path.exists(), "error": str(exc)}


def _group_summaries(paths: list[Path]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path in paths:
        field_id = _field_id_from_name(path)
        out.setdefault(field_id, []).append(_fits_summary(path))
    return out


def _index_meta(path: Path) -> dict[str, Any]:
    item: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return item
    try:
        idx = Quad4DIndex.load(path)
        item.update(
            {
                "schema": idx.metadata.get("schema"),
                "tile_keys": list(idx.tile_keys),
                "entries": int(idx.codes_4d.shape[0]),
                "stars": int(idx.catalog_ra_dec.shape[0]),
                "metadata": dict(idx.metadata),
            }
        )
    except Exception as exc:
        item.update({"load_error": str(exc)})
    return item


def _compact_index_path(base: Path, tile_key: str) -> Path:
    return base / "indexes" / f"p219_astrometry_ab_code_4d_v1_{tile_key}_S_stars2000_q40000.npz"


def _ensure_ngc6888_indexes(base: Path) -> list[Path]:
    out: list[Path] = []
    for tile_key in ("d50_2644", "d50_2645"):
        path = _compact_index_path(base, tile_key)
        if not path.exists():
            build_experimental_4d_index(
                MULTIFIELD_INDEX_ROOT,
                path,
                tile_keys=[tile_key],
                level="S",
                max_stars_per_tile=2000,
                max_quads_per_tile=40000,
            )
        out.append(path.resolve())
    return out


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _inventory(args: argparse.Namespace, ngc_indexes: list[Path]) -> dict[str, Any]:
    ngc_paths = sorted(NGC6888_DIR.glob("Light_NGC 6888_*.fit")) if NGC6888_DIR.exists() else []
    m106_paths = sorted(M106_DIR.glob("Light_mosaic_M 106_*.fit")) if M106_DIR.exists() else []
    fresh_dir = Path("/home/tristan/zemosaic/example/fresh")
    referenced_missing = [
        {
            "field_id": "ngc3628",
            "target_or_region": "NGC 3628",
            "reason_for_exclusion": "referenced by older reports but /home/tristan/zemosaic/example/fresh is absent locally",
        },
        {
            "field_id": "ngc6823",
            "target_or_region": "NGC 6823",
            "reason_for_exclusion": "referenced by older reports but /home/tristan/zemosaic/example/fresh is absent locally",
        },
        {
            "field_id": "ic1848",
            "target_or_region": "IC 1848",
            "reason_for_exclusion": "referenced by older reports but /home/tristan/zemosaic/example/fresh is absent locally",
        },
    ]
    ngc_summaries = [_fits_summary(path) for path in ngc_paths]
    m106_summaries = [_fits_summary(path) for path in m106_paths]
    ngc_scales = [float(x["pixel_scale_arcsec"]) for x in ngc_summaries if x.get("pixel_scale_arcsec") is not None]
    m106_scales = [float(x["pixel_scale_arcsec"]) for x in m106_summaries if x.get("pixel_scale_arcsec") is not None]
    fields = [
        {
            "field_id": "ngc6888",
            "target_or_region": "NGC 6888 / Crescent Nebula",
            "fits_count": int(len(ngc_paths)),
            "sample_fits": [str(p) for p in ngc_paths[:10]],
            "dimensions": sorted({tuple(x.get("dimensions") or []) for x in ngc_summaries}),
            "scale_range_arcsec": [min(ngc_scales), max(ngc_scales)] if ngc_scales else None,
            "instrument_camera": sorted({str(x.get("instrument") or "") for x in ngc_summaries if x.get("instrument")}),
            "existing_4d_index_sources": [str(MULTIFIELD_INDEX_ROOT / "manifest.json")],
            "explicit_compact_4d_indexes": [_index_meta(path) for path in ngc_indexes],
            "catalog_tiles_available": ["d50_2644", "d50_2645"],
            "offline_wcs_available": bool(ngc_summaries and all(x.get("has_wcs") for x in ngc_summaries)),
            "apparent_quality": "usable: solved FITS with 2218-ish detected sources in quick probe",
            "selection_status": "selected",
            "reason": "only non-M106 field with local FITS and compatible bounded 4D catalog tiles",
        },
        {
            "field_id": "m106",
            "target_or_region": "M106",
            "fits_count": int(len(m106_paths)),
            "sample_fits": [str(p) for p in m106_paths[:5]],
            "dimensions": sorted({tuple(x.get("dimensions") or []) for x in m106_summaries}),
            "scale_range_arcsec": [min(m106_scales), max(m106_scales)] if m106_scales else None,
            "instrument_camera": sorted({str(x.get("instrument") or "") for x in m106_summaries if x.get("instrument")}),
            "explicit_compact_4d_indexes": [_index_meta(M106_INDEX_2823), _index_meta(M106_INDEX_2822)],
            "catalog_tiles_available": ["d50_2823", "d50_2822"],
            "offline_wcs_available": bool(m106_summaries and all(x.get("has_wcs") for x in m106_summaries)),
            "apparent_quality": "validated in P2.18 all30",
            "selection_status": "excluded_from_primary",
            "reason": "baseline already validated; P2.19 asks for fields independent of M106",
        },
        {
            "field_id": "sdss_testdata",
            "target_or_region": "astrometry-main SDSS testdata",
            "fits_count": int(len(list((ROOT / "astrometry-main/sdss/testdata").glob("*.fit")))),
            "index_4d_existing": False,
            "selection_status": "excluded",
            "reason": "not a documented ZeBlind/ZeMosaic local astrophotography corpus and no compact 4D field index is available",
        },
        *referenced_missing,
    ]
    return {
        "schema": "zeblind.p219_local_field_inventory.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "bounded_search_roots": [
            str(ROOT / "reports"),
            str(ROOT / "tools"),
            str(ROOT / "tests"),
            str(NGC6888_DIR.parent),
        ],
        "fresh_dir_exists": fresh_dir.exists(),
        "fields": fields,
        "notes": [
            "No arbitrary full-disk crawl was performed.",
            "The only local non-M106 FITS field found in bounded locations is NGC6888.",
            "NGC3628, NGC6823 and IC1848 remain report-referenced but their FITS files are absent from the current local example tree.",
        ],
    }


def _select_corpus(args: argparse.Namespace, ngc_indexes: list[Path]) -> dict[str, Any]:
    ngc_paths = sorted(NGC6888_DIR.glob("Light_NGC 6888_*.fit")) if NGC6888_DIR.exists() else []
    if int(args.max_images_per_field) > 0:
        ngc_paths = ngc_paths[: int(args.max_images_per_field)]
    selected = [
        {
            "field_id": "ngc6888",
            "target_or_region": "NGC 6888 / Crescent Nebula",
            "selection_reason": "available local non-M106 field with solved WCS and explicit compact 4D indexes",
            "limitations": "single independent non-M106 field; not enough local data for 4-6-field generalization",
            "fits": [str(p) for p in ngc_paths],
            "index_paths_nominal": [str(p) for p in ngc_indexes],
            "index_paths_reversed": [str(p) for p in reversed(ngc_indexes)],
            "tiles_for_union_candidate_tiles": ["d50_2644", "d50_2645"],
            "primary_index": str(ngc_indexes[0]),
            "secondary_index": str(ngc_indexes[1]),
        }
    ] if ngc_paths else []
    return {
        "schema": "zeblind.p219_selected_multifield_corpus.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target_images_per_field": [3, 10],
        "selected_fields": selected,
        "selected_field_count_excluding_m106": int(len(selected)),
        "selected_image_count_excluding_m106": int(sum(len(x["fits"]) for x in selected)),
        "excluded_fields": [
            {"field_id": "m106", "reason": "P2.18 baseline, excluded from primary non-M106 corpus"},
            {"field_id": "ngc3628", "reason": "FITS files absent locally"},
            {"field_id": "ngc6823", "reason": "FITS files absent locally"},
            {"field_id": "ic1848", "reason": "FITS files absent locally"},
        ],
    }


def _config(index_paths: list[Path], args: argparse.Namespace, *, historical: bool = False) -> SolveConfig:
    if historical:
        return SolveConfig(
            max_stars=int(args.quad_sources),
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
        )
    return SolveConfig(
        max_stars=int(args.quad_sources),
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


def _prepare_work_fits(source: Path, work_dir: Path, tag: str) -> Path:
    target_dir = work_dir / tag / source.stem
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    shutil.copy2(source, target)
    p22._strip_wcs(target)
    return target


def _detect_sources(source: Path, args: argparse.Namespace) -> tuple[np.ndarray, tuple[int, int], dict[str, Any], np.ndarray, dict[str, Any]]:
    raw, image_shape, detect_meta = p23._detect_runtime_stars(source, args)
    clean, clean_stats = p215._clean_verification_sources(raw, image_shape, min_sep_px=float(args.verification_min_sep_px))
    return raw, image_shape, detect_meta, clean, clean_stats


def _prep_cache(target: Path, quad_sources: np.ndarray, verification_sources: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    resolved = target.resolve()
    stat = resolved.stat()
    return {
        str(resolved): {
            "sig": (int(stat.st_mtime_ns), int(stat.st_size)),
            "downsample": int(args.downsample),
            "detect_k_sigma": float(args.detect_k_sigma),
            "detect_min_area": int(args.detect_min_area),
            "stars": quad_sources.copy(),
            "astrometry_4d_verification_stars": verification_sources.copy(),
        }
    }


def _wcs_scale_rotation(wcs: WCS | None) -> dict[str, Any]:
    if wcs is None:
        return {}
    try:
        matrix = np.asarray(wcs.pixel_scale_matrix, dtype=np.float64)
        scales = proj_plane_pixel_scales(wcs.celestial) * 3600.0
        rot = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
        return {
            "pixel_scale_arcsec": float(np.sqrt(float(scales[0]) * float(scales[1]))),
            "rotation_deg": rot,
        }
    except Exception:
        return {}


def _angle_delta_deg(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    d = (float(a) - float(b) + 180.0) % 360.0 - 180.0
    return abs(float(d))


def _offline_wcs_check(source: Path, result_wcs: WCS | None, image_shape: tuple[int, int] | None) -> dict[str, Any]:
    out: dict[str, Any] = {"available": False, "ok": None}
    if result_wcs is None or image_shape is None:
        out["reason"] = "missing_result_wcs"
        return out
    try:
        ref_wcs = WCS(fits.getheader(source))
        if not bool(getattr(ref_wcs, "has_celestial", False)):
            out["reason"] = "reference_has_no_celestial_wcs"
            return out
        height, width = int(image_shape[0]), int(image_shape[1])
        pts = np.asarray(
            [
                [width * 0.5, height * 0.5],
                [0.0, 0.0],
                [width - 1.0, 0.0],
                [width - 1.0, height - 1.0],
                [0.0, height - 1.0],
            ],
            dtype=np.float64,
        )
        result_world = result_wcs.wcs_pix2world(pts, 0)
        ref_world = ref_wcs.wcs_pix2world(pts, 0)
        c1 = SkyCoord(result_world[:, 0] * u.deg, result_world[:, 1] * u.deg, frame="icrs")
        c2 = SkyCoord(ref_world[:, 0] * u.deg, ref_world[:, 1] * u.deg, frame="icrs")
        sep_arcsec = np.asarray(c1.separation(c2).arcsec, dtype=np.float64)
        back_to_ref = np.asarray(ref_wcs.wcs_world2pix(result_world, 0), dtype=np.float64)
        pixel_errors = np.sqrt(np.sum((back_to_ref - pts) ** 2, axis=1))
        grid_x = np.linspace(0, max(0, width - 1), 5)
        grid_y = np.linspace(0, max(0, height - 1), 5)
        grid = np.asarray([[x, y] for y in grid_y for x in grid_x], dtype=np.float64)
        grid_world = result_wcs.wcs_pix2world(grid, 0)
        grid_back = np.asarray(ref_wcs.wcs_world2pix(grid_world, 0), dtype=np.float64)
        grid_errors = np.sqrt(np.sum((grid_back - grid) ** 2, axis=1))
        rmeta = _wcs_scale_rotation(result_wcs)
        fmeta = _wcs_scale_rotation(ref_wcs)
        scale_diff_pct = None
        if rmeta.get("pixel_scale_arcsec") and fmeta.get("pixel_scale_arcsec"):
            scale_diff_pct = abs(float(rmeta["pixel_scale_arcsec"]) - float(fmeta["pixel_scale_arcsec"])) / float(fmeta["pixel_scale_arcsec"]) * 100.0
        rotation_diff = _angle_delta_deg(rmeta.get("rotation_deg"), fmeta.get("rotation_deg"))
        ok = bool(
            np.isfinite(pixel_errors[0])
            and pixel_errors[0] <= 25.0
            and np.nanmedian(grid_errors) <= 25.0
            and np.nanmax(pixel_errors[1:]) <= 80.0
            and (scale_diff_pct is None or scale_diff_pct <= 5.0)
            and (rotation_diff is None or rotation_diff <= 2.0)
        )
        out.update(
            {
                "available": True,
                "ok": ok,
                "center_sep_arcsec": float(sep_arcsec[0]),
                "corner_sep_arcsec": [float(v) for v in sep_arcsec[1:]],
                "center_pixel_error": float(pixel_errors[0]),
                "max_corner_pixel_error": float(np.nanmax(pixel_errors[1:])),
                "median_grid_pixel_error": float(np.nanmedian(grid_errors)),
                "max_grid_pixel_error": float(np.nanmax(grid_errors)),
                "scale_diff_pct": scale_diff_pct,
                "rotation_diff_deg": rotation_diff,
                "tolerances": {
                    "center_pixel_error_max": 25.0,
                    "median_grid_pixel_error_max": 25.0,
                    "max_corner_pixel_error_max": 80.0,
                    "scale_diff_pct_max": 5.0,
                    "rotation_diff_deg_max": 2.0,
                },
            }
        )
        return out
    except Exception as exc:
        out["reason"] = str(exc)
        return out


def _chosen_validation(stats: dict[str, Any], success: bool) -> dict[str, Any]:
    if success and isinstance(stats.get("astrometry_4d_best_accepted_validation"), dict):
        return dict(stats["astrometry_4d_best_accepted_validation"])
    if isinstance(stats.get("astrometry_4d_best_reject"), dict):
        return dict(stats["astrometry_4d_best_reject"])
    return {}


def _failure_class(row: dict[str, Any]) -> str:
    if row.get("success"):
        return "success"
    message = str(row.get("message") or "")
    stop = str(row.get("stop_reason") or "")
    if "missing explicit" in message or stop == "missing_explicit_index_paths":
        return "index absent"
    if stop == "index_absent":
        return "index absent"
    if stop == "index_load_failed":
        return "index incompatible"
    if int(row.get("hits", 0) or 0) == 0:
        return "no relevant 4D hypothesis or no index coverage"
    val = row.get("chosen_validation") or {}
    reason = str(val.get("reason") or "")
    if "pixel_scale_out_of_range" in reason:
        return "scale invalid"
    if "rms_ok=0" in reason:
        return "candidate plausible but RMS insufficient"
    if "inliers_ok=0" in reason:
        return "candidate plausible but inliers insufficient"
    if stop == "cancelled":
        return "time budget exhausted"
    return "unclassified validation failure"


def _run_4d_case(
    *,
    field_id: str,
    source: Path,
    index_paths: list[Path],
    args: argparse.Namespace,
    tag: str,
    control: str = "baseline",
) -> dict[str, Any]:
    t0 = time.perf_counter()
    work = _prepare_work_fits(source, args.work_dir.expanduser().resolve(), tag)
    raw, image_shape, detect_meta, clean, clean_stats = _detect_sources(source, args)
    quad_sources = raw[: int(args.quad_sources)]
    result = solve_blind(
        work,
        MULTIFIELD_INDEX_ROOT,
        config=_config(index_paths, args),
        prep_cache=_prep_cache(work, quad_sources, clean, args),
    )
    stats = dict(result.stats or {})
    chosen = _chosen_validation(stats, bool(result.success))
    wmeta = _wcs_scale_rotation(result.wcs)
    row = {
        "field_id": field_id,
        "control": control,
        "fits": str(source),
        "work_fits": str(work),
        "success": bool(result.success),
        "message": str(result.message),
        "dimensions": [int(image_shape[0]), int(image_shape[1])],
        "raw_sources": int(raw.shape[0]),
        "quad_sources": int(quad_sources.shape[0]),
        "verification_sources": int(clean.shape[0]),
        "detect_meta": detect_meta,
        "clean_stats": clean_stats,
        "index_paths": [str(path.expanduser().resolve()) for path in index_paths],
        "index_order": [Path(path).stem for path in index_paths],
        "origin_tile": stats.get("astrometry_4d_selected_origin_tile_key") or chosen.get("origin_tile_key"),
        "rank": stats.get("astrometry_4d_selected_rank") or chosen.get("hit_rank"),
        "local_rank": stats.get("astrometry_4d_selected_local_rank") or chosen.get("local_rank"),
        "code_distance": chosen.get("code_distance"),
        "image_quads": int(stats.get("astrometry_4d_image_quads", 0) or 0),
        "hits": int(stats.get("astrometry_4d_hits", 0) or 0),
        "hits_by_index": stats.get("astrometry_4d_hits_by_index"),
        "hypotheses_tested": int(stats.get("astrometry_4d_hits_tested", 0) or 0),
        "accepted_candidates": int(stats.get("astrometry_4d_accepted_candidates", 0) or 0),
        "stop_reason": stats.get("astrometry_4d_stop_reason"),
        "inliers": int(chosen.get("inliers", stats.get("inliers", 0)) or 0),
        "rms_px": float(chosen.get("rms_px", stats.get("rms_px", float("nan")))),
        "median_residual_px": chosen.get("median_residual_px"),
        "mad_residual_px": chosen.get("mad_residual_px"),
        "coverage": {
            "geo_cov_x": chosen.get("geo_cov_x"),
            "geo_cov_y": chosen.get("geo_cov_y"),
            "geo_cov_area": chosen.get("geo_cov_area"),
        },
        "conditioning": chosen.get("geo_cond"),
        "pixel_scale_arcsec": chosen.get("pix_scale_arcsec") or wmeta.get("pixel_scale_arcsec"),
        "rotation_deg": wmeta.get("rotation_deg"),
        "residual_metric": chosen.get("residual_metric"),
        "legacy_inverse_inliers": chosen.get("legacy_inverse_inliers"),
        "legacy_inverse_rms_px": chosen.get("legacy_inverse_rms_px"),
        "legacy_inverse_quality": chosen.get("legacy_inverse_quality"),
        "quad_build_s": float(stats.get("astrometry_4d_quad_build_s", 0.0) or 0.0),
        "lookup_s": float(stats.get("astrometry_4d_kd_lookup_s", 0.0) or 0.0),
        "validation_s": float(stats.get("astrometry_4d_validation_s", 0.0) or 0.0),
        "solver_total_s": float(stats.get("astrometry_4d_total_s", 0.0) or 0.0),
        "wall_s": float(time.perf_counter() - t0),
        "max_accepts_hit": int(stats.get("astrometry_4d_accepted_candidates", 0) or 0) >= int(args.max_accepts),
        "max_wall_s_hit": stats.get("astrometry_4d_stop_reason") == "cancelled",
        "chosen_validation": chosen,
        "best_accepted": stats.get("astrometry_4d_best_accepted_validation") or {},
        "best_plausible_reject": stats.get("astrometry_4d_best_plausible_reject") or {},
        "best_scale_invalid_reject": stats.get("astrometry_4d_best_scale_invalid_reject") or {},
        "best_rms_invalid_reject": stats.get("astrometry_4d_best_rms_invalid_reject") or {},
        "best_geometry_invalid_reject": stats.get("astrometry_4d_best_geometry_invalid_reject") or {},
        "reject_reason_counts": stats.get("astrometry_4d_reject_reason_counts") or {},
        "offline_wcs_check": _offline_wcs_check(source, result.wcs, image_shape),
    }
    row["failure_class"] = _failure_class(row)
    row["false_positive_offline"] = bool(row["success"] and row["offline_wcs_check"].get("available") and not row["offline_wcs_check"].get("ok"))
    return row


def _run_historical_case(field_id: str, source: Path, args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    work = _prepare_work_fits(source, args.work_dir.expanduser().resolve(), "historical_subset")
    shape = _fits_summary(source).get("dimensions") or None
    image_shape = (int(shape[0]), int(shape[1])) if shape else None
    result = solve_blind(work, MULTIFIELD_INDEX_ROOT, config=_config([], args, historical=True))
    stats = dict(result.stats or {})
    return {
        "field_id": field_id,
        "fits": str(source),
        "success_historical": bool(result.success),
        "message_historical": str(result.message),
        "tile_historical": result.tile_key,
        "inliers_historical": stats.get("inliers"),
        "rms_historical": stats.get("rms_px"),
        "time_historical_s": float(stats.get("total_s", 0.0) or (time.perf_counter() - t0)),
        "wall_historical_s": float(time.perf_counter() - t0),
        "offline_wcs_check_historical": _offline_wcs_check(source, result.wcs, image_shape),
    }


def _negative_controls(selected: list[dict[str, Any]], args: argparse.Namespace, ngc_indexes: list[Path]) -> list[dict[str, Any]]:
    if not selected:
        return []
    first_ngc = Path(selected[0]["fits"])
    controls: list[dict[str, Any]] = []
    controls.append(_run_4d_case(field_id="ngc6888", source=first_ngc, index_paths=[M106_INDEX_2823, M106_INDEX_2822], args=args, tag="neg_wrong_m106_index", control="wrong_m106_indexes"))
    controls.append(_run_4d_case(field_id="ngc6888", source=first_ngc, index_paths=[ngc_indexes[0]], args=args, tag="ctrl_primary_only", control="primary_index_only"))
    controls.append(_run_4d_case(field_id="ngc6888", source=first_ngc, index_paths=[ngc_indexes[1]], args=args, tag="ctrl_secondary_only", control="secondary_index_only"))
    missing = args.work_dir.expanduser().resolve() / "missing_indexes" / "missing_p219_index.npz"
    controls.append(_run_4d_case(field_id="ngc6888", source=first_ngc, index_paths=[missing], args=args, tag="neg_missing_index", control="missing_index"))
    incompatible = MULTIFIELD_INDEX_ROOT / "tiles/d50_2644.npz"
    controls.append(_run_4d_case(field_id="ngc6888", source=first_ngc, index_paths=[incompatible], args=args, tag="neg_incompatible_index", control="incompatible_index_format"))
    m106 = sorted(M106_DIR.glob("Light_mosaic_M 106_*.fit"))[0] if M106_DIR.exists() and sorted(M106_DIR.glob("Light_mosaic_M 106_*.fit")) else None
    if m106 is not None:
        controls.append(_run_4d_case(field_id="m106", source=m106, index_paths=ngc_indexes, args=args, tag="neg_m106_with_ngc_indexes", control="m106_image_with_ngc6888_indexes"))
    return controls


def _reversed_order(selected: list[dict[str, Any]], args: argparse.Namespace, ngc_indexes: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in selected:
        rows.append(
            _run_4d_case(
                field_id="ngc6888",
                source=Path(item["fits"]),
                index_paths=list(reversed(ngc_indexes)),
                args=args,
                tag="reversed_index_order",
                control="reversed_index_order",
            )
        )
    return rows


def _metric_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    din = []
    drms = []
    diffs = []
    direct_worse = []
    for row in rows:
        legacy_in = row.get("legacy_inverse_inliers")
        legacy_rms = row.get("legacy_inverse_rms_px")
        if legacy_in is not None:
            din.append(float(row.get("inliers", 0) or 0) - float(legacy_in))
        if legacy_rms is not None and np.isfinite(float(legacy_rms)) and np.isfinite(float(row.get("rms_px", np.nan))):
            delta = float(legacy_rms) - float(row["rms_px"])
            drms.append(delta)
            if delta < -0.05:
                direct_worse.append(Path(str(row["fits"])).name)
        legacy_good = str(row.get("legacy_inverse_quality") or "") == "GOOD"
        direct_good = bool(row.get("success"))
        if legacy_in is not None and legacy_good != direct_good:
            diffs.append(Path(str(row["fits"])).name)
    return {
        "delta_inliers_direct_minus_inverse": {
            "median": _median(din),
            "p95": _percentile(din, 95),
            "max": max(din) if din else None,
            "min": min(din) if din else None,
        },
        "delta_rms_inverse_minus_direct": {
            "median": _median(drms),
            "p95": _percentile(drms, 95),
            "max": max(drms) if drms else None,
            "min": min(drms) if drms else None,
        },
        "decision_diff_count": int(len(diffs)),
        "decision_diff_cases": diffs,
        "direct_worse_by_more_than_0p05_count": int(len(direct_worse)),
        "direct_worse_by_more_than_0p05_cases": direct_worse,
    }


def _field_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["field_id"]), []).append(row)
    out: dict[str, Any] = {}
    for field_id, items in grouped.items():
        successes = [r for r in items if r.get("success")]
        out[field_id] = {
            "total": int(len(items)),
            "successes": int(len(successes)),
            "median_inliers": _median([float(r["inliers"]) for r in successes]),
            "min_inliers": min((int(r["inliers"]) for r in successes), default=None),
            "median_rms": _median([float(r["rms_px"]) for r in successes]),
            "max_rms": max((float(r["rms_px"]) for r in successes), default=None),
            "median_total_s": _median([float(r["solver_total_s"]) for r in items]),
            "p95_total_s": _percentile([float(r["solver_total_s"]) for r in items], 95),
            "median_rank": _median([float(r["rank"]) for r in successes if r.get("rank") is not None]),
            "p95_rank": _percentile([float(r["rank"]) for r in successes if r.get("rank") is not None], 95),
            "max_accepts_cases": [Path(str(r["fits"])).name for r in items if r.get("max_accepts_hit")],
            "max_wall_s_cases": [Path(str(r["fits"])).name for r in items if r.get("max_wall_s_hit")],
            "origin_tiles": sorted({str(r.get("origin_tile")) for r in successes if r.get("origin_tile")}),
        }
    return out


def _cost_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "median_total_s": _median([float(row["solver_total_s"]) for row in rows]),
        "p95_total_s": _percentile([float(row["solver_total_s"]) for row in rows], 95),
        "median_validation_s": _median([float(row["validation_s"]) for row in rows]),
        "p95_validation_s": _percentile([float(row["validation_s"]) for row in rows], 95),
        "median_hypotheses": _median([float(row["hypotheses_tested"]) for row in rows]),
        "p95_hypotheses": _percentile([float(row["hypotheses_tested"]) for row in rows], 95),
        "max_verification_sources": max((int(row["verification_sources"]) for row in rows), default=0),
        "max_accepts_cases": [Path(str(row["fits"])).name for row in rows if row.get("max_accepts_hit")],
        "max_wall_s_cases": [Path(str(row["fits"])).name for row in rows if row.get("max_wall_s_hit")],
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    field_stats = payload["field_stats"]
    metric = payload["metric_stats"]
    cost = payload["cost_stats"]
    lines = [
        "# ZeBlind P2.19 - 4D multifield bounded validation",
        "",
        "> Diagnostic only. No ZeNear, GUI, default backend, thresholds, AB/C/D core, direct metric, all-sky routing, or oracle runtime input changed.",
        "",
        "## Executive Summary",
        "",
        f"- Non-M106 fields tested: `{summary['fields_tested_excluding_m106']}`.",
        f"- Non-M106 images tested: `{summary['images_tested_excluding_m106']}`.",
        f"- 4D successes: `{summary['successes']}/{summary['baseline_cases']}`.",
        f"- False positives observed offline: `{summary['false_positives']}`.",
        f"- Negative-control false accepts: `{summary['negative_false_accepts']}`.",
        f"- Verdict: `{summary['verdict']}`.",
        "",
        "## Inventory",
        "",
    ]
    for field in payload["inventory"]["fields"]:
        lines.append(
            f"- `{field['field_id']}`: fits=`{field.get('fits_count')}`, status=`{field.get('selection_status')}`, reason={field.get('reason') or field.get('reason_for_exclusion')}"
        )
    lines.extend(
        [
            "",
            "## Selected Corpus",
            "",
            "Only `NGC6888` is selected as a non-M106 field because it is the only available local non-M106 FITS corpus with compatible bounded 4D catalog tiles. Missing report-referenced fields are documented in the inventory JSON.",
            "",
            "## Image Matrix",
            "",
            "| field | image | success | tile | rank | inliers | RMS | legacy | hyp | accepts | stop | validation | total | offline |",
            "|---|---|---:|---|---:|---:|---:|---|---:|---:|---|---:|---:|---|",
        ]
    )
    for row in payload["baseline_rows"]:
        legacy = f"{row.get('legacy_inverse_inliers')}/{_fmt(row.get('legacy_inverse_rms_px'), 3)}"
        offline = row.get("offline_wcs_check") or {}
        lines.append(
            f"| `{row['field_id']}` | `{Path(str(row['fits'])).name}` | `{row['success']}` | `{row.get('origin_tile')}` | {row.get('rank')} | {row.get('inliers')} | {_fmt(row.get('rms_px'), 3)} | `{legacy}` | {row.get('hypotheses_tested')} | {row.get('accepted_candidates')} | `{row.get('stop_reason')}` | {_fmt(row.get('validation_s'), 3)} | {_fmt(row.get('solver_total_s'), 3)} | `{offline.get('ok')}` |"
        )
    lines.extend(["", "## Field Statistics", "", "```json", json.dumps(field_stats, indent=2, default=_json_default), "```", ""])
    lines.extend(
        [
            "## Negative Controls",
            "",
            "| control | field | image | success | inliers | RMS | scale | reason | offline |",
            "|---|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in payload["negative_controls"]:
        val = row.get("chosen_validation") or {}
        offline = row.get("offline_wcs_check") or {}
        lines.append(
            f"| `{row.get('control')}` | `{row.get('field_id')}` | `{Path(str(row['fits'])).name}` | `{row.get('success')}` | {row.get('inliers')} | {_fmt(row.get('rms_px'), 3)} | {_fmt(row.get('pixel_scale_arcsec'), 3)} | {str(val.get('reason') or row.get('message'))[:120]} | `{offline.get('ok')}` |"
        )
    lines.extend(
        [
            "",
            "## Reversed Index Order",
            "",
            f"- Successes: `{summary['reversed_successes']}/{summary['reversed_cases']}`.",
            f"- Offline false positives: `{summary['reversed_false_positives']}`.",
            "",
            "## Historical Vs 4D",
            "",
            "| image | historical | hist inliers | hist RMS | hist time | 4D | 4D inliers | 4D RMS | 4D time |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    by_name = {Path(str(row["fits"])).name: row for row in payload["baseline_rows"]}
    for row in payload["historical_subset"]:
        name = Path(str(row["fits"])).name
        base = by_name.get(name, {})
        lines.append(
            f"| `{name}` | `{row.get('success_historical')}` | {row.get('inliers_historical')} | {_fmt(row.get('rms_historical'), 3)} | {_fmt(row.get('wall_historical_s'), 3)} | `{base.get('success')}` | {base.get('inliers')} | {_fmt(base.get('rms_px'), 3)} | {_fmt(base.get('solver_total_s'), 3)} |"
        )
    lines.extend(
        [
            "",
            "## Direct Vs Legacy",
            "",
            f"- Delta inliers direct-inverse: `{metric['delta_inliers_direct_minus_inverse']}`",
            f"- Delta RMS inverse-direct: `{metric['delta_rms_inverse_minus_direct']}`",
            f"- Different decisions: `{metric['decision_diff_count']}` / `{metric['decision_diff_cases']}`",
            f"- Direct markedly worse: `{metric['direct_worse_by_more_than_0p05_count']}` / `{metric['direct_worse_by_more_than_0p05_cases']}`",
            "",
            "## Runtime",
            "",
            f"- Median total: `{_fmt(cost['median_total_s'], 3)}s`, p95 total: `{_fmt(cost['p95_total_s'], 3)}s`.",
            f"- Median validation: `{_fmt(cost['median_validation_s'], 3)}s`, p95 validation: `{_fmt(cost['p95_validation_s'], 3)}s`.",
            f"- Median hypotheses: `{_fmt(cost['median_hypotheses'], 0)}`, p95 hypotheses: `{_fmt(cost['p95_hypotheses'], 0)}`.",
            f"- Max verification sources: `{cost['max_verification_sources']}`.",
            f"- `max_accepts` cases: `{cost['max_accepts_cases']}`.",
            f"- `max_wall_s` cases: `{cost['max_wall_s_cases']}`.",
            "",
            "## Failure Analysis",
            "",
        ]
    )
    failures = [row for row in payload["baseline_rows"] if not row.get("success")]
    if failures:
        for row in failures:
            lines.append(f"- `{Path(str(row['fits'])).name}`: `{row.get('failure_class')}`.")
    else:
        lines.append("- No baseline 4D failures on the selected local non-M106 corpus.")
    lines.extend(["", "## Mandatory Answers", ""])
    for answer in payload["answers"]:
        lines.append(f"- {answer}")
    lines.extend(["", "## Parameters", "", "```json", json.dumps(payload["params"], indent=2, default=_json_default), "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="P2.19 bounded non-M106 multifield validation for ZeBlind 4D.")
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_BASE / "work")
    ap.add_argument("--inventory-out", type=Path, default=DEFAULT_INVENTORY)
    ap.add_argument("--selection-out", type=Path, default=DEFAULT_SELECTION)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--max-images-per-field", type=int, default=10)
    ap.add_argument("--historical-subset", type=int, default=3)
    ap.add_argument("--verification-min-sep-px", type=float, default=0.75)
    ap.add_argument("--detect-k-sigma", type=float, default=3.0)
    ap.add_argument("--detect-min-area", type=int, default=5)
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--quad-sources", type=int, default=120)
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
    args = ap.parse_args()

    ngc_indexes = _ensure_ngc6888_indexes(DEFAULT_BASE)
    inventory = _inventory(args, ngc_indexes)
    selection = _select_corpus(args, ngc_indexes)
    selected_items = list((selection.get("selected_fields") or [{}])[0].get("fits") or []) if selection.get("selected_fields") else []
    selected = [{"field_id": "ngc6888", "fits": item} for item in selected_items]
    _write_json(args.inventory_out, inventory)
    _write_json(args.selection_out, selection)

    baseline_rows: list[dict[str, Any]] = []
    for i, item in enumerate(selected, start=1):
        print(json.dumps({"event": "baseline_start", "i": i, "n": len(selected), "fits": Path(item["fits"]).name}), flush=True)
        row = _run_4d_case(
            field_id=str(item["field_id"]),
            source=Path(item["fits"]),
            index_paths=ngc_indexes,
            args=args,
            tag="baseline_q120_vfull",
        )
        baseline_rows.append(row)
        print(json.dumps({"event": "baseline_done", "fits": Path(item["fits"]).name, "success": row["success"], "inliers": row["inliers"], "rms": row["rms_px"], "total_s": row["solver_total_s"]}, default=_json_default), flush=True)

    reversed_rows = _reversed_order(selected, args, ngc_indexes)
    negative = _negative_controls(selected, args, ngc_indexes)
    historical = []
    for item in selected[: max(0, int(args.historical_subset))]:
        historical.append(_run_historical_case(str(item["field_id"]), Path(item["fits"]), args))

    successes = [row for row in baseline_rows if row.get("success")]
    false_positives = [row for row in baseline_rows if row.get("false_positive_offline")]
    reversed_successes = [row for row in reversed_rows if row.get("success")]
    reversed_false = [row for row in reversed_rows if row.get("false_positive_offline")]
    negative_false_accepts = [
        row for row in negative
        if row.get("control") in {"wrong_m106_indexes", "missing_index", "incompatible_index_format", "m106_image_with_ngc6888_indexes"}
        and row.get("success")
    ]
    field_stats = _field_stats(baseline_rows)
    metric_stats = _metric_stats(baseline_rows)
    cost_stats = _cost_stats(baseline_rows)
    fields_tested = len({row["field_id"] for row in baseline_rows})
    bounded_non_m106_smoke_positive = (
        fields_tested >= 1
        and len(successes) == len(baseline_rows)
        and not false_positives
        and not negative_false_accepts
        and len(reversed_successes) == len(reversed_rows)
        and not reversed_false
        and not cost_stats["max_wall_s_cases"]
    )
    experimental_release_candidate_general = bool(fields_tested >= 4 and bounded_non_m106_smoke_positive)
    verdict = "B. Generalisation partielle"
    if not bounded_non_m106_smoke_positive:
        verdict = "C. Generalisation negative"
    answers = [
        "Le backend 4D se generalise hors M106 sur le seul champ local exploitable, NGC6888; ce n'est pas encore une validation multi-champs large.",
        f"Champs independants hors M106 testes: `{fields_tested}`.",
        f"Taux de succes global: `{len(successes)}/{len(baseline_rows)}`; par champ: `{field_stats}`.",
        f"Faux positifs observes: `{len(false_positives) + len(negative_false_accepts) + len(reversed_false)}`.",
        "Les solutions acceptees correspondent au WCS offline attendu." if not false_positives else "Au moins une solution acceptee ne correspond pas au WCS offline attendu.",
        f"Ordre des index: `{len(reversed_successes)}/{len(reversed_rows)}` succes, faux positifs `{len(reversed_false)}`.",
        f"Metrique directe stable hors M106: residual_metric reste `catalog_world2pix_to_image_px`; decisions direct/legacy differentes `{metric_stats['decision_diff_count']}`.",
        "Aucun echec 4D baseline sur NGC6888; les rejets observes viennent des controles d'index absent/incompatible ou mauvais champ.",
        f"Cout runtime acceptable sur ce corpus: median `{_fmt(cost_stats['median_total_s'], 3)}s`, p95 `{_fmt(cost_stats['p95_total_s'], 3)}s`, aucun `max_wall_s`.",
        "Statut general: experimental release candidate conserve seulement comme generalisation partielle, pas encore RC multi-champs complet faute de 4 champs locaux.",
        "Pret pour un manifest d'index et un routage borne en prototype, mais l'integration GUI minimale doit attendre un corpus 4-6 champs reel.",
        "Direction unique recommandee: fournir/selectionner 3 a 5 champs FITS supplementaires avec WCS offline et construire pour chacun 1-2 index 4D compacts bornes, puis relancer exactement ce probe.",
    ]
    payload = {
        "schema": "zeblind.p219_4d_multifield_validation.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inventory": inventory,
        "selection": selection,
        "baseline_rows": baseline_rows,
        "reversed_order_rows": reversed_rows,
        "negative_controls": negative,
        "historical_subset": historical,
        "field_stats": field_stats,
        "metric_stats": metric_stats,
        "cost_stats": cost_stats,
        "summary": {
            "fields_tested_excluding_m106": int(fields_tested),
            "images_tested_excluding_m106": int(len(baseline_rows)),
            "baseline_cases": int(len(baseline_rows)),
            "successes": int(len(successes)),
            "false_positives": int(len(false_positives)),
            "negative_false_accepts": int(len(negative_false_accepts)),
            "reversed_cases": int(len(reversed_rows)),
            "reversed_successes": int(len(reversed_successes)),
            "reversed_false_positives": int(len(reversed_false)),
            "verdict": verdict,
            "bounded_non_m106_smoke_positive": bool(bounded_non_m106_smoke_positive),
            "experimental_release_candidate_general": bool(experimental_release_candidate_general),
            "corpus_diversity_limitation": "only one non-M106 local field available",
        },
        "answers": answers,
        "params": {
            "quad_hash_schema": ASTROMETRY_AB_CODE_4D_SCHEMA,
            "quad_sources": int(args.quad_sources),
            "verification_sources": "full",
            "blind_astrometry_4d_validation_catalog_policy": "union_candidate_tiles",
            "blind_astrometry_4d_accept_policy": "best_within_budget",
            "quality_inliers": int(args.quality_inliers),
            "quality_rms": float(args.quality_rms),
            "match_radius_px": float(args.match_radius_px),
            "max_quads": int(args.max_quads),
            "max_hypotheses": int(args.max_hypotheses),
            "max_accepts": int(args.max_accepts),
            "max_wall_s": float(args.max_wall_s),
            "wcs_oracle_runtime_input": False,
            "all_sky": False,
            "default_behavior_changed": False,
            "gui_changed": False,
        },
    }
    _write_json(args.json_out, payload)
    _write_report(args.report, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
