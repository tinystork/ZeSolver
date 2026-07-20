#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy import units as u

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zeblindsolver.astap_4d_builder import Astap4DBuildConfig, build_4d_index_from_astap
from zeblindsolver.index_manifest_4d import (
    MANIFEST_SCHEMA,
    MANIFEST_VERSION,
    load_4d_index_manifest,
    sha256_file,
)
from zeblindsolver.quad_index_4d import (
    ASTROMETRY_AB_CODE_4D_DEFAULT_TOL,
    ASTROMETRY_AB_CODE_4D_SCHEMA,
    Quad4DIndex,
    scientific_payload_fingerprint,
)
from zeblindsolver.zeblindsolver import SolveConfig, solve_blind


P1D3B_TILE_KEYS = ("d50_2602", "d50_2644", "d50_2645", "d50_2702", "d50_2822", "d50_2823")
P1D3B_PRODUCT_ORDER = ("d50_2823", "d50_2822", "d50_2644", "d50_2645", "d50_2602", "d50_2702")
P1D3B_MINI_LABELS = (
    "232329",
    "232431",
    "232144",
    "232205",
    "232247",
    "232350",
    "232102",
    "232513",
    "232534",
    "232658",
)
P1D3B_DIFFICULT_LABELS = ("233828", "234013")
P1D3B_BUILD_CONFIG = Astap4DBuildConfig(
    family="d50",
    tile_keys=(),
    level="S",
    mag_cap=15.0,
    source_max_stars=2000,
    source_star_truncation_mode="native_prefix",
    max_stars_per_tile=2000,
    max_quads_per_tile=40000,
    sampler_tag="catalog_ring_coverage",
    code_tol_recommended=ASTROMETRY_AB_CODE_4D_DEFAULT_TOL,
    dtype="float32",
    quad_schema=ASTROMETRY_AB_CODE_4D_SCHEMA,
)


@dataclass(frozen=True)
class RuntimePolicy:
    max_stars: int = 120
    max_quads: int = 2500
    detect_k_sigma: float = 3.0
    detect_min_area: int = 5
    downsample: int = 1
    quality_inliers: int = 40
    quality_rms: float = 1.2
    pixel_scale_min_arcsec: float = 1.79
    pixel_scale_max_arcsec: float = 2.99
    search_budget_s: float = 45.0
    max_accepts: int = 64
    code_tol: float = 0.015
    max_hits_4d: int = 2000
    max_hits_per_image_quad: int = 8
    max_hypotheses: int = 2000
    image_strategy: str = "log_spaced"
    match_radius_px: float = 3.0
    validation_catalog_policy: str = "union_candidate_tiles"
    source_policy: str = "diagnostic_unfiltered"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return sha256_file(path)


def snapshot_tree(root: Path, *, include_npz_sha: bool = False) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if not root.exists():
        return {"path": str(root), "exists": False}
    files = [path for path in root.rglob("*") if path.is_file()]
    payload: dict[str, Any] = {
        "path": str(root),
        "exists": True,
        "file_count": len(files),
        "total_size_bytes": int(sum(path.stat().st_size for path in files)),
        "mtime_ns_sum": int(sum(path.stat().st_mtime_ns for path in files)),
    }
    if include_npz_sha:
        payload["npz_sha256"] = {
            str(path.relative_to(root)): _sha256(path)
            for path in sorted(files)
            if path.suffix.lower() == ".npz"
        }
    return payload


def snapshot_file(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.exists():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(path.stat().st_size),
        "mtime_ns": int(path.stat().st_mtime_ns),
        "sha256": _sha256(path),
    }


def build_runtime_config(index_paths: Iterable[Path | str], *, accept_policy: str, policy: RuntimePolicy | None = None) -> SolveConfig:
    runtime = policy or RuntimePolicy()
    return SolveConfig(
        max_stars=int(runtime.max_stars),
        max_quads=int(runtime.max_quads),
        detect_k_sigma=float(runtime.detect_k_sigma),
        detect_min_area=int(runtime.detect_min_area),
        downsample=int(runtime.downsample),
        quality_inliers=int(runtime.quality_inliers),
        quality_rms=float(runtime.quality_rms),
        pixel_scale_min_arcsec=float(runtime.pixel_scale_min_arcsec),
        pixel_scale_max_arcsec=float(runtime.pixel_scale_max_arcsec),
        blind_global_hard_budget_s=0.0,
        blind_astrometry_4d_search_budget_s=float(runtime.search_budget_s),
        blind_reuse_existing_solved_wcs=False,
        quad_hash_schema=ASTROMETRY_AB_CODE_4D_SCHEMA,
        blind_astrometry_4d_index_enabled=False,
        blind_astrometry_4d_index_path="",
        blind_astrometry_4d_index_paths=tuple(str(Path(path).expanduser().resolve()) for path in index_paths),
        blind_astrometry_4d_validation_catalog_policy=str(runtime.validation_catalog_policy),
        blind_astrometry_4d_source_policy=str(runtime.source_policy),
        blind_astrometry_4d_accept_policy=str(accept_policy),
        blind_astrometry_4d_max_accepts=int(runtime.max_accepts),
        blind_astrometry_4d_code_tol=float(runtime.code_tol),
        blind_astrometry_4d_max_hits=int(runtime.max_hits_4d),
        blind_astrometry_4d_max_hits_per_image_quad=int(runtime.max_hits_per_image_quad),
        blind_astrometry_4d_max_hypotheses=int(runtime.max_hypotheses),
        blind_astrometry_4d_image_strategy=str(runtime.image_strategy),
        blind_astrometry_4d_match_radius_px=float(runtime.match_radius_px),
    )


def build_config_for_tile(tile_key: str) -> Astap4DBuildConfig:
    return Astap4DBuildConfig(
        **{
            **asdict(P1D3B_BUILD_CONFIG),
            "tile_keys": (str(tile_key),),
        }
    )


def assert_p1d3b_build_config() -> dict[str, Any]:
    cfg = P1D3B_BUILD_CONFIG
    expected = {
        "family": "d50",
        "mag_cap": 15.0,
        "source_max_stars": 2000,
        "source_star_truncation_mode": "native_prefix",
        "max_stars_per_tile": 2000,
        "max_quads_per_tile": 40000,
        "sampler_tag": "catalog_ring_coverage",
        "dtype": "float32",
        "quad_schema": ASTROMETRY_AB_CODE_4D_SCHEMA,
    }
    actual = asdict(cfg)
    mismatches = {key: {"expected": value, "actual": actual.get(key)} for key, value in expected.items() if actual.get(key) != value}
    if mismatches:
        raise RuntimeError(f"P1D-3B build configuration mismatch: {mismatches}")
    return actual


def build_direct_indexes_twice(astap_root: Path, out_dir: Path, *, tile_order: Iterable[str]) -> dict[str, Any]:
    astap_root = astap_root.expanduser().resolve()
    first_dir = out_dir / "direct_build_a"
    second_dir = out_dir / "direct_build_b"
    fingerprints: dict[str, dict[str, Any]] = {}
    first_paths: dict[str, Path] = {}
    second_paths: dict[str, Path] = {}
    for tile_key in tile_order:
        cfg = build_config_for_tile(tile_key)
        first = first_dir / f"{tile_key}_direct_S_q40000.npz"
        second = second_dir / f"{tile_key}_direct_S_q40000.npz"
        t0 = time.perf_counter()
        build_4d_index_from_astap(astap_root, first, config=cfg)
        first_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        build_4d_index_from_astap(astap_root, second, config=cfg)
        second_s = time.perf_counter() - t1
        fp_a = scientific_payload_fingerprint(first)
        fp_b = scientific_payload_fingerprint(second)
        if fp_a != fp_b:
            raise RuntimeError(f"direct index scientific fingerprint mismatch for {tile_key}: {fp_a} != {fp_b}")
        index = Quad4DIndex.load(first)
        build_params = dict(index.metadata.get("build_parameters") or {})
        if build_params.get("mag_cap") != cfg.mag_cap:
            raise RuntimeError(f"metadata mag_cap mismatch for {tile_key}: {build_params.get('mag_cap')} != {cfg.mag_cap}")
        fingerprints[tile_key] = {
            "first_path": str(first),
            "second_path": str(second),
            "scientific_fingerprint": fp_a,
            "file_sha256": _sha256(first),
            "star_count": int(index.catalog_ra_dec.shape[0]),
            "quad_count": int(index.codes_4d.shape[0]),
            "first_build_s": first_s,
            "second_build_s": second_s,
            "metadata_build_parameters": build_params,
        }
        first_paths[tile_key] = first
        second_paths[tile_key] = second
    return {"first_paths": first_paths, "second_paths": second_paths, "fingerprints": fingerprints}


def reuse_direct_indexes(build_dir: Path, *, tile_order: Iterable[str]) -> dict[str, Any]:
    build_dir = build_dir.expanduser().resolve()
    first_dir = build_dir / "direct_build_a"
    second_dir = build_dir / "direct_build_b"
    fingerprints: dict[str, dict[str, Any]] = {}
    first_paths: dict[str, Path] = {}
    second_paths: dict[str, Path] = {}
    for tile_key in tile_order:
        first = first_dir / f"{tile_key}_direct_S_q40000.npz"
        second = second_dir / f"{tile_key}_direct_S_q40000.npz"
        if not first.exists() or not second.exists():
            raise FileNotFoundError(f"missing reused direct index pair for {tile_key}: {first}, {second}")
        fp_a = scientific_payload_fingerprint(first)
        fp_b = scientific_payload_fingerprint(second)
        if fp_a != fp_b:
            raise RuntimeError(f"reused direct index scientific fingerprint mismatch for {tile_key}: {fp_a} != {fp_b}")
        index = Quad4DIndex.load(first)
        fingerprints[tile_key] = {
            "first_path": str(first),
            "second_path": str(second),
            "scientific_fingerprint": fp_a,
            "file_sha256": _sha256(first),
            "star_count": int(index.catalog_ra_dec.shape[0]),
            "quad_count": int(index.codes_4d.shape[0]),
            "first_build_s": None,
            "second_build_s": None,
            "metadata_build_parameters": dict(index.metadata.get("build_parameters") or {}),
            "reused": True,
        }
        first_paths[tile_key] = first
        second_paths[tile_key] = second
    return {"first_paths": first_paths, "second_paths": second_paths, "fingerprints": fingerprints}


def manifest_entry_from_index(entry_id: str, index_path: Path, *, priority: int, source_label: str) -> dict[str, Any]:
    index = Quad4DIndex.load(index_path)
    meta = dict(index.metadata)
    return {
        "id": entry_id,
        "enabled": True,
        "priority": int(priority),
        "path": str(index_path.expanduser().resolve()),
        "filename": index_path.name,
        "sha256": _sha256(index_path),
        "file_size_bytes": int(index_path.stat().st_size),
        "quad_schema": ASTROMETRY_AB_CODE_4D_SCHEMA,
        "index_version": int(meta.get("version", 1)),
        "level": str(meta.get("level") or "S"),
        "tile_keys": list(index.tile_keys),
        "star_count": int(index.catalog_ra_dec.shape[0]),
        "quad_count": int(index.codes_4d.shape[0]),
        "sampler_tag": str(meta.get("sampler_tag") or "catalog_ring_coverage"),
        "code_tol_recommended": float(meta.get("code_tol_recommended", ASTROMETRY_AB_CODE_4D_DEFAULT_TOL)),
        "catalog_source": source_label,
        "generation_metadata": {
            "source_catalog": str(meta.get("source_catalog") or source_label),
            "max_stars_per_tile": int(meta.get("max_stars_per_tile", 0) or 0),
            "max_quads_per_tile": int(meta.get("max_quads_per_tile", 0) or 0),
            "star_count": int(index.catalog_ra_dec.shape[0]),
            "entry_count": int(index.codes_4d.shape[0]),
            "dtype": str(meta.get("dtype") or ""),
        },
    }


def write_strict_manifest(path: Path, entries: list[dict[str, Any]], *, description: str) -> None:
    payload = {
        "schema": MANIFEST_SCHEMA,
        "manifest_version": MANIFEST_VERSION,
        "description": description,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "indexes": entries,
    }
    _write_json(path, payload)
    load_4d_index_manifest(path)


def make_comparison_manifests(
    product_manifest: Path,
    direct_paths_by_tile: dict[str, Path],
    out_dir: Path,
) -> dict[str, Any]:
    loaded = load_4d_index_manifest(product_manifest)
    product_by_tile = {entry.tile_keys[0]: entry for entry in loaded.entries}
    missing = [tile for tile in P1D3B_PRODUCT_ORDER if tile not in product_by_tile]
    if missing:
        raise RuntimeError(f"product manifest missing P1D-3B tile(s): {missing}")
    baseline_entries: list[dict[str, Any]] = []
    direct_entries: list[dict[str, Any]] = []
    for priority, tile_key in enumerate(P1D3B_PRODUCT_ORDER):
        product = product_by_tile[tile_key]
        base_entry = dict(product.manifest_entry)
        base_entry["priority"] = priority
        base_entry["path"] = str(product.path)
        baseline_entries.append(base_entry)
        direct_path = direct_paths_by_tile[tile_key]
        direct_entries.append(
            manifest_entry_from_index(
                product.id,
                direct_path,
                priority=priority,
                source_label="astap_raw",
            )
        )
    baseline_manifest = out_dir / "baseline_manifest.json"
    direct_manifest = out_dir / "direct_manifest.json"
    write_strict_manifest(
        baseline_manifest,
        baseline_entries,
        description="P1D-3B baseline manifest referencing unchanged product NPZ.",
    )
    write_strict_manifest(
        direct_manifest,
        direct_entries,
        description="P1D-3B direct manifest referencing temporary ASTAP-built NPZ.",
    )
    return {
        "baseline_manifest": baseline_manifest,
        "direct_manifest": direct_manifest,
        "baseline_paths": tuple(entry.path for entry in load_4d_index_manifest(baseline_manifest).entries),
        "direct_paths": tuple(entry.path for entry in load_4d_index_manifest(direct_manifest).entries),
    }


def discover_m106_cases(data_dir: Path, *, labels: Iterable[str] | None = None) -> list[dict[str, Any]]:
    data_dir = data_dir.expanduser().resolve()
    wanted = {str(label) for label in labels} if labels else None
    rows: list[dict[str, Any]] = []
    for path in sorted(data_dir.glob("*.fit*")):
        match = re.search(r"-(\d{6})", path.name)
        if not match:
            continue
        label = match.group(1)
        if wanted is not None and label not in wanted:
            continue
        rows.append({"label": label, "path": path, "corpus": "m106"})
    return rows


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


def _load_wcs_and_shape(path: Path) -> tuple[Any | None, tuple[int, int] | None]:
    from astropy.wcs import WCS

    try:
        with fits.open(path, memmap=False) as hdul:
            shape = tuple(int(v) for v in hdul[0].data.shape[-2:])
            wcs = WCS(hdul[0].header).celestial
            if not bool(getattr(wcs, "has_celestial", False)):
                return None, shape
            return wcs, shape
    except Exception:
        return None, None


def _pixel_scale_arcsec(wcs: Any) -> float:
    try:
        matrix = np.asarray(wcs.pixel_scale_matrix, dtype=np.float64)
        det = float(np.linalg.det(matrix))
        if not math.isfinite(det) or abs(det) <= 0.0:
            return float("nan")
        return float(math.sqrt(abs(det)) * 3600.0)
    except Exception:
        return float("nan")


def _rotation_deg(wcs: Any) -> float | None:
    try:
        matrix = np.asarray(wcs.pixel_scale_matrix, dtype=np.float64)
        return float(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])))
    except Exception:
        return None


def _parity(wcs: Any) -> str | None:
    try:
        det = float(np.linalg.det(np.asarray(wcs.pixel_scale_matrix, dtype=np.float64)))
        if det > 0:
            return "positive"
        if det < 0:
            return "negative"
        return "degenerate"
    except Exception:
        return None


def _wcs_points(shape: tuple[int, int]) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    return np.asarray(
        [
            [width / 2.0, height / 2.0],
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [0.0, height - 1.0],
            [width - 1.0, height - 1.0],
        ],
        dtype=np.float64,
    )


def wcs_metrics(candidate: Path, reference: Path | None = None, compare_to: Path | None = None) -> dict[str, Any]:
    cand_wcs, shape = _load_wcs_and_shape(candidate)
    if cand_wcs is None or shape is None:
        return {"has_wcs": False}
    result: dict[str, Any] = {
        "has_wcs": True,
        "scale_arcsec_px": _pixel_scale_arcsec(cand_wcs),
        "rotation_deg": _rotation_deg(cand_wcs),
        "parity": _parity(cand_wcs),
    }
    pts = _wcs_points(shape)
    result["center_ra_dec"] = [float(v) for v in np.asarray(cand_wcs.wcs_pix2world(pts[:1], 0)[0], dtype=np.float64)]
    if reference is not None:
        ref_wcs, _ = _load_wcs_and_shape(reference)
        if ref_wcs is not None:
            ref = ref_wcs.pixel_to_world(pts[:, 0], pts[:, 1])
            cand = cand_wcs.pixel_to_world(pts[:, 0], pts[:, 1])
            sep = ref.separation(cand).arcsec
            ref_scale = _pixel_scale_arcsec(ref_wcs)
            scale_ratio = (
                float(result["scale_arcsec_px"] / ref_scale)
                if math.isfinite(ref_scale) and ref_scale > 0
                else None
            )
            oracle_usable = scale_ratio is not None and 0.5 <= float(scale_ratio) <= 2.0
            result.update(
                {
                    "oracle_available": True,
                    "oracle_usable": bool(oracle_usable),
                    "oracle_unusable_reason": None if oracle_usable else "reference_pixel_scale_incompatible",
                    "oracle_center_sep_arcsec": float(sep[0]),
                    "oracle_corner_max_sep_arcsec": float(np.max(sep[1:])),
                    "oracle_corner_median_sep_arcsec": float(np.median(sep[1:])),
                    "oracle_scale_ref_arcsec_px": float(ref_scale),
                    "oracle_scale_ratio": scale_ratio,
                }
            )
    if compare_to is not None:
        other_wcs, _ = _load_wcs_and_shape(compare_to)
        if other_wcs is not None:
            other = other_wcs.pixel_to_world(pts[:, 0], pts[:, 1])
            cand = cand_wcs.pixel_to_world(pts[:, 0], pts[:, 1])
            sep = other.separation(cand).arcsec
            result.update(
                {
                    "compare_center_sep_arcsec": float(sep[0]),
                    "compare_corner_max_sep_arcsec": float(np.max(sep[1:])),
                    "compare_corner_median_sep_arcsec": float(np.median(sep[1:])),
                }
            )
    return result


def _copy_case(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    _strip_wcs(target)


def run_single_solve(
    case: dict[str, Any],
    *,
    mode: str,
    index_paths: tuple[Path, ...],
    accept_policy: str,
    legacy_index_root: Path,
    work_dir: Path,
    runtime_policy: RuntimePolicy,
) -> dict[str, Any]:
    source = Path(case["path"]).expanduser().resolve()
    label = str(case["label"])
    target = work_dir / accept_policy / mode / label / source.name
    _copy_case(source, target)
    cfg = build_runtime_config(index_paths, accept_policy=accept_policy, policy=runtime_policy)
    start = time.perf_counter()
    result = solve_blind(target, legacy_index_root.expanduser().resolve(), config=cfg)
    wall_s = time.perf_counter() - start
    stats = dict(result.stats or {})
    metrics = wcs_metrics(target, source) if result.success else {"has_wcs": False}
    return {
        "label": label,
        "corpus": str(case.get("corpus") or ""),
        "mode": mode,
        "accept_policy": accept_policy,
        "copy_path": str(target),
        "success": bool(result.success),
        "failure_reason": "" if result.success else str(result.message),
        "message": str(result.message),
        "selected_index": stats.get("astrometry_4d_selected_index_path"),
        "selected_tile": result.tile_key or stats.get("astrometry_4d_selected_origin_tile_key"),
        "candidate_rank": stats.get("astrometry_4d_selected_rank"),
        "local_rank": stats.get("astrometry_4d_selected_local_rank"),
        "hits_4d": int(stats.get("astrometry_4d_hits", 0) or 0),
        "hits_tested": int(stats.get("astrometry_4d_hits_tested", 0) or 0),
        "validated_candidates": int(stats.get("astrometry_4d_accepted_candidates", 0) or 0),
        "inliers": int(stats.get("inliers", 0) or 0),
        "rms_px": float(stats.get("rms_px", float("nan"))),
        "pix_scale_arcsec": float(stats.get("pix_scale_arcsec", float("nan"))),
        "rotation_deg": metrics.get("rotation_deg"),
        "parity": metrics.get("parity"),
        "total_s": float(stats.get("astrometry_4d_total_s", wall_s) or wall_s),
        "wall_s": float(wall_s),
        "search_s": float(stats.get("astrometry_4d_kd_lookup_s", 0.0) or 0.0),
        "validation_s": float(stats.get("astrometry_4d_validation_s", 0.0) or 0.0),
        "quad_build_s": float(stats.get("astrometry_4d_quad_build_s", 0.0) or 0.0),
        "first_plausible_rank": stats.get("astrometry_4d_first_plausible_rank"),
        "first_accepted_rank": stats.get("astrometry_4d_first_accepted_rank"),
        "best_accepted_rank": stats.get("astrometry_4d_best_accepted_rank"),
        "reject_reason_counts": stats.get("astrometry_4d_reject_reason_counts") or stats.get("reject_reason_counts") or {},
        "stop_reason": stats.get("astrometry_4d_stop_reason"),
        "wcs_metrics": metrics,
    }


def solution_runtime_valid(row: dict[str, Any], policy: RuntimePolicy | None = None) -> bool:
    runtime = policy or RuntimePolicy()
    if not bool(row.get("success")):
        return False
    inliers = int(row.get("inliers", 0) or 0)
    rms = float(row.get("rms_px", float("nan")))
    metrics = dict(row.get("wcs_metrics") or {})
    scale = float(row.get("pix_scale_arcsec", float("nan")))
    if not math.isfinite(scale):
        scale = float(metrics.get("scale_arcsec_px", float("nan")))
    if inliers < int(runtime.quality_inliers):
        return False
    if not math.isfinite(rms) or rms > float(runtime.quality_rms):
        return False
    if not math.isfinite(scale) or not (runtime.pixel_scale_min_arcsec <= scale <= runtime.pixel_scale_max_arcsec):
        return False
    return True


def solution_valid(row: dict[str, Any], policy: RuntimePolicy | None = None) -> bool:
    if not solution_runtime_valid(row, policy):
        return False
    metrics = dict(row.get("wcs_metrics") or {})
    if bool(metrics.get("oracle_usable", False)):
        center_sep = metrics.get("oracle_center_sep_arcsec")
        corner_sep = metrics.get("oracle_corner_max_sep_arcsec")
        if center_sep is not None and float(center_sep) > 30.0:
            return False
        if corner_sep is not None and float(corner_sep) > 120.0:
            return False
    return True


def classify_pair(baseline: dict[str, Any], direct: dict[str, Any], policy: RuntimePolicy | None = None) -> str:
    base_runtime_ok = solution_runtime_valid(baseline, policy)
    direct_runtime_ok = solution_runtime_valid(direct, policy)
    if baseline.get("success") and direct.get("success") and base_runtime_ok and direct_runtime_ok:
        direct_metrics = dict(direct.get("wcs_metrics") or {})
        compare_center = direct_metrics.get("compare_center_sep_arcsec")
        compare_corner = direct_metrics.get("compare_corner_max_sep_arcsec")
        if compare_center is not None and compare_corner is not None:
            if float(compare_center) <= 5.0 and float(compare_corner) <= 20.0:
                return "SAME_SUCCESS_EQUIVALENT_WCS"
    base_ok = solution_valid(baseline, policy)
    direct_ok = solution_valid(direct, policy)
    if baseline.get("success") and not base_ok:
        return "INVALID_BASELINE_SOLUTION"
    if direct.get("success") and not direct_ok:
        return "INVALID_DIRECT_SOLUTION"
    if base_ok and direct_ok:
        direct_metrics = dict(direct.get("wcs_metrics") or {})
        compare_center = direct_metrics.get("compare_center_sep_arcsec")
        compare_corner = direct_metrics.get("compare_corner_max_sep_arcsec")
        if compare_center is not None and compare_corner is not None:
            if float(compare_center) <= 5.0 and float(compare_corner) <= 20.0:
                return "SAME_SUCCESS_EQUIVALENT_WCS"
        return "SAME_SUCCESS_DIFFERENT_VALID_WCS"
    if base_ok and not direct_ok:
        return "DIRECT_LOSS"
    if direct_ok and not base_ok:
        return "DIRECT_GAIN_VALIDATED"
    base_reason = str(baseline.get("failure_reason") or baseline.get("stop_reason") or "")
    direct_reason = str(direct.get("failure_reason") or direct.get("stop_reason") or "")
    if base_reason == direct_reason:
        return "BOTH_FAIL_SAME_REASON"
    return "BOTH_FAIL_DIFFERENT_REASON"


def compare_runtime(
    cases: list[dict[str, Any]],
    *,
    baseline_paths: tuple[Path, ...],
    direct_paths: tuple[Path, ...],
    legacy_index_root: Path,
    work_dir: Path,
    policies: Iterable[str],
    runtime_policy: RuntimePolicy,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for accept_policy in policies:
        for ordinal, case in enumerate(cases, start=1):
            _progress("solve_pair_start", policy=accept_policy, ordinal=ordinal, total=len(cases), label=case["label"])
            baseline = run_single_solve(
                case,
                mode="baseline",
                index_paths=baseline_paths,
                accept_policy=accept_policy,
                legacy_index_root=legacy_index_root,
                work_dir=work_dir,
                runtime_policy=runtime_policy,
            )
            direct = run_single_solve(
                case,
                mode="direct",
                index_paths=direct_paths,
                accept_policy=accept_policy,
                legacy_index_root=legacy_index_root,
                work_dir=work_dir,
                runtime_policy=runtime_policy,
            )
            if baseline["success"] and direct["success"]:
                direct["wcs_metrics"] = wcs_metrics(Path(direct["copy_path"]), Path(case["path"]), Path(baseline["copy_path"]))
            rows.append(
                {
                    "label": str(case["label"]),
                    "corpus": str(case.get("corpus") or ""),
                    "accept_policy": accept_policy,
                    "classification": classify_pair(baseline, direct, runtime_policy),
                    "baseline": baseline,
                    "direct": direct,
                }
            )
            _progress(
                "solve_pair_done",
                policy=accept_policy,
                ordinal=ordinal,
                total=len(cases),
                label=case["label"],
                classification=rows[-1]["classification"],
                baseline_success=baseline["success"],
                direct_success=direct["success"],
            )
    return rows


def summarize_runtime(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"classifications": {}, "by_policy_mode": {}}
    for row in rows:
        cls = str(row["classification"])
        summary["classifications"][cls] = int(summary["classifications"].get(cls, 0)) + 1
        for mode in ("baseline", "direct"):
            key = f"{row['accept_policy']}:{mode}"
            bucket = summary["by_policy_mode"].setdefault(key, {"success": 0, "fail": 0, "total_s": [], "search_s": [], "validation_s": []})
            data = row[mode]
            bucket["success" if data.get("success") else "fail"] += 1
            bucket["total_s"].append(float(data.get("total_s", 0.0) or 0.0))
            bucket["search_s"].append(float(data.get("search_s", 0.0) or 0.0))
            bucket["validation_s"].append(float(data.get("validation_s", 0.0) or 0.0))
    for bucket in summary["by_policy_mode"].values():
        for metric in ("total_s", "search_s", "validation_s"):
            values = sorted(float(v) for v in bucket[metric])
            if not values:
                bucket[metric] = {}
                continue
            p95_idx = min(len(values) - 1, int(math.ceil(0.95 * len(values))) - 1)
            bucket[metric] = {
                "median": float(statistics.median(values)),
                "p95": float(values[p95_idx]),
                "max": float(values[-1]),
            }
    return summary


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    rows = report.get("runtime_rows") or []
    lines = [
        "# P1D-3B Direct Blind 4D Runtime Validation",
        "",
        f"- status: `{report.get('status')}`",
        f"- gate: `{report.get('gate')}`",
        f"- work_dir: `{report.get('work_dir')}`",
        f"- corpus cases: `{len(report.get('cases') or [])}`",
        f"- policies: `{', '.join(report.get('policies') or [])}`",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(report.get("build_config"), indent=2, sort_keys=True, default=_json_default),
        "```",
        "",
        "## Runtime Summary",
        "",
        "```json",
        json.dumps(report.get("runtime_summary"), indent=2, sort_keys=True, default=_json_default),
        "```",
        "",
        "## Per FITS",
        "",
        "| policy | label | classification | baseline | direct | baseline tile | direct tile | baseline inliers/rms | direct inliers/rms | direct vs baseline center/corner |",
        "|---|---:|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        base = row["baseline"]
        direct = row["direct"]
        metrics = dict(direct.get("wcs_metrics") or {})
        compare = ""
        if metrics.get("compare_center_sep_arcsec") is not None:
            compare = f"{float(metrics['compare_center_sep_arcsec']):.3f}/{float(metrics.get('compare_corner_max_sep_arcsec', 0.0)):.3f}"
        lines.append(
            "| {policy} | {label} | `{cls}` | {bs} | {ds} | {bt} | {dt} | {bi}/{br:.3f} | {di}/{dr:.3f} | {cmp} |".format(
                policy=row["accept_policy"],
                label=row["label"],
                cls=row["classification"],
                bs="ok" if base.get("success") else "fail",
                ds="ok" if direct.get("success") else "fail",
                bt=base.get("selected_tile") or "",
                dt=direct.get("selected_tile") or "",
                bi=int(base.get("inliers", 0) or 0),
                br=float(base.get("rms_px", float("nan"))),
                di=int(direct.get("inliers", 0) or 0),
                dr=float(direct.get("rms_px", float("nan"))),
                cmp=compare,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_labels(raw: str | None) -> tuple[str, ...] | None:
    if not raw:
        return None
    labels = tuple(part.strip() for part in raw.split(",") if part.strip())
    return labels or None


def _progress(event: str, **payload: Any) -> None:
    data = {"event": event}
    data.update(payload)
    print(json.dumps(data, default=_json_default), flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate direct ASTAP Blind 4D indexes against product 4D NPZ at runtime.")
    parser.add_argument("--astap-root", type=Path, default=Path("/opt/astap"))
    parser.add_argument("--product-manifest", type=Path, default=ROOT / "config" / "zeblind_4d_experimental_manifest.json")
    parser.add_argument("--legacy-index-root", type=Path, default=Path("/home/tristan/zesolver_index"))
    parser.add_argument("--product-index-root", type=Path, default=ROOT / "indexes" / "astrometry_4d")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "reports" / "eq_ircut_cleanbench_20260518_230249" / "data")
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--reuse-build-dir", type=Path, default=None, help="Reuse a previous P1D-3B build-only directory after rechecking A/B scientific fingerprints.")
    parser.add_argument("--labels", default="", help="Comma-separated FITS time labels. Defaults to all M106 cases.")
    parser.add_argument("--mini", action="store_true", help="Use the canonical P2 mini corpus labels.")
    parser.add_argument("--policies", default="first_accept,best_within_budget")
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--report-md", type=Path, default=None)
    parser.add_argument("--build-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build_config = assert_p1d3b_build_config()
    work_dir = args.work_dir.expanduser().resolve() if args.work_dir is not None else Path(tempfile.mkdtemp(prefix="p1d3b_direct_runtime_"))
    if work_dir.exists() and any(work_dir.iterdir()):
        raise SystemExit(f"work directory must be empty: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)
    policies = tuple(part.strip() for part in str(args.policies).split(",") if part.strip())
    before = {
        "astap": snapshot_tree(args.astap_root),
        "legacy_index": snapshot_tree(args.legacy_index_root),
        "product_index_root": snapshot_tree(args.product_index_root, include_npz_sha=True),
        "product_manifest": snapshot_file(args.product_manifest),
    }
    if args.reuse_build_dir is not None:
        _progress("reuse_direct_indexes", build_dir=args.reuse_build_dir)
        direct_build = reuse_direct_indexes(args.reuse_build_dir, tile_order=P1D3B_PRODUCT_ORDER)
    else:
        _progress("build_direct_indexes", work_dir=work_dir)
        direct_build = build_direct_indexes_twice(args.astap_root, work_dir, tile_order=P1D3B_PRODUCT_ORDER)
    _progress("write_manifests")
    manifests = make_comparison_manifests(args.product_manifest, direct_build["first_paths"], work_dir)
    loaded_baseline = load_4d_index_manifest(manifests["baseline_manifest"])
    loaded_direct = load_4d_index_manifest(manifests["direct_manifest"])
    labels = _parse_labels(args.labels)
    if args.mini:
        labels = P1D3B_MINI_LABELS
    cases = discover_m106_cases(args.data_dir, labels=labels)
    if not cases and not args.build_only:
        raise SystemExit(f"no M106 FITS cases discovered under {args.data_dir}")
    missing_mini = sorted(set(P1D3B_MINI_LABELS) - {case["label"] for case in cases}) if args.mini else []
    if args.build_only:
        runtime_rows: list[dict[str, Any]] = []
    else:
        _progress("run_runtime", cases=len(cases), policies=policies)
        runtime_rows = compare_runtime(
            cases,
            baseline_paths=loaded_baseline.enabled_index_paths,
            direct_paths=loaded_direct.enabled_index_paths,
            legacy_index_root=args.legacy_index_root,
            work_dir=work_dir / "fits_copies",
            policies=policies,
            runtime_policy=RuntimePolicy(),
        )
    after = {
        "astap": snapshot_tree(args.astap_root),
        "legacy_index": snapshot_tree(args.legacy_index_root),
        "product_index_root": snapshot_tree(args.product_index_root, include_npz_sha=True),
        "product_manifest": snapshot_file(args.product_manifest),
    }
    runtime_summary = summarize_runtime(runtime_rows)
    blockers = [
        row for row in runtime_rows if row["classification"] in {"DIRECT_LOSS", "INVALID_DIRECT_SOLUTION"}
    ]
    false_positive_blockers = [row for row in runtime_rows if row["classification"] == "INVALID_DIRECT_SOLUTION"]
    integrity_unchanged = before == after
    gate_ready = bool(runtime_rows) and not blockers and not false_positive_blockers and integrity_unchanged
    report = {
        "status": "ok" if gate_ready else "not_ready",
        "gate": "READY_FOR_P1D4_LIBRARY_OWNED_BLIND4D_MANIFEST" if gate_ready else "NOT_READY_FOR_P1D4_LIBRARY_OWNED_BLIND4D_MANIFEST",
        "work_dir": str(work_dir),
        "build_config": build_config,
        "runtime_policy": asdict(RuntimePolicy()),
        "policies": list(policies),
        "cases": [{"label": case["label"], "path": str(case["path"]), "corpus": case["corpus"]} for case in cases],
        "missing_canonical_cases": {"mini": missing_mini},
        "direct_fingerprints": direct_build["fingerprints"],
        "manifests": {
            "baseline_manifest": str(manifests["baseline_manifest"]),
            "direct_manifest": str(manifests["direct_manifest"]),
            "baseline_index_paths": [str(path) for path in loaded_baseline.enabled_index_paths],
            "direct_index_paths": [str(path) for path in loaded_direct.enabled_index_paths],
        },
        "runtime_rows": runtime_rows,
        "runtime_summary": runtime_summary,
        "integrity": {"before": before, "after": after, "unchanged": integrity_unchanged},
        "warnings": [
            "FAST-style external integrity uses counts, sizes, mtimes and product NPZ SHA256; ASTAP shard cryptographic hashes are not recomputed."
        ],
        "limits": [
            "This tool does not replace product NPZ or strict product manifests.",
            "Negative controls are reported only when their configured FITS data are supplied to this tool.",
        ],
    }
    report_json = args.report_json or (work_dir / "p1d3b_direct_runtime_report.json")
    report_md = args.report_md or (work_dir / "p1d3b_direct_runtime_report.md")
    _write_json(report_json, report)
    write_markdown_report(report_md, report)
    _progress("done", gate=report["gate"], report_json=report_json, report_md=report_md)
    return 0 if gate_ready or args.build_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
