from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zeblindsolver.index_manifest_4d import Loaded4DManifest
from zeblindsolver.profiles import ZEBLIND_4D_EXPERIMENTAL_PROFILE

from zesolver.catalog_resources import SolverCatalogResources


@dataclass(frozen=True, slots=True)
class BlindConfigInputs:
    """Legacy-compatible inputs required to build a BlindSolveConfig."""

    blind_backend_profile: str
    blind_4d_manifest_path: Path | None
    blind_4d_loaded_manifest: Loaded4DManifest | None
    blind_index_path: Path | None
    blind_max_candidates: int
    blind_max_stars: int
    blind_max_quads: int
    blind_pixel_tolerance: float
    blind_quality_inliers: int
    blind_quality_rms: float
    blind_fast_mode: bool
    blind_index_scale_overlap_prefilter_enabled: bool
    blind_index_scale_overlap_proxy_lo_frac: float
    blind_index_scale_overlap_proxy_hi_frac: float
    log_level: str
    downsample: int
    hint_ra_deg: float | None
    hint_dec_deg: float | None
    hint_radius_deg: float | None
    hint_focal_mm: float | None
    hint_pixel_um: float | None
    hint_resolution_arcsec: float | None
    hint_resolution_min_arcsec: float | None
    hint_resolution_max_arcsec: float | None
    dev_bucket_limit_override: int
    dev_vote_percentile: int
    dev_collect_matches_vectorized_experimental: bool
    dev_bucket_cap_S: int
    dev_bucket_cap_M: int
    dev_bucket_cap_L: int
    dev_detect_k_sigma: float
    dev_detect_min_area: int
    dev_verify_logodds_enabled: bool
    dev_verify_logodds_bail: float
    dev_verify_logodds_stoplooking: float
    dev_verify_logodds_min_validations: int
    dev_hard_max_candidates_tried: int
    dev_hard_max_validations: int
    dev_depth_ladder_enabled: bool
    dev_depth_ladder_caps: tuple[int, ...]
    catalog_library_path: Path | None


def build_blind_config_inputs(
    request: Any,
    *,
    resources: SolverCatalogResources,
    configuration: Any,
    loaded_manifest: Loaded4DManifest,
) -> BlindConfigInputs:
    values = dict(configuration.legacy_solve_config_values)
    return BlindConfigInputs(
        blind_backend_profile=ZEBLIND_4D_EXPERIMENTAL_PROFILE,
        blind_4d_manifest_path=loaded_manifest.manifest_path,
        blind_4d_loaded_manifest=loaded_manifest,
        blind_index_path=loaded_manifest.manifest_path.parent,
        blind_max_candidates=int(values.get("blind_max_candidates", 10) or 10),
        blind_max_stars=int(values.get("blind_max_stars", 500) or 500),
        blind_max_quads=int(values.get("blind_max_quads", 8000) or 8000),
        blind_pixel_tolerance=float(values.get("blind_pixel_tolerance", 2.5) or 2.5),
        blind_quality_inliers=int(values.get("blind_quality_inliers", 40) or 40),
        blind_quality_rms=float(values.get("blind_quality_rms", 1.2) or 1.2),
        blind_fast_mode=bool(values.get("blind_fast_mode", True)),
        blind_index_scale_overlap_prefilter_enabled=bool(values.get("blind_index_scale_overlap_prefilter_enabled", False)),
        blind_index_scale_overlap_proxy_lo_frac=float(values.get("blind_index_scale_overlap_proxy_lo_frac", 0.05) or 0.05),
        blind_index_scale_overlap_proxy_hi_frac=float(values.get("blind_index_scale_overlap_proxy_hi_frac", 0.95) or 0.95),
        log_level=str(values.get("log_level", "INFO") or "INFO"),
        downsample=int(values.get("downsample", 1) or 1),
        hint_ra_deg=request.ra_hint_deg,
        hint_dec_deg=request.dec_hint_deg,
        hint_radius_deg=request.radius_hint_deg,
        hint_focal_mm=request.focal_length_mm,
        hint_pixel_um=request.pixel_size_um,
        hint_resolution_arcsec=request.pixel_scale_arcsec,
        hint_resolution_min_arcsec=request.pixel_scale_min_arcsec,
        hint_resolution_max_arcsec=request.pixel_scale_max_arcsec,
        dev_bucket_limit_override=int(values.get("dev_bucket_limit_override", 0) or 0),
        dev_vote_percentile=int(values.get("dev_vote_percentile", 40) or 40),
        dev_collect_matches_vectorized_experimental=bool(values.get("dev_collect_matches_vectorized_experimental", False)),
        dev_bucket_cap_S=int(values.get("dev_bucket_cap_S", 0) or 0),
        dev_bucket_cap_M=int(values.get("dev_bucket_cap_M", 0) or 0),
        dev_bucket_cap_L=int(values.get("dev_bucket_cap_L", 0) or 0),
        dev_detect_k_sigma=float(values.get("dev_detect_k_sigma", 3.0) or 3.0),
        dev_detect_min_area=int(values.get("dev_detect_min_area", 5) or 5),
        dev_verify_logodds_enabled=bool(values.get("dev_verify_logodds_enabled", False)),
        dev_verify_logodds_bail=float(values.get("dev_verify_logodds_bail", -24.0) or -24.0),
        dev_verify_logodds_stoplooking=float(values.get("dev_verify_logodds_stoplooking", 24.0) or 24.0),
        dev_verify_logodds_min_validations=int(values.get("dev_verify_logodds_min_validations", 8) or 8),
        dev_hard_max_candidates_tried=int(values.get("dev_hard_max_candidates_tried", 0) or 0),
        dev_hard_max_validations=int(values.get("dev_hard_max_validations", 0) or 0),
        dev_depth_ladder_enabled=bool(values.get("dev_depth_ladder_enabled", False)),
        dev_depth_ladder_caps=_depth_ladder_caps(values.get("dev_depth_ladder_caps", (80, 160, 500))),
        catalog_library_path=resources.library_path,
    )


def _depth_ladder_caps(value: object) -> tuple[int, ...]:
    if isinstance(value, str):
        parts: tuple[object, ...] = tuple(value.replace(";", ",").split(","))
    elif isinstance(value, (list, tuple)):
        parts = tuple(value)
    else:
        parts = (80, 160, 500)
    caps_list: list[int] = []
    for item in parts:
        try:
            cap = int(item)
        except (TypeError, ValueError):
            continue
        if cap > 0:
            caps_list.append(cap)
    caps = tuple(caps_list)
    return caps or (80, 160, 500)
