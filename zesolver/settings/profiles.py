from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from zeblindsolver.quad_index_4d import ASTROMETRY_AB_CODE_4D_SCHEMA


@dataclass(frozen=True, slots=True)
class NearSolverProfile:
    profile_id: str
    profile_version: int
    values: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class BlindSolverProfile:
    profile_id: str
    profile_version: int
    values: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class PipelineProfile:
    profile_id: str
    profile_version: int
    values: Mapping[str, object]


ZENEAR_V1 = NearSolverProfile(
    profile_id="zenear-v1",
    profile_version=1,
    values={
        "near_max_tile_candidates": 48,
        "near_tile_cache_size": 128,
        "near_detect_k_sigma": 4.5,
        "near_detect_min_area": 8,
        "near_detect_max_labels": 1200,
        "near_detect_gpu_slots": 1,
        "near_warm_start": True,
        "near_quality_inliers": 60,
        "near_quality_rms": 1.0,
        "near_pixel_tolerance": 3.0,
        "near_ransac_trials": 1200,
        "near_max_img_stars": 800,
        "near_max_cat_stars": 2000,
        "near_try_parity_flip": True,
        "near_search_margin": 1.2,
        "near_astap_iso_strict": True,
        "near_defer_blind_fallback": False,
        "near_allow_second_rescue": False,
        "search_radius_scale": 1.8,
        "search_radius_attempts": 3,
        "max_search_radius_deg": None,
    },
)

ZEBLIND4D_V1 = BlindSolverProfile(
    profile_id="zeblind4d-v1",
    profile_version=1,
    values={
        "blind_backend_profile": "zeblind_4d_experimental",
        "blind_max_stars": 500,
        "blind_max_quads": 8000,
        "blind_max_candidates": 10,
        "blind_pixel_tolerance": 2.5,
        "blind_quality_inliers": 40,
        "blind_quality_rms": 1.2,
        "blind_fast_mode": True,
        "blind_index_scale_overlap_prefilter_enabled": False,
        "blind_index_scale_overlap_proxy_lo_frac": 0.05,
        "blind_index_scale_overlap_proxy_hi_frac": 0.95,
        "quad_hash_schema": ASTROMETRY_AB_CODE_4D_SCHEMA,
        "quad_sources": 120,
        "verification_sources": "full",
        "validation_catalog_policy": "union_candidate_tiles",
        "accept_policy": "best_within_budget",
        "match_radius_px": 3.0,
        "max_quads": 2500,
        "max_hypotheses": 2000,
        "max_accepts": 64,
        "max_wall_s": 45.0,
        "source_policy": "diagnostic_unfiltered",
        "image_strategy": "log_spaced",
        "code_tol": 0.015,
        "max_hits": 2000,
        "max_hits_per_image_quad": 8,
    },
)

PIPELINE_V1 = PipelineProfile(
    profile_id="pipeline-v1",
    profile_version=1,
    values={
        "order": ("near", "blind4d", "astrometry_optional"),
        "near_defer_blind_fallback": False,
        "blind_skip_if_valid": True,
        "astrometry_fallback_after_blind": True,
        "stop_on_cancel": True,
        "legacy_historical_profile": "diagnostic_only",
    },
)

_NEAR = {ZENEAR_V1.profile_id: ZENEAR_V1}
_BLIND = {ZEBLIND4D_V1.profile_id: ZEBLIND4D_V1}
_PIPELINE = {PIPELINE_V1.profile_id: PIPELINE_V1}


def get_near_profile(profile_id: str) -> NearSolverProfile:
    try:
        return _NEAR[profile_id]
    except KeyError as exc:
        raise KeyError(f"unknown near profile: {profile_id}") from exc


def get_blind_profile(profile_id: str) -> BlindSolverProfile:
    try:
        return _BLIND[profile_id]
    except KeyError as exc:
        raise KeyError(f"unknown blind profile: {profile_id}") from exc


def get_pipeline_profile(profile_id: str) -> PipelineProfile:
    try:
        return _PIPELINE[profile_id]
    except KeyError as exc:
        raise KeyError(f"unknown pipeline profile: {profile_id}") from exc
