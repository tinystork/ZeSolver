from __future__ import annotations

from typing import Any

from zeblindsolver.index_manifest_4d import IndexManifestError, Loaded4DManifest, load_4d_index_manifest
from zeblindsolver.profiles import HISTORICAL_PROFILE, ZEBLIND_4D_EXPERIMENTAL_PROFILE, get_solver_profile
from zeblindsolver.zeblindsolver import SolveConfig as BlindSolveConfig


def ensure_loaded_4d_manifest(config: Any) -> Loaded4DManifest:
    if config.blind_backend_profile != ZEBLIND_4D_EXPERIMENTAL_PROFILE:
        raise IndexManifestError("4D manifest requested while blind profile is not zeblind_4d_experimental")
    if config.blind_4d_loaded_manifest is not None:
        return config.blind_4d_loaded_manifest
    if config.blind_4d_manifest_path is None:
        raise IndexManifestError("blind_4d_manifest_required")
    return load_4d_index_manifest(config.blind_4d_manifest_path)


def build_blind_solve_config(
    app_config: Any,
    *,
    ra_hint: float | None = None,
    dec_hint: float | None = None,
    log_level: str | None = None,
    loaded_manifest: Loaded4DManifest | None = None,
) -> BlindSolveConfig:
    profile_name = str(getattr(app_config, "blind_backend_profile", ZEBLIND_4D_EXPERIMENTAL_PROFILE) or ZEBLIND_4D_EXPERIMENTAL_PROFILE).strip().lower()
    profile = get_solver_profile(profile_name)
    final_ra = getattr(app_config, "hint_ra_deg", None)
    final_dec = getattr(app_config, "hint_dec_deg", None)
    if final_ra is None:
        final_ra = ra_hint
    if final_dec is None:
        final_dec = dec_hint

    base = BlindSolveConfig(
        max_candidates=int(getattr(app_config, "blind_max_candidates", 10) or 10),
        max_stars=int(getattr(app_config, "blind_max_stars", 500) or 500),
        max_quads=int(getattr(app_config, "blind_max_quads", 8000) or 8000),
        detect_k_sigma=float(getattr(app_config, "dev_detect_k_sigma", 3.0) or 3.0),
        detect_min_area=int(getattr(app_config, "dev_detect_min_area", 5) or 5),
        bucket_cap_S=int(getattr(app_config, "dev_bucket_cap_S", 0) or 0),
        bucket_cap_M=int(getattr(app_config, "dev_bucket_cap_M", 0) or 0),
        bucket_cap_L=int(getattr(app_config, "dev_bucket_cap_L", 0) or 0),
        sip_order=2,
        quality_rms=float(getattr(app_config, "blind_quality_rms", 1.2) or 1.2),
        quality_inliers=int(getattr(app_config, "blind_quality_inliers", 40) or 40),
        pixel_tolerance=float(getattr(app_config, "blind_pixel_tolerance", 2.5) or 2.5),
        fast_mode=bool(getattr(app_config, "blind_fast_mode", True)),
        log_level=str(log_level or getattr(app_config, "log_level", "INFO") or "INFO").upper(),
        bucket_limit_override=int(getattr(app_config, "dev_bucket_limit_override", 0) or 0),
        vote_percentile=int(getattr(app_config, "dev_vote_percentile", 40) or 40),
        collect_matches_vectorized_experimental=bool(getattr(app_config, "dev_collect_matches_vectorized_experimental", False)),
        ra_hint_deg=final_ra,
        dec_hint_deg=final_dec,
        radius_hint_deg=getattr(app_config, "hint_radius_deg", getattr(app_config, "solver_hint_radius_deg", None)),
        focal_length_mm=getattr(app_config, "hint_focal_mm", getattr(app_config, "solver_hint_focal_mm", None)),
        pixel_size_um=getattr(app_config, "hint_pixel_um", getattr(app_config, "solver_hint_pixel_um", None)),
        pixel_scale_arcsec=getattr(app_config, "hint_resolution_arcsec", getattr(app_config, "solver_hint_resolution_arcsec", None)),
        pixel_scale_min_arcsec=getattr(app_config, "hint_resolution_min_arcsec", getattr(app_config, "solver_hint_resolution_min_arcsec", None)),
        pixel_scale_max_arcsec=getattr(app_config, "hint_resolution_max_arcsec", getattr(app_config, "solver_hint_resolution_max_arcsec", None)),
        downsample=max(1, int(getattr(app_config, "downsample", getattr(app_config, "solver_downsample", 1)) or 1)),
        verify_logodds_enabled=bool(getattr(app_config, "dev_verify_logodds_enabled", False)),
        verify_logodds_bail=float(getattr(app_config, "dev_verify_logodds_bail", -24.0) or -24.0),
        verify_logodds_stoplooking=float(getattr(app_config, "dev_verify_logodds_stoplooking", 24.0) or 24.0),
        verify_logodds_min_validations=int(getattr(app_config, "dev_verify_logodds_min_validations", 8) or 8),
        hard_max_candidates_tried=int(getattr(app_config, "dev_hard_max_candidates_tried", 0) or 0),
        hard_max_validations=int(getattr(app_config, "dev_hard_max_validations", 0) or 0),
        depth_ladder_enabled=bool(getattr(app_config, "dev_depth_ladder_enabled", False)),
        depth_ladder_caps=tuple(
            int(v)
            for v in getattr(app_config, "dev_depth_ladder_caps", (80, 160, 500))
            if isinstance(v, (int, float)) and int(v) > 0
        ) or (80, 160, 500),
        blind_index_scale_overlap_prefilter_enabled=bool(getattr(app_config, "blind_index_scale_overlap_prefilter_enabled", False)),
        blind_index_scale_overlap_proxy_lo_frac=float(getattr(app_config, "blind_index_scale_overlap_proxy_lo_frac", 0.05) or 0.05),
        blind_index_scale_overlap_proxy_hi_frac=float(getattr(app_config, "blind_index_scale_overlap_proxy_hi_frac", 0.95) or 0.95),
    )
    if profile.name == HISTORICAL_PROFILE:
        return base
    manifest = loaded_manifest
    if manifest is None:
        manifest = getattr(app_config, "blind_4d_loaded_manifest", None)
    if manifest is None:
        manifest_path = getattr(app_config, "blind_4d_manifest_path", None)
        if manifest_path is None:
            raise IndexManifestError("blind_4d_manifest_required")
        manifest = load_4d_index_manifest(manifest_path)
    return profile.apply_to_config(base, index_paths=manifest.enabled_index_paths)
