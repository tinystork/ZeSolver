from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .product import ProductSettings
from .profiles import BlindSolverProfile, NearSolverProfile, PipelineProfile, get_blind_profile, get_near_profile, get_pipeline_profile
from .runtime import DeveloperOverrides, RuntimeOptions


@dataclass(frozen=True, slots=True)
class ResolvedSolverConfiguration:
    product_settings: ProductSettings
    runtime_options: RuntimeOptions
    near_profile: NearSolverProfile
    blind_profile: BlindSolverProfile
    pipeline_profile: PipelineProfile
    developer_overrides_active: bool
    developer_override_values: Mapping[str, object]
    legacy_solve_config_values: Mapping[str, object]
    report_metadata: Mapping[str, object]


def build_solver_configuration(
    *,
    product_settings: ProductSettings,
    runtime_options: RuntimeOptions,
    near_profile: str = "zenear-v1",
    blind_profile: str = "zeblind4d-v1",
    pipeline_profile: str = "pipeline-v1",
    developer_overrides: DeveloperOverrides | None = None,
) -> ResolvedSolverConfiguration:
    near = get_near_profile(near_profile)
    blind = get_blind_profile(blind_profile)
    pipeline = get_pipeline_profile(pipeline_profile)
    overrides = developer_overrides or DeveloperOverrides()
    values: dict[str, object] = {}
    values.update(near.values)
    values.update(
        {
            "catalog_library_path": product_settings.catalog_library_path,
            "fov_deg": product_settings.fov_deg,
            "downsample": product_settings.downsample,
            "overwrite": product_settings.overwrite_wcs,
            "workers": runtime_options.worker_count_resolved if runtime_options.worker_count_resolved is not None else product_settings.workers,
            "formats": product_settings.input_formats,
            "blind_enabled": product_settings.blind_enabled,
            "blind_only": product_settings.blind_only,
            "near_catalog_mode": product_settings.near_catalog_mode,
            "blind_backend_profile": blind.values["blind_backend_profile"],
            "astrometry_api_url": product_settings.astrometry_api_url,
            "astrometry_api_key": product_settings.astrometry_api_key,
            "astrometry_timeout_s": product_settings.astrometry_timeout_s,
            "astrometry_parallel_jobs": product_settings.astrometry_parallel_jobs,
            "astrometry_use_hints": product_settings.astrometry_use_hints,
            "astrometry_fallback_after_blind": product_settings.web_fallback,
            "log_level": product_settings.log_level,
            "hint_ra_deg": product_settings.hint_ra_deg,
            "hint_dec_deg": product_settings.hint_dec_deg,
            "hint_radius_deg": product_settings.hint_radius_deg,
            "hint_focal_mm": product_settings.hint_focal_mm,
            "hint_pixel_um": product_settings.hint_pixel_um,
            "hint_resolution_arcsec": product_settings.hint_resolution_arcsec,
            "hint_resolution_min_arcsec": product_settings.hint_resolution_min_arcsec,
            "hint_resolution_max_arcsec": product_settings.hint_resolution_max_arcsec,
        }
    )
    for key, value in blind.values.items():
        if key.startswith("blind_") or key in {"quad_hash_schema", "quad_sources", "max_quads", "max_hypotheses", "max_accepts", "max_wall_s", "match_radius_px", "code_tol", "max_hits", "max_hits_per_image_quad"}:
            values[key] = value
    values.update(pipeline.values)
    if overrides.active:
        values.update(overrides.values)
    return ResolvedSolverConfiguration(
        product_settings=product_settings,
        runtime_options=runtime_options,
        near_profile=near,
        blind_profile=blind,
        pipeline_profile=pipeline,
        developer_overrides_active=overrides.active,
        developer_override_values=dict(overrides.values) if overrides.active else {},
        legacy_solve_config_values=values,
        report_metadata={
            "near_profile": near.profile_id,
            "blind_profile": blind.profile_id,
            "pipeline_profile": pipeline.profile_id,
            "developer_overrides_active": overrides.active,
        },
    )
