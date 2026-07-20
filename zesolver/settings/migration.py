from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .product import ProductSettings, ProfileSelection


@dataclass(frozen=True, slots=True)
class SettingsMigrationResult:
    product: ProductSettings
    migrated: tuple[str, ...]
    ignored: tuple[str, ...]
    deprecated: tuple[str, ...]
    warnings: tuple[str, ...]
    user_action_required: bool = False
    historical_diagnostic_profile: str | None = None


def migrate_persistent_settings_v2(settings) -> SettingsMigrationResult:
    migrated: list[str] = []
    ignored: list[str] = []
    deprecated: list[str] = []
    warnings: list[str] = []

    profile = str(getattr(settings, "blind_backend_profile", "zeblind_4d_experimental") or "zeblind_4d_experimental").strip().lower()
    historical = "historical" if profile == "historical" else None
    if historical:
        deprecated.append("blind_backend_profile=historical")
        warnings.append("historical_profile_preserved_as_diagnostic")
    elif profile not in {"zeblind_4d_experimental", "historical"}:
        warnings.append("invalid_blind_backend_profile_reset_to_zeblind4d-v1")

    workers_raw = int(getattr(settings, "solver_workers", 0) or 0)
    workers: int | str = "auto" if workers_raw <= 0 else workers_raw
    catalog_path = getattr(settings, "catalog_library_path", None)
    product = ProductSettings(
        catalog_library_path=Path(catalog_path).expanduser() if catalog_path else None,
        output_mode="overwrite" if bool(getattr(settings, "solver_overwrite", True)) else "preserve",
        overwrite_wcs=bool(getattr(settings, "solver_overwrite", True)),
        workers=workers,
        gpu_mode=str(getattr(settings, "near_detect_backend", "auto") or "auto"),
        web_fallback=bool(getattr(settings, "astrometry_fallback_local", True)),
        language="auto",
        log_level=str(getattr(settings, "log_level", "INFO") or "INFO").upper(),
        input_formats=tuple(
            item.strip().lower()
            for item in str(getattr(settings, "solver_formats", "") or "").split(",")
            if item.strip()
        ),
        blind_enabled=bool(getattr(settings, "solver_blind_enabled", True)),
        blind_only=False,
        near_catalog_mode=str(getattr(settings, "near_catalog_mode", "auto") or "auto").strip().lower().replace("_", "-"),
        blind4d_catalog_mode=str(getattr(settings, "blind4d_catalog_mode", "auto") or "auto").strip().lower().replace("_", "-"),
        downsample=max(1, int(getattr(settings, "solver_downsample", 1) or 1)),
        fov_deg=float(getattr(settings, "solver_fov_deg", 1.5) or 1.5),
        hint_ra_deg=getattr(settings, "solver_hint_ra_deg", None),
        hint_dec_deg=getattr(settings, "solver_hint_dec_deg", None),
        hint_radius_deg=getattr(settings, "solver_hint_radius_deg", None),
        hint_focal_mm=getattr(settings, "solver_hint_focal_mm", None),
        hint_pixel_um=getattr(settings, "solver_hint_pixel_um", None),
        hint_resolution_arcsec=getattr(settings, "solver_hint_resolution_arcsec", None),
        hint_resolution_min_arcsec=getattr(settings, "solver_hint_resolution_min_arcsec", None),
        hint_resolution_max_arcsec=getattr(settings, "solver_hint_resolution_max_arcsec", None),
        astrometry_api_url=str(getattr(settings, "astrometry_api_url", "https://nova.astrometry.net/api") or "https://nova.astrometry.net/api"),
        astrometry_api_key=(getattr(settings, "astrometry_api_key", None) or None),
        astrometry_parallel_jobs=max(1, int(getattr(settings, "astrometry_parallel_jobs", 2) or 2)),
        astrometry_timeout_s=max(30, int(getattr(settings, "astrometry_timeout_s", 600) or 600)),
        astrometry_use_hints=bool(getattr(settings, "astrometry_use_hints", True)),
        profiles=ProfileSelection(historical_diagnostic=historical),
    )
    migrated.extend(
        (
            "catalog_library_path",
            "solver_overwrite",
            "solver_workers",
            "near_detect_backend",
            "solver_blind_enabled",
            "near_catalog_mode",
            "blind4d_catalog_mode",
            "solver_downsample",
            "solver_fov_deg",
            "astrometry_*",
        )
    )
    for name in (
        "db_root",
        "index_root",
        "blind_4d_manifest_path",
        "solver_family",
        "dev_family_selection",
    ):
        if getattr(settings, name, None):
            deprecated.append(name)
    for name in (
        "blind_max_stars",
        "blind_max_quads",
        "blind_max_candidates",
        "near_quality_inliers",
        "near_quality_rms",
        "dev_bucket_limit_override",
        "benchmark_inputs",
    ):
        if getattr(settings, name, None) is not None:
            ignored.append(name)
    return SettingsMigrationResult(
        product=product,
        migrated=tuple(dict.fromkeys(migrated)),
        ignored=tuple(dict.fromkeys(ignored)),
        deprecated=tuple(dict.fromkeys(deprecated)),
        warnings=tuple(dict.fromkeys(warnings)),
        user_action_required=False,
        historical_diagnostic_profile=historical,
    )
