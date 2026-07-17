from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SETTINGS_SCHEMA_VERSION_V2 = 2


@dataclass(frozen=True, slots=True)
class ProfileSelection:
    near: str = "zenear-v1"
    blind: str = "zeblind4d-v1"
    pipeline: str = "pipeline-v1"
    historical_diagnostic: str | None = None


@dataclass(frozen=True, slots=True)
class ProductSettings:
    catalog_library_path: Path | None = None
    output_mode: str = "overwrite"
    overwrite_wcs: bool = True
    workers: int | str = "auto"
    gpu_mode: str = "auto"
    web_fallback: bool = True
    language: str = "auto"
    log_level: str = "INFO"
    input_formats: tuple[str, ...] = ()
    blind_enabled: bool = True
    blind_only: bool = False
    downsample: int = 1
    fov_deg: float = 1.5
    hint_ra_deg: float | None = None
    hint_dec_deg: float | None = None
    hint_radius_deg: float | None = None
    hint_focal_mm: float | None = None
    hint_pixel_um: float | None = None
    hint_resolution_arcsec: float | None = None
    hint_resolution_min_arcsec: float | None = None
    hint_resolution_max_arcsec: float | None = None
    astrometry_api_url: str = "https://nova.astrometry.net/api"
    astrometry_api_key: str | None = None
    astrometry_parallel_jobs: int = 2
    astrometry_timeout_s: int = 600
    astrometry_use_hints: bool = True
    profiles: ProfileSelection = ProfileSelection()

    def to_v2_payload(self) -> dict[str, object]:
        return {
            "settings_schema_version": SETTINGS_SCHEMA_VERSION_V2,
            "product": {
                "catalog_library_path": str(self.catalog_library_path) if self.catalog_library_path else None,
                "output_mode": self.output_mode,
                "overwrite_wcs": self.overwrite_wcs,
                "workers": self.workers,
                "gpu_mode": self.gpu_mode,
                "web_fallback": self.web_fallback,
                "language": self.language,
                "log_level": self.log_level,
                "input_formats": list(self.input_formats),
                "blind_enabled": self.blind_enabled,
                "blind_only": self.blind_only,
                "downsample": self.downsample,
                "fov_deg": self.fov_deg,
            },
            "profiles": {
                "near": self.profiles.near,
                "blind": self.profiles.blind,
                "pipeline": self.profiles.pipeline,
            },
        }
