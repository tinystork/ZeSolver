from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from zesolver.engine_selection import EngineMode, EngineSelection
from zesolver.settings import ProductSettings, RuntimeOptions


@dataclass(frozen=True, slots=True)
class GuiSettingsState:
    input_dir: Path | None = None
    output_dir: Path | None = None
    catalog_library_path: Path | None = None
    backend: str = "local"
    engine_mode: EngineMode = EngineMode.AUTO
    overwrite_wcs: bool = True
    workers: int = 1
    preserve_order: bool = True
    use_blind: bool = True
    use_web_fallback: bool = False
    formats: tuple[str, ...] = ()
    max_files: int | None = None
    worker_strategy: str = "threads"
    requires_raster_sidecar: bool = False
    requires_adaptive_hints: bool = False
    blind4d_all_sky: bool = False
    log_level: str = "INFO"
    language: str = "auto"
    fov_deg: float = 1.5
    downsample: int = 1
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
    legacy_config: object | None = None
    catalog_resources: object | None = None


@dataclass(frozen=True, slots=True)
class GuiSolveRequest:
    input_paths: tuple[Path, ...]
    engine_mode: EngineMode
    backend: str
    overwrite_wcs: bool
    workers: int
    preserve_order: bool
    use_blind: bool
    use_web_fallback: bool
    product_settings: ProductSettings
    runtime_options: RuntimeOptions
    worker_strategy: str = "threads"
    requires_raster_sidecar: bool = False
    requires_adaptive_hints: bool = False
    blind4d_all_sky: bool = False
    legacy_config: object | None = None
    catalog_resources: object | None = None
    metadata_overrides: Mapping[str, object] = field(default_factory=dict)

    def for_phase(self, phase: str) -> "GuiSolveRequest":
        if phase == "blind":
            product = replace(self.product_settings, blind_enabled=True, blind_only=True)
            return replace(self, product_settings=product)
        if phase == "near":
            product = replace(self.product_settings, blind_only=False)
            return replace(self, product_settings=product)
        return self


@dataclass(frozen=True, slots=True)
class GuiFileResult:
    path: Path
    status: str
    message: str
    backend: str | None = None
    inliers: int | None = None
    rms_px: float | None = None
    pixel_scale_arcsec: float | None = None
    wcs_written: bool = False
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    run_info: tuple[tuple[str, dict[str, Any]], ...] = ()
    selected_engine: EngineMode | None = None

    @property
    def legacy_status(self) -> str:
        value = str(self.status or "").strip().upper()
        if value == "SOLVED":
            return "solved"
        if value == "CANCELLED":
            return "cancelled"
        if value in {"INVALID_INPUT", "SKIPPED"}:
            return "skipped"
        if value in {"UNSOLVED", "FAILED", "CATALOG_UNAVAILABLE", "REJECTED_FALSE_SOLUTION"}:
            return "failed"
        return str(self.status or "").strip().lower() or "failed"


@dataclass(frozen=True, slots=True)
class GuiRunSummary:
    selected_engine: EngineMode
    selection_reason: str
    results: tuple[GuiFileResult, ...]
    cancelled: bool
    duration_s: float
    warnings: tuple[str, ...] = ()
    selection: EngineSelection | None = None
