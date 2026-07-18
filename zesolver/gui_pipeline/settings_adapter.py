from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from zesolver.engine_selection import EngineMode, EngineSelectionRequest, RASTER_EXTENSIONS
from zesolver.settings import ProductSettings, RuntimeOptions

from .requests import GuiSettingsState, GuiSolveRequest


def engine_mode_from_environment(default: EngineMode = EngineMode.AUTO) -> EngineMode:
    raw = os.environ.get("ZESOLVER_GUI_ENGINE")
    if not raw:
        return default
    return EngineMode(str(raw).strip().lower())


def build_product_settings(state: GuiSettingsState) -> ProductSettings:
    return ProductSettings(
        catalog_library_path=state.catalog_library_path,
        output_mode="overwrite" if state.overwrite_wcs else "preserve",
        overwrite_wcs=state.overwrite_wcs,
        workers=state.workers,
        web_fallback=state.use_web_fallback,
        language=state.language,
        log_level=state.log_level,
        input_formats=tuple(state.formats),
        blind_enabled=state.use_blind,
        blind_only=False,
        downsample=state.downsample,
        fov_deg=state.fov_deg,
        hint_ra_deg=state.hint_ra_deg,
        hint_dec_deg=state.hint_dec_deg,
        hint_radius_deg=state.hint_radius_deg,
        hint_focal_mm=state.hint_focal_mm,
        hint_pixel_um=state.hint_pixel_um,
        hint_resolution_arcsec=state.hint_resolution_arcsec,
        hint_resolution_min_arcsec=state.hint_resolution_min_arcsec,
        hint_resolution_max_arcsec=state.hint_resolution_max_arcsec,
        astrometry_api_url=state.astrometry_api_url,
        astrometry_api_key=state.astrometry_api_key,
        astrometry_parallel_jobs=state.astrometry_parallel_jobs,
        astrometry_timeout_s=state.astrometry_timeout_s,
        astrometry_use_hints=state.astrometry_use_hints,
    )


def build_runtime_options(state: GuiSettingsState, *, cancel_token: object | None = None) -> RuntimeOptions:
    return RuntimeOptions(
        input_dir=state.input_dir,
        output_dir=state.output_dir,
        worker_count_resolved=state.workers,
        max_files=state.max_files,
        cancel_token=cancel_token,
    )


def build_gui_solve_request(
    paths: Sequence[Path],
    state: GuiSettingsState,
    *,
    cancel_token: object | None = None,
) -> GuiSolveRequest:
    product = build_product_settings(state)
    runtime = build_runtime_options(state, cancel_token=cancel_token)
    return GuiSolveRequest(
        input_paths=tuple(Path(path) for path in paths),
        engine_mode=state.engine_mode,
        backend=state.backend,
        overwrite_wcs=state.overwrite_wcs,
        workers=state.workers,
        preserve_order=state.preserve_order,
        use_blind=state.use_blind,
        use_web_fallback=state.use_web_fallback,
        product_settings=product,
        runtime_options=runtime,
        worker_strategy=state.worker_strategy,
        requires_raster_sidecar=state.requires_raster_sidecar,
        requires_adaptive_hints=state.requires_adaptive_hints,
        blind4d_all_sky=state.blind4d_all_sky,
        legacy_config=state.legacy_config,
        catalog_resources=state.catalog_resources,
    )


def build_gui_solve_request_from_legacy_config(
    paths: Sequence[Path],
    config: object,
    *,
    engine_mode: EngineMode | str | None = None,
    backend: str = "local",
    catalog_resources: object | None = None,
    cancel_token: object | None = None,
    preserve_order: bool = True,
) -> GuiSolveRequest:
    mode = engine_mode_from_environment() if engine_mode is None else EngineMode(str(engine_mode).strip().lower())
    state = GuiSettingsState(
        input_dir=getattr(config, "input_dir", None),
        catalog_library_path=getattr(config, "catalog_library_path", None),
        backend=backend,
        engine_mode=mode,
        overwrite_wcs=bool(getattr(config, "overwrite", True)),
        workers=max(1, int(getattr(config, "workers", 1) or 1)),
        preserve_order=preserve_order,
        use_blind=bool(getattr(config, "blind_enabled", True)),
        use_web_fallback=bool(getattr(config, "astrometry_fallback_after_blind", False))
        and bool(getattr(config, "astrometry_api_key", None)),
        formats=tuple(getattr(config, "formats", ()) or ()),
        max_files=getattr(config, "max_files", None),
        worker_strategy=_worker_strategy_from_environment(paths),
        requires_raster_sidecar=_has_raster(paths),
        requires_adaptive_hints=False,
        blind4d_all_sky=False,
        log_level=str(getattr(config, "log_level", "INFO") or "INFO"),
        fov_deg=float(getattr(config, "fov_deg", 1.5) or 1.5),
        downsample=int(getattr(config, "downsample", 1) or 1),
        hint_ra_deg=getattr(config, "hint_ra_deg", None),
        hint_dec_deg=getattr(config, "hint_dec_deg", None),
        hint_radius_deg=getattr(config, "hint_radius_deg", None),
        hint_focal_mm=getattr(config, "hint_focal_mm", None),
        hint_pixel_um=getattr(config, "hint_pixel_um", None),
        hint_resolution_arcsec=getattr(config, "hint_resolution_arcsec", None),
        hint_resolution_min_arcsec=getattr(config, "hint_resolution_min_arcsec", None),
        hint_resolution_max_arcsec=getattr(config, "hint_resolution_max_arcsec", None),
        astrometry_api_url=str(getattr(config, "astrometry_api_url", "https://nova.astrometry.net/api") or "https://nova.astrometry.net/api"),
        astrometry_api_key=getattr(config, "astrometry_api_key", None),
        astrometry_parallel_jobs=int(getattr(config, "astrometry_parallel_jobs", 2) or 2),
        astrometry_timeout_s=int(getattr(config, "astrometry_timeout_s", 600) or 600),
        astrometry_use_hints=bool(getattr(config, "astrometry_use_hints", True)),
        legacy_config=config,
        catalog_resources=catalog_resources,
    )
    request = build_gui_solve_request(paths, state, cancel_token=cancel_token)
    product = replace(
        request.product_settings,
        web_fallback=state.use_web_fallback,
    )
    return replace(request, product_settings=product)


def build_engine_selection_request(request: GuiSolveRequest) -> EngineSelectionRequest:
    representative = _representative_path(request.input_paths)
    unknown: list[str] = []
    suffixes = {Path(path).suffix.lower() for path in request.input_paths}
    if len(suffixes) > 1 and not any(suffix in RASTER_EXTENSIONS for suffix in suffixes):
        unknown.append("mixed_file_types")
    return EngineSelectionRequest(
        input_path=representative,
        requested_mode=request.engine_mode,
        is_batch=len(request.input_paths) > 1,
        workers=request.workers,
        worker_strategy=request.worker_strategy,
        backend=request.backend,
        fallback_web=request.use_web_fallback,
        requires_raster_sidecar=request.requires_raster_sidecar,
        requires_adaptive_hints=request.requires_adaptive_hints,
        unknown_capabilities=tuple(unknown),
        blind4d_all_sky=request.blind4d_all_sky,
    )


def _representative_path(paths: tuple[Path, ...]) -> Path | None:
    if not paths:
        return None
    for path in paths:
        if Path(path).suffix.lower() in RASTER_EXTENSIONS:
            return Path(path)
    return Path(paths[0])


def _has_raster(paths: Sequence[Path]) -> bool:
    return any(Path(path).suffix.lower() in RASTER_EXTENSIONS for path in paths)


def _worker_strategy_from_environment(paths: Sequence[Path]) -> str:
    raw = os.environ.get("ZE_NEAR_PARALLEL_MODE", "thread")
    mode = str(raw or "thread").strip().lower()
    if mode in {"process", "hybrid"}:
        return mode
    return "threads"
