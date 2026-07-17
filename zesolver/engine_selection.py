from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class EngineMode(str, Enum):
    AUTO = "auto"
    PIPELINE = "pipeline"
    LEGACY = "legacy"


@dataclass(frozen=True, slots=True)
class EngineSelectionRequest:
    input_path: Path | str | None = None
    requested_mode: EngineMode | str = EngineMode.AUTO
    is_batch: bool = False
    workers: int = 1
    worker_strategy: str = "threads"
    backend: str = "local"
    fallback_web: bool = False
    requires_raster_sidecar: bool = False
    requires_adaptive_hints: bool = False
    unknown_capabilities: tuple[str, ...] = field(default_factory=tuple)
    blind4d_all_sky: bool = False


@dataclass(frozen=True, slots=True)
class EngineSelection:
    requested_mode: EngineMode
    selected_mode: EngineMode
    supported: bool
    reason: str
    warnings: tuple[str, ...] = ()


FITS_EXTENSIONS = frozenset({".fit", ".fits", ".fts"})
RASTER_EXTENSIONS = frozenset({".tif", ".tiff", ".png", ".jpg", ".jpeg"})


def select_engine(request: EngineSelectionRequest) -> EngineSelection:
    requested = _coerce_mode(request.requested_mode)
    unsupported = _unsupported_pipeline_reasons(request)
    warnings = _selection_warnings(request)

    if requested == EngineMode.LEGACY:
        return EngineSelection(
            requested_mode=requested,
            selected_mode=EngineMode.LEGACY,
            supported=True,
            reason="legacy_requested",
            warnings=warnings,
        )

    if requested == EngineMode.PIPELINE:
        if unsupported:
            return EngineSelection(
                requested_mode=requested,
                selected_mode=EngineMode.PIPELINE,
                supported=False,
                reason="pipeline_unsupported: " + "; ".join(unsupported),
                warnings=warnings,
            )
        return EngineSelection(
            requested_mode=requested,
            selected_mode=EngineMode.PIPELINE,
            supported=True,
            reason="pipeline_requested_supported",
            warnings=warnings,
        )

    if unsupported:
        return EngineSelection(
            requested_mode=requested,
            selected_mode=EngineMode.LEGACY,
            supported=True,
            reason="auto_legacy: " + "; ".join(unsupported),
            warnings=warnings,
        )
    return EngineSelection(
        requested_mode=requested,
        selected_mode=EngineMode.PIPELINE,
        supported=True,
        reason="auto_pipeline: fits_local_supported",
        warnings=warnings,
    )


def _coerce_mode(value: EngineMode | str) -> EngineMode:
    if isinstance(value, EngineMode):
        return value
    try:
        return EngineMode(str(value).strip().lower())
    except ValueError as exc:
        raise ValueError(f"unknown engine mode: {value}") from exc


def _unsupported_pipeline_reasons(request: EngineSelectionRequest) -> tuple[str, ...]:
    reasons: list[str] = []
    suffix = _suffix(request.input_path)
    if suffix and suffix in RASTER_EXTENSIONS:
        reasons.append(f"raster_not_supported_by_pipeline:{suffix}")
    elif suffix and suffix not in FITS_EXTENSIONS:
        reasons.append(f"unknown_file_type:{suffix}")
    elif not suffix:
        reasons.append("input_type_unknown")

    backend = str(request.backend or "local").strip().lower()
    if backend not in {"local", "near", "blind4d", "near_blind4d"}:
        reasons.append(f"backend_not_supported_by_pipeline:{backend}")
    if backend in {"web", "astrometry", "astrometry.net"}:
        reasons.append("web_backend_requires_legacy")
    if request.fallback_web:
        reasons.append("fallback_web_requires_legacy")
    if request.requires_raster_sidecar:
        reasons.append("raster_sidecar_requires_legacy")
    if request.requires_adaptive_hints:
        reasons.append("adaptive_hints_require_legacy")
    if request.is_batch and str(request.worker_strategy or "threads").strip().lower() not in {"threads", "thread"}:
        reasons.append(f"batch_worker_strategy_requires_legacy:{request.worker_strategy}")
    if request.unknown_capabilities:
        reasons.append("unknown_capability:" + ",".join(request.unknown_capabilities))
    return tuple(dict.fromkeys(reasons))


def _selection_warnings(request: EngineSelectionRequest) -> tuple[str, ...]:
    warnings: list[str] = []
    if not request.blind4d_all_sky:
        warnings.append("blind4d_coverage_partial_not_all_sky")
    return tuple(warnings)


def _suffix(path: Path | str | None) -> str:
    if path is None:
        return ""
    return Path(path).suffix.lower()
