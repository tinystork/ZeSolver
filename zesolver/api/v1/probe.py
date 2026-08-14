"""Lightweight capability probing for the ZeSolver v1 API.

The default probe is intentionally cheap: no catalog scan, no GPU/CuPy import,
no network, no writes, no worker processes.  Catalog and GPU availability are
only inspected when explicitly requested.
"""

from __future__ import annotations

from pathlib import Path

from .errors import InvalidRequestError
from .models import (
    API_VERSION,
    _SUPPORTED_CAPABILITIES,
    ApiInfo,
    CapabilityAvailability,
    CapabilityState,
    CapabilityUnavailableReason,
    RuntimeProbe,
)


def _product_version() -> str | None:
    """Return the installed product version without importing the heavy package."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except Exception:  # pragma: no cover - stdlib always present
        return None
    try:
        return version("ZeSolver")
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def get_api_info() -> ApiInfo:
    """Return static API metadata with no dynamic/runtime state.

    This function performs no catalog scan, no GPU initialization, no GUI
    config read/write and no network access.
    """
    return ApiInfo(
        api_version=API_VERSION,
        product_version=_product_version(),
        supported_capabilities=_SUPPORTED_CAPABILITIES,
    )


def probe(
    *,
    check_catalogs: bool = False,
    check_gpu: bool = False,
    timeout_s: float = 2.0,
    resources_path: Path | None = None,
) -> RuntimeProbe:
    """Probe runtime capabilities.

    * With ``check_catalogs=False`` (default) catalog-related capabilities are
      reported as :attr:`CapabilityAvailability.NOT_CHECKED`.
    * With ``check_gpu=False`` (default) the ``gpu`` capability is reported as
      :attr:`CapabilityAvailability.NOT_CHECKED`.

    No catalog GB scan, download, write, network or worker process is ever
    performed here.
    """
    if timeout_s <= 0:
        raise InvalidRequestError("timeout_s must be > 0")

    capabilities: list[CapabilityState] = []

    if check_catalogs:
        near_state = _catalog_capability_state(
            resources_path, "near_solve", near=True
        )
        blind_state = _catalog_capability_state(
            resources_path, "blind_solve", near=False
        )
        capabilities.append(near_state)
        capabilities.append(blind_state)
    else:
        capabilities.append(
            CapabilityState(
                "near_solve", True, CapabilityAvailability.NOT_CHECKED
            )
        )
        capabilities.append(
            CapabilityState(
                "blind_solve", True, CapabilityAvailability.NOT_CHECKED
            )
        )

    capabilities.append(
        CapabilityState("wcs_write", True, CapabilityAvailability.AVAILABLE)
    )
    capabilities.append(
        CapabilityState("cancel", True, CapabilityAvailability.AVAILABLE)
    )

    if check_gpu:
        capabilities.append(_gpu_capability_state())
    else:
        capabilities.append(
            CapabilityState("gpu", True, CapabilityAvailability.NOT_CHECKED)
        )

    return RuntimeProbe(
        api_version=API_VERSION,
        product_version=_product_version(),
        capabilities=tuple(capabilities),
        warnings=(),
    )


def _catalog_capability_state(
    resources_path: Path | None, capability_id: str, *, near: bool
) -> CapabilityState:
    if resources_path is None:
        return CapabilityState(
            capability_id,
            True,
            CapabilityAvailability.NOT_CHECKED,
            detail="resources_path not provided",
        )
    path = Path(resources_path).expanduser()
    if not path.exists():
        return CapabilityState(
            capability_id,
            True,
            CapabilityAvailability.UNAVAILABLE,
            CapabilityUnavailableReason.MISSING_RESOURCE,
            detail=f"catalog path does not exist: {path}",
        )
    try:
        from zesolver.catalog_library.manifest import load_manifest

        manifest = load_manifest(path)
    except Exception as exc:
        return CapabilityState(
            capability_id,
            True,
            CapabilityAvailability.UNAVAILABLE,
            CapabilityUnavailableReason.MISSING_RESOURCE,
            detail=f"catalog manifest unavailable: {type(exc).__name__}",
        )
    if near:
        available = bool(getattr(manifest, "sources", ()))
    else:
        available = any(
            getattr(index, "engine", None) == "blind4d"
            for index in getattr(manifest, "derived_indexes", ())
        )
    if available:
        return CapabilityState(
            capability_id, True, CapabilityAvailability.AVAILABLE
        )
    return CapabilityState(
        capability_id,
        True,
        CapabilityAvailability.UNAVAILABLE,
        CapabilityUnavailableReason.MISSING_RESOURCE,
        detail="capability not present in catalog manifest",
    )


def _gpu_capability_state() -> CapabilityState:
    try:
        from zesolver.gpu_support.models import CapabilityState as GpuState
        from zesolver.gpu_support.models import EffectiveBackend
        from zesolver.gpu_support.probe import probe_gpu_capability

        report = probe_gpu_capability(run_self_test=False)
    except Exception as exc:  # pragma: no cover - defensive
        return CapabilityState(
            "gpu",
            True,
            CapabilityAvailability.UNAVAILABLE,
            CapabilityUnavailableReason.UNKNOWN,
            detail=f"gpu probe failed: {exc}",
        )

    if report.effective_backend is EffectiveBackend.CUDA:
        return CapabilityState(
            "gpu",
            True,
            CapabilityAvailability.AVAILABLE,
            detail=f"CUDA via {report.cupy_package_name}",
        )
    if report.cuda_runtime_state is GpuState.UNSUPPORTED:
        return CapabilityState(
            "gpu",
            True,
            CapabilityAvailability.UNAVAILABLE,
            CapabilityUnavailableReason.UNSUPPORTED_PLATFORM,
            detail=report.human_message or "cuda unsupported on platform",
        )
    return CapabilityState(
        "gpu",
        True,
        CapabilityAvailability.UNAVAILABLE,
        CapabilityUnavailableReason.GPU_UNAVAILABLE,
        detail=report.human_message or str(report.reason_code),
    )
