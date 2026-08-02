"""CuPy/CUDA self-test used by the GPU diagnostic service."""

from __future__ import annotations

from typing import Any

from .models import CapabilityState, ReasonCode


def classify_cupy_exception(exc: BaseException) -> ReasonCode:
    text = str(exc).lower()
    name = f"{exc.__class__.__module__}.{exc.__class__.__name__}".lower()
    combined = f"{name} {text}"
    if "no module named 'cupy'" in combined or "no module named cupy" in combined:
        return ReasonCode.CUPY_NOT_INSTALLED
    if "outofmemory" in combined or "out of memory" in combined:
        return ReasonCode.CUDA_ALLOCATION_FAILED
    if "no cuda device" in combined or "devicecount" in combined and "0" in combined:
        return ReasonCode.NO_CUDA_DEVICE
    if "driver" in combined:
        return ReasonCode.CUDA_DRIVER_UNAVAILABLE
    if "runtime" in combined or "cudaerror" in combined or "cuda error" in combined:
        return ReasonCode.CUDA_RUNTIME_UNAVAILABLE
    return ReasonCode.CUPY_IMPORT_FAILED


def run_cupy_self_test(*, device: int | None = None, import_module: Any = None) -> dict[str, Any]:
    """Import CuPy and prove a tiny CUDA allocation/calculation works."""
    try:
        if import_module is None:
            import importlib

            import_module = importlib.import_module
        cupy = import_module("cupy")
        runtime = import_module("cupy.cuda.runtime")
    except Exception as exc:
        return {
            "state": CapabilityState.UNAVAILABLE,
            "reason_code": classify_cupy_exception(exc),
            "error": str(exc),
        }

    try:
        ndev = int(runtime.getDeviceCount())
        if ndev <= 0:
            return {
                "state": CapabilityState.UNAVAILABLE,
                "reason_code": ReasonCode.NO_CUDA_DEVICE,
                "device_count": 0,
            }
        dev = int(device) if device is not None else 0
        if dev < 0 or dev >= ndev:
            return {
                "state": CapabilityState.BROKEN,
                "reason_code": ReasonCode.NO_CUDA_DEVICE,
                "device_count": ndev,
                "error": f"CUDA device {dev} out of range (devices={ndev})",
            }
        with cupy.cuda.Device(dev):
            props = runtime.getDeviceProperties(dev)
            name = None
            if isinstance(props, dict):
                raw_name = props.get("name")
                if isinstance(raw_name, (bytes, bytearray)):
                    name = raw_name.decode(errors="ignore")
                elif raw_name:
                    name = str(raw_name)
            arr = cupy.arange(16, dtype=cupy.float32)
            result = cupy.sum(arr)
            cupy.cuda.Stream.null.synchronize()
            raw_value = result.get()
            try:
                value = float(raw_value)
            except Exception:
                value = float(raw_value.reshape(-1)[0])
            if value != 120.0:
                return {
                    "state": CapabilityState.BROKEN,
                    "reason_code": ReasonCode.CUDA_INITIALIZATION_FAILED,
                    "device_count": ndev,
                    "device_name": name,
                    "error": f"unexpected CUDA self-test value {value!r}",
                }
            try:
                pool = cupy.get_default_memory_pool()
                pool.free_all_blocks()
            except Exception:
                pass
        version = str(getattr(cupy, "__version__", "") or "")
        return {
            "state": CapabilityState.AVAILABLE,
            "reason_code": ReasonCode.GPU_READY,
            "device_count": ndev,
            "device_name": name,
            "cupy_version": version or None,
        }
    except Exception as exc:
        return {
            "state": CapabilityState.BROKEN,
            "reason_code": classify_cupy_exception(exc),
            "error": str(exc),
        }
