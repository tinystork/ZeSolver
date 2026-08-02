"""Fast, GUI-free GPU capability probing."""

from __future__ import annotations

import importlib
import importlib.metadata
import platform as platform_module
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .models import (
    CapabilityState,
    DistributionKind,
    EffectiveBackend,
    GpuCapabilityReport,
    GpuRuntimeContext,
    ReasonCode,
    ensure_runtime_context,
)
from .self_test import run_cupy_self_test


CUPY_VARIANTS = ("cupy", "cupy-cuda12x", "cupy-cuda13x", "cupy-rocm-5-0", "cupy-rocm-6-0")


@dataclass(frozen=True)
class ProbeHooks:
    platform_system: Callable[[], str] = platform_module.system
    machine: Callable[[], str] = platform_module.machine
    package_version: Callable[[str], str] = importlib.metadata.version
    import_module: Callable[[str], Any] = importlib.import_module
    which: Callable[[str], str | None] = shutil.which
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run


def _installed_cupy_packages(hooks: ProbeHooks) -> dict[str, str]:
    found: dict[str, str] = {}
    for package in CUPY_VARIANTS:
        try:
            found[package] = str(hooks.package_version(package))
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception:
            continue
    return found


def _detect_nvidia_smi(hooks: ProbeHooks, *, timeout_s: float = 2.0) -> dict[str, Any]:
    exe = hooks.which("nvidia-smi")
    if not exe:
        return {"state": CapabilityState.UNKNOWN, "reason": "nvidia-smi not found"}
    cmd = [
        exe,
        "--query-gpu=name,driver_version",
        "--format=csv,noheader",
    ]
    try:
        completed = hooks.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            shell=False,
        )
    except Exception as exc:
        return {"state": CapabilityState.UNKNOWN, "reason": str(exc), "command": cmd}
    stdout = str(completed.stdout or "").strip()
    stderr = str(completed.stderr or "").strip()
    if completed.returncode != 0 or not stdout:
        return {
            "state": CapabilityState.UNKNOWN,
            "reason": stderr or f"nvidia-smi returned {completed.returncode}",
            "command": cmd,
        }
    names: list[str] = []
    driver = None
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if parts and parts[0]:
            names.append(parts[0])
        if len(parts) > 1 and parts[1]:
            driver = parts[1]
    return {
        "state": CapabilityState.AVAILABLE if names else CapabilityState.UNKNOWN,
        "device_names": tuple(names),
        "driver_version": driver,
        "command": cmd,
    }


def probe_gpu_capability(
    runtime_context: GpuRuntimeContext | None = None,
    *,
    hooks: ProbeHooks | None = None,
    run_self_test: bool = True,
) -> GpuCapabilityReport:
    context = ensure_runtime_context(runtime_context)
    hooks = hooks or ProbeHooks()
    system = str(hooks.platform_system() or sys.platform).lower()
    arch = str(hooks.machine() or platform_module.machine() or "unknown")
    details: dict[str, Any] = {"python": sys.version.split()[0]}

    if system == "darwin":
        return GpuCapabilityReport(
            platform="darwin",
            architecture=arch,
            distribution_kind=context.distribution_kind,
            nvidia_gpu_state=CapabilityState.UNSUPPORTED,
            driver_state=CapabilityState.UNSUPPORTED,
            driver_version=None,
            cupy_package_state=CapabilityState.UNSUPPORTED,
            cupy_package_name=None,
            cupy_version=None,
            cuda_runtime_state=CapabilityState.UNSUPPORTED,
            device_count=0,
            allocation_test_state=CapabilityState.UNSUPPORTED,
            effective_backend=EffectiveBackend.CPU,
            reason_code=ReasonCode.CUDA_UNSUPPORTED_ON_PLATFORM,
            human_message="CUDA is not supported for ZeSolver in this macOS configuration. CPU mode is available.",
            technical_details=details,
        )

    installed = _installed_cupy_packages(hooks)
    details["cupy_packages"] = dict(installed)
    if len(installed) > 1:
        smi = _detect_nvidia_smi(hooks)
        return GpuCapabilityReport(
            platform=system,
            architecture=arch,
            distribution_kind=context.distribution_kind,
            nvidia_gpu_state=smi.get("state", CapabilityState.UNKNOWN),
            driver_state=CapabilityState.UNKNOWN,
            driver_version=smi.get("driver_version"),
            cupy_package_state=CapabilityState.CONFLICT,
            cupy_package_name=", ".join(sorted(installed)),
            cupy_version=None,
            cuda_runtime_state=CapabilityState.CONFLICT,
            device_count=None,
            allocation_test_state=CapabilityState.CONFLICT,
            effective_backend=EffectiveBackend.CPU,
            reason_code=ReasonCode.CUPY_PACKAGE_CONFLICT,
            human_message="Multiple CuPy variants are installed. ZeSolver will continue on CPU until the environment is repaired.",
            technical_details=details,
        )

    package_name = next(iter(installed), None)
    package_version = installed.get(package_name) if package_name else None
    if package_name and run_self_test:
        result = run_cupy_self_test(import_module=hooks.import_module)
        state = result.get("state", CapabilityState.UNKNOWN)
        reason = result.get("reason_code", ReasonCode.UNKNOWN)
        device_count = result.get("device_count")
        if state == CapabilityState.AVAILABLE:
            names = (str(result.get("device_name")),) if result.get("device_name") else ()
            return GpuCapabilityReport(
                platform=system,
                architecture=arch,
                distribution_kind=context.distribution_kind,
                nvidia_gpu_state=CapabilityState.AVAILABLE,
                driver_state=CapabilityState.AVAILABLE,
                driver_version=None,
                cupy_package_state=CapabilityState.AVAILABLE,
                cupy_package_name=package_name,
                cupy_version=str(result.get("cupy_version") or package_version or ""),
                cuda_runtime_state=CapabilityState.AVAILABLE,
                device_count=int(device_count or 0),
                device_names=names,
                allocation_test_state=CapabilityState.AVAILABLE,
                effective_backend=EffectiveBackend.CUDA,
                reason_code=ReasonCode.GPU_READY,
                human_message="CUDA acceleration is available for ZeNear star detection.",
                technical_details={**details, "self_test": dict(result)},
            )
        return GpuCapabilityReport(
            platform=system,
            architecture=arch,
            distribution_kind=context.distribution_kind,
            nvidia_gpu_state=CapabilityState.UNKNOWN,
            driver_state=CapabilityState.UNKNOWN,
            driver_version=None,
            cupy_package_state=CapabilityState.BROKEN,
            cupy_package_name=package_name,
            cupy_version=package_version,
            cuda_runtime_state=CapabilityState.BROKEN,
            device_count=int(device_count) if isinstance(device_count, int) else None,
            allocation_test_state=CapabilityState.BROKEN,
            effective_backend=EffectiveBackend.CPU,
            reason_code=reason if isinstance(reason, ReasonCode) else ReasonCode.CUPY_IMPORT_FAILED,
            human_message="GPU support is installed but did not pass the CUDA self-test. ZeSolver will continue on CPU.",
            technical_details={**details, "self_test": dict(result)},
        )

    smi = _detect_nvidia_smi(hooks)
    nvidia_state = smi.get("state", CapabilityState.UNKNOWN)
    reason = ReasonCode.CUPY_NOT_INSTALLED if nvidia_state == CapabilityState.AVAILABLE else ReasonCode.NVIDIA_NOT_DETECTED
    return GpuCapabilityReport(
        platform=system,
        architecture=arch,
        distribution_kind=context.distribution_kind,
        nvidia_gpu_state=nvidia_state,
        driver_state=CapabilityState.AVAILABLE if nvidia_state == CapabilityState.AVAILABLE else CapabilityState.UNKNOWN,
        driver_version=smi.get("driver_version"),
        cupy_package_state=CapabilityState.UNAVAILABLE,
        cupy_package_name=None,
        cupy_version=None,
        cuda_runtime_state=CapabilityState.UNAVAILABLE,
        device_count=None,
        device_names=tuple(smi.get("device_names") or ()),
        allocation_test_state=CapabilityState.UNAVAILABLE,
        effective_backend=EffectiveBackend.CPU,
        reason_code=reason,
        human_message=(
            "An NVIDIA GPU was detected, but CuPy is not installed. CPU mode is available."
            if reason == ReasonCode.CUPY_NOT_INSTALLED
            else "No usable NVIDIA/CUDA stack was detected. CPU mode is available."
        ),
        technical_details={**details, "nvidia_smi": smi},
    )
