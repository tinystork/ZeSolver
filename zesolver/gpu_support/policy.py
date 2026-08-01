"""GPU provisioning policy for ZeSolver."""

from __future__ import annotations

import sys
from pathlib import Path

from .models import (
    CapabilityState,
    DistributionKind,
    EffectiveBackend,
    GpuCapabilityReport,
    GpuPackageProfile,
    GpuProvisioningPlan,
    GpuRuntimeContext,
    ProvisioningStatus,
    ReasonCode,
    ensure_runtime_context,
)


DEFAULT_CUPY_PROFILE = GpuPackageProfile(
    package_requirement="cupy-cuda12x[ctk]",
    supported_platforms=("linux", "win32", "windows"),
    supported_python_versions=("3.10", "3.11", "3.12", "3.13"),
    minimum_driver_rule="NVIDIA driver with CUDA 12 runtime compatibility",
    tested_by_zesolver=False,
    display_name="CuPy CUDA 12.x",
    documentation_note="Candidate profile for source-managed beta environments; local CuPy-only validation required CTK components, and Windows validation is required before making it a public packaging promise.",
)

ALLOWED_GPU_PACKAGES = (DEFAULT_CUPY_PROFILE.package_requirement,)


def _same_python(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    try:
        return Path(left).expanduser().absolute() == Path(right).expanduser().absolute()
    except Exception:
        return str(left) == str(right)


def gpu_profile_for_runtime(report: GpuCapabilityReport) -> GpuPackageProfile | None:
    platform = str(report.platform or "").lower()
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    if platform not in DEFAULT_CUPY_PROFILE.supported_platforms:
        return None
    if py not in DEFAULT_CUPY_PROFILE.supported_python_versions:
        return None
    return DEFAULT_CUPY_PROFILE


def build_gpu_provisioning_plan(
    report: GpuCapabilityReport,
    runtime_context: GpuRuntimeContext | None = None,
) -> GpuProvisioningPlan:
    context = ensure_runtime_context(runtime_context)
    reason = str(getattr(context, "environment_reason", "") or "").strip()
    if report.effective_backend == EffectiveBackend.CUDA:
        return GpuProvisioningPlan(
            ProvisioningStatus.ALREADY_AVAILABLE,
            None,
            requires_consent=False,
            restart_required=False,
            message="CUDA acceleration is already available.",
        )
    if report.cupy_package_state == CapabilityState.CONFLICT:
        return GpuProvisioningPlan(
            ProvisioningStatus.CONFLICT_DETECTED,
            None,
            message="Multiple CuPy variants are installed. ZeSolver will not modify this environment automatically.",
        )
    if context.distribution_kind != DistributionKind.SOURCE_MANAGED or not context.allow_environment_mutation:
        if context.distribution_kind == DistributionKind.FROZEN_STANDALONE:
            message = "Executable standalone: GPU provisioning is diagnostic-only in this edition; CPU mode remains available."
        elif context.distribution_kind == DistributionKind.EMBEDDED_HOST:
            message = "Embedded host runtime: the host application is responsible for GPU provisioning; CPU mode remains available."
        else:
            message = reason or "System or unproven Python environment: GPU provisioning is disabled; CPU mode remains available."
        return GpuProvisioningPlan(
            ProvisioningStatus.ENVIRONMENT_NOT_MUTABLE
            if context.distribution_kind != DistributionKind.FROZEN_STANDALONE
            else ProvisioningStatus.GUIDANCE_ONLY,
            None,
            message=message,
            technical_details={"distribution_kind": context.distribution_kind.value, "environment_reason": reason},
        )
    if not _same_python(context.python_executable, sys.executable):
        return GpuProvisioningPlan(
            ProvisioningStatus.ENVIRONMENT_NOT_MUTABLE,
            None,
            message="GPU provisioning disabled: the target interpreter is not the active ZeSolver Python executable.",
            technical_details={
                "distribution_kind": context.distribution_kind.value,
                "environment_reason": reason,
                "python_executable": context.python_executable,
                "active_executable": sys.executable,
            },
        )
    if report.platform == "darwin" or report.reason_code == ReasonCode.CUDA_UNSUPPORTED_ON_PLATFORM:
        return GpuProvisioningPlan(
            ProvisioningStatus.ENVIRONMENT_NOT_MUTABLE,
            None,
            message="CUDA provisioning is not supported for this macOS configuration.",
        )
    if report.nvidia_gpu_state != CapabilityState.AVAILABLE:
        return GpuProvisioningPlan(
            ProvisioningStatus.ENVIRONMENT_NOT_MUTABLE,
            None,
            message="No NVIDIA GPU was detected, so no CuPy installation is proposed by default.",
        )
    if report.cupy_package_state not in {CapabilityState.UNAVAILABLE, CapabilityState.UNKNOWN}:
        return GpuProvisioningPlan(
            ProvisioningStatus.ENVIRONMENT_NOT_MUTABLE,
            None,
            message="The current GPU package state is not suitable for guided installation.",
        )
    profile = gpu_profile_for_runtime(report)
    if profile is None:
        return GpuProvisioningPlan(
            ProvisioningStatus.ENVIRONMENT_NOT_MUTABLE,
            None,
            message="No validated ZeSolver CuPy package profile matches this runtime.",
        )
    python = sys.executable
    return GpuProvisioningPlan(
        ProvisioningStatus.AVAILABLE,
        profile,
        command=(python, "-m", "pip", "install", profile.package_requirement),
        requires_consent=True,
        restart_required=True,
        message=f"Install optional {profile.display_name} support for ZeNear detection.",
        technical_details={
            "allowlist": ALLOWED_GPU_PACKAGES,
            "environment_reason": reason,
            "gpu_temp_dir": getattr(context, "gpu_temp_dir", None),
        },
    )
