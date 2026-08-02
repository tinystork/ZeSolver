"""Reusable GPU diagnostic/provisioning API for ZeSolver hosts."""

from .models import (
    CapabilityState,
    DistributionKind,
    EffectiveBackend,
    GpuCapabilityReport,
    GpuPackageProfile,
    GpuProvisioningPlan,
    GpuProvisioningProgress,
    GpuProvisioningResult,
    GpuRuntimeContext,
    ProvisioningStatus,
    ReasonCode,
)
from .policy import ALLOWED_GPU_PACKAGES, DEFAULT_CUPY_PROFILE, build_gpu_provisioning_plan
from .probe import ProbeHooks, probe_gpu_capability
from .provisioning import GuidanceOnlyProvisioner, HostProvisionerAdapter, PythonEnvironmentProvisioner
from .runtime import default_gpu_temp_dir, detect_gpu_runtime_context
from .self_test import classify_cupy_exception, run_cupy_self_test

__all__ = [
    "ALLOWED_GPU_PACKAGES",
    "DEFAULT_CUPY_PROFILE",
    "CapabilityState",
    "DistributionKind",
    "EffectiveBackend",
    "GpuCapabilityReport",
    "GpuPackageProfile",
    "GpuProvisioningPlan",
    "GpuProvisioningProgress",
    "GpuProvisioningResult",
    "GpuRuntimeContext",
    "GuidanceOnlyProvisioner",
    "HostProvisionerAdapter",
    "ProbeHooks",
    "ProvisioningStatus",
    "PythonEnvironmentProvisioner",
    "ReasonCode",
    "build_gpu_provisioning_plan",
    "classify_cupy_exception",
    "default_gpu_temp_dir",
    "detect_gpu_runtime_context",
    "probe_gpu_capability",
    "run_cupy_self_test",
]
