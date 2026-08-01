"""Reusable GPU diagnostic/provisioning API for ZeSolver hosts."""

from .models import (
    CapabilityState,
    DistributionKind,
    EffectiveBackend,
    GpuCapabilityReport,
    GpuPackageProfile,
    GpuProvisioningPlan,
    GpuProvisioningResult,
    GpuRuntimeContext,
    ProvisioningStatus,
    ReasonCode,
)
from .policy import ALLOWED_GPU_PACKAGES, DEFAULT_CUPY_PROFILE, build_gpu_provisioning_plan
from .probe import ProbeHooks, probe_gpu_capability
from .provisioning import GuidanceOnlyProvisioner, HostProvisionerAdapter, PythonEnvironmentProvisioner
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
    "probe_gpu_capability",
    "run_cupy_self_test",
]
