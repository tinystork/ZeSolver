"""Structured GPU diagnostic and provisioning models.

This module is deliberately GUI-free so ZeSolver, future frozen builds and
embedded hosts can share the same GPU contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SCHEMA = "zesolver.gpu_diagnostic.v1"


class CapabilityState(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"
    BROKEN = "broken"
    CONFLICT = "conflict"
    UNSUPPORTED = "unsupported"


class DistributionKind(str, Enum):
    SOURCE_MANAGED = "source_managed"
    FROZEN_STANDALONE = "frozen_standalone"
    EMBEDDED_HOST = "embedded_host"
    UNKNOWN = "unknown"


class EffectiveBackend(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"


class ReasonCode(str, Enum):
    GPU_READY = "GPU_READY"
    CUDA_UNSUPPORTED_ON_PLATFORM = "CUDA_UNSUPPORTED_ON_PLATFORM"
    CUPY_NOT_INSTALLED = "CUPY_NOT_INSTALLED"
    CUPY_IMPORT_FAILED = "CUPY_IMPORT_FAILED"
    CUPY_PACKAGE_CONFLICT = "CUPY_PACKAGE_CONFLICT"
    CUDA_DRIVER_UNAVAILABLE = "CUDA_DRIVER_UNAVAILABLE"
    CUDA_RUNTIME_UNAVAILABLE = "CUDA_RUNTIME_UNAVAILABLE"
    NO_CUDA_DEVICE = "NO_CUDA_DEVICE"
    CUDA_INITIALIZATION_FAILED = "CUDA_INITIALIZATION_FAILED"
    CUDA_ALLOCATION_FAILED = "CUDA_ALLOCATION_FAILED"
    NVIDIA_NOT_DETECTED = "NVIDIA_NOT_DETECTED"
    ENVIRONMENT_NOT_MUTABLE = "ENVIRONMENT_NOT_MUTABLE"
    UNKNOWN = "UNKNOWN"


class ProvisioningStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    ALREADY_AVAILABLE = "ALREADY_AVAILABLE"
    INSTALLED_RESTART_REQUIRED = "INSTALLED_RESTART_REQUIRED"
    DECLINED = "DECLINED"
    CANCELLED = "CANCELLED"
    INSTALL_FAILED = "INSTALL_FAILED"
    CONFLICT_DETECTED = "CONFLICT_DETECTED"
    ENVIRONMENT_NOT_MUTABLE = "ENVIRONMENT_NOT_MUTABLE"
    GUIDANCE_ONLY = "GUIDANCE_ONLY"


@dataclass(frozen=True)
class GpuRuntimeContext:
    distribution_kind: DistributionKind = DistributionKind.UNKNOWN
    allow_environment_mutation: bool = False
    python_executable: str | None = None
    host_name: str | None = None
    host_can_provision: bool = False
    environment_reason: str = ""
    gpu_temp_dir: str | None = None


@dataclass(frozen=True)
class GpuPackageProfile:
    package_requirement: str
    supported_platforms: tuple[str, ...]
    supported_python_versions: tuple[str, ...]
    minimum_driver_rule: str
    tested_by_zesolver: bool
    display_name: str
    documentation_note: str


@dataclass(frozen=True)
class GpuCapabilityReport:
    platform: str
    architecture: str
    distribution_kind: DistributionKind
    nvidia_gpu_state: CapabilityState
    driver_state: CapabilityState
    driver_version: str | None
    cupy_package_state: CapabilityState
    cupy_package_name: str | None
    cupy_version: str | None
    cuda_runtime_state: CapabilityState
    device_count: int | None
    device_names: tuple[str, ...] = ()
    allocation_test_state: CapabilityState = CapabilityState.UNKNOWN
    effective_backend: EffectiveBackend = EffectiveBackend.CPU
    reason_code: ReasonCode = ReasonCode.UNKNOWN
    human_message: str = ""
    technical_details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["schema"] = SCHEMA
        for key in (
            "distribution_kind",
            "nvidia_gpu_state",
            "driver_state",
            "cupy_package_state",
            "cuda_runtime_state",
            "allocation_test_state",
            "effective_backend",
            "reason_code",
        ):
            value = data.get(key)
            if isinstance(value, Enum):
                data[key] = value.value
        return data


@dataclass(frozen=True)
class GpuProvisioningPlan:
    status: ProvisioningStatus
    profile: GpuPackageProfile | None
    command: tuple[str, ...] = ()
    requires_consent: bool = True
    restart_required: bool = True
    message: str = ""
    technical_details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass(frozen=True)
class GpuProvisioningResult:
    status: ProvisioningStatus
    message: str
    returncode: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    restart_required: bool = False
    technical_details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GpuProvisioningProgress:
    phase: str
    message: str
    line: str = ""

    def to_text(self) -> str:
        text = self.line or self.message
        return f"[{self.phase}] {text}" if self.phase else text

    def to_dict(self) -> dict[str, str]:
        return {"phase": self.phase, "message": self.message, "line": self.line}

    def __str__(self) -> str:
        return self.to_text()

    def __len__(self) -> int:
        return len(self.to_text())

    def __contains__(self, value: object) -> bool:
        return str(value) in self.to_text()


ProgressCallback = Callable[[GpuProvisioningProgress | str], None]
CancelToken = Callable[[], bool]
HostProvisioningCallback = Callable[[GpuProvisioningPlan], GpuProvisioningResult]


def normalize_distribution_kind(value: object) -> DistributionKind:
    if isinstance(value, DistributionKind):
        return value
    text = str(value or "").strip().lower().replace("-", "_")
    for item in DistributionKind:
        if item.value == text or item.name.lower() == text:
            return item
    return DistributionKind.UNKNOWN


def ensure_runtime_context(context: GpuRuntimeContext | None) -> GpuRuntimeContext:
    return context if isinstance(context, GpuRuntimeContext) else GpuRuntimeContext()


def path_text(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return str(path)
