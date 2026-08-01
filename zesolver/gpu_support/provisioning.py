"""Interchangeable GPU provisioners."""

from __future__ import annotations

import subprocess
import time
from typing import Callable

from .models import (
    CancelToken,
    DistributionKind,
    GpuCapabilityReport,
    GpuProvisioningPlan,
    GpuProvisioningResult,
    GpuRuntimeContext,
    HostProvisioningCallback,
    ProgressCallback,
    ProvisioningStatus,
    ensure_runtime_context,
)
from .policy import ALLOWED_GPU_PACKAGES, build_gpu_provisioning_plan


def _tail(text: str, *, limit: int = 4000) -> str:
    text = str(text or "")
    return text[-limit:]


class PythonEnvironmentProvisioner:
    def can_provision(self, context: GpuRuntimeContext, report: GpuCapabilityReport) -> bool:
        plan = build_gpu_provisioning_plan(report, context)
        return plan.status == ProvisioningStatus.AVAILABLE and bool(plan.command)

    def build_plan(self, context: GpuRuntimeContext, report: GpuCapabilityReport) -> GpuProvisioningPlan:
        return build_gpu_provisioning_plan(report, context)

    def provision(
        self,
        plan: GpuProvisioningPlan,
        progress_callback: ProgressCallback | None = None,
        cancel_token: CancelToken | None = None,
        *,
        timeout_s: float = 900.0,
    ) -> GpuProvisioningResult:
        if plan.status != ProvisioningStatus.AVAILABLE or not plan.command:
            return GpuProvisioningResult(ProvisioningStatus.ENVIRONMENT_NOT_MUTABLE, plan.message)
        package = plan.command[-1]
        if package not in ALLOWED_GPU_PACKAGES:
            return GpuProvisioningResult(ProvisioningStatus.INSTALL_FAILED, "Package is not in the ZeSolver GPU allowlist.")
        if cancel_token and cancel_token():
            return GpuProvisioningResult(ProvisioningStatus.CANCELLED, "GPU installation cancelled before start.")
        if progress_callback:
            progress_callback("Starting optional GPU package installation...")
        try:
            proc = subprocess.Popen(
                list(plan.command),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,
            )
            started = time.monotonic()
            while proc.poll() is None:
                if cancel_token and cancel_token():
                    proc.terminate()
                    try:
                        stdout, stderr = proc.communicate(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        stdout, stderr = proc.communicate(timeout=10)
                    return GpuProvisioningResult(
                        ProvisioningStatus.CANCELLED,
                        "GPU installation cancelled.",
                        returncode=proc.returncode,
                        stdout_tail=_tail(stdout),
                        stderr_tail=_tail(stderr),
                    )
                if time.monotonic() - started > timeout_s:
                    proc.terminate()
                    stdout, stderr = proc.communicate(timeout=10)
                    return GpuProvisioningResult(
                        ProvisioningStatus.INSTALL_FAILED,
                        "GPU installation timed out.",
                        returncode=proc.returncode,
                        stdout_tail=_tail(stdout),
                        stderr_tail=_tail(stderr),
                    )
                time.sleep(0.1)
            stdout, stderr = proc.communicate()
        except Exception as exc:
            return GpuProvisioningResult(ProvisioningStatus.INSTALL_FAILED, str(exc))
        if proc.returncode != 0:
            return GpuProvisioningResult(
                ProvisioningStatus.INSTALL_FAILED,
                "GPU package installation failed.",
                returncode=proc.returncode,
                stdout_tail=_tail(stdout),
                stderr_tail=_tail(stderr),
            )
        return GpuProvisioningResult(
            ProvisioningStatus.INSTALLED_RESTART_REQUIRED,
            "Installation finished. Restart ZeSolver to activate GPU acceleration.",
            returncode=0,
            stdout_tail=_tail(stdout),
            stderr_tail=_tail(stderr),
            restart_required=True,
        )


class GuidanceOnlyProvisioner:
    def can_provision(self, context: GpuRuntimeContext, report: GpuCapabilityReport) -> bool:
        return False

    def build_plan(self, context: GpuRuntimeContext, report: GpuCapabilityReport) -> GpuProvisioningPlan:
        return build_gpu_provisioning_plan(report, context)

    def provision(
        self,
        plan: GpuProvisioningPlan,
        progress_callback: ProgressCallback | None = None,
        cancel_token: CancelToken | None = None,
    ) -> GpuProvisioningResult:
        return GpuProvisioningResult(ProvisioningStatus.ENVIRONMENT_NOT_MUTABLE, plan.message)


class HostProvisionerAdapter:
    def __init__(self, callback: HostProvisioningCallback | None = None) -> None:
        self.callback = callback

    def can_provision(self, context: GpuRuntimeContext, report: GpuCapabilityReport) -> bool:
        return context.distribution_kind == DistributionKind.EMBEDDED_HOST and bool(self.callback or context.host_can_provision)

    def build_plan(self, context: GpuRuntimeContext, report: GpuCapabilityReport) -> GpuProvisioningPlan:
        return build_gpu_provisioning_plan(report, context)

    def provision(
        self,
        plan: GpuProvisioningPlan,
        progress_callback: ProgressCallback | None = None,
        cancel_token: CancelToken | None = None,
    ) -> GpuProvisioningResult:
        if self.callback is None:
            return GpuProvisioningResult(ProvisioningStatus.ENVIRONMENT_NOT_MUTABLE, "Host provisioning callback is not configured.")
        return self.callback(plan)


def select_provisioner(context: GpuRuntimeContext | None) -> object:
    ctx = ensure_runtime_context(context)
    if ctx.distribution_kind == DistributionKind.EMBEDDED_HOST:
        return HostProvisionerAdapter()
    if ctx.distribution_kind == DistributionKind.SOURCE_MANAGED and ctx.allow_environment_mutation:
        return PythonEnvironmentProvisioner()
    return GuidanceOnlyProvisioner()
