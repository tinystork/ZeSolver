"""Interchangeable GPU provisioners."""

from __future__ import annotations

from collections import deque
import logging
import subprocess
import threading
import time

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


def _format_command(command: tuple[str, ...]) -> str:
    return " ".join(str(part) for part in command)


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
            logging.info("GPU_PROVISIONING_START command=%s shell=false", _format_command(tuple(plan.command)))
            proc = subprocess.Popen(
                list(plan.command),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                shell=False,
            )
            logging.info("GPU_PROVISIONING_PROCESS_STARTED pid=%s", getattr(proc, "pid", None))
            started = time.monotonic()
            output_tail: deque[str] = deque(maxlen=400)
            output_lock = threading.Lock()

            def _reader() -> None:
                stream = proc.stdout
                if stream is None:
                    return
                while True:
                    chunk = stream.read(4096)
                    if chunk == "":
                        break
                    with output_lock:
                        output_tail.append(chunk.rstrip("\r\n"))
                    if progress_callback:
                        progress_callback(chunk.rstrip("\r\n"))

            reader = threading.Thread(target=_reader, name="zesolver-gpu-provision-output", daemon=True)
            reader.start()
            while proc.poll() is None:
                if cancel_token and cancel_token():
                    logging.info("GPU_PROVISIONING_CANCEL_REQUESTED pid=%s", getattr(proc, "pid", None))
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        logging.warning("GPU_PROVISIONING_CANCEL_KILL pid=%s", getattr(proc, "pid", None))
                        proc.kill()
                        proc.wait(timeout=10)
                    reader.join(timeout=5)
                    with output_lock:
                        stdout = "\n".join(output_tail)
                    logging.info("GPU_PROVISIONING_CANCELLED returncode=%s", proc.returncode)
                    return GpuProvisioningResult(
                        ProvisioningStatus.CANCELLED,
                        "GPU installation cancelled.",
                        returncode=proc.returncode,
                        stdout_tail=_tail(stdout),
                        stderr_tail="",
                    )
                if time.monotonic() - started > timeout_s:
                    logging.warning("GPU_PROVISIONING_TIMEOUT pid=%s timeout_s=%s", getattr(proc, "pid", None), timeout_s)
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=10)
                    reader.join(timeout=5)
                    with output_lock:
                        stdout = "\n".join(output_tail)
                    logging.warning("GPU_PROVISIONING_TIMEOUT_END returncode=%s", proc.returncode)
                    return GpuProvisioningResult(
                        ProvisioningStatus.INSTALL_FAILED,
                        "GPU installation timed out.",
                        returncode=proc.returncode,
                        stdout_tail=_tail(stdout),
                        stderr_tail="",
                    )
                time.sleep(0.1)
            proc.wait()
            reader.join(timeout=5)
            with output_lock:
                stdout = "\n".join(output_tail)
        except Exception as exc:
            logging.warning("GPU_PROVISIONING_EXCEPTION error=%s", exc)
            return GpuProvisioningResult(ProvisioningStatus.INSTALL_FAILED, str(exc))
        logging.info("GPU_PROVISIONING_END returncode=%s", proc.returncode)
        if proc.returncode != 0:
            logging.warning("GPU_PROVISIONING_INSTALL_FAILED returncode=%s", proc.returncode)
            return GpuProvisioningResult(
                ProvisioningStatus.INSTALL_FAILED,
                "INSTALL_FAILED: GPU package installation failed.",
                returncode=proc.returncode,
                stdout_tail=_tail(stdout),
                stderr_tail="",
            )
        return GpuProvisioningResult(
            ProvisioningStatus.INSTALLED_RESTART_REQUIRED,
            "Installation finished. Restart ZeSolver to activate GPU acceleration.",
            returncode=0,
            stdout_tail=_tail(stdout),
            stderr_tail="",
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
