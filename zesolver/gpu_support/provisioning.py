"""Interchangeable GPU provisioners."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import subprocess
import threading
import time

from .models import (
    CancelToken,
    DistributionKind,
    GpuCapabilityReport,
    GpuProvisioningPlan,
    GpuProvisioningProgress,
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


def _emit_progress(progress_callback: ProgressCallback | None, phase: str, message: str, line: str = "") -> None:
    if progress_callback:
        progress_callback(GpuProvisioningProgress(phase, message, line))


@dataclass(frozen=True)
class _CommandResult:
    returncode: int | None
    stdout_tail: str
    stderr_tail: str
    timed_out: bool = False
    cancelled: bool = False


def _pip_phase_for_line(line: str, current_phase: str) -> str:
    text = line.lower()
    if any(token in text for token in ("collecting ", "downloading ", "obtaining ", "metadata")):
        return "download"
    if any(token in text for token in ("installing collected packages", "building wheel", "successfully installed")):
        return "install"
    return current_phase


def _run_command_streaming(
    command: tuple[str, ...],
    *,
    phase: str,
    progress_callback: ProgressCallback | None,
    cancel_token: CancelToken | None,
    timeout_s: float,
    phase_from_line: object | None = None,
    env: dict[str, str] | None = None,
) -> _CommandResult:
    logging.info("GPU_PROVISIONING_COMMAND_START phase=%s command=%s shell=false", phase, _format_command(command))
    proc = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        bufsize=0,
        shell=False,
        env=env,
    )
    logging.info("GPU_PROVISIONING_PROCESS_STARTED phase=%s pid=%s", phase, getattr(proc, "pid", None))
    output_tail: deque[str] = deque(maxlen=400)
    output_lock = threading.Lock()
    current_phase = phase

    def _reader() -> None:
        nonlocal current_phase
        stream = proc.stdout
        if stream is None:
            return
        fd = stream.fileno()
        pending = ""
        while True:
            chunk = os.read(fd, 8192)
            if not chunk:
                break
            text = chunk.decode(errors="replace")
            pieces = text.splitlines(keepends=True)
            for piece in pieces:
                pending += piece
                if pending.endswith(("\n", "\r")):
                    line = pending.rstrip("\r\n")
                    pending = ""
                    if callable(phase_from_line):
                        current_phase = str(phase_from_line(line, current_phase))
                    with output_lock:
                        output_tail.append(line)
                    _emit_progress(progress_callback, current_phase, line or current_phase, line)
                elif len(pending) >= 4096:
                    line = pending
                    pending = ""
                    if callable(phase_from_line):
                        current_phase = str(phase_from_line(line, current_phase))
                    with output_lock:
                        output_tail.append(line)
                    _emit_progress(progress_callback, current_phase, line, line)
        if pending:
            line = pending.rstrip("\r\n")
            if callable(phase_from_line):
                current_phase = str(phase_from_line(line, current_phase))
            with output_lock:
                output_tail.append(line)
            _emit_progress(progress_callback, current_phase, line or current_phase, line)

    reader = threading.Thread(target=_reader, name=f"zesolver-gpu-{phase}-output", daemon=True)
    reader.start()
    started = time.monotonic()
    while proc.poll() is None:
        if cancel_token and cancel_token():
            logging.info("GPU_PROVISIONING_CANCEL_REQUESTED phase=%s pid=%s", phase, getattr(proc, "pid", None))
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logging.warning("GPU_PROVISIONING_CANCEL_KILL phase=%s pid=%s", phase, getattr(proc, "pid", None))
                proc.kill()
                proc.wait(timeout=10)
            reader.join(timeout=5)
            with output_lock:
                stdout = "\n".join(output_tail)
            logging.info("GPU_PROVISIONING_CANCELLED phase=%s returncode=%s", phase, proc.returncode)
            return _CommandResult(proc.returncode, _tail(stdout), "", cancelled=True)
        if time.monotonic() - started > timeout_s:
            logging.warning("GPU_PROVISIONING_TIMEOUT phase=%s pid=%s timeout_s=%s", phase, getattr(proc, "pid", None), timeout_s)
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
            reader.join(timeout=5)
            with output_lock:
                stdout = "\n".join(output_tail)
            logging.warning("GPU_PROVISIONING_TIMEOUT_END phase=%s returncode=%s", phase, proc.returncode)
            return _CommandResult(proc.returncode, _tail(stdout), "", timed_out=True)
        time.sleep(0.05)
    proc.wait()
    reader.join(timeout=5)
    with output_lock:
        stdout = "\n".join(output_tail)
    logging.info("GPU_PROVISIONING_COMMAND_END phase=%s returncode=%s", phase, proc.returncode)
    return _CommandResult(proc.returncode, _tail(stdout), "")


def _parse_self_test_report(stdout_tail: str) -> dict[str, object]:
    text = str(stdout_tail or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return {}
    try:
        parsed = json.loads(text[start : end + 1])
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _pip_environment(plan: GpuProvisioningPlan) -> dict[str, str] | None:
    temp_dir = str((plan.technical_details or {}).get("gpu_temp_dir") or "").strip()
    if not temp_dir:
        return None
    path = Path(temp_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["TMPDIR"] = str(path)
    env["TMP"] = str(path)
    env["TEMP"] = str(path)
    return env


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
        pip_check_command: tuple[str, ...] | None = None,
        self_test_command: tuple[str, ...] | None = None,
    ) -> GpuProvisioningResult:
        if plan.status != ProvisioningStatus.AVAILABLE or not plan.command:
            return GpuProvisioningResult(ProvisioningStatus.ENVIRONMENT_NOT_MUTABLE, plan.message)
        package = plan.command[-1]
        if package not in ALLOWED_GPU_PACKAGES:
            return GpuProvisioningResult(ProvisioningStatus.INSTALL_FAILED, "Package is not in the ZeSolver GPU allowlist.")
        if cancel_token and cancel_token():
            return GpuProvisioningResult(ProvisioningStatus.CANCELLED, "GPU installation cancelled before start.")
        _emit_progress(progress_callback, "preparation", "Preparing optional GPU package installation.")
        pip_env = _pip_environment(plan)
        try:
            logging.info("GPU_PROVISIONING_START command=%s shell=false", _format_command(tuple(plan.command)))
            install = _run_command_streaming(
                tuple(plan.command),
                phase="download",
                progress_callback=progress_callback,
                cancel_token=cancel_token,
                timeout_s=timeout_s,
                phase_from_line=_pip_phase_for_line,
                env=pip_env,
            )
        except Exception as exc:
            logging.warning("GPU_PROVISIONING_EXCEPTION error=%s", exc)
            return GpuProvisioningResult(ProvisioningStatus.INSTALL_FAILED, str(exc))
        if install.cancelled:
            return GpuProvisioningResult(
                ProvisioningStatus.CANCELLED,
                "GPU installation cancelled.",
                returncode=install.returncode,
                stdout_tail=install.stdout_tail,
                stderr_tail=install.stderr_tail,
            )
        if install.timed_out:
            return GpuProvisioningResult(
                ProvisioningStatus.INSTALL_FAILED,
                "GPU installation timed out.",
                returncode=install.returncode,
                stdout_tail=install.stdout_tail,
                stderr_tail=install.stderr_tail or install.stdout_tail,
            )
        logging.info("GPU_PROVISIONING_END returncode=%s", install.returncode)
        if install.returncode != 0:
            logging.warning("GPU_PROVISIONING_INSTALL_FAILED returncode=%s", install.returncode)
            return GpuProvisioningResult(
                ProvisioningStatus.INSTALL_FAILED,
                "INSTALL_FAILED: GPU package installation failed.",
                returncode=install.returncode,
                stdout_tail=install.stdout_tail,
                stderr_tail=install.stderr_tail or install.stdout_tail,
            )
        python = str(plan.command[0])
        pip_check = pip_check_command or (python, "-m", "pip", "check")
        _emit_progress(progress_callback, "pip_check", "Running pip check.")
        check = _run_command_streaming(
            tuple(pip_check),
            phase="pip_check",
            progress_callback=progress_callback,
            cancel_token=cancel_token,
            timeout_s=min(timeout_s, 120.0),
            env=pip_env,
        )
        if check.cancelled:
            return GpuProvisioningResult(
                ProvisioningStatus.CANCELLED,
                "GPU installation cancelled during pip check.",
                returncode=check.returncode,
                stdout_tail=check.stdout_tail,
                stderr_tail=check.stderr_tail,
            )
        if check.returncode != 0 or check.timed_out:
            logging.warning("GPU_PROVISIONING_PIP_CHECK_FAILED returncode=%s timeout=%s", check.returncode, check.timed_out)
            return GpuProvisioningResult(
                ProvisioningStatus.INSTALL_FAILED,
                "INSTALL_FAILED: pip check failed after GPU package installation.",
                returncode=check.returncode,
                stdout_tail=_tail(f"{install.stdout_tail}\n{check.stdout_tail}"),
                stderr_tail=check.stderr_tail or check.stdout_tail,
            )
        diagnostic = self_test_command or (python, "-m", "zesolver.gpu_diagnostic", "--json", "--self-test")
        _emit_progress(progress_callback, "self_test", "Running CUDA self-test in a fresh Python process.")
        self_test = _run_command_streaming(
            tuple(diagnostic),
            phase="self_test",
            progress_callback=progress_callback,
            cancel_token=cancel_token,
            timeout_s=min(timeout_s, 180.0),
        )
        report = _parse_self_test_report(self_test.stdout_tail)
        backend = str(report.get("effective_backend") or "")
        reason = str(report.get("reason_code") or "")
        if self_test.cancelled:
            return GpuProvisioningResult(
                ProvisioningStatus.CANCELLED,
                "GPU installation cancelled during CUDA self-test.",
                returncode=self_test.returncode,
                stdout_tail=self_test.stdout_tail,
                stderr_tail=self_test.stderr_tail,
                technical_details={"self_test": report},
            )
        if self_test.returncode != 0 or self_test.timed_out or backend != "cuda" or reason != "GPU_READY":
            logging.warning(
                "GPU_PROVISIONING_SELF_TEST_FAILED returncode=%s timeout=%s backend=%s reason=%s",
                self_test.returncode,
                self_test.timed_out,
                backend,
                reason,
            )
            return GpuProvisioningResult(
                ProvisioningStatus.INSTALL_FAILED,
                "INSTALL_FAILED: CUDA self-test did not pass after installation.",
                returncode=self_test.returncode,
                stdout_tail=_tail(f"{install.stdout_tail}\n{check.stdout_tail}\n{self_test.stdout_tail}"),
                stderr_tail=self_test.stderr_tail or self_test.stdout_tail,
                technical_details={"self_test": report},
            )
        names = report.get("device_names") if isinstance(report.get("device_names"), list) else []
        gpu_name = ", ".join(str(item) for item in names) if names else "-"
        cupy_version = str(report.get("cupy_version") or "-")
        device_count = report.get("device_count")
        _emit_progress(progress_callback, "restart_required", "CUDA_SELF_TEST_OK - Restart required.")
        return GpuProvisioningResult(
            ProvisioningStatus.INSTALLED_RESTART_REQUIRED,
            f"Installation finished. CUDA self-test OK. GPU: {gpu_name}; CuPy: {cupy_version}; devices: {device_count}. Restart ZeSolver to activate GPU acceleration.",
            returncode=0,
            stdout_tail=_tail(f"{install.stdout_tail}\n{check.stdout_tail}\n{self_test.stdout_tail}"),
            stderr_tail="",
            restart_required=True,
            technical_details={"self_test": report},
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
