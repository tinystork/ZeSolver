from __future__ import annotations

import importlib.metadata
import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from zesolver.gpu_diagnostic import main as gpu_diag_main
from zesolver.gpu_support import (
    CapabilityState,
    DistributionKind,
    EffectiveBackend,
    GpuRuntimeContext,
    ProbeHooks,
    ProvisioningStatus,
    ReasonCode,
    build_gpu_provisioning_plan,
    probe_gpu_capability,
    run_cupy_self_test,
)
from zesolver.gpu_support.provisioning import HostProvisionerAdapter, PythonEnvironmentProvisioner
from zeblindsolver import metadata_solver as ms
from zesolver.resource_telemetry import BatchResourceTelemetry


def _missing_package(_name: str) -> str:
    raise importlib.metadata.PackageNotFoundError


def _hooks(
    *,
    system: str = "Linux",
    machine: str = "x86_64",
    versions=None,
    which=None,
    run=None,
    import_module=None,
) -> ProbeHooks:
    versions = versions or _missing_package
    return ProbeHooks(
        platform_system=lambda: system,
        machine=lambda: machine,
        package_version=versions,
        which=(which or (lambda _name: None)),
        run=(run or subprocess.run),
        import_module=(import_module or __import__),
    )


def test_gpu_probe_macos_is_cpu_only_and_not_provisionable() -> None:
    ctx = GpuRuntimeContext(DistributionKind.SOURCE_MANAGED, allow_environment_mutation=True)
    report = probe_gpu_capability(ctx, hooks=_hooks(system="Darwin"), run_self_test=False)
    plan = build_gpu_provisioning_plan(report, ctx)

    assert report.reason_code == ReasonCode.CUDA_UNSUPPORTED_ON_PLATFORM
    assert report.effective_backend == EffectiveBackend.CPU
    assert plan.command == ()


def test_gpu_probe_nvidia_without_cupy_can_build_source_plan() -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout="RTX 4080, 555.42\n", stderr="")

    ctx = GpuRuntimeContext(DistributionKind.SOURCE_MANAGED, allow_environment_mutation=True, python_executable="/venv/bin/python")
    report = probe_gpu_capability(
        ctx,
        hooks=_hooks(which=lambda name: "/usr/bin/nvidia-smi", run=fake_run),
        run_self_test=False,
    )
    plan = build_gpu_provisioning_plan(report, ctx)

    assert report.nvidia_gpu_state == CapabilityState.AVAILABLE
    assert report.reason_code == ReasonCode.CUPY_NOT_INSTALLED
    assert plan.status == ProvisioningStatus.AVAILABLE
    assert plan.command == ("/venv/bin/python", "-m", "pip", "install", "cupy-cuda12x[ctk]")
    assert plan.requires_consent is True


def test_gpu_probe_frozen_standalone_never_builds_pip_plan() -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout="RTX 4080, 555.42\n", stderr="")

    ctx = GpuRuntimeContext(DistributionKind.FROZEN_STANDALONE, allow_environment_mutation=False, python_executable="/app/ZeSolver")
    report = probe_gpu_capability(
        ctx,
        hooks=_hooks(which=lambda name: "/usr/bin/nvidia-smi", run=fake_run),
        run_self_test=False,
    )
    plan = build_gpu_provisioning_plan(report, ctx)

    assert report.reason_code == ReasonCode.CUPY_NOT_INSTALLED
    assert plan.command == ()
    assert plan.status in {ProvisioningStatus.GUIDANCE_ONLY, ProvisioningStatus.ENVIRONMENT_NOT_MUTABLE}


def test_gpu_probe_embedded_host_reports_without_mutating() -> None:
    ctx = GpuRuntimeContext(DistributionKind.EMBEDDED_HOST, allow_environment_mutation=False, host_name="ZeMosaic")
    report = probe_gpu_capability(ctx, hooks=_hooks(system="Linux"), run_self_test=False)
    plan = build_gpu_provisioning_plan(report, ctx)

    assert report.effective_backend == EffectiveBackend.CPU
    assert plan.command == ()


def test_host_adapter_delegates_to_host_callback() -> None:
    ctx = GpuRuntimeContext(DistributionKind.EMBEDDED_HOST, host_can_provision=True)
    calls = []

    def callback(plan):
        calls.append(plan)
        from zesolver.gpu_support import GpuProvisioningResult

        return GpuProvisioningResult(ProvisioningStatus.GUIDANCE_ONLY, "host handled")

    adapter = HostProvisionerAdapter(callback)
    report = probe_gpu_capability(ctx, hooks=_hooks(system="Linux"), run_self_test=False)
    plan = build_gpu_provisioning_plan(report, ctx)
    result = adapter.provision(plan)

    assert calls == [plan]
    assert result.message == "host handled"


def test_gpu_probe_detects_cupy_package_conflict() -> None:
    def versions(name: str) -> str:
        if name in {"cupy", "cupy-cuda12x"}:
            return "13.0"
        raise importlib.metadata.PackageNotFoundError

    report = probe_gpu_capability(GpuRuntimeContext(), hooks=_hooks(versions=versions), run_self_test=False)

    assert report.cupy_package_state == CapabilityState.CONFLICT
    assert report.reason_code == ReasonCode.CUPY_PACKAGE_CONFLICT


def test_cupy_self_test_success_with_fake_runtime() -> None:
    class FakeDevice:
        def __init__(self, _dev):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeArray:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)

        def get(self):
            return self.values

    class FakeCupy:
        __version__ = "14.1"

        float32 = np.float32
        cuda = SimpleNamespace(Device=FakeDevice, Stream=SimpleNamespace(null=SimpleNamespace(synchronize=lambda: None)))

        @staticmethod
        def arange(*args, **kwargs):
            return np.arange(*args, **kwargs)

        @staticmethod
        def sum(arr):
            return FakeArray([np.sum(arr)])

        @staticmethod
        def get_default_memory_pool():
            return SimpleNamespace(free_all_blocks=lambda: None)

    class FakeRuntime:
        @staticmethod
        def getDeviceCount():
            return 1

        @staticmethod
        def getDeviceProperties(_dev):
            return {"name": b"Fake GPU"}

    def importer(name: str):
        if name == "cupy":
            return FakeCupy
        if name == "cupy.cuda.runtime":
            return FakeRuntime
        raise ImportError(name)

    result = run_cupy_self_test(import_module=importer)

    assert result["state"] == CapabilityState.AVAILABLE
    assert result["reason_code"] == ReasonCode.GPU_READY
    assert result["device_count"] == 1


def test_python_environment_provisioner_success_requires_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProc:
        returncode = 0

        def __init__(self, *args, **kwargs):
            self.args = args
            self._polled = False

        def poll(self):
            if self._polled:
                return 0
            self._polled = True
            return None

        def communicate(self, timeout=None):
            self.returncode = 0
            return "installed", ""

        def terminate(self):
            self.returncode = -15

    monkeypatch.setattr(subprocess, "Popen", FakeProc)
    plan = build_gpu_provisioning_plan(
        probe_gpu_capability(
            GpuRuntimeContext(DistributionKind.SOURCE_MANAGED, True, "/py"),
            hooks=_hooks(which=lambda name: "/usr/bin/nvidia-smi", run=lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout="GPU, 555\n", stderr="")),
            run_self_test=False,
        ),
        GpuRuntimeContext(DistributionKind.SOURCE_MANAGED, True, "/py"),
    )

    result = PythonEnvironmentProvisioner().provision(plan, timeout_s=5)

    assert result.status == ProvisioningStatus.INSTALLED_RESTART_REQUIRED
    assert result.restart_required is True


def test_gpu_diagnostic_json_cli(capsys: pytest.CaptureFixture[str]) -> None:
    code = gpu_diag_main(["--json", "--distribution-kind", "unknown"])
    data = json.loads(capsys.readouterr().out)

    assert code == 0
    assert data["schema"] == "zesolver.gpu_diagnostic.v1"
    assert data["effective_backend"] in {"cpu", "cuda"}


@pytest.fixture(autouse=True)
def _reset_gpu_state():
    ms.reset_zenear_gpu_runtime_state()
    yield
    ms.reset_zenear_gpu_runtime_state()


def _image() -> np.ndarray:
    data = np.ones((80, 80), dtype=np.float32) * 1000.0
    yy, xx = np.ogrid[:80, :80]
    data += 5000.0 * np.exp(-((xx - 40) ** 2 + (yy - 36) ** 2) / (2.0 * 2.0**2)).astype(np.float32)
    return data


def test_auto_without_cupy_disables_gpu_once_for_complete_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"probe": 0}

    def missing_cupy(_device=None):
        calls["probe"] += 1
        raise ModuleNotFoundError("No module named 'cupy'")

    monkeypatch.setattr(ms, "_cuda_runtime_probe", missing_cupy)
    telemetry = BatchResourceTelemetry()
    token = __import__("zesolver.resource_telemetry", fromlist=["set_active_batch_telemetry"]).set_active_batch_telemetry(telemetry)
    try:
        first = ms.detect_stars_astap_strict(_image(), backend="auto", bin_factor=1, max_stars=10)
        second = ms.detect_stars_astap_strict(_image(), backend="auto", bin_factor=1, max_stars=10)
        summary = telemetry.near_detection_summary(terminal_status="completed")
    finally:
        __import__("zesolver.resource_telemetry", fromlist=["reset_active_batch_telemetry"]).reset_active_batch_telemetry(token)

    assert calls["probe"] == 1
    assert first.backend_used == "cpu"
    assert first.fallback_used is True
    assert first.fallback_reason == "CUPY_NOT_INSTALLED"
    assert second.backend_used == "cpu"
    assert second.fallback_used is False
    assert summary["fallbacks"] == 1
    assert summary["gpu_disabled_for_batch"] is True
    assert summary["gpu_disabled_reason"] == "CUPY_NOT_INSTALLED"
