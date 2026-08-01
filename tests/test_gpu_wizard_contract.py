from __future__ import annotations

from pathlib import Path


def test_startup_wizard_gpu_page_uses_non_gui_service_contract() -> None:
    source = (Path(__file__).resolve().parents[1] / "zesolver" / "gui_startup_wizard.py").read_text(encoding="utf-8")

    assert "def default_gpu_runtime_context()" in source
    assert "ZESOLVER_ALLOW_GPU_PROVISIONING" in source
    assert "self.addPage(self._gpu_page())" in source
    assert "probe_gpu_capability(context)" in source
    assert "build_gpu_provisioning_plan" in source
    assert "PythonEnvironmentProvisioner" in source
    assert "gpu_user_cpu_selected" in source


def test_settings_performance_tab_exposes_gpu_diagnostic_button() -> None:
    source = (Path(__file__).resolve().parents[1] / "zesolver.py").read_text(encoding="utf-8")

    assert "settings_perf_gpu_diag_btn" in source
    assert "def _run_gpu_diagnostic_from_settings" in source
    assert "probe_gpu_capability(context)" in source
