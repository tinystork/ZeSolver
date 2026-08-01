from __future__ import annotations

import os
import threading
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtCore = pytest.importorskip("PySide6.QtCore")
QtGui = pytest.importorskip("PySide6.QtGui")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from zesolver.gpu_support import GpuProvisioningPlan, GpuProvisioningResult, ProvisioningStatus
from zesolver.gui_startup_wizard import StartupAstapProbe, StartupCatalogProbe, StartupGpuProvisionWorker, StartupWizardDecision, ZeSolverStartupWizard
from zesolver.settings_store import PersistentSettings

_QT_APP = None


@pytest.fixture()
def qt_app():
    global _QT_APP
    instance = QtWidgets.QApplication.instance()
    if instance is not None and not isinstance(instance, QtWidgets.QApplication):
        pytest.skip("A non-widget Qt application is already active")
    _QT_APP = instance or _QT_APP or QtWidgets.QApplication([])
    return _QT_APP


def _wait_until(app, predicate, timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            app.processEvents()
            return True
        time.sleep(0.01)
    app.processEvents()
    return bool(predicate())


def _decision() -> StartupWizardDecision:
    return StartupWizardDecision(
        True,
        "fresh",
        StartupCatalogProbe("none"),
        StartupAstapProbe("none"),
        "test",
        False,
        0,
    )


def _plan() -> GpuProvisioningPlan:
    return GpuProvisioningPlan(
        ProvisioningStatus.AVAILABLE,
        None,
        command=("python", "-m", "pip", "install", "cupy-cuda12x[ctk]"),
    )


def _wizard(qt_app):
    saved: list[PersistentSettings] = []
    dialog = ZeSolverStartupWizard(
        settings=PersistentSettings(),
        decision=_decision(),
        save_settings=saved.append,
    )
    dialog._gpu_plan = _plan()
    return dialog, saved


def test_gpu_worker_result_ready_does_not_mask_native_qthread_finished(qt_app, monkeypatch: pytest.MonkeyPatch) -> None:
    import zesolver.gui_startup_wizard as wizard_module

    class FakeProvisioner:
        def provision(self, plan, progress_callback=None, cancel_token=None):
            if progress_callback:
                progress_callback("fake pip output")
            return GpuProvisioningResult(
                ProvisioningStatus.INSTALLED_RESTART_REQUIRED,
                "installed",
                restart_required=True,
            )

    monkeypatch.setattr(wizard_module, "PythonEnvironmentProvisioner", lambda: FakeProvisioner())
    worker = StartupGpuProvisionWorker(_plan())
    results: list[GpuProvisioningResult] = []
    native_finished: list[bool] = []

    worker.resultReady.connect(results.append)
    worker.finished.connect(lambda: native_finished.append(True))
    worker.start()

    assert _wait_until(qt_app, lambda: bool(native_finished), timeout_s=3.0)
    assert results and results[0].status == ProvisioningStatus.INSTALLED_RESTART_REQUIRED
    assert native_finished == [True]


def test_gpu_result_ready_keeps_worker_until_native_thread_finished(qt_app) -> None:
    dialog, saved = _wizard(qt_app)

    class StillRunningWorker:
        def isRunning(self):
            return True

        def request_cancel(self):
            pass

    worker = StillRunningWorker()
    dialog._gpu_worker = worker  # type: ignore[assignment]
    dialog._set_gpu_provisioning_active(True)

    dialog._on_gpu_provision_result_ready(
        GpuProvisioningResult(
            ProvisioningStatus.INSTALLED_RESTART_REQUIRED,
            "Installation finished.",
            restart_required=True,
        )
    )

    assert dialog._gpu_worker is worker
    assert dialog._gpu_provisioning_active is True
    assert saved[-1].gpu_restart_required is True
    assert "Relancez ZeSolver" in dialog.gpu_detail_view.toPlainText()

    dialog._on_gpu_worker_thread_finished()

    assert dialog._gpu_worker is None
    assert dialog._gpu_provisioning_active is False


def test_gpu_provision_success_cleans_worker_after_native_finished(qt_app, monkeypatch: pytest.MonkeyPatch) -> None:
    import zesolver.gui_startup_wizard as wizard_module

    class FakeProvisioner:
        def provision(self, plan, progress_callback=None, cancel_token=None):
            if progress_callback:
                progress_callback("install ok")
            return GpuProvisioningResult(
                ProvisioningStatus.INSTALLED_RESTART_REQUIRED,
                "Installation terminee.",
                restart_required=True,
            )

    monkeypatch.setattr(wizard_module, "PythonEnvironmentProvisioner", lambda: FakeProvisioner())
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *_args, **_kwargs: QtWidgets.QMessageBox.StandardButton.Yes)
    dialog, saved = _wizard(qt_app)

    dialog._install_gpu_support()

    assert _wait_until(qt_app, lambda: dialog._gpu_worker is None, timeout_s=3.0)
    assert saved[-1].gpu_restart_required is True
    assert "Relancez ZeSolver" in dialog.gpu_detail_view.toPlainText()
    assert "Operation GPU terminee." in dialog.gpu_detail_view.toPlainText()


def test_gpu_provision_error_is_displayed_without_crash(qt_app, monkeypatch: pytest.MonkeyPatch) -> None:
    import zesolver.gui_startup_wizard as wizard_module

    class FakeProvisioner:
        def provision(self, plan, progress_callback=None, cancel_token=None):
            if progress_callback:
                progress_callback("pip failed")
            return GpuProvisioningResult(
                ProvisioningStatus.INSTALL_FAILED,
                "INSTALL_FAILED: pip failed",
                returncode=7,
            )

    monkeypatch.setattr(wizard_module, "PythonEnvironmentProvisioner", lambda: FakeProvisioner())
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *_args, **_kwargs: QtWidgets.QMessageBox.StandardButton.Yes)
    dialog, saved = _wizard(qt_app)

    dialog._install_gpu_support()

    assert _wait_until(qt_app, lambda: dialog._gpu_worker is None, timeout_s=3.0)
    assert saved[-1].gpu_user_cpu_selected is True
    assert "INSTALL_FAILED" in dialog.gpu_status_label.text()
    assert "pip failed" in dialog.gpu_detail_view.toPlainText()


def test_gpu_provision_cancel_stops_worker_before_reject(qt_app, monkeypatch: pytest.MonkeyPatch) -> None:
    import zesolver.gui_startup_wizard as wizard_module

    started = threading.Event()

    class SlowProvisioner:
        def provision(self, plan, progress_callback=None, cancel_token=None):
            started.set()
            while not (cancel_token and cancel_token()):
                time.sleep(0.01)
            if progress_callback:
                progress_callback("cancelled")
            return GpuProvisioningResult(ProvisioningStatus.CANCELLED, "GPU installation cancelled.")

    monkeypatch.setattr(wizard_module, "PythonEnvironmentProvisioner", lambda: SlowProvisioner())
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *_args, **_kwargs: QtWidgets.QMessageBox.StandardButton.Yes)
    dialog, saved = _wizard(qt_app)
    dialog._install_gpu_support()

    assert started.wait(2.0)
    assert dialog._is_gpu_provisioning_running()
    assert not dialog.button(QtWidgets.QWizard.NextButton).isEnabled()

    dialog.reject()

    assert _wait_until(qt_app, lambda: dialog._gpu_worker is None, timeout_s=3.0)
    assert saved[-1].gpu_user_cpu_selected is True
    assert "cancelled" in dialog.gpu_detail_view.toPlainText()


def test_gpu_close_event_ignores_when_worker_does_not_stop(qt_app) -> None:
    dialog, _saved = _wizard(qt_app)

    class StuckWorker:
        cancel_requested = False

        def isRunning(self):
            return True

        def request_cancel(self):
            self.cancel_requested = True

        def wait(self, _timeout):
            return False

    worker = StuckWorker()
    dialog._gpu_worker = worker  # type: ignore[assignment]
    event = QtGui.QCloseEvent()

    dialog.closeEvent(event)

    assert worker.cancel_requested is True
    assert not event.isAccepted()
    assert dialog._gpu_worker is worker
