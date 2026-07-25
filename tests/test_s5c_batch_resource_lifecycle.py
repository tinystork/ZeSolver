from __future__ import annotations

import time
import threading
from pathlib import Path

import numpy as np
from astropy.io import fits

from zesolver.core import EngineSolveResult, SolveRequest, SolveStatus
from zesolver.core.batch import BatchSolverPipeline, BatchSolveRequest
from zesolver.gui_pipeline.pipeline_runner import PipelineGuiRunner
from zesolver.gui_pipeline.requests import GuiSettingsState
from zesolver.gui_pipeline.settings_adapter import build_gui_solve_request
from zesolver.resource_telemetry import increment_batch_counter

from batch_pipeline_fixtures import result_for
from solver_pipeline_fixtures import near_resources, sample_wcs


def _fits(path: Path) -> Path:
    fits.PrimaryHDU(data=np.ones((16, 16), dtype=np.uint16)).writeto(path)
    return path


def _state(tmp_path: Path, resources, *, workers: int = 4) -> GuiSettingsState:
    return GuiSettingsState(
        workers=workers,
        preserve_order=True,
        use_blind=True,
        catalog_resources=resources,
        legacy_config=object(),
    )


def _counters(summary) -> dict[str, int]:
    assert summary.telemetry is not None
    counters = summary.telemetry["counters"]
    assert isinstance(counters, dict)
    return counters


def test_s5c_all_near_never_resolves_or_loads_blind(tmp_path: Path, monkeypatch) -> None:
    files = tuple(_fits(tmp_path / f"{idx}.fit") for idx in range(4))
    resources = near_resources(tmp_path, blind_count=1)

    def fake_near(self, request, *, resources, configuration):
        return EngineSolveResult(status=SolveStatus.SOLVED, backend="NEAR", wcs=sample_wcs())

    def fail_blind(self, request, *, resources, configuration):  # pragma: no cover - should not run
        raise AssertionError("blind must stay lazy for an all-Near batch")

    monkeypatch.setattr("zesolver.core.pipeline.ExistingNearSolverPort.solve", fake_near)
    monkeypatch.setattr("zesolver.core.blind_port.ProductionBlindSolverPort.solve", fail_blind)

    request = build_gui_solve_request(files, _state(tmp_path, resources, workers=6))
    summary = PipelineGuiRunner().run(request)
    counters = _counters(summary)

    assert [item.status for item in summary.results] == ["SOLVED"] * 4
    assert counters["blind_runtime_resolution_count"] == 0
    assert counters["blind_index_payload_load_count"] == 0
    assert counters["blind_kdtree_build_count"] == 0


def test_s5c_mixed_batch_uses_one_shared_blind_runtime(tmp_path: Path, monkeypatch) -> None:
    files = tuple(_fits(tmp_path / f"{idx}.fit") for idx in range(4))
    resources = near_resources(tmp_path, blind_count=1)

    def fake_near(self, request, *, resources, configuration):
        if request.request_id in {"0", "1"}:
            return EngineSolveResult(status=SolveStatus.SOLVED, backend="NEAR", wcs=sample_wcs())
        return EngineSolveResult(status=SolveStatus.UNSOLVED, backend="NEAR", error="false hint")

    def fake_blind(self, request, *, resources, configuration):
        self._runtime_selection(resources, configuration)
        return EngineSolveResult(status=SolveStatus.SOLVED, backend="BLIND4D", wcs=sample_wcs())

    monkeypatch.setattr("zesolver.core.pipeline.ExistingNearSolverPort.solve", fake_near)
    monkeypatch.setattr("zesolver.core.blind_port.ProductionBlindSolverPort.solve", fake_blind)

    emitted: list[object] = []
    request = build_gui_solve_request(files, _state(tmp_path, resources, workers=6))
    summary = PipelineGuiRunner(result_callback=emitted.append).run(request)
    counters = _counters(summary)

    assert [item.backend for item in summary.results] == ["NEAR", "NEAR", "BLIND4D", "BLIND4D"]
    assert counters["blind_runtime_resolution_count"] == 1
    assert counters["blind_index_payload_load_count"] == 0
    assert counters["blind_kdtree_build_count"] == 0
    assert len({item.path for item in emitted}) == 4


def test_s5c_batch_pipeline_is_worker_local_not_file_local() -> None:
    requests = tuple(SolveRequest(Path(f"{idx}.fit"), None, True, request_id=str(idx)) for idx in range(6))
    constructed: list[str] = []

    class Pipeline:
        def __init__(self, phase: str) -> None:
            increment_batch_counter("solver_pipeline_constructor_count")
            self.phase = phase
            constructed.append(phase)

        def solve(self, request: SolveRequest):
            time.sleep(0.01)
            return result_for(request, SolveStatus.SOLVED, "NEAR")

    def factory(phase: str):
        return Pipeline(phase)

    result = BatchSolverPipeline(solver_pipeline_factory=factory).solve(
        BatchSolveRequest(requests=requests, workers=2)
    )

    assert len(result.results) == 6
    assert len(constructed) <= 2
    assert result.telemetry is not None
    assert result.telemetry["counters"]["solver_pipeline_constructor_count"] <= 2


def test_s5c_two_runs_same_process_keep_counters_bounded(tmp_path: Path, monkeypatch) -> None:
    files = tuple(_fits(tmp_path / f"run_{idx}.fit") for idx in range(2))
    resources = near_resources(tmp_path, blind_count=1)

    monkeypatch.setattr(
        "zesolver.core.pipeline.ExistingNearSolverPort.solve",
        lambda self, request, *, resources, configuration: EngineSolveResult(
            status=SolveStatus.SOLVED,
            backend="NEAR",
            wcs=sample_wcs(),
        ),
    )

    request = build_gui_solve_request(files, _state(tmp_path, resources, workers=2))
    first = PipelineGuiRunner().run(request)
    second = PipelineGuiRunner().run(request)

    c1 = _counters(first)
    c2 = _counters(second)
    assert c1["blind_runtime_resolution_count"] == 0
    assert c2["blind_runtime_resolution_count"] == 0
    assert c2["solver_pipeline_constructor_count"] <= c1["solver_pipeline_constructor_count"] + 2


def test_s5c_qt_event_loop_stays_responsive_during_slow_preflight(tmp_path: Path, monkeypatch) -> None:
    QtCore = __import__("PySide6.QtCore", fromlist=["QtCore"])
    QtWidgets = __import__("PySide6.QtWidgets", fromlist=["QtWidgets"])
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    files = tuple(_fits(tmp_path / f"qt_{idx}.fit") for idx in range(2))
    resources = near_resources(tmp_path, blind_count=0)
    ticks = {"count": 0}
    phases: list[str] = []

    def slow_resolve(**_kwargs):
        deadline = time.perf_counter() + 0.35
        while time.perf_counter() < deadline:
            time.sleep(0.01)
        return resources

    monkeypatch.setattr("zesolver.gui_pipeline.pipeline_runner.resolve_catalog_resources", slow_resolve)
    monkeypatch.setattr(
        "zesolver.core.pipeline.ExistingNearSolverPort.solve",
        lambda self, request, *, resources, configuration: EngineSolveResult(
            status=SolveStatus.SOLVED,
            backend="NEAR",
            wcs=sample_wcs(),
        ),
    )

    timer = QtCore.QTimer()
    timer.setInterval(25)
    timer.timeout.connect(lambda: ticks.__setitem__("count", ticks["count"] + 1))
    request = build_gui_solve_request(files, _state(tmp_path, None, workers=1))
    done = threading.Event()
    errors: list[BaseException] = []

    def target() -> None:
        try:
            PipelineGuiRunner(progress_callback=lambda progress: phases.append(progress.current_phase or "")).run(request)
        except BaseException as exc:  # pragma: no cover - assertion aid
            errors.append(exc)
        finally:
            done.set()

    timer.start()
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    deadline = time.perf_counter() + 2.0
    while not done.is_set() and time.perf_counter() < deadline:
        app.processEvents()
        time.sleep(0.005)
    timer.stop()
    thread.join(timeout=1.0)

    assert not errors
    assert done.is_set()
    assert ticks["count"] >= 5
    assert "Préparation de la bibliothèque" in phases
