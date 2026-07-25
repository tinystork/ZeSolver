from __future__ import annotations

import threading
import time
from pathlib import Path

from zesolver.core.batch import BatchSolverPipeline, BatchSolveRequest
from zesolver.core.models import SolveRequest, SolveResult, SolveStatus
from zesolver.gui_pipeline.pipeline_runner import PipelineGuiRunner
from zesolver.gui_pipeline.requests import GuiSettingsState
from zesolver.gui_pipeline.settings_adapter import build_gui_solve_request

from batch_pipeline_fixtures import result_for


def _request(name: str) -> SolveRequest:
    return SolveRequest(Path(f"{name}.fit"), None, True, request_id=name)


class _BlockingBlindPipeline:
    def __init__(
        self,
        phase: str,
        statuses: dict[str, SolveStatus],
        *,
        started: dict[str, threading.Event],
        finished: dict[str, threading.Event],
        release: dict[str, threading.Event],
        calls: list[str],
    ) -> None:
        self.phase = phase
        self.statuses = statuses
        self.started = started
        self.finished = finished
        self.release = release
        self.calls = calls

    def solve(self, req: SolveRequest) -> SolveResult:
        request_id = str(req.request_id)
        self.calls.append(f"{self.phase}:{request_id}")
        if self.phase == "near":
            return result_for(req, SolveStatus.UNSOLVED, None, "needs blind")
        self.started[request_id].set()
        self.release[request_id].wait(timeout=5.0)
        status = self.statuses.get(request_id, SolveStatus.SOLVED)
        backend = "BLIND4D" if status is SolveStatus.SOLVED else None
        result = result_for(req, status, backend, None if status is SolveStatus.SOLVED else "blind failed")
        self.finished[request_id].set()
        return result


def _blocking_factory(
    statuses: dict[str, SolveStatus],
    *,
    started: dict[str, threading.Event],
    finished: dict[str, threading.Event],
    release: dict[str, threading.Event],
    calls: list[str],
):
    def _factory(phase: str):
        return _BlockingBlindPipeline(
            phase,
            statuses,
            started=started,
            finished=finished,
            release=release,
            calls=calls,
        )

    return _factory


def test_s5f_blind_result_is_emitted_while_later_blind_work_continues() -> None:
    reqs = tuple(_request(name) for name in ("blind-A", "blind-B", "blind-C"))
    started = {req.request_id: threading.Event() for req in reqs}
    finished = {req.request_id: threading.Event() for req in reqs}
    release = {req.request_id: threading.Event() for req in reqs}
    calls: list[str] = []
    emitted: list[tuple[str | None, int]] = []
    done = threading.Event()
    errors: list[BaseException] = []
    holder: dict[str, object] = {}

    def sink(result: SolveResult, progress) -> None:
        emitted.append((result.request_id, progress.solved + progress.failed + progress.cancelled + progress.skipped))

    def target() -> None:
        try:
            holder["result"] = BatchSolverPipeline(
                solver_pipeline_factory=_blocking_factory(
                    {"blind-A": SolveStatus.SOLVED, "blind-B": SolveStatus.SOLVED, "blind-C": SolveStatus.SOLVED},
                    started=started,
                    finished=finished,
                    release=release,
                    calls=calls,
                ),
                progress_sink=sink,
            ).solve(BatchSolveRequest(requests=reqs, workers=1))
        except BaseException as exc:  # pragma: no cover - assertion aid
            errors.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    assert started["blind-A"].wait(timeout=2.0)
    release["blind-A"].set()
    assert finished["blind-A"].wait(timeout=2.0)

    deadline = time.perf_counter() + 2.0
    while not emitted and time.perf_counter() < deadline:
        time.sleep(0.01)

    assert emitted == [("blind-A", 1)]
    assert not finished["blind-B"].is_set()
    assert not done.is_set()

    release["blind-B"].set()
    release["blind-C"].set()
    thread.join(timeout=5.0)
    assert not errors
    assert done.is_set()
    assert [item[0] for item in emitted] == ["blind-A", "blind-B", "blind-C"]
    result = holder["result"]
    assert result.telemetry is not None
    phases = [item["phase"] for item in result.telemetry["events"]]
    assert "blind_result_ready" in phases
    assert "blind_result_emitted" in phases
    assert all(
        float(item["emit_lag_s"]) >= 0.0
        for item in result.telemetry["events"]
        if item["phase"] == "blind_result_emitted"
    )


def test_s5f_blind_success_and_failure_emit_immediately_without_duplicates() -> None:
    reqs = tuple(_request(name) for name in ("A", "B", "C"))
    script = {"A": SolveStatus.SOLVED, "B": SolveStatus.FAILED, "C": SolveStatus.SOLVED}
    emitted: list[str | None] = []

    class Pipeline:
        def __init__(self, phase: str) -> None:
            self.phase = phase

        def solve(self, req: SolveRequest) -> SolveResult:
            if self.phase == "near":
                return result_for(req, SolveStatus.UNSOLVED, None, "needs blind")
            status = script[str(req.request_id)]
            return result_for(req, status, "BLIND4D" if status is SolveStatus.SOLVED else None, None)

    BatchSolverPipeline(
        solver_pipeline_factory=lambda phase: Pipeline(phase),
        progress_sink=lambda result, _progress: emitted.append(result.request_id),
    ).solve(BatchSolveRequest(requests=reqs, workers=1))

    assert emitted == ["A", "B", "C"]
    assert len(emitted) == len(set(emitted))


def test_s5f_mixed_batch_progress_counts_near_then_each_blind_result() -> None:
    reqs = tuple(_request(name) for name in ("near-1", "near-2", "blind-1", "blind-2", "blind-3"))
    emitted: list[tuple[str | None, int]] = []

    class Pipeline:
        def __init__(self, phase: str) -> None:
            self.phase = phase

        def solve(self, req: SolveRequest) -> SolveResult:
            request_id = str(req.request_id)
            if self.phase == "near" and request_id.startswith("near"):
                return result_for(req, SolveStatus.SOLVED, "NEAR")
            if self.phase == "near":
                return result_for(req, SolveStatus.UNSOLVED, None, "needs blind")
            return result_for(req, SolveStatus.SOLVED, "BLIND4D")

    def sink(result: SolveResult, progress) -> None:
        completed = progress.solved + progress.failed + progress.cancelled + progress.skipped
        emitted.append((result.request_id, completed))

    BatchSolverPipeline(solver_pipeline_factory=lambda phase: Pipeline(phase), progress_sink=sink).solve(
        BatchSolveRequest(requests=reqs, workers=1)
    )

    assert [item[1] for item in emitted] == [1, 2, 3, 4, 5]
    assert {item[0] for item in emitted[:2]} == {"near-1", "near-2"}
    assert [item[0] for item in emitted[2:]] == ["blind-1", "blind-2", "blind-3"]


def test_s5f_callbacks_use_completion_order_but_final_result_preserves_input_order() -> None:
    reqs = tuple(_request(name) for name in ("A", "B", "C"))
    started = {req.request_id: threading.Event() for req in reqs}
    finished = {req.request_id: threading.Event() for req in reqs}
    release = {req.request_id: threading.Event() for req in reqs}
    calls: list[str] = []
    emitted: list[str | None] = []
    done = threading.Event()
    holder: dict[str, object] = {}

    def target() -> None:
        holder["result"] = BatchSolverPipeline(
            solver_pipeline_factory=_blocking_factory(
                {"A": SolveStatus.SOLVED, "B": SolveStatus.SOLVED, "C": SolveStatus.SOLVED},
                started=started,
                finished=finished,
                release=release,
                calls=calls,
            ),
            progress_sink=lambda result, _progress: emitted.append(result.request_id),
        ).solve(BatchSolveRequest(requests=reqs, workers=1, preserve_order=True))
        done.set()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    assert started["A"].wait(timeout=2.0)
    release["A"].set()
    assert started["B"].wait(timeout=2.0)
    release["B"].set()
    assert started["C"].wait(timeout=2.0)
    release["C"].set()
    thread.join(timeout=5.0)

    assert done.is_set()
    assert emitted == ["A", "B", "C"]
    result = holder["result"]
    assert [item.request_id for item in result.results] == ["A", "B", "C"]


def test_s5f_stop_during_blind_keeps_first_result_and_cancels_rest_once() -> None:
    reqs = tuple(_request(name) for name in ("A", "B", "C"))
    emitted: list[tuple[str | None, SolveStatus]] = []
    cancel = threading.Event()

    class Pipeline:
        def __init__(self, phase: str) -> None:
            self.phase = phase

        def solve(self, req: SolveRequest) -> SolveResult:
            if self.phase == "near":
                return result_for(req, SolveStatus.UNSOLVED, None, "needs blind")
            return result_for(req, SolveStatus.SOLVED, "BLIND4D")

    def sink(result: SolveResult, _progress) -> None:
        emitted.append((result.request_id, result.status))
        if result.request_id == "A":
            cancel.set()

    result = BatchSolverPipeline(solver_pipeline_factory=lambda phase: Pipeline(phase), progress_sink=sink).solve(
        BatchSolveRequest(requests=reqs, workers=1, cancel_token=cancel)
    )

    assert emitted[0] == ("A", SolveStatus.SOLVED)
    assert len({item[0] for item in emitted}) == len(emitted)
    assert [item.request_id for item in result.results] == ["A", "B", "C"]
    assert any(item.status is SolveStatus.CANCELLED for item in result.results[1:])


def test_s5f_stop_on_error_emits_failure_immediately_and_backfills_once() -> None:
    reqs = tuple(_request(name) for name in ("A", "B", "C"))
    emitted: list[tuple[str | None, SolveStatus]] = []

    class Pipeline:
        def __init__(self, phase: str) -> None:
            self.phase = phase

        def solve(self, req: SolveRequest) -> SolveResult:
            if self.phase == "near":
                return result_for(req, SolveStatus.UNSOLVED, None, "needs blind")
            status = SolveStatus.FAILED if req.request_id == "A" else SolveStatus.SOLVED
            return result_for(req, status, "BLIND4D" if status is SolveStatus.SOLVED else None, "blind failed")

    result = BatchSolverPipeline(
        solver_pipeline_factory=lambda phase: Pipeline(phase),
        progress_sink=lambda solve_result, _progress: emitted.append((solve_result.request_id, solve_result.status)),
    ).solve(BatchSolveRequest(requests=reqs, workers=1, stop_on_error=True))

    assert emitted[0] == ("A", SolveStatus.FAILED)
    assert len({item[0] for item in emitted}) == len(emitted)
    assert [item.request_id for item in result.results] == ["A", "B", "C"]
    assert all(item.status in {SolveStatus.FAILED, SolveStatus.CANCELLED} for item in result.results)


def test_s5f_pipeline_gui_runner_forwards_blind_result_before_run_returns(tmp_path: Path) -> None:
    files = tuple(tmp_path / f"{name}.fit" for name in ("A", "B", "C"))
    request = build_gui_solve_request(files, GuiSettingsState(workers=1, preserve_order=True))
    started = {"0": threading.Event(), "1": threading.Event(), "2": threading.Event()}
    release = {"0": threading.Event(), "1": threading.Event(), "2": threading.Event()}
    callbacks: list[str] = []
    progress_completed: list[int] = []
    done = threading.Event()
    errors: list[BaseException] = []

    class Pipeline:
        def __init__(self, phase: str) -> None:
            self.phase = phase

        def solve(self, req: SolveRequest) -> SolveResult:
            request_id = str(req.request_id)
            if self.phase == "near":
                return result_for(req, SolveStatus.UNSOLVED, None, "needs blind")
            started[request_id].set()
            release[request_id].wait(timeout=5.0)
            return result_for(req, SolveStatus.SOLVED, "BLIND4D")

    def target() -> None:
        try:
            PipelineGuiRunner(
                solver_pipeline_factory=lambda phase, _request: Pipeline(phase),
                result_callback=lambda item: callbacks.append(Path(item.path).stem),
                progress_callback=lambda progress: progress_completed.append(progress.completed),
            ).run(request)
        except BaseException as exc:  # pragma: no cover - assertion aid
            errors.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    assert started["0"].wait(timeout=2.0)
    release["0"].set()
    deadline = time.perf_counter() + 2.0
    while not callbacks and time.perf_counter() < deadline:
        time.sleep(0.01)

    assert callbacks == ["A"]
    assert progress_completed[-1] == 1
    assert not done.is_set()

    release["1"].set()
    release["2"].set()
    thread.join(timeout=5.0)
    assert not errors
    assert callbacks == ["A", "B", "C"]


def test_s5f_qt_offscreen_receives_blind_result_while_batch_continues(tmp_path: Path) -> None:
    QtCore = __import__("PySide6.QtCore", fromlist=["QtCore"])
    QtWidgets = __import__("PySide6.QtWidgets", fromlist=["QtWidgets"])
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    files = tuple(tmp_path / f"{name}.fit" for name in ("A", "B", "C"))
    request = build_gui_solve_request(files, GuiSettingsState(workers=1, preserve_order=True))
    started = {"0": threading.Event(), "1": threading.Event(), "2": threading.Event()}
    release = {"0": threading.Event(), "1": threading.Event(), "2": threading.Event()}
    callbacks: list[str] = []
    completions: list[int] = []
    ticks = {"count": 0}
    done = threading.Event()
    errors: list[BaseException] = []

    class Pipeline:
        def __init__(self, phase: str) -> None:
            self.phase = phase

        def solve(self, req: SolveRequest) -> SolveResult:
            request_id = str(req.request_id)
            if self.phase == "near":
                return result_for(req, SolveStatus.UNSOLVED, None, "needs blind")
            started[request_id].set()
            release[request_id].wait(timeout=5.0)
            return result_for(req, SolveStatus.SOLVED, "BLIND4D")

    timer = QtCore.QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: ticks.__setitem__("count", ticks["count"] + 1))

    def target() -> None:
        try:
            PipelineGuiRunner(
                solver_pipeline_factory=lambda phase, _request: Pipeline(phase),
                result_callback=lambda item: callbacks.append(Path(item.path).stem),
                progress_callback=lambda progress: completions.append(progress.completed),
            ).run(request)
        except BaseException as exc:  # pragma: no cover - assertion aid
            errors.append(exc)
        finally:
            done.set()

    timer.start()
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    assert started["0"].wait(timeout=2.0)
    release["0"].set()
    deadline = time.perf_counter() + 2.0
    while not callbacks and time.perf_counter() < deadline:
        app.processEvents()
        time.sleep(0.005)

    assert callbacks == ["A"]
    assert completions[-1] == 1
    tick_deadline = time.perf_counter() + 0.25
    while ticks["count"] == 0 and time.perf_counter() < tick_deadline:
        app.processEvents()
        time.sleep(0.005)
    assert ticks["count"] > 0
    assert not done.is_set()

    release["1"].set()
    release["2"].set()
    deadline = time.perf_counter() + 5.0
    while not done.is_set() and time.perf_counter() < deadline:
        app.processEvents()
        time.sleep(0.005)
    timer.stop()
    thread.join(timeout=1.0)

    assert not errors
    assert done.is_set()
    assert callbacks == ["A", "B", "C"]
