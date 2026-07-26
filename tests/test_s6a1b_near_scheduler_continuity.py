from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
from astropy.io import fits

from catalog_resource_helpers import write_catalog_library
from near_catalog_provider_helpers import write_astap_1476_tile
from zesolver.catalog_resources import resolve_catalog_resources
from zesolver.core.batch import BatchSolverPipeline, BatchSolveRequest
from zesolver.core.models import SolveRequest, SolveResult, SolveStatus
from zesolver.gui_pipeline.pipeline_runner import PipelineGuiRunner
from zesolver.gui_pipeline.requests import GuiSettingsState
from zesolver.gui_pipeline.settings_adapter import build_gui_solve_request

from batch_pipeline_fixtures import result_for


def _request(name: str) -> SolveRequest:
    return SolveRequest(Path(f"{name}.fit"), None, True, request_id=name)


class _NearPort:
    def __init__(self) -> None:
        self.near_port_id = 1001
        self._near_runtime = _Runtime()


class _Runtime:
    runtime_id = 2002
    provider = type("Provider", (), {"provider_id": 3003})()


class _SyntheticNearPipeline:
    def __init__(self, phase: str, durations: dict[str, float], calls: list[tuple[str, str, int]]) -> None:
        self.phase = phase
        self.durations = durations
        self.near_solver = _NearPort()
        self.calls = calls

    def solve(self, req: SolveRequest) -> SolveResult:
        self.calls.append((self.phase, str(req.request_id), id(self)))
        if self.phase != "near":
            return result_for(req, SolveStatus.FAILED, None, "unexpected blind")
        delay = float(self.durations.get(str(req.request_id), 0.0))
        if delay > 0.0:
            time.sleep(delay)
        return result_for(req, SolveStatus.SOLVED, "NEAR")


def _factory(durations: dict[str, float], calls: list[tuple[str, str, int]]):
    def _make(phase: str):
        return _SyntheticNearPipeline(phase, durations, calls)

    return _make


def _near_scheduler(result) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    assert result.telemetry is not None
    scheduler = result.telemetry["scheduler"]
    assert isinstance(scheduler, dict)
    near = scheduler["near"]
    assert isinstance(near, dict)
    return near["summary"], near["tasks"]


def test_s6a1b_pool_threads_persist_and_do_not_process_in_blocks_of_six() -> None:
    reqs = tuple(_request(str(idx)) for idx in range(18))
    durations = {str(idx): 0.03 for idx in range(18)}
    calls: list[tuple[str, str, int]] = []

    result = BatchSolverPipeline(solver_pipeline_factory=_factory(durations, calls)).solve(
        BatchSolveRequest(requests=reqs, workers=6, preserve_order=True)
    )
    summary, tasks = _near_scheduler(result)

    assert result.progress.solved == 18
    assert summary["worker_threads_unique"] <= 6
    assert summary["solver_pipelines_unique"] <= 6
    assert summary["near_port_ids_unique"] == 1
    assert summary["runtime_ids_unique"] == 1
    assert summary["provider_ids_unique"] == 1
    assert int(summary["max_tasks_active"]) > 1
    assert float(summary["average_tasks_active"]) > 1.0
    by_thread: dict[object, set[object]] = {}
    counts: dict[object, int] = {}
    for task in tasks:
        ident = task.get("thread_ident")
        by_thread.setdefault(ident, set()).add(task.get("pipeline_id"))
        counts[ident] = counts.get(ident, 0) + 1
    assert all(len(pipelines) == 1 for pipelines in by_thread.values())
    assert any(count > 1 for count in counts.values())
    assert [item.request_id for item in result.results] == [str(idx) for idx in range(18)]


def test_s6a1b_fast_worker_starts_next_task_before_slow_first_wave_finishes() -> None:
    reqs = tuple(_request(str(idx)) for idx in range(12))
    durations = {
        "0": 0.20,
        "1": 0.40,
        "2": 0.60,
        "3": 0.80,
        "4": 1.00,
        "5": 1.20,
        **{str(idx): 0.02 for idx in range(6, 12)},
    }

    result = BatchSolverPipeline(solver_pipeline_factory=_factory(durations, [])).solve(
        BatchSolveRequest(requests=reqs, workers=6)
    )
    summary, tasks = _near_scheduler(result)
    by_request = {str(task["request_id"]): task for task in tasks}

    assert by_request["6"]["task_started_at"] < by_request["5"]["task_finished_at"]
    assert by_request["6"]["task_started_at"] < by_request["1"]["task_finished_at"]
    assert summary["median_worker_handoff_gap_ms"] is not None
    assert float(summary["median_worker_handoff_gap_ms"]) < 100.0


def test_s6a1b_startup_stagger_applies_only_to_initial_fill() -> None:
    reqs = tuple(_request(str(idx)) for idx in range(8))
    durations = {str(idx): 0.01 for idx in range(8)}

    result = BatchSolverPipeline(solver_pipeline_factory=_factory(durations, [])).solve(
        BatchSolveRequest(requests=reqs, workers=6, startup_stagger_ms=200)
    )
    _summary, tasks = _near_scheduler(result)
    by_request = {str(task["request_id"]): task for task in tasks}

    assert float(by_request["1"]["queue_wait_ms"]) >= 150.0
    assert float(by_request["6"]["queue_wait_ms"]) < 150.0


def test_s6a1b_stop_during_startup_stagger_is_not_blocked_by_sleep() -> None:
    reqs = tuple(_request(str(idx)) for idx in range(6))
    cancel = threading.Event()
    emitted: list[str | None] = []
    started = time.perf_counter()

    def sink(result: SolveResult, _progress) -> None:
        emitted.append(result.request_id)
        cancel.set()

    result = BatchSolverPipeline(
        solver_pipeline_factory=_factory({str(idx): 0.01 for idx in range(6)}, []),
        progress_sink=sink,
    ).solve(BatchSolveRequest(requests=reqs, workers=6, startup_stagger_ms=500, cancel_token=cancel))

    assert emitted
    assert time.perf_counter() - started < 1.5
    assert result.cancelled is True
    assert any(item.status is SolveStatus.CANCELLED for item in result.results)


def test_s6a1b_runtime_close_event_survives_more_than_128_acquisitions(tmp_path: Path, monkeypatch) -> None:
    library = write_catalog_library(tmp_path / "library", include_source=True, index_paths=[])
    write_astap_1476_tile(
        library / "sources" / "astap" / "d50",
        family="d50",
        tile_code="2823",
        ra_deg=np.asarray([184.6, 184.7], dtype=np.float64),
        dec_deg=np.asarray([47.2, 47.3], dtype=np.float64),
    )
    resources = resolve_catalog_resources(catalog_library=library)
    files = tuple(_fits(tmp_path / f"frame_{idx}.fit") for idx in range(140))
    seen: list[int] = []

    def fake_near_solve(fits_path, index_root, *, catalog_provider=None, **kwargs):
        seen.append(id(catalog_provider))
        return {"success": False, "message": "synthetic miss", "stats": {}, "wrote_wcs": False}

    monkeypatch.setattr("zesolver.core.pipeline.near_solve", fake_near_solve)

    summary = PipelineGuiRunner().run(build_gui_solve_request(files, _state(resources, workers=6)))
    counters = summary.telemetry["counters"]
    phases = [item["phase"] for item in summary.telemetry["events"]]

    assert len(seen) == 140
    assert len(set(seen)) == 1
    assert counters["near_catalog_runtime_reused"] == 140
    assert counters["near_catalog_runtime_closed"] == 1
    assert "near_catalog_runtime_closed" in phases
    assert "near_catalog_runtime_reused" not in phases


def test_s6a1b_preserve_order_does_not_change_progressive_completion_order() -> None:
    reqs = tuple(_request(name) for name in ("slow", "fast", "middle"))
    durations = {"slow": 0.20, "fast": 0.01, "middle": 0.08}
    emitted: list[str | None] = []

    result = BatchSolverPipeline(
        solver_pipeline_factory=_factory(durations, []),
        progress_sink=lambda solve_result, _progress: emitted.append(solve_result.request_id),
    ).solve(BatchSolveRequest(requests=reqs, workers=3, preserve_order=True))

    assert emitted[0] == "fast"
    assert [item.request_id for item in result.results] == ["slow", "fast", "middle"]


def _fits(path: Path) -> Path:
    fits.PrimaryHDU(data=np.ones((8, 8), dtype=np.uint16)).writeto(path)
    return path


def _state(resources, *, workers: int) -> GuiSettingsState:
    return GuiSettingsState(
        workers=workers,
        preserve_order=True,
        use_blind=False,
        catalog_resources=resources,
        legacy_config=object(),
    )
