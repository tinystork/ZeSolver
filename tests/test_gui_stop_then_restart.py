from __future__ import annotations

import threading
from pathlib import Path

from zesolver.engine_selection import EngineMode
from zesolver.gui_pipeline.controller import GuiSolveController
from zesolver.gui_pipeline.requests import GuiRunSummary, GuiSettingsState
from zesolver.gui_pipeline.settings_adapter import build_gui_solve_request


class _RestartRunner:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.cancelled = threading.Event()

    def run(self, request):
        self.started.set()
        self.cancelled.wait(timeout=2)
        return GuiRunSummary(EngineMode.LEGACY, "done", (), self.cancelled.is_set(), 0.0)

    def cancel(self) -> None:
        self.cancelled.set()


def test_gui_controller_can_restart_after_cancelled_run() -> None:
    runners: list[_RestartRunner] = []

    def factory() -> _RestartRunner:
        runner = _RestartRunner()
        runners.append(runner)
        return runner

    controller = GuiSolveController(pipeline_runner_factory=factory, legacy_runner_factory=factory)
    request = build_gui_solve_request(
        [Path("a.fit")],
        GuiSettingsState(engine_mode=EngineMode.LEGACY),
    )
    thread = threading.Thread(target=lambda: controller.run(request), daemon=True)
    thread.start()
    assert runners[0].started.wait(timeout=2)
    controller.cancel()
    thread.join(timeout=2)
    assert not thread.is_alive()

    summary = controller.run(request)
    assert summary.cancelled is False
    assert len(runners) == 2
    assert runners[0] is not runners[1]
