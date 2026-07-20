from __future__ import annotations

import threading
import time
from pathlib import Path

from zesolver.engine_selection import EngineMode
from zesolver.gui_pipeline.controller import GuiSolveController
from zesolver.gui_pipeline.requests import GuiRunSummary, GuiSettingsState
from zesolver.gui_pipeline.settings_adapter import build_gui_solve_request


class _BlockingRunner:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.cancelled = threading.Event()

    def run(self, request):
        self.started.set()
        assert self.cancelled.wait(timeout=5)
        return GuiRunSummary(EngineMode.LEGACY, "cancelled", (), True, 0.0)

    def cancel(self) -> None:
        self.cancelled.set()


def test_controller_cancel_reaches_active_runner_quickly() -> None:
    runner = _BlockingRunner()
    controller = GuiSolveController(
        pipeline_runner_factory=lambda: runner,
        legacy_runner_factory=lambda: runner,
    )
    request = build_gui_solve_request(
        [Path("a.fit")],
        GuiSettingsState(engine_mode=EngineMode.LEGACY),
    )
    thread = threading.Thread(target=lambda: controller.run(request), daemon=True)
    thread.start()
    assert runner.started.wait(timeout=2)
    t0 = time.perf_counter()
    controller.cancel()
    assert runner.cancelled.wait(timeout=0.1)
    thread.join(timeout=2)
    assert time.perf_counter() - t0 < 0.5
    assert not thread.is_alive()
