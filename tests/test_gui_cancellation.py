from __future__ import annotations

from pathlib import Path

from zesolver.gui_pipeline.controller import GuiSolveController
from zesolver.gui_pipeline.requests import GuiRunSummary, GuiSettingsState
from zesolver.gui_pipeline.settings_adapter import build_gui_solve_request


class _CancellingRunner:
    def __init__(self) -> None:
        self.cancelled = False

    def run(self, request):
        self.cancel()
        return GuiRunSummary(request.engine_mode, "cancelled", (), True, 0.0)

    def cancel(self):
        self.cancelled = True


def test_cancellation_summary_is_preserved() -> None:
    runner = _CancellingRunner()
    controller = GuiSolveController(
        pipeline_runner_factory=lambda: runner,
        legacy_runner_factory=lambda: runner,
    )
    summary = controller.run(build_gui_solve_request([Path("a.fit")], GuiSettingsState()))
    assert summary.cancelled
    assert runner.cancelled
