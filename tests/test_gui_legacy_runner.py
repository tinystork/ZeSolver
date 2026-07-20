from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from zesolver.gui_pipeline.legacy_runner import LegacyGuiRunner
from zesolver.gui_pipeline.requests import GuiSettingsState
from zesolver.gui_pipeline.settings_adapter import build_gui_solve_request


def test_legacy_runner_wraps_existing_results() -> None:
    def run_legacy(request, cancel_event):
        yield SimpleNamespace(path=request.input_paths[0], status="solved", message="ok", run_info=[])

    runner = LegacyGuiRunner(run_legacy=run_legacy)
    summary = runner.run(build_gui_solve_request([Path("a.fit")], GuiSettingsState()))
    assert len(summary.results) == 1
    assert summary.results[0].status == "SOLVED"
