from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from zesolver.gui_pipeline.legacy_runner import LegacyGuiRunner
from zesolver.gui_pipeline.requests import GuiSettingsState
from zesolver.gui_pipeline.settings_adapter import build_gui_solve_request


def test_legacy_late_yield_does_not_duplicate_live_result(tmp_path) -> None:
    file_path = tmp_path / "one.fit"
    request = build_gui_solve_request([file_path], GuiSettingsState())
    callback_results = []

    def run_legacy(_request, _cancel_event, live_result_callback):
        result = SimpleNamespace(path=file_path, status="solved", message="near solution found")
        live_result_callback(result)
        yield result

    summary = LegacyGuiRunner(run_legacy=run_legacy, result_callback=callback_results.append).run(request)

    assert len(callback_results) == 1
    assert len(summary.results) == 1
    assert callback_results[0].path == Path(file_path)
