from __future__ import annotations

from types import SimpleNamespace

from zesolver.gui_pipeline.legacy_runner import LegacyGuiRunner
from zesolver.gui_pipeline.requests import GuiSettingsState
from zesolver.gui_pipeline.settings_adapter import build_gui_solve_request


def test_legacy_progress_reports_completed_and_remaining_inputs(tmp_path) -> None:
    files = tuple(tmp_path / f"{idx}.fit" for idx in range(5))
    request = build_gui_solve_request(files, GuiSettingsState())
    progress_events = []

    def run_legacy(_request, _cancel_event, live_result_callback):
        for path in files:
            result = SimpleNamespace(path=path, status="solved", message="near solution found")
            live_result_callback(result)
            yield result

    LegacyGuiRunner(run_legacy=run_legacy, progress_callback=progress_events.append).run(request)

    assert [event.completed for event in progress_events] == [1, 2, 3, 4, 5]
    assert [event.total - event.completed for event in progress_events] == [4, 3, 2, 1, 0]
