from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

from zesolver.gui_pipeline.legacy_runner import LegacyGuiRunner
from zesolver.gui_pipeline.requests import GuiSettingsState
from zesolver.gui_pipeline.settings_adapter import build_gui_solve_request


def test_legacy_live_callback_reaches_gui_before_iterator_yields(tmp_path) -> None:
    files = tuple(tmp_path / f"{idx}.fit" for idx in range(2))
    request = build_gui_solve_request(files, GuiSettingsState())
    live_seen = threading.Event()
    release_yield = threading.Event()
    callback_results = []
    progress_events = []
    summary_box = {}

    def run_legacy(_request, _cancel_event, live_result_callback):
        first = SimpleNamespace(path=files[0], status="solved", message="near solution found")
        live_result_callback(first)
        live_seen.set()
        assert release_yield.wait(2)
        yield first
        second = SimpleNamespace(path=files[1], status="solved", message="near solution found")
        live_result_callback(second)
        yield second

    runner = LegacyGuiRunner(
        run_legacy=run_legacy,
        result_callback=callback_results.append,
        progress_callback=progress_events.append,
    )
    thread = threading.Thread(target=lambda: summary_box.setdefault("summary", runner.run(request)))
    thread.start()

    assert live_seen.wait(2)
    assert len(callback_results) == 1
    assert progress_events[-1].completed == 1
    assert progress_events[-1].total == 2
    assert thread.is_alive()

    release_yield.set()
    thread.join(2)

    assert not thread.is_alive()
    assert len(callback_results) == 2
    assert [item.path for item in callback_results] == [Path(files[0]), Path(files[1])]
    assert len(summary_box["summary"].results) == 2
