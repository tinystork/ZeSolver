from __future__ import annotations

from pathlib import Path

from zesolver.gui_pipeline.progress_adapter import gui_progress_from_results
from zesolver.gui_pipeline.requests import GuiFileResult


def test_progress_is_monotone_for_result_sequence() -> None:
    results: list[GuiFileResult] = []
    completed: list[int] = []
    for idx, status in enumerate(("SOLVED", "FAILED", "CANCELLED")):
        results.append(GuiFileResult(Path(f"{idx}.fit"), status, status))
        completed.append(gui_progress_from_results(3, tuple(results)).completed)
    assert completed == sorted(completed)
    assert completed[-1] == 3
