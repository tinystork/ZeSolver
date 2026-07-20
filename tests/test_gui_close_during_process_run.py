from __future__ import annotations

from tests.legacy_stop_helpers import BlockingImageSolver, active_child_count, run_legacy_stop_case


def test_close_style_cancellation_cleans_process_run(monkeypatch, tmp_path) -> None:
    before = active_child_count()
    elapsed, results = run_legacy_stop_case(
        monkeypatch,
        tmp_path,
        BlockingImageSolver,
        files=4,
        workers=2,
        wait_started=2,
        grace="0.2",
    )
    assert elapsed < 5.0
    assert len(results) == 4
    assert active_child_count() <= before
