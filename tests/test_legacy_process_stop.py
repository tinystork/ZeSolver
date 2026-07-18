from __future__ import annotations

from tests.legacy_stop_helpers import CooperativeImageSolver, active_child_count, run_legacy_stop_case


def test_legacy_process_stop_marks_every_file_cancelled(monkeypatch, tmp_path) -> None:
    before = active_child_count()
    elapsed, results = run_legacy_stop_case(
        monkeypatch,
        tmp_path,
        CooperativeImageSolver,
        files=5,
        workers=2,
        wait_started=2,
    )
    assert elapsed < 5.0
    assert len(results) == 5
    assert active_child_count() <= before
