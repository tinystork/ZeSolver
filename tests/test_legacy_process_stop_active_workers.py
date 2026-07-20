from __future__ import annotations

from tests.legacy_stop_helpers import BlockingImageSolver, run_legacy_stop_case


def test_uncooperative_active_workers_are_bounded(monkeypatch, tmp_path) -> None:
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
