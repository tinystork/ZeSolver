from __future__ import annotations

from tests.legacy_stop_helpers import CooperativeImageSolver, run_legacy_stop_case


def test_pending_futures_are_returned_as_cancelled(monkeypatch, tmp_path) -> None:
    _elapsed, results = run_legacy_stop_case(
        monkeypatch,
        tmp_path,
        CooperativeImageSolver,
        files=6,
        workers=1,
        wait_started=1,
    )
    assert [item.status for item in results].count("cancelled") == 6
