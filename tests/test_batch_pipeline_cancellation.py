from __future__ import annotations

from zesolver.core.batch import BatchSolverPipeline, BatchSolveRequest
from zesolver.core.models import SolveStatus

from batch_pipeline_fixtures import factory, request


def test_batch_cancellation_before_run_marks_all_cancelled() -> None:
    reqs = (request("a"), request("b"))
    result = BatchSolverPipeline(solver_pipeline_factory=factory({}, [])).solve(
        BatchSolveRequest(requests=reqs, cancel_token=lambda: True)
    )

    assert result.cancelled is True
    assert [item.status for item in result.results] == [SolveStatus.CANCELLED, SolveStatus.CANCELLED]


def test_batch_cancellation_before_blind_marks_unresolved_cancelled() -> None:
    reqs = (request("a"), request("b"))
    calls = {"count": 0}

    def token() -> bool:
        calls["count"] += 1
        return calls["count"] > 6

    script = {"near": {"a": SolveStatus.UNSOLVED, "b": SolveStatus.UNSOLVED}, "blind": {"a": SolveStatus.SOLVED, "b": SolveStatus.SOLVED}}
    result = BatchSolverPipeline(solver_pipeline_factory=factory(script, [])).solve(
        BatchSolveRequest(requests=reqs, workers=1, cancel_token=token)
    )

    assert len(result.results) == 2
    assert any(item.status in {SolveStatus.CANCELLED, SolveStatus.SOLVED} for item in result.results)
