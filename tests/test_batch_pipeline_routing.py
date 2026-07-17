from __future__ import annotations

from zesolver.core.batch import BatchSolverPipeline, BatchSolveRequest
from zesolver.core.models import SolveStatus

from batch_pipeline_fixtures import factory, request


def test_batch_routes_only_near_failures_to_blind() -> None:
    reqs = (request("near_ok"), request("needs_blind"), request("blind_fails"))
    script = {
        "near": {
            "near_ok": SolveStatus.SOLVED,
            "needs_blind": SolveStatus.UNSOLVED,
            "blind_fails": SolveStatus.UNSOLVED,
        },
        "blind": {
            "needs_blind": SolveStatus.SOLVED,
            "blind_fails": SolveStatus.UNSOLVED,
        },
    }
    calls: list[str] = []

    result = BatchSolverPipeline(solver_pipeline_factory=factory(script, calls)).solve(
        BatchSolveRequest(requests=reqs, workers=1)
    )

    assert [item.status for item in result.results] == [SolveStatus.SOLVED, SolveStatus.SOLVED, SolveStatus.UNSOLVED]
    assert "blind:near_ok" not in calls
    assert "blind:needs_blind" in calls
    assert "blind:blind_fails" in calls
    assert result.progress.solved == 2
    assert result.progress.failed == 1


def test_batch_no_input_lost_or_duplicated() -> None:
    reqs = tuple(request(str(i)) for i in range(6))
    script = {"near": {str(i): SolveStatus.UNSOLVED for i in range(6)}, "blind": {str(i): SolveStatus.SOLVED for i in range(6)}}

    result = BatchSolverPipeline(solver_pipeline_factory=factory(script, [])).solve(
        BatchSolveRequest(requests=reqs, workers=3)
    )

    assert len(result.results) == 6
    assert sorted(item.request_id for item in result.results) == [str(i) for i in range(6)]
