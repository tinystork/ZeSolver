from __future__ import annotations

from zesolver.core.batch import BatchSolverPipeline, BatchSolveRequest
from zesolver.core.models import SolveStatus

from batch_pipeline_fixtures import factory, request


def test_batch_preserve_order_true_keeps_input_order() -> None:
    reqs = (request("a"), request("b"), request("c"))
    script = {"near": {"a": SolveStatus.SOLVED, "b": SolveStatus.SOLVED, "c": SolveStatus.SOLVED}}
    calls: list[str] = []

    result = BatchSolverPipeline(solver_pipeline_factory=factory(script, calls)).solve(
        BatchSolveRequest(requests=reqs, workers=2, preserve_order=True)
    )

    assert [item.request_id for item in result.results] == ["a", "b", "c"]
    assert result.progress.solved == 3
    assert len({item.request_id for item in result.results}) == 3


def test_batch_preserve_order_false_uses_completion_order() -> None:
    reqs = (request("a"), request("b"))
    script = {"near": {"a": SolveStatus.SOLVED, "b": SolveStatus.SOLVED}}
    calls: list[str] = []

    result = BatchSolverPipeline(solver_pipeline_factory=factory(script, calls, delay=0.01)).solve(
        BatchSolveRequest(requests=reqs, workers=2, preserve_order=False)
    )

    assert sorted(item.request_id for item in result.results) == ["a", "b"]
    assert result.progress.solved == 2
