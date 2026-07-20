from __future__ import annotations

from zesolver.core.batch import BatchSolverPipeline, BatchSolveRequest
from zesolver.core.models import SolveStatus

from batch_pipeline_fixtures import factory, request


def test_batch_engine_exception_is_failed_result() -> None:
    reqs = (request("boom"),)
    script = {"near": {"boom": SolveStatus.FAILED}, "blind": {"boom": SolveStatus.FAILED}}

    result = BatchSolverPipeline(solver_pipeline_factory=factory(script, [])).solve(
        BatchSolveRequest(requests=reqs, workers=1)
    )

    assert result.results[0].status is SolveStatus.FAILED
    assert "ENGINE_FAILED" in str(result.results[0].error)


def test_batch_progress_callback_receives_final_results() -> None:
    reqs = (request("a"), request("b"))
    script = {"near": {"a": SolveStatus.SOLVED, "b": SolveStatus.UNSOLVED}, "blind": {"b": SolveStatus.SOLVED}}
    seen: list[tuple[str | None, int]] = []

    def sink(result, progress):
        seen.append((result.request_id, progress.solved))

    BatchSolverPipeline(solver_pipeline_factory=factory(script, []), progress_sink=sink).solve(
        BatchSolveRequest(requests=reqs, workers=1)
    )

    assert seen[-1] == ("b", 2)
