from __future__ import annotations

from zesolver.core.batch import BatchSolverPipeline, BatchSolveRequest


def test_batch_empty_returns_empty_result() -> None:
    runner = BatchSolverPipeline(solver_pipeline_factory=lambda phase: None)
    result = runner.solve(BatchSolveRequest(requests=()))

    assert result.results == ()
    assert result.progress.total == 0
    assert result.cancelled is False
