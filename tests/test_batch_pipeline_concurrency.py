from __future__ import annotations

from zesolver.core.batch import BatchSolverPipeline, BatchSolveRequest
from zesolver.core.models import SolveStatus

from batch_pipeline_fixtures import factory, request


def test_batch_worker_count_one_and_many_produce_same_results() -> None:
    reqs = tuple(request(str(i)) for i in range(4))
    script = {"near": {str(i): SolveStatus.SOLVED for i in range(4)}}

    one = BatchSolverPipeline(solver_pipeline_factory=factory(script, [])).solve(
        BatchSolveRequest(requests=reqs, workers=1)
    )
    many = BatchSolverPipeline(solver_pipeline_factory=factory(script, [])).solve(
        BatchSolveRequest(requests=reqs, workers=4)
    )

    assert [item.status for item in one.results] == [item.status for item in many.results]
    assert [item.request_id for item in many.results] == [str(i) for i in range(4)]
