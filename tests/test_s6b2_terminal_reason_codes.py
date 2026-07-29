from __future__ import annotations

from zesolver.core.batch import BatchSolverPipeline, BatchSolveRequest
from zesolver.core.models import SolveStatus
from zesolver.core.terminal_reasons import TerminalReasonCode, is_unresolved_move_eligible

from batch_pipeline_fixtures import factory, request


def test_near_and_blind_exhausted_is_scientific_unresolved():
    reqs = (request("a"),)
    calls: list[str] = []
    script = {"near": {"a": SolveStatus.UNSOLVED}, "blind": {"a": SolveStatus.UNSOLVED}}

    result = BatchSolverPipeline(solver_pipeline_factory=factory(script, calls)).solve(
        BatchSolveRequest(requests=reqs, workers=1, blind_enabled=True)
    )

    assert calls == ["near:a", "blind:a"]
    assert result.results[0].terminal_reason_code == TerminalReasonCode.ALL_ENABLED_SOLVERS_EXHAUSTED.value
    assert is_unresolved_move_eligible(result.results[0].terminal_reason_code)


def test_cancelled_is_not_move_eligible():
    req = request("a")
    result = BatchSolverPipeline(solver_pipeline_factory=factory({}, [])).solve(
        BatchSolveRequest(requests=(req,), cancel_token=lambda: True)
    )

    assert result.results[0].status is SolveStatus.CANCELLED
    assert result.results[0].terminal_reason_code == TerminalReasonCode.CANCELLED.value
    assert not is_unresolved_move_eligible(result.results[0].terminal_reason_code)


def test_runtime_failure_is_not_move_eligible():
    reqs = (request("boom"),)
    script = {"near": {"boom": SolveStatus.FAILED}}

    result = BatchSolverPipeline(solver_pipeline_factory=factory(script, [])).solve(
        BatchSolveRequest(requests=reqs, workers=1)
    )

    assert result.results[0].status is SolveStatus.FAILED
    assert result.results[0].terminal_reason_code == TerminalReasonCode.RUNTIME_ERROR.value
    assert not is_unresolved_move_eligible(result.results[0].terminal_reason_code)

