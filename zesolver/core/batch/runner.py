from __future__ import annotations

import concurrent.futures
import time
from collections.abc import Callable
from typing import Protocol

from zesolver.core.models import SolveRequest, SolveResult, SolveStatus
from zesolver.core.result_adapter import failure_result

from .models import BatchProgress, BatchSolveRequest, BatchSolveResult


class CancellationToken(Protocol):
    def is_set(self) -> bool:
        ...


class ProgressSink(Protocol):
    def __call__(self, result: SolveResult, progress: BatchProgress) -> None:
        ...


class BatchSolverPipeline:
    """Two-phase batch runner built on SolverPipeline factories."""

    def __init__(
        self,
        *,
        solver_pipeline_factory: Callable[..., object],
        progress_sink: ProgressSink | None = None,
        profile_ids: dict[str, str] | None = None,
    ) -> None:
        self.solver_pipeline_factory = solver_pipeline_factory
        self.progress_sink = progress_sink
        self.profile_ids = dict(profile_ids or {})
        self.execution_order: list[str] = []

    def solve(self, batch_request: BatchSolveRequest) -> BatchSolveResult:
        started = time.perf_counter()
        requests = tuple(batch_request.requests)
        total = len(requests)
        final: dict[int, SolveResult] = {}
        emitted: list[tuple[int, SolveResult]] = []
        unresolved: dict[int, SolveRequest] = {}
        cancelled = self._cancelled(batch_request)

        if total == 0:
            return BatchSolveResult(
                results=(),
                progress=_progress(total=0, final=()),
                cancelled=False,
                duration_s=time.perf_counter() - started,
            )

        if cancelled:
            for idx, request in enumerate(requests):
                result = _synthetic_failure(request, SolveStatus.CANCELLED, "CANCELLED")
                final[idx] = result
                emitted.append((idx, result))
                self._emit(result, total=total, final=tuple(final.values()))
            return self._finish(batch_request, started, final, emitted, cancelled=True)

        near_phase = self._run_phase(
            "near",
            requests_by_index={idx: request for idx, request in enumerate(requests)},
            batch_request=batch_request,
        )
        for idx, result in near_phase.items():
            if result.status is SolveStatus.SOLVED and result.backend == "NEAR":
                final[idx] = result
                emitted.append((idx, result))
                self._emit(result, total=total, final=tuple(final.values()))
            elif result.status is SolveStatus.CANCELLED:
                final[idx] = result
                emitted.append((idx, result))
                cancelled = True
                self._emit(result, total=total, final=tuple(final.values()))
            else:
                unresolved[idx] = requests[idx]

        if self._cancelled(batch_request):
            cancelled = True
            for idx, request in unresolved.items():
                result = _synthetic_failure(request, SolveStatus.CANCELLED, "CANCELLED_BEFORE_BLIND")
                final[idx] = result
                emitted.append((idx, result))
                self._emit(result, total=total, final=tuple(final.values()))
            return self._finish(batch_request, started, final, emitted, cancelled=True)

        if unresolved and not cancelled:
            blind_phase = self._run_phase("blind", requests_by_index=unresolved, batch_request=batch_request)
            for idx in tuple(unresolved):
                result = blind_phase.get(idx)
                if result is None:
                    result = _synthetic_failure(unresolved[idx], SolveStatus.FAILED, "WORKER_FAILED_TO_RETURN_RESULT")
                final[idx] = result
                emitted.append((idx, result))
                self._emit(result, total=total, final=tuple(final.values()))
                if batch_request.stop_on_error and result.status is not SolveStatus.SOLVED:
                    cancelled = True
                    break

        if len(final) < total and cancelled:
            for idx, request in enumerate(requests):
                if idx in final:
                    continue
                result = _synthetic_failure(request, SolveStatus.CANCELLED, "CANCELLED_AFTER_ERROR")
                final[idx] = result
                emitted.append((idx, result))
                self._emit(result, total=total, final=tuple(final.values()))

        return self._finish(batch_request, started, final, emitted, cancelled=cancelled)

    def _run_phase(
        self,
        phase: str,
        *,
        requests_by_index: dict[int, SolveRequest],
        batch_request: BatchSolveRequest,
    ) -> dict[int, SolveResult]:
        if not requests_by_index:
            return {}
        workers = max(1, int(batch_request.workers or 1))
        results: dict[int, SolveResult] = {}

        def _task(item: tuple[int, SolveRequest]) -> tuple[int, SolveResult]:
            idx, request = item
            if self._cancelled(batch_request):
                return idx, _synthetic_failure(request, SolveStatus.CANCELLED, f"CANCELLED_BEFORE_{phase.upper()}")
            self.execution_order.append(f"{phase}:{request.request_id or request.input_path.name}")
            try:
                pipeline = self._new_pipeline(phase)
                result = pipeline.solve(request)
            except Exception as exc:
                result = _synthetic_failure(request, SolveStatus.FAILED, f"ENGINE_FAILED: {exc}")
            return idx, result

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(_task, item): item[0] for item in requests_by_index.items()}
            for future in concurrent.futures.as_completed(future_map):
                idx = future_map[future]
                try:
                    result_idx, result = future.result()
                except Exception as exc:
                    request = requests_by_index[idx]
                    result_idx = idx
                    result = _synthetic_failure(request, SolveStatus.FAILED, f"WORKER_CRASHED: {exc}")
                results[result_idx] = result
                if batch_request.stop_on_error and result.status is not SolveStatus.SOLVED:
                    for pending in future_map:
                        pending.cancel()
                    break
                if self._cancelled(batch_request):
                    for pending in future_map:
                        pending.cancel()
                    break
        return results

    def _new_pipeline(self, phase: str):
        try:
            return self.solver_pipeline_factory(phase)
        except TypeError:
            return self.solver_pipeline_factory()

    def _cancelled(self, batch_request: BatchSolveRequest) -> bool:
        token = batch_request.cancel_token
        if token is None:
            return False
        if callable(token):
            return bool(token())
        is_set = getattr(token, "is_set", None)
        if callable(is_set):
            return bool(is_set())
        return bool(token)

    def _emit(self, result: SolveResult, *, total: int, final: tuple[SolveResult, ...]) -> None:
        if self.progress_sink is None:
            return
        self.progress_sink(result, _progress(total=total, final=final))

    def _finish(
        self,
        batch_request: BatchSolveRequest,
        started: float,
        final: dict[int, SolveResult],
        emitted: list[tuple[int, SolveResult]],
        *,
        cancelled: bool,
    ) -> BatchSolveResult:
        total = len(batch_request.requests)
        if batch_request.preserve_order:
            ordered = tuple(final[idx] for idx in sorted(final))
        else:
            ordered = tuple(result for _idx, result in emitted)
        return BatchSolveResult(
            results=ordered,
            progress=_progress(total=total, final=tuple(final.values())),
            cancelled=cancelled,
            duration_s=time.perf_counter() - started,
        )


def _progress(*, total: int, final: tuple[SolveResult, ...]) -> BatchProgress:
    solved = sum(1 for item in final if item.status is SolveStatus.SOLVED)
    cancelled = sum(1 for item in final if item.status is SolveStatus.CANCELLED)
    skipped = sum(1 for item in final if item.status is SolveStatus.INVALID_INPUT and item.error == "SKIPPED")
    failed = len(final) - solved - cancelled - skipped
    return BatchProgress(
        total=total,
        queued=max(0, total - len(final)),
        running=0,
        solved=solved,
        failed=failed,
        skipped=skipped,
        cancelled=cancelled,
    )


def _synthetic_failure(request: SolveRequest, status: SolveStatus, error: str) -> SolveResult:
    return failure_result(
        request,
        status=status,
        profile_ids={},
        catalog_status=None,
        error=error,
    )
