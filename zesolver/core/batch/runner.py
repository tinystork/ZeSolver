from __future__ import annotations

import concurrent.futures
import threading
import time
from collections.abc import Callable
from typing import Protocol

from zesolver.core.models import SolveRequest, SolveResult, SolveStatus
from zesolver.core.result_adapter import failure_result
from zesolver.resource_telemetry import (
    BatchResourceTelemetry,
    active_batch_telemetry,
    reset_active_batch_telemetry,
    set_active_batch_telemetry,
)

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
        telemetry = active_batch_telemetry() or BatchResourceTelemetry()
        telemetry_token = set_active_batch_telemetry(telemetry)
        telemetry.mark_rss("batch_start")
        try:
            requests = tuple(batch_request.requests)
            total = len(requests)
            final: dict[int, SolveResult] = {}
            emitted: list[tuple[int, SolveResult]] = []
            emitted_indices: set[int] = set()
            unresolved: dict[int, SolveRequest] = {}
            cancelled = self._cancelled(batch_request)

            def _record_terminal(idx: int, result: SolveResult) -> bool:
                if idx in emitted_indices:
                    return False
                final[idx] = result
                emitted.append((idx, result))
                emitted_indices.add(idx)
                self._emit(result, total=total, final=tuple(final.values()))
                return True

            if total == 0:
                return BatchSolveResult(
                    results=(),
                    progress=_progress(total=0, final=()),
                    cancelled=False,
                    duration_s=time.perf_counter() - started,
                    telemetry=telemetry.snapshot(),
                )

            if cancelled:
                for idx, request in enumerate(requests):
                    result = _synthetic_failure(request, SolveStatus.CANCELLED, "CANCELLED")
                    _record_terminal(idx, result)
                return self._finish(batch_request, started, final, emitted, cancelled=True, telemetry=telemetry)

            telemetry.mark_rss("before_near")

            def _near_completed(idx: int, result: SolveResult) -> None:
                nonlocal cancelled
                if result.status is SolveStatus.SOLVED and result.backend == "NEAR":
                    _record_terminal(idx, result)
                elif result.status is SolveStatus.CANCELLED:
                    _record_terminal(idx, result)
                    cancelled = True
                else:
                    unresolved[idx] = requests[idx]

            self._run_phase(
                "near",
                requests_by_index={idx: request for idx, request in enumerate(requests)},
                batch_request=batch_request,
                on_result=_near_completed,
            )
            telemetry.mark_rss("after_near")

            if self._cancelled(batch_request):
                cancelled = True
                for idx, request in unresolved.items():
                    result = _synthetic_failure(request, SolveStatus.CANCELLED, "CANCELLED_BEFORE_BLIND")
                    _record_terminal(idx, result)
                return self._finish(batch_request, started, final, emitted, cancelled=True, telemetry=telemetry)

            if unresolved and not cancelled:
                telemetry.mark_rss("before_blind")
                blind_phase: dict[int, SolveResult] = {}

                def _blind_completed(idx: int, result: SolveResult) -> None:
                    nonlocal cancelled
                    unresolved.pop(idx, None)
                    if _record_terminal(idx, result):
                        if result.status is SolveStatus.CANCELLED:
                            cancelled = True
                        if batch_request.stop_on_error and result.status is not SolveStatus.SOLVED:
                            cancelled = True

                blind_phase = self._run_phase(
                    "blind",
                    requests_by_index=unresolved,
                    batch_request=batch_request,
                    on_result=_blind_completed,
                )
                for idx in tuple(unresolved):
                    if idx in emitted_indices:
                        continue
                    result = blind_phase.get(idx)
                    if result is None:
                        if cancelled or self._cancelled(batch_request):
                            result = _synthetic_failure(unresolved[idx], SolveStatus.CANCELLED, "CANCELLED_DURING_BLIND")
                        else:
                            result = _synthetic_failure(
                                unresolved[idx],
                                SolveStatus.FAILED,
                                "WORKER_FAILED_TO_RETURN_RESULT",
                            )
                    _record_terminal(idx, result)
                    if batch_request.stop_on_error and result.status is not SolveStatus.SOLVED:
                        cancelled = True
                        break
                telemetry.mark_rss("after_blind")

            if len(final) < total and cancelled:
                for idx, request in enumerate(requests):
                    if idx in final:
                        continue
                    result = _synthetic_failure(request, SolveStatus.CANCELLED, "CANCELLED_AFTER_ERROR")
                    _record_terminal(idx, result)

            return self._finish(batch_request, started, final, emitted, cancelled=cancelled, telemetry=telemetry)
        finally:
            reset_active_batch_telemetry(telemetry_token)

    def _run_phase(
        self,
        phase: str,
        *,
        requests_by_index: dict[int, SolveRequest],
        batch_request: BatchSolveRequest,
        on_result: Callable[[int, SolveResult], None] | None = None,
    ) -> dict[int, SolveResult]:
        if not requests_by_index:
            return {}
        workers = 1 if phase == "blind" else max(1, int(batch_request.workers or 1))
        results: dict[int, SolveResult] = {}
        local = threading.local()
        telemetry = active_batch_telemetry()

        def _task(item: tuple[int, SolveRequest]) -> tuple[int, SolveResult]:
            idx, request = item
            token = set_active_batch_telemetry(telemetry) if telemetry is not None else None
            if telemetry is not None:
                telemetry.note_worker_thread()
            try:
                if self._cancelled(batch_request):
                    return idx, _synthetic_failure(request, SolveStatus.CANCELLED, f"CANCELLED_BEFORE_{phase.upper()}")
                self.execution_order.append(f"{phase}:{request.request_id or request.input_path.name}")
                try:
                    pipeline = getattr(local, "pipeline", None)
                    if pipeline is None:
                        pipeline = self._new_pipeline(phase)
                        local.pipeline = pipeline
                    result = pipeline.solve(request)
                except Exception as exc:
                    result = _synthetic_failure(request, SolveStatus.FAILED, f"ENGINE_FAILED: {exc}")
                return idx, result
            finally:
                if token is not None:
                    reset_active_batch_telemetry(token)

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
                ready_at = time.perf_counter()
                if telemetry is not None and phase == "blind":
                    telemetry.event(
                        "blind_result_ready",
                        index=result_idx,
                        request_id=result.request_id,
                        status=result.status.name,
                    )
                if on_result is not None:
                    on_result(result_idx, result)
                if telemetry is not None and phase == "blind":
                    telemetry.event(
                        "blind_result_emitted",
                        index=result_idx,
                        request_id=result.request_id,
                        status=result.status.name,
                        emit_lag_s=round(time.perf_counter() - ready_at, 6),
                    )
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
        telemetry: BatchResourceTelemetry | None = None,
    ) -> BatchSolveResult:
        total = len(batch_request.requests)
        if batch_request.preserve_order:
            ordered = tuple(final[idx] for idx in sorted(final))
        else:
            ordered = tuple(result for _idx, result in emitted)
        if telemetry is not None:
            telemetry.mark_rss("batch_end")
            telemetry.diagnostic_gc()
        return BatchSolveResult(
            results=ordered,
            progress=_progress(total=total, final=tuple(final.values())),
            cancelled=cancelled,
            duration_s=time.perf_counter() - started,
            telemetry=(telemetry.snapshot() if telemetry is not None else None),
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
