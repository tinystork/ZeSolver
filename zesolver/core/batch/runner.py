from __future__ import annotations

import concurrent.futures
import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Protocol

from zesolver.core.models import SolveRequest, SolveResult, SolveStatus
from zesolver.core.result_adapter import failure_result
from zesolver.core.terminal_reasons import TerminalReasonCode
from zesolver.resource_telemetry import (
    BatchResourceTelemetry,
    active_batch_telemetry,
    reset_active_batch_telemetry,
    set_active_batch_telemetry,
)
from zesolver.unresolved_output import move_unresolved_results

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
        from zeblindsolver.metadata_solver import reset_zenear_gpu_runtime_state

        reset_zenear_gpu_runtime_state()
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
                result = _normalize_terminal_result(result)
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
                elif result.status is SolveStatus.UNSOLVED:
                    if batch_request.blind_enabled:
                        unresolved[idx] = requests[idx]
                    else:
                        terminal = replace(
                            result,
                            terminal_reason_code=TerminalReasonCode.NEAR_UNRESOLVED_BLIND_UNAVAILABLE.value,
                        )
                        _record_terminal(idx, terminal)
                else:
                    _record_terminal(idx, result)
                    if batch_request.stop_on_error:
                        cancelled = True

            self._run_phase(
                "near",
                requests_by_index={idx: request for idx, request in enumerate(requests)},
                batch_request=batch_request,
                on_result=_near_completed,
            )
            telemetry.mark_rss("after_near")

            if self._cancelled(batch_request):
                cancelled = True
                for idx, request in enumerate(requests):
                    if idx in final:
                        continue
                    result = _synthetic_failure(request, SolveStatus.CANCELLED, "CANCELLED_BEFORE_BLIND")
                    _record_terminal(idx, result)
                return self._finish(batch_request, started, final, emitted, cancelled=True, telemetry=telemetry)

            if unresolved and not cancelled and batch_request.blind_enabled:
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

            if unresolved and not cancelled:
                for idx in tuple(unresolved):
                    if idx in emitted_indices:
                        continue
                    result = failure_result(
                        unresolved[idx],
                        status=SolveStatus.UNSOLVED,
                        profile_ids=self.profile_ids,
                        catalog_status=None,
                        error="no_solver_produced_solution",
                        terminal_reason_code=TerminalReasonCode.NEAR_UNRESOLVED_BLIND_UNAVAILABLE.value,
                    )
                    _record_terminal(idx, result)

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
        items = tuple(requests_by_index.items())
        total_items = len(items)
        ranks = {idx: pos for pos, (idx, _request) in enumerate(items)}
        records: dict[int, dict[str, object]] = {
            idx: {
                "request_index": idx,
                "request_id": request.request_id,
                "path": str(request.input_path),
                "phase": phase,
            }
            for idx, request in items
        }
        phase_started_at = time.perf_counter()
        state_lock = threading.Lock()
        started_count = 0
        finished_count = 0
        active_count = 0
        max_active = 0
        active_area_s = 0.0
        full_occupancy_s = 0.0
        workers_minus_one_s = 0.0
        last_active_change = phase_started_at
        last_finish_by_thread: dict[int, tuple[float, bool]] = {}
        handoff_gaps_s: list[float] = []

        def _update_active_integral(now: float) -> None:
            nonlocal active_area_s, full_occupancy_s, workers_minus_one_s, last_active_change
            delta = max(0.0, now - last_active_change)
            active_area_s += delta * active_count
            if active_count >= workers:
                full_occupancy_s += delta
            if active_count >= max(1, workers - 1):
                workers_minus_one_s += delta
            last_active_change = now

        def _wait_initial_stagger(rank: int) -> None:
            delay_ms = max(0, int(getattr(batch_request, "startup_stagger_ms", 0) or 0))
            if phase != "near" or delay_ms <= 0 or rank >= workers:
                return
            deadline = phase_started_at + (float(delay_ms) / 1000.0) * float(rank)
            while True:
                if self._cancelled(batch_request):
                    return
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    return
                time.sleep(min(0.02, remaining))

        def _mark_task_started(idx: int, request: SolveRequest, pipeline: object) -> None:
            nonlocal started_count, active_count, max_active
            now = time.perf_counter()
            ident = threading.get_ident()
            with state_lock:
                _update_active_integral(now)
                started_count += 1
                active_count += 1
                max_active = max(max_active, active_count)
                previous = last_finish_by_thread.get(ident)
                if previous is not None:
                    previous_finished_at, queue_was_nonempty = previous
                    if queue_was_nonempty:
                        gap = max(0.0, now - previous_finished_at)
                        handoff_gaps_s.append(gap)
                        records[idx]["worker_handoff_gap_ms"] = round(gap * 1000.0, 3)
                identity = _pipeline_identity(pipeline)
                records[idx].update(
                    {
                        "task_started_at": now,
                        "thread_ident": ident,
                        "thread_name": threading.current_thread().name,
                        "pipeline_id": id(pipeline),
                        **identity,
                    }
                )
            if telemetry is not None:
                telemetry.bind_scheduler_task(phase, idx, ident=ident)

        def _mark_task_finished(idx: int) -> None:
            nonlocal finished_count, active_count
            now = time.perf_counter()
            ident = threading.get_ident()
            detect_window = telemetry.detect_window(phase, idx) if telemetry is not None else {}
            with state_lock:
                _update_active_integral(now)
                active_count = max(0, active_count - 1)
                finished_count += 1
                last_finish_by_thread[ident] = (now, started_count < total_items)
                records[idx].update(
                    {
                        "task_finished_at": now,
                        "near_detect_started_at": detect_window.get("near_detect_started_at"),
                        "near_detect_finished_at": detect_window.get("near_detect_finished_at"),
                    }
                )
            if telemetry is not None:
                telemetry.unbind_scheduler_task(ident=ident)

        def _task(item: tuple[int, SolveRequest]) -> tuple[int, SolveResult]:
            idx, request = item
            token = set_active_batch_telemetry(telemetry) if telemetry is not None else None
            if telemetry is not None:
                telemetry.note_worker_thread()
            try:
                _wait_initial_stagger(ranks.get(idx, 0))
                if self._cancelled(batch_request):
                    return idx, _synthetic_failure(request, SolveStatus.CANCELLED, f"CANCELLED_BEFORE_{phase.upper()}")
                self.execution_order.append(f"{phase}:{request.request_id or request.input_path.name}")
                try:
                    pipeline = getattr(local, "pipeline", None)
                    if pipeline is None:
                        pipeline = self._new_pipeline(phase)
                        local.pipeline = pipeline
                    _mark_task_started(idx, request, pipeline)
                    records[idx]["near_solve_started_at"] = time.perf_counter() if phase == "near" else None
                    result = pipeline.solve(request)
                    records[idx].update(_pipeline_identity(pipeline))
                    records[idx]["near_solve_finished_at"] = time.perf_counter() if phase == "near" else None
                except Exception as exc:
                    result = _synthetic_failure(request, SolveStatus.FAILED, f"ENGINE_FAILED: {exc}")
                return idx, result
            finally:
                if "task_started_at" in records[idx]:
                    _mark_task_finished(idx)
                if token is not None:
                    reset_active_batch_telemetry(token)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {}
            for item in items:
                idx = item[0]
                records[idx]["future_submitted_at"] = time.perf_counter()
                future_map[pool.submit(_task, item)] = idx
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
                emitted_at = time.perf_counter()
                if result_idx in records:
                    records[result_idx]["result_emitted_at"] = emitted_at
                    finished_at = records[result_idx].get("task_finished_at")
                    if isinstance(finished_at, (int, float)):
                        records[result_idx]["result_emit_delay_ms"] = round(max(0.0, emitted_at - float(finished_at)) * 1000.0, 3)
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
        phase_finished_at = time.perf_counter()
        with state_lock:
            _update_active_integral(phase_finished_at)
            phase_summary = _scheduler_summary(
                phase=phase,
                workers=workers,
                total_items=total_items,
                phase_started_at=phase_started_at,
                phase_finished_at=phase_finished_at,
                records=records,
                max_active=max_active,
                active_area_s=active_area_s,
                full_occupancy_s=full_occupancy_s,
                workers_minus_one_s=workers_minus_one_s,
                handoff_gaps_s=handoff_gaps_s,
            )
            task_trace = tuple(_finalize_task_record(record) for _idx, record in sorted(records.items()))
        if telemetry is not None:
            telemetry.scheduler_phase(phase, summary=phase_summary, tasks=task_trace)
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
            cancelled = bool(is_set())
        else:
            cancelled = bool(token)
        if cancelled:
            telemetry = active_batch_telemetry()
            if telemetry is not None:
                telemetry.mark_cancel_requested(source="batch_cancel_token")
        return cancelled

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
        if telemetry is not None:
            telemetry.mark_rss("batch_end")
            telemetry.diagnostic_gc()
            terminal = "cancelled" if cancelled else ("failed" if _progress(total=total, final=tuple(final.values())).failed else "completed")
            move_summary = move_unresolved_results(
                input_root=batch_request.input_root or _common_input_root(result.input_path for result in final.values()),
                results=tuple(final.values()),
                terminal_status=("cancelled" if cancelled else "completed"),
                requested=bool(batch_request.move_unresolved_files),
                run_id=batch_request.run_id,
                started_at=batch_request.started_at,
                finished_at=None,
                log_warning=logging.warning,
            )
            if move_summary.records:
                moved_by_input = {
                    record.original_relative_path: record
                    for record in move_summary.records
                    if record.move_status == "moved" and record.destination_relative_path
                }
                rewritten: dict[int, SolveResult] = {}
                input_root = batch_request.input_root or _common_input_root(result.input_path for result in final.values())
                for idx, result in final.items():
                    try:
                        rel = result.input_path.resolve().relative_to(input_root.resolve()).as_posix()
                    except Exception:
                        rel = result.input_path.name
                    record = moved_by_input.get(rel)
                    if record is not None and record.destination_relative_path:
                        dst = input_root / record.destination_relative_path
                        rewritten[idx] = replace(
                            result,
                            moved_to=dst,
                            error=f"Non résolu — déplacé vers {record.destination_relative_path}",
                        )
                    else:
                        rewritten[idx] = result
                final.clear()
                final.update(rewritten)
            telemetry.record_unresolved_output(move_summary.telemetry(requested=bool(batch_request.move_unresolved_files)))
            if bool(batch_request.move_unresolved_files):
                logging.info(
                    "Rangement des non-résolus : éligibles=%d déplacés=%d erreurs=%d destination=%s",
                    move_summary.eligible,
                    move_summary.moved,
                    move_summary.move_failed,
                    move_summary.directory,
                )
            elif move_summary.eligible:
                logging.info(
                    "%d image(s) restent non résolues. Rangement automatique désactivé.",
                    move_summary.eligible,
                )
            _log_near_detection_summary(telemetry, terminal_status=terminal)
        if batch_request.preserve_order:
            ordered = tuple(final[idx] for idx in sorted(final))
        else:
            by_index = dict(final)
            ordered = tuple(by_index.get(idx, result) for idx, result in emitted)
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


def _normalize_terminal_result(result: SolveResult) -> SolveResult:
    if result.terminal_reason_code:
        return result
    if result.status is SolveStatus.UNSOLVED:
        return replace(result, terminal_reason_code=TerminalReasonCode.ALL_ENABLED_SOLVERS_EXHAUSTED.value)
    if result.status is SolveStatus.CANCELLED:
        return replace(result, terminal_reason_code=TerminalReasonCode.CANCELLED.value)
    if result.status is SolveStatus.INVALID_INPUT:
        reason = TerminalReasonCode.SKIPPED_EXISTING_WCS.value if result.error == "SKIPPED" else TerminalReasonCode.INPUT_UNREADABLE.value
        return replace(result, terminal_reason_code=reason)
    if result.status is SolveStatus.FAILED:
        return replace(result, terminal_reason_code=TerminalReasonCode.RUNTIME_ERROR.value)
    return result


def _synthetic_failure(request: SolveRequest, status: SolveStatus, error: str) -> SolveResult:
    reason = TerminalReasonCode.CANCELLED.value if status is SolveStatus.CANCELLED else TerminalReasonCode.RUNTIME_ERROR.value
    return failure_result(
        request,
        status=status,
        profile_ids={},
        catalog_status=None,
        error=error,
        terminal_reason_code=reason,
    )


def _common_input_root(paths) -> object:
    from pathlib import Path

    values = [Path(path) for path in paths]
    if not values:
        return Path(".").resolve()
    try:
        import os

        return Path(os.path.commonpath([str(path.parent.resolve()) for path in values]))
    except Exception:
        return values[0].parent


def _log_near_detection_summary(telemetry: BatchResourceTelemetry, *, terminal_status: str) -> None:
    try:
        summary = telemetry.near_detection_summary(terminal_status=terminal_status)
        detect = summary.get("detect_duration_ms") if isinstance(summary.get("detect_duration_ms"), dict) else {}
        slot = summary.get("gpu_slot_wait_ms") if isinstance(summary.get("gpu_slot_wait_ms"), dict) else {}
        logging.info(
            "ZeNear detection summary: requested=%s cuda_images=%d cpu_images=%d fallbacks=%d gpu_errors=%d gpu_oom=%d device=%s detect_median_ms=%s detect_p95_ms=%s gpu_slot_wait_p95_ms=%s vram_peak=%s terminal=%s",
            summary.get("requested"),
            int(summary.get("images_cuda", 0) or 0),
            int(summary.get("images_cpu", 0) or 0),
            int(summary.get("fallbacks", 0) or 0),
            int(summary.get("gpu_errors", 0) or 0),
            int(summary.get("gpu_oom", 0) or 0),
            ",".join(str(item) for item in summary.get("devices_used", []) or []) or "-",
            detect.get("median"),
            detect.get("p95"),
            slot.get("p95"),
            summary.get("vram_peak_bytes"),
            terminal_status,
        )
    except Exception:
        pass


def _pipeline_identity(pipeline: object) -> dict[str, object]:
    near_solver = getattr(pipeline, "near_solver", None)
    near_runtime_holder = getattr(near_solver, "_near_runtime", None)
    batch_runtime_id = None
    if near_runtime_holder is not None and hasattr(near_runtime_holder, "_runtime"):
        batch_runtime_id = id(near_runtime_holder)
        runtime = getattr(near_runtime_holder, "_runtime", None)
    else:
        runtime = near_runtime_holder
    provider = getattr(runtime, "provider", None)
    return {
        "near_port_id": getattr(near_solver, "near_port_id", id(near_solver) if near_solver is not None else None),
        "near_batch_runtime_id": batch_runtime_id,
        "runtime_id": getattr(runtime, "runtime_id", id(runtime) if runtime is not None else None),
        "provider_id": getattr(provider, "provider_id", id(provider) if provider is not None else None),
    }


def _finalize_task_record(record: dict[str, object]) -> dict[str, object]:
    out = dict(record)
    submitted = _float_or_none(out.get("future_submitted_at"))
    started = _float_or_none(out.get("task_started_at"))
    near_started = _float_or_none(out.get("near_solve_started_at"))
    near_finished = _float_or_none(out.get("near_solve_finished_at"))
    detect_started = _float_or_none(out.get("near_detect_started_at"))
    detect_finished = _float_or_none(out.get("near_detect_finished_at"))
    finished = _float_or_none(out.get("task_finished_at"))
    emitted = _float_or_none(out.get("result_emitted_at"))
    if submitted is not None and started is not None:
        out["queue_wait_ms"] = round(max(0.0, started - submitted) * 1000.0, 3)
    if near_started is not None and near_finished is not None:
        out["near_duration_ms"] = round(max(0.0, near_finished - near_started) * 1000.0, 3)
    if detect_started is not None and detect_finished is not None:
        out["detect_duration_ms"] = round(max(0.0, detect_finished - detect_started) * 1000.0, 3)
    if started is not None and finished is not None:
        out["task_duration_ms"] = round(max(0.0, finished - started) * 1000.0, 3)
    if finished is not None and emitted is not None:
        out["result_emit_delay_ms"] = round(max(0.0, emitted - finished) * 1000.0, 3)
    return out


def _scheduler_summary(
    *,
    phase: str,
    workers: int,
    total_items: int,
    phase_started_at: float,
    phase_finished_at: float,
    records: dict[int, dict[str, object]],
    max_active: int,
    active_area_s: float,
    full_occupancy_s: float,
    workers_minus_one_s: float,
    handoff_gaps_s: list[float],
) -> dict[str, object]:
    duration_s = max(0.0, phase_finished_at - phase_started_at)
    finalized = tuple(_finalize_task_record(record) for record in records.values())
    thread_ids = {record.get("thread_ident") for record in finalized if record.get("thread_ident") is not None}
    pipeline_ids = {record.get("pipeline_id") for record in finalized if record.get("pipeline_id") is not None}
    near_port_ids = {record.get("near_port_id") for record in finalized if record.get("near_port_id") is not None}
    runtime_ids = {record.get("runtime_id") for record in finalized if record.get("runtime_id") is not None}
    provider_ids = {record.get("provider_id") for record in finalized if record.get("provider_id") is not None}
    batch_runtime_ids = {record.get("near_batch_runtime_id") for record in finalized if record.get("near_batch_runtime_id") is not None}
    tasks_per_thread: dict[str, int] = {}
    for record in finalized:
        ident = record.get("thread_ident")
        if ident is not None:
            key = str(ident)
            tasks_per_thread[key] = tasks_per_thread.get(key, 0) + 1
    result_times = sorted(
        value
        for value in (_float_or_none(record.get("result_emitted_at")) for record in finalized)
        if value is not None
    )
    intervals_ms = [
        max(0.0, (result_times[pos] - result_times[pos - 1]) * 1000.0)
        for pos in range(1, len(result_times))
    ]
    task_durations = _numbers(finalized, "task_duration_ms")
    near_durations = _numbers(finalized, "near_duration_ms")
    detect_durations = _numbers(finalized, "detect_duration_ms")
    handoff_ms = [value * 1000.0 for value in handoff_gaps_s]
    return {
        "phase": phase,
        "workers_requested": workers,
        "task_count": total_items,
        "worker_threads_unique": len(thread_ids),
        "solver_pipelines_unique": len(pipeline_ids),
        "near_port_ids_unique": len(near_port_ids),
        "near_batch_runtime_ids_unique": len(batch_runtime_ids),
        "runtime_ids_unique": len(runtime_ids),
        "provider_ids_unique": len(provider_ids),
        "tasks_per_thread": tasks_per_thread,
        "max_tasks_active": max_active,
        "average_tasks_active": round(active_area_s / duration_s, 3) if duration_s > 0.0 else 0.0,
        "time_at_full_occupancy_s": round(full_occupancy_s, 6),
        "time_at_workers_minus_one_or_more_s": round(workers_minus_one_s, 6),
        "median_worker_handoff_gap_ms": _percentile(handoff_ms, 50),
        "p95_worker_handoff_gap_ms": _percentile(handoff_ms, 95),
        "max_worker_handoff_gap_ms": round(max(handoff_ms), 3) if handoff_ms else None,
        "idle_time_while_queue_nonempty_s": round(sum(handoff_gaps_s), 6),
        "median_task_duration_ms": _percentile(task_durations, 50),
        "p95_task_duration_ms": _percentile(task_durations, 95),
        "median_near_duration_ms": _percentile(near_durations, 50),
        "p95_near_duration_ms": _percentile(near_durations, 95),
        "median_detect_duration_ms": _percentile(detect_durations, 50),
        "p95_detect_duration_ms": _percentile(detect_durations, 95),
        "median_result_interval_ms": _percentile(intervals_ms, 50),
        "p90_result_interval_ms": _percentile(intervals_ms, 90),
        "result_intervals_lt_250ms": sum(1 for item in intervals_ms if item < 250.0),
        "result_pauses_gt_1s": sum(1 for item in intervals_ms if item > 1000.0),
        "result_pauses_gt_2s": sum(1 for item in intervals_ms if item > 2000.0),
        "duration_s": round(duration_s, 6),
    }


def _numbers(records: tuple[dict[str, object], ...], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = _float_or_none(record.get(key))
        if value is not None:
            values.append(value)
    return values


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    return None


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    rank = (len(ordered) - 1) * max(0.0, min(100.0, float(pct))) / 100.0
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return round(ordered[low], 3)
    weight = rank - low
    return round(ordered[low] * (1.0 - weight) + ordered[high] * weight, 3)
