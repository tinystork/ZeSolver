from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from zesolver.catalog_resources import resolve_catalog_resources
from zesolver.core import ProductionBlindSolverPort, SolverPipeline
from zesolver.core.batch import BatchSolverPipeline, BatchSolveRequest
from zesolver.core.models import SolveRequest, SolveResult
from zesolver.resource_telemetry import BatchResourceTelemetry, reset_active_batch_telemetry, set_active_batch_telemetry

from .progress_adapter import GuiProgress, gui_progress_from_batch
from .requests import GuiFileResult, GuiRunSummary, GuiSolveRequest
from .result_adapter import gui_result_from_solve_result


class PipelineGuiRunner:
    def __init__(
        self,
        *,
        progress_callback: Callable[[GuiProgress], None] | None = None,
        result_callback: Callable[[GuiFileResult], None] | None = None,
        solver_pipeline_factory: Callable[[str, GuiSolveRequest], object] | None = None,
    ) -> None:
        self._cancel_event = threading.Event()
        self._running = False
        self._progress_callback = progress_callback
        self._result_callback = result_callback
        self._solver_pipeline_factory = solver_pipeline_factory

    def cancel(self) -> None:
        self._cancel_event.set()

    def is_running(self) -> bool:
        return self._running

    def run(self, request: GuiSolveRequest) -> GuiRunSummary:
        self._running = True
        emitted: list[GuiFileResult] = []
        emitted_paths: set[Path] = set()
        telemetry = BatchResourceTelemetry()
        telemetry_token = set_active_batch_telemetry(telemetry)
        try:
            solve_requests = tuple(
                SolveRequest(
                    input_path=Path(path),
                    output_path=None,
                    overwrite_wcs=request.overwrite_wcs,
                    metadata_overrides=request.metadata_overrides,
                    request_id=str(idx),
                )
                for idx, path in enumerate(request.input_paths)
            )
            if self._progress_callback is not None:
                self._progress_callback(
                    GuiProgress(
                        total=len(solve_requests),
                        completed=0,
                        solved=0,
                        failed=0,
                        skipped=0,
                        cancelled=0,
                        current_phase="Préparation de la bibliothèque",
                    )
                )

            shared_catalog_resources = request.catalog_resources
            shared_blind_solver = None
            if self._solver_pipeline_factory is None:
                if shared_catalog_resources is None:
                    legacy = request.legacy_config
                    shared_catalog_resources = resolve_catalog_resources(
                        catalog_library=request.product_settings.catalog_library_path,
                        legacy_db_root=getattr(legacy, "db_root", None) if legacy is not None else None,
                        legacy_families=tuple(getattr(legacy, "families", ()) or ()) if legacy is not None else None,
                        legacy_blind4d_manifest=getattr(legacy, "blind_4d_manifest_path", None) if legacy is not None else None,
                        legacy_index_root=getattr(legacy, "blind_index_path", None) if legacy is not None else None,
                        enable_environment_discovery=legacy is None,
                    )
                shared_blind_solver = ProductionBlindSolverPort()
            telemetry.mark_rss("after_preflight")

            def make_pipeline(phase: str) -> SolverPipeline:
                phase_request = request.for_phase(phase)
                if self._solver_pipeline_factory is not None:
                    return self._solver_pipeline_factory(phase, phase_request)
                if phase == "near" and self._progress_callback is not None:
                    self._progress_callback(
                        GuiProgress(
                            total=len(solve_requests),
                            completed=len(emitted_paths),
                            solved=0,
                            failed=0,
                            skipped=0,
                            cancelled=0,
                            current_phase="Préparation de ZeNear",
                        )
                    )
                if phase == "blind" and self._progress_callback is not None:
                    self._progress_callback(
                        GuiProgress(
                            total=len(solve_requests),
                            completed=len(emitted_paths),
                            solved=0,
                            failed=0,
                            skipped=0,
                            cancelled=0,
                            current_phase="Préparation de ZeBlind",
                        )
                    )
                return SolverPipeline(
                    product_settings=phase_request.product_settings,
                    runtime_options=phase_request.runtime_options,
                    catalog_resources=shared_catalog_resources,
                    blind_solver=shared_blind_solver,
                )

            def on_progress(result: SolveResult, progress) -> None:
                gui_result = gui_result_from_solve_result(result, selected_engine=request.engine_mode)
                emitted.append(gui_result)
                try:
                    key = Path(gui_result.path).resolve()
                except Exception:
                    key = Path(gui_result.path)
                if self._result_callback is not None and key not in emitted_paths:
                    emitted_paths.add(key)
                    self._result_callback(gui_result)
                if self._progress_callback is not None:
                    self._progress_callback(gui_progress_from_batch(result, progress))

            batch = BatchSolverPipeline(solver_pipeline_factory=make_pipeline, progress_sink=on_progress)
            batch_result = batch.solve(
                BatchSolveRequest(
                    requests=solve_requests,
                    workers=max(1, int(request.workers or 1)),
                    preserve_order=request.preserve_order,
                    cancel_token=self._cancel_event,
                )
            )
            telemetry.event("batch_complete", telemetry=batch_result.telemetry or {})
            final = tuple(gui_result_from_solve_result(item, selected_engine=request.engine_mode) for item in batch_result.results)
            if len(emitted_paths) != len(final):
                for item in final:
                    try:
                        key = Path(item.path).resolve()
                    except Exception:
                        key = Path(item.path)
                    if self._result_callback is not None and key not in emitted_paths:
                        emitted_paths.add(key)
                        self._result_callback(item)
            return GuiRunSummary(
                selected_engine=request.engine_mode,
                selection_reason="pipeline_runner",
                results=final,
                cancelled=batch_result.cancelled,
                duration_s=batch_result.duration_s,
                telemetry=batch_result.telemetry,
            )
        finally:
            reset_active_batch_telemetry(telemetry_token)
            self._running = False
