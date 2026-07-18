from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from zesolver.core import SolverPipeline
from zesolver.core.batch import BatchSolverPipeline, BatchSolveRequest
from zesolver.core.models import SolveRequest, SolveResult

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

            def make_pipeline(phase: str) -> SolverPipeline:
                phase_request = request.for_phase(phase)
                if self._solver_pipeline_factory is not None:
                    return self._solver_pipeline_factory(phase, phase_request)
                return SolverPipeline(
                    product_settings=phase_request.product_settings,
                    runtime_options=phase_request.runtime_options,
                    catalog_resources=phase_request.catalog_resources,
                )

            def on_progress(result: SolveResult, progress) -> None:
                gui_result = gui_result_from_solve_result(result, selected_engine=request.engine_mode)
                emitted.append(gui_result)
                if self._result_callback is not None:
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
            final = tuple(gui_result_from_solve_result(item, selected_engine=request.engine_mode) for item in batch_result.results)
            if len(emitted) != len(final):
                for item in final:
                    if self._result_callback is not None:
                        self._result_callback(item)
            return GuiRunSummary(
                selected_engine=request.engine_mode,
                selection_reason="pipeline_runner",
                results=final,
                cancelled=batch_result.cancelled,
                duration_s=batch_result.duration_s,
            )
        finally:
            self._running = False
