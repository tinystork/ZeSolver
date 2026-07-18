from __future__ import annotations

import time
from collections.abc import Callable

from zesolver.engine_selection import EngineMode, EngineSelection, select_engine

from .requests import GuiRunSummary, GuiSolveRequest
from .settings_adapter import build_engine_selection_request


class GuiEngineSelectionError(RuntimeError):
    def __init__(self, selection: EngineSelection) -> None:
        super().__init__(selection.reason)
        self.selection = selection


class GuiSolveController:
    def __init__(
        self,
        *,
        pipeline_runner_factory: Callable[[], object],
        legacy_runner_factory: Callable[[], object],
        selection_logger: Callable[[EngineSelection], None] | None = None,
    ) -> None:
        self._pipeline_runner_factory = pipeline_runner_factory
        self._legacy_runner_factory = legacy_runner_factory
        self._selection_logger = selection_logger
        self._runner: object | None = None
        self._running = False
        self.last_selection: EngineSelection | None = None

    def run(self, request: GuiSolveRequest) -> GuiRunSummary:
        if self._running:
            raise RuntimeError("gui_run_already_active")
        started = time.perf_counter()
        selection = select_engine(build_engine_selection_request(request))
        self.last_selection = selection
        if self._selection_logger is not None:
            self._selection_logger(selection)
        if not selection.supported:
            raise GuiEngineSelectionError(selection)

        runner = self._pipeline_runner_factory() if selection.selected_mode is EngineMode.PIPELINE else self._legacy_runner_factory()
        self._runner = runner
        self._running = True
        try:
            summary = runner.run(request)
            return GuiRunSummary(
                selected_engine=selection.selected_mode,
                selection_reason=selection.reason,
                results=summary.results,
                cancelled=summary.cancelled,
                duration_s=summary.duration_s if summary.duration_s is not None else time.perf_counter() - started,
                warnings=tuple(selection.warnings) + tuple(summary.warnings),
                selection=selection,
            )
        finally:
            self._running = False
            self._runner = None

    def cancel(self) -> None:
        runner = self._runner
        if runner is None:
            return
        cancel = getattr(runner, "cancel", None)
        if callable(cancel):
            cancel()

    def is_running(self) -> bool:
        return self._running
