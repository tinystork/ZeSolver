from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable

from .progress_adapter import GuiProgress, gui_progress_from_results
from .requests import GuiFileResult, GuiRunSummary, GuiSolveRequest
from .result_adapter import gui_result_from_legacy


class LegacyGuiRunner:
    def __init__(
        self,
        *,
        run_legacy: Callable[[GuiSolveRequest, threading.Event], Iterable[object]],
        progress_callback: Callable[[GuiProgress], None] | None = None,
        result_callback: Callable[[GuiFileResult], None] | None = None,
    ) -> None:
        self._run_legacy = run_legacy
        self._cancel_event = threading.Event()
        self._running = False
        self._progress_callback = progress_callback
        self._result_callback = result_callback

    def cancel(self) -> None:
        self._cancel_event.set()

    def is_running(self) -> bool:
        return self._running

    def run(self, request: GuiSolveRequest) -> GuiRunSummary:
        started = time.perf_counter()
        self._running = True
        results: list[GuiFileResult] = []
        try:
            for raw in self._run_legacy(request, self._cancel_event):
                gui = gui_result_from_legacy(raw, selected_engine=request.engine_mode)
                results.append(gui)
                if self._result_callback is not None:
                    self._result_callback(gui)
                if self._progress_callback is not None:
                    self._progress_callback(
                        gui_progress_from_results(
                            len(request.input_paths),
                            tuple(results),
                            current_path=gui.path,
                            phase="LEGACY",
                        )
                    )
                if self._cancel_event.is_set():
                    break
            return GuiRunSummary(
                selected_engine=request.engine_mode,
                selection_reason="legacy_runner",
                results=tuple(results),
                cancelled=self._cancel_event.is_set(),
                duration_s=time.perf_counter() - started,
            )
        finally:
            self._running = False
