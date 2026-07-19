from __future__ import annotations

import threading
import time
import inspect
from collections.abc import Callable, Iterable
from pathlib import Path

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
        results_by_path: dict[Path, GuiFileResult] = {}
        ordered_paths: list[Path] = []
        notified_paths: set[Path] = set()

        def _key(path: Path) -> Path:
            try:
                return Path(path).resolve()
            except Exception:
                return Path(path)

        def _record(raw: object, *, notify: bool) -> GuiFileResult:
            gui = gui_result_from_legacy(raw, selected_engine=request.engine_mode)
            key = _key(gui.path)
            if key not in results_by_path:
                results_by_path[key] = gui
                ordered_paths.append(key)
            else:
                gui = results_by_path[key]
            if notify and key not in notified_paths:
                notified_paths.add(key)
                if self._result_callback is not None:
                    self._result_callback(gui)
                if self._progress_callback is not None:
                    self._progress_callback(
                        gui_progress_from_results(
                            len(request.input_paths),
                            tuple(results_by_path[path] for path in ordered_paths),
                            current_path=gui.path,
                            phase="LEGACY",
                        )
                    )
            return gui

        def _live_result(raw: object) -> None:
            _record(raw, notify=True)

        def _iter_legacy() -> Iterable[object]:
            try:
                signature = inspect.signature(self._run_legacy)
                params = tuple(signature.parameters.values())
                supports_live = any(param.kind is inspect.Parameter.VAR_POSITIONAL for param in params)
                if not supports_live:
                    positional = {
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    }
                    supports_live = sum(1 for param in params if param.kind in positional) >= 3
            except (TypeError, ValueError):
                supports_live = True
            if supports_live:
                return self._run_legacy(request, self._cancel_event, _live_result)
            return self._run_legacy(request, self._cancel_event)

        try:
            for raw in _iter_legacy():
                _record(raw, notify=True)
                if self._cancel_event.is_set():
                    break
            return GuiRunSummary(
                selected_engine=request.engine_mode,
                selection_reason="legacy_runner",
                results=tuple(results_by_path[path] for path in ordered_paths),
                cancelled=self._cancel_event.is_set(),
                duration_s=time.perf_counter() - started,
            )
        finally:
            self._running = False
