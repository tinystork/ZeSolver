from __future__ import annotations

from pathlib import Path

import pytest

from zesolver.engine_selection import EngineMode
from zesolver.gui_pipeline.controller import GuiEngineSelectionError, GuiSolveController
from zesolver.gui_pipeline.requests import GuiRunSummary, GuiSettingsState
from zesolver.gui_pipeline.settings_adapter import build_gui_solve_request


class _Runner:
    def __init__(self, label: EngineMode, calls: list[str]) -> None:
        self.label = label
        self.calls = calls
        self.cancelled = False

    def run(self, request):
        self.calls.append(self.label.value)
        return GuiRunSummary(self.label, "ok", (), self.cancelled, 0.01)

    def cancel(self) -> None:
        self.cancelled = True


def test_controller_routes_auto_fits_to_pipeline() -> None:
    calls: list[str] = []
    controller = GuiSolveController(
        pipeline_runner_factory=lambda: _Runner(EngineMode.PIPELINE, calls),
        legacy_runner_factory=lambda: _Runner(EngineMode.LEGACY, calls),
    )
    request = build_gui_solve_request([Path("a.fit")], GuiSettingsState(engine_mode=EngineMode.AUTO))
    summary = controller.run(request)
    assert summary.selected_engine is EngineMode.PIPELINE
    assert calls == ["pipeline"]


def test_controller_routes_auto_raster_to_legacy() -> None:
    calls: list[str] = []
    controller = GuiSolveController(
        pipeline_runner_factory=lambda: _Runner(EngineMode.PIPELINE, calls),
        legacy_runner_factory=lambda: _Runner(EngineMode.LEGACY, calls),
    )
    request = build_gui_solve_request([Path("a.jpg")], GuiSettingsState(engine_mode=EngineMode.AUTO))
    summary = controller.run(request)
    assert summary.selected_engine is EngineMode.LEGACY
    assert "raster_not_supported_by_pipeline" in summary.selection_reason
    assert calls == ["legacy"]


def test_controller_rejects_pipeline_raster_without_fallback() -> None:
    calls: list[str] = []
    controller = GuiSolveController(
        pipeline_runner_factory=lambda: _Runner(EngineMode.PIPELINE, calls),
        legacy_runner_factory=lambda: _Runner(EngineMode.LEGACY, calls),
    )
    request = build_gui_solve_request([Path("a.png")], GuiSettingsState(engine_mode=EngineMode.PIPELINE))
    with pytest.raises(GuiEngineSelectionError):
        controller.run(request)
    assert calls == []
