from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from zesolver.engine_selection import EngineMode, select_engine
from zesolver.gui_pipeline.requests import GuiSettingsState
from zesolver.gui_pipeline.settings_adapter import build_engine_selection_request, build_gui_solve_request


def test_snapshot_settings_are_immutable() -> None:
    state = GuiSettingsState(input_dir=Path("/tmp/in"), workers=2)
    with pytest.raises(FrozenInstanceError):
        state.workers = 3  # type: ignore[misc]


def test_auto_selects_pipeline_for_local_fits() -> None:
    request = build_gui_solve_request(
        [Path("light.fit")],
        GuiSettingsState(engine_mode=EngineMode.AUTO, backend="local", workers=2),
    )
    selection = select_engine(build_engine_selection_request(request))
    assert selection.selected_mode is EngineMode.PIPELINE
    assert selection.supported


def test_auto_selects_legacy_for_raster() -> None:
    request = build_gui_solve_request(
        [Path("light.png")],
        GuiSettingsState(engine_mode=EngineMode.AUTO, backend="local"),
    )
    selection = select_engine(build_engine_selection_request(request))
    assert selection.selected_mode is EngineMode.LEGACY
    assert "raster_not_supported_by_pipeline" in selection.reason


def test_pipeline_explicit_rejects_raster() -> None:
    request = build_gui_solve_request(
        [Path("light.tif")],
        GuiSettingsState(engine_mode=EngineMode.PIPELINE, backend="local"),
    )
    selection = select_engine(build_engine_selection_request(request))
    assert selection.selected_mode is EngineMode.PIPELINE
    assert not selection.supported


def test_legacy_explicit_stays_legacy() -> None:
    request = build_gui_solve_request(
        [Path("light.fit")],
        GuiSettingsState(engine_mode=EngineMode.LEGACY, backend="local"),
    )
    selection = select_engine(build_engine_selection_request(request))
    assert selection.selected_mode is EngineMode.LEGACY
    assert selection.reason == "legacy_requested"
