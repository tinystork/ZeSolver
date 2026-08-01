from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from zesolver.engine_selection import EngineMode, select_engine
from zesolver.gui_pipeline.requests import GuiSettingsState
from zesolver.gui_pipeline.settings_adapter import (
    build_engine_selection_request,
    build_gui_solve_request,
    build_gui_solve_request_from_legacy_config,
)


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


def test_legacy_config_unknown_catalog_resources_keeps_blind4d_coverage_unknown() -> None:
    class Config:
        input_dir = Path("/tmp")
        catalog_library_path = Path("/catalog")
        overwrite = True
        workers = 2
        blind_enabled = True
        astrometry_fallback_after_blind = False
        astrometry_api_key = None
        formats = ("fit",)
        max_files = None
        log_level = "INFO"
        fov_deg = 1.5
        downsample = 1

    request = build_gui_solve_request_from_legacy_config([Path("m31.fit")], Config(), catalog_resources=None)
    selection_request = build_engine_selection_request(request)
    selection = select_engine(selection_request)

    assert request.blind4d_all_sky is None
    assert selection_request.blind4d_all_sky is None
    assert selection.selected_mode is EngineMode.PIPELINE
    assert "blind4d_coverage_partial_not_all_sky" not in selection.warnings


def test_engine_selection_request_marks_blind_disabled() -> None:
    request = build_gui_solve_request(
        [Path("light.fit")],
        GuiSettingsState(engine_mode=EngineMode.AUTO, backend="local", use_blind=False, blind4d_all_sky=False),
    )

    selection_request = build_engine_selection_request(request)

    assert selection_request.blind4d_enabled is False
