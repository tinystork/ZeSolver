from __future__ import annotations

from pathlib import Path

from zesolver.engine_selection import EngineMode
from zesolver.gui_pipeline.requests import GuiSettingsState
from zesolver.gui_pipeline.settings_adapter import build_gui_solve_request


def test_gui_solve_request_carries_catalog_library_path_without_legacy_paths(tmp_path: Path) -> None:
    library = tmp_path / "Bibliothèque ZeSolver"
    request = build_gui_solve_request(
        [tmp_path / "frame.fit"],
        GuiSettingsState(
            catalog_library_path=library,
            engine_mode=EngineMode.AUTO,
            backend="local",
            use_blind=True,
        ),
    )

    assert request.product_settings.catalog_library_path == library
    assert request.product_settings.near_catalog_mode == "auto"
    assert request.product_settings.blind4d_catalog_mode == "auto"


def test_gui_solve_request_preserves_explicit_rollback_modes(tmp_path: Path) -> None:
    class LegacyConfig:
        near_catalog_mode = "legacy-index"
        blind4d_catalog_mode = "external-manifest"

    library = tmp_path / "library"
    request = build_gui_solve_request(
        [tmp_path / "frame.fit"],
        GuiSettingsState(
            catalog_library_path=library,
            legacy_config=LegacyConfig(),
            use_blind=True,
        ),
    )

    assert request.product_settings.catalog_library_path == library
    assert request.product_settings.near_catalog_mode == "legacy-index"
    assert request.product_settings.blind4d_catalog_mode == "external-manifest"
