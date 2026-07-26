from __future__ import annotations

import json
from pathlib import Path

from catalog_resource_helpers import strict_entry, write_catalog_library, write_fake_4d_index, write_strict_manifest
from zesolver.catalog_resources import Blind4DCatalogMode, resolve_blind4d_runtime, resolve_catalog_resources
from zesolver.engine_selection import EngineMode
from zesolver.gui_pipeline.requests import GuiSettingsState
from zesolver.gui_pipeline.settings_adapter import build_gui_solve_request


def _library_and_external_manifest(tmp_path: Path) -> tuple[Path, Path, Path]:
    library_index = write_fake_4d_index(tmp_path / "library-indexes" / "d50_2823_S_q.npz", "d50_2823")
    external_index = write_fake_4d_index(tmp_path / "d50_2823_S_q.npz", "d50_2823")
    library = write_catalog_library(tmp_path / "library", index_paths=[library_index])
    catalog = tmp_path / "library" / "catalog.json"
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    payload["runtime_order"] = {"blind4d": ["blind4d-0"]}
    catalog.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    external = write_strict_manifest(
        tmp_path / "external_manifest.json",
        [strict_entry("external", external_index, "d50_2823")],
    )
    return library, library_index, external


def test_gui_settings_auto_preserves_stale_manifest_but_runtime_uses_active_library(tmp_path: Path) -> None:
    library, library_index, external = _library_and_external_manifest(tmp_path)

    request = build_gui_solve_request(
        [tmp_path / "frame.fit"],
        GuiSettingsState(
            catalog_library_path=library,
            engine_mode=EngineMode.AUTO,
            backend="local",
            use_blind=True,
            legacy_config=type(
                "LegacyConfig",
                (),
                {
                    "near_catalog_mode": "auto",
                    "blind4d_catalog_mode": "auto",
                    "blind_4d_manifest_path": external,
                },
            )(),
        ),
    )
    resources = resolve_catalog_resources(
        catalog_library=request.product_settings.catalog_library_path,
        legacy_blind4d_manifest=external,
    )
    runtime = resolve_blind4d_runtime(
        resources,
        mode=request.product_settings.blind4d_catalog_mode,
        external_manifest_path=external,
    )

    assert request.product_settings.blind4d_catalog_mode == "auto"
    assert resources.blind4d_manifest_path is None
    assert runtime.available
    assert runtime.mode_requested is Blind4DCatalogMode.AUTO
    assert runtime.mode_effective is Blind4DCatalogMode.LIBRARY_VIEW
    assert runtime.index_paths == (library_index.resolve(),)
    assert runtime.telemetry()["blind4d_external_fallback_used"] is False


def test_gui_settings_external_manifest_remains_an_explicit_override(tmp_path: Path) -> None:
    library, _library_index, external = _library_and_external_manifest(tmp_path)

    request = build_gui_solve_request(
        [tmp_path / "frame.fit"],
        GuiSettingsState(
            catalog_library_path=library,
            engine_mode=EngineMode.AUTO,
            backend="local",
            use_blind=True,
            legacy_config=type(
                "LegacyConfig",
                (),
                {
                    "near_catalog_mode": "auto",
                    "blind4d_catalog_mode": "external-manifest",
                    "blind_4d_manifest_path": external,
                },
            )(),
        ),
    )
    resources = resolve_catalog_resources(
        catalog_library=request.product_settings.catalog_library_path,
        legacy_blind4d_manifest=external,
    )
    runtime = resolve_blind4d_runtime(
        resources,
        mode=request.product_settings.blind4d_catalog_mode,
        external_manifest_path=external,
    )

    assert request.product_settings.blind4d_catalog_mode == "external-manifest"
    assert runtime.available
    assert runtime.mode_requested is Blind4DCatalogMode.EXTERNAL_MANIFEST
    assert runtime.mode_effective is Blind4DCatalogMode.EXTERNAL_MANIFEST
    assert runtime.loaded_manifest is not None
    assert runtime.loaded_manifest.manifest_path == external.resolve()
