from __future__ import annotations

import json
from pathlib import Path

import pytest

from zesolver.settings_store import PersistentSettings, load_persistent_settings, save_persistent_settings


SOURCE = (Path(__file__).resolve().parents[1] / "zesolver.py").read_text(encoding="utf-8")


def test_gui_source_contains_catalog_library_control_and_translations() -> None:
    required = [
        "settings_catalog_library_edit",
        "settings_catalog_library_browse",
        "settings_catalog_library_validate_btn",
        "settings_catalog_library_clear_btn",
        "settings_catalog_library_status_label",
        "Bibliothèque ZeSolver",
        "ZeSolver library",
        "READY_PARTIAL",
        "Couverture globale Blind 4D",
        "Global Blind 4D coverage",
        "settings_catalog_library_cleared",
    ]
    missing = [needle for needle in required if needle not in SOURCE]
    assert not missing


def test_gui_reads_catalog_library_path_from_visible_widget() -> None:
    assert 'catalog_library_path = self._catalog_library_path_from_ui()' in SOURCE
    assert 'catalog_library_path=getattr(self._settings, "catalog_library_path", None)' not in SOURCE
    assert "self._settings.catalog_library_path = catalog_library_path" in SOURCE


def test_gui_catalog_library_path_roundtrip_and_empty_becomes_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import zesolver.settings_store as store

    settings_file = tmp_path / "settings.json"
    library = tmp_path / "library with spaces et accent é"
    monkeypatch.setattr(store, "SETTINGS_PATH", settings_file)
    monkeypatch.setattr(store, "_resolve_settings_path", lambda: settings_file)

    save_persistent_settings(PersistentSettings(catalog_library_path=str(library), db_root="/legacy/db", index_root="/legacy/index"))
    loaded = load_persistent_settings()
    assert loaded.catalog_library_path == str(library)
    assert loaded.db_root == "/legacy/db"
    assert loaded.index_root == "/legacy/index"

    save_persistent_settings(PersistentSettings(catalog_library_path=None, db_root="/legacy/db", index_root="/legacy/index"))
    payload = json.loads(settings_file.read_text(encoding="utf-8"))
    assert payload["catalog_library_path"] is None
    assert load_persistent_settings().catalog_library_path is None


def test_gui_catalog_library_branch_does_not_require_legacy_paths_in_build_config() -> None:
    assert "if not db_root_text and catalog_resources_for_config is None" in SOURCE
    assert "if not index_root_text and catalog_resources_for_config is None" in SOURCE
    assert "catalog_resources_for_config.near.root" in SOURCE
    assert "blind_index_path=index_root" in SOURCE


def test_gui_logs_catalog_library_preflight_and_runtime_sources() -> None:
    required = [
        "Catalog resources: ",
        "Near catalog preflight: ",
        "ZeBlind 4D preflight: ",
        "catalog_preflight_timings",
        "catalog_library_selected_log",
        "catalog_library_status_log",
        "blind4d_runtime.telemetry(include_paths=False)",
    ]
    missing = [needle for needle in required if needle not in SOURCE]
    assert not missing


def test_gui_wizard_prefers_valid_catalog_library_before_legacy_prompts() -> None:
    library_branch = SOURCE.index("library_text = self._catalog_library_path_from_ui()")
    legacy_branch = SOURCE.index("db_text = (self.settings_db_edit.text().strip()")
    assert library_branch < legacy_branch
    assert "simple_wizard_library_ready" in SOURCE
    assert "simple_wizard_invalid_library" in SOURCE
    assert "simple_wizard_legacy_compat" in SOURCE
