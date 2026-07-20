from __future__ import annotations

import json
from pathlib import Path

import pytest

from zesolver.settings_store import load_persistent_settings


def test_old_legacy_settings_load_with_auto_catalog_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import zesolver.settings_store as store

    settings_file = tmp_path / "settings.json"
    settings_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "db_root": "/legacy astap",
                "index_root": "/legacy index",
                "blind_4d_manifest_path": "/legacy/manifest.json",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(store, "SETTINGS_PATH", settings_file)
    monkeypatch.setattr(store, "_resolve_settings_path", lambda: settings_file)

    settings = load_persistent_settings()

    assert settings.catalog_library_path is None
    assert settings.db_root == "/legacy astap"
    assert settings.index_root == "/legacy index"
    assert settings.blind_4d_manifest_path == "/legacy/manifest.json"
    assert settings.near_catalog_mode == "auto"
    assert settings.blind4d_catalog_mode == "auto"

