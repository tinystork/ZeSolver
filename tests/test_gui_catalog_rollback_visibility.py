from __future__ import annotations

from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[1] / "zesolver.py").read_text(encoding="utf-8")


def test_rollback_modes_are_visible_and_reset_without_erasing_paths() -> None:
    required = [
        "near_catalog_mode_combo",
        "blind4d_catalog_mode_combo",
        "settings_rollback_active",
        "ROLLBACK HISTORIQUE ACTIF",
        "HISTORICAL ROLLBACK ACTIVE",
        "settings_restore_auto_modes_btn",
        "self._settings.near_catalog_mode = \"auto\"",
        "self._settings.blind4d_catalog_mode = \"auto\"",
    ]
    missing = [needle for needle in required if needle not in SOURCE]
    assert not missing
    assert "self._settings.db_root = None" not in SOURCE
    assert "self._settings.index_root = None" not in SOURCE


def test_inactive_legacy_sentinels_do_not_block_catalog_library_auto_mode() -> None:
    assert "catalog_resources_for_save is None or near_catalog_mode == \"legacy-index\"" in SOURCE
    assert "catalog_resources_for_config is None or near_catalog_mode == \"legacy-index\"" in SOURCE
    assert "blind4d_catalog_mode == \"external-manifest\"" in SOURCE
    assert "near_catalog_mode=near_catalog_mode" in SOURCE
    assert "blind4d_catalog_mode=blind4d_catalog_mode" in SOURCE

