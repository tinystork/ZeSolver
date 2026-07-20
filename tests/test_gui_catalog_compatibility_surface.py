from __future__ import annotations

from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[1] / "zesolver.py").read_text(encoding="utf-8")
PROFILES = (Path(__file__).resolve().parents[1] / "zesolver" / "gui_profiles.py").read_text(encoding="utf-8")
SECTIONS = (Path(__file__).resolve().parents[1] / "zesolver" / "gui_settings_sections.py").read_text(encoding="utf-8")


def test_normal_surface_keeps_catalog_library_and_moves_legacy_into_compatibility_group() -> None:
    required = [
        "catalog_compat_group",
        "catalog_compat_group_title",
        "Compatibilité historique et diagnostic",
        "Legacy compatibility and diagnostics",
        "settings_legacy_astap_label",
        "settings_legacy_index_label",
        "settings_blind4d_external_label",
        "settings_restore_auto_modes",
        "catalog_maintenance_group",
        "Outils avancés et maintenance des catalogues",
    ]
    missing = [needle for needle in required if needle not in SOURCE]
    assert not missing
    assert '"catalog_compat_group"' in PROFILES
    assert '"catalog_maintenance_group"' in PROFILES


def test_database_tab_and_legacy_groups_are_hidden_outside_expert_mode() -> None:
    assert "self._set_tab_visible(self.database_scroll, expert)" in SOURCE
    assert "hidden_tabs = [getattr(self, n, None) for n in (\"database_scroll\"" in SOURCE
    assert "apply_settings_easy_visibility(self, expert=expert)" in SOURCE


def test_qt_layout_parent_warning_source_is_removed() -> None:
    assert "column.addLayout(form)" not in SECTIONS
    assert "build_presets_fov_reco_groups(self, QtWidgets, preset_utils, column, form)" in SOURCE
    assert "column.addLayout(form)" in SOURCE

