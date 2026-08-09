from __future__ import annotations

from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[1] / "zesolver/_app.py").read_text(encoding="utf-8")
HELPER_SOURCE = (Path(__file__).resolve().parents[1] / "zesolver" / "gui_settings_sections.py").read_text(encoding="utf-8")


def test_easy_expert_database_astap_widgets_use_shared_controller() -> None:
    required = [
        "def _set_astap_root(",
        "self.easy_astap_edit.textChanged.connect(lambda text: self._set_astap_root(text, source=\"easy\", validate=False))",
        "self.db_tab_edit.textChanged.connect(lambda text: self._set_astap_root(text, source=\"database-tab\", validate=False))",
        "self._set_astap_root(db_root, source=\"cli\", validate=False)",
    ]
    missing = [needle for needle in required if needle not in SOURCE]
    assert not missing
    assert "owner.settings_db_edit.textChanged.connect(\n        lambda text: owner._set_astap_root(text, source=\"settings\", validate=False)" in HELPER_SOURCE


def test_astap_validation_and_clear_are_centralized() -> None:
    required = [
        "def _verify_astap_root_from_gui(",
        "validation = validate_astap_root(text)",
        "def _clear_astap_root(self) -> None:",
        "self._set_astap_root(\"\", source=\"clear\", validate=False)",
        "self._astap_validation_state = \"not_verified\"",
    ]
    missing = [needle for needle in required if needle not in SOURCE]
    assert not missing


def test_wizard_accepts_astap_only_without_legacy_index() -> None:
    astap_branch = SOURCE.index("simple_wizard_astap_ready")
    legacy_branch = SOURCE.index("self._current_near_catalog_mode_from_ui() == \"legacy-index\"")
    assert astap_branch < legacy_branch
    assert "validate_astap_root(db_path)" in SOURCE

