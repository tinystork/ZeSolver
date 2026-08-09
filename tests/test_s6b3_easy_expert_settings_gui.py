from __future__ import annotations

from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[1] / "zesolver/_app.py").read_text(encoding="utf-8")
PROFILE_SOURCE = (Path(__file__).resolve().parents[1] / "zesolver" / "gui_profiles.py").read_text(encoding="utf-8")


def test_easy_astap_controls_are_visible_outside_legacy_compat_group() -> None:
    easy_pos = SOURCE.index("self.easy_astap_label = QtWidgets.QLabel()")
    compat_pos = SOURCE.index("self.catalog_compat_group = QtWidgets.QGroupBox")
    assert easy_pos < compat_pos
    assert '"catalog_compat_group"' in PROFILE_SOURCE
    assert '"easy_astap_edit"' not in PROFILE_SOURCE


def test_easy_astap_controls_have_browse_verify_clear_and_status() -> None:
    required = [
        "self.easy_astap_edit = QtWidgets.QLineEdit",
        "self.easy_astap_browse = QtWidgets.QPushButton",
        "self.easy_astap_verify_btn = QtWidgets.QPushButton",
        "self.easy_astap_clear_btn = QtWidgets.QPushButton",
        "self.easy_astap_status_label = QtWidgets.QLabel",
        "easy_astap_valid",
        "easy_astap_invalid",
    ]
    missing = [needle for needle in required if needle not in SOURCE]
    assert not missing


def test_instrument_auto_is_visible_in_easy_and_expert() -> None:
    required = [
        "self.instrument_combo = QtWidgets.QComboBox()",
        "self.instrument_label_widget",
        "instrument_auto",
        "instrument_custom",
        "fov_focal_spin",
        "widget.setEnabled(expert_fields_enabled)",
    ]
    missing = [needle for needle in required if needle not in SOURCE]
    assert not missing


def test_language_keys_exist_for_french_and_english() -> None:
    for needle in (
        '"instrument_auto": "Auto — métadonnées FITS"',
        '"instrument_auto": "Auto — FITS metadata"',
        '"easy_astap_label": "Base ASTAP — ZeNear uniquement"',
        '"easy_astap_label": "ASTAP database — ZeNear only"',
    ):
        assert needle in SOURCE

