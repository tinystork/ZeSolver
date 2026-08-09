from __future__ import annotations

import json
from pathlib import Path

import pytest

from zesolver.settings_store import PersistentSettings, load_persistent_settings, save_persistent_settings


SOURCE = (Path(__file__).resolve().parents[1] / "zesolver/_app.py").read_text(encoding="utf-8")
HELPER_SOURCE = (Path(__file__).resolve().parents[1] / "zesolver" / "gui_settings_sections.py").read_text(encoding="utf-8")


def _settings_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import zesolver.settings_store as store

    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(store, "SETTINGS_PATH", settings_file)
    monkeypatch.setattr(store, "_resolve_settings_path", lambda: settings_file)
    return settings_file


def test_new_installation_defaults_to_instrument_auto(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _settings_file(tmp_path, monkeypatch)

    settings = load_persistent_settings()

    assert settings.instrument_mode == "auto"


def test_existing_preset_migrates_to_preset_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings_file = _settings_file(tmp_path, monkeypatch)
    settings_file.write_text(json.dumps({"last_preset_id": "seestar_s50"}), encoding="utf-8")

    settings = load_persistent_settings()

    assert settings.instrument_mode == "preset"
    assert settings.last_preset_id == "seestar_s50"


def test_existing_custom_migrates_to_custom_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings_file = _settings_file(tmp_path, monkeypatch)
    settings_file.write_text(json.dumps({"last_fov_focal_mm": 420.0, "last_fov_pixel_um": 3.76}), encoding="utf-8")

    settings = load_persistent_settings()

    assert settings.instrument_mode == "custom"
    assert settings.last_fov_focal_mm == pytest.approx(420.0)


def test_instrument_mode_roundtrip_preserves_preset_and_custom_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _settings_file(tmp_path, monkeypatch)
    save_persistent_settings(
        PersistentSettings(
            instrument_mode="auto",
            last_preset_id="seestar_s30",
            last_fov_focal_mm=333.0,
            last_fov_pixel_um=2.4,
        )
    )

    loaded = load_persistent_settings()

    assert loaded.instrument_mode == "auto"
    assert loaded.last_preset_id == "seestar_s30"
    assert loaded.last_fov_focal_mm == pytest.approx(333.0)


def test_gui_uses_stable_instrument_ids_not_translated_labels() -> None:
    required = [
        "combo.addItem(self._text(\"instrument_auto\"), \"__auto__\")",
        "combo.addItem(preset.label, preset.id)",
        "combo.addItem(self._text(\"instrument_custom\"), \"__custom__\")",
        "def _current_instrument_mode(self) -> str:",
    ]
    missing = [needle for needle in required if needle not in SOURCE]
    assert not missing
    assert "owner._instrument_mode = \"preset\"" in HELPER_SOURCE
    assert "owner._instrument_mode = \"custom\"" in HELPER_SOURCE

