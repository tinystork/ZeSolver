import json
from pathlib import Path

import pytest


def test_settings_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Import lazily to avoid GUI side effects
    import importlib
    zs = importlib.import_module("zesolver")

    # Redirect settings file to a temp path
    settings_file = tmp_path / ".zesolver_settings.json"
    monkeypatch.setattr(zs, "SETTINGS_PATH", settings_file, raising=True)

    # Build a settings object with preset/FOV fields
    s = zs.PersistentSettings(
        db_root=str(tmp_path / "database"),
        index_root=str(tmp_path / "index"),
        last_preset_id="c11_0p63_asi533",
        last_fov_focal_mm=2800.0,
        last_fov_pixel_um=3.76,
        last_fov_res_w=3008,
        last_fov_res_h=3008,
        last_fov_reducer=0.63,
        last_fov_binning=1,
    )
    zs.save_persistent_settings(s)

    # Sanity check: written file exists and contains expected keys
    data = json.loads(settings_file.read_text(encoding="utf-8"))
    assert data["last_preset_id"] == "c11_0p63_asi533"
    assert data["last_fov_focal_mm"] == 2800.0

    # Load back and compare select fields
    s2 = zs.load_persistent_settings()
    assert s2.last_preset_id == s.last_preset_id
    assert s2.last_fov_focal_mm == pytest.approx(s.last_fov_focal_mm)
    assert s2.last_fov_pixel_um == pytest.approx(s.last_fov_pixel_um)
    assert s2.last_fov_res_w == s.last_fov_res_w
    assert s2.last_fov_res_h == s.last_fov_res_h
    assert s2.last_fov_reducer == pytest.approx(s.last_fov_reducer)
    assert s2.last_fov_binning == s.last_fov_binning

