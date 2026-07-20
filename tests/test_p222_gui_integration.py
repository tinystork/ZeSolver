from __future__ import annotations

import json
from pathlib import Path

import pytest

from zeblindsolver.profiles import ZEBLIND_4D_EXPERIMENTAL_PROFILE
from zesolver.blind4d_runtime import ENV_4D_MANIFEST_PATH, resolve_default_4d_manifest_path
from zesolver.settings_store import PersistentSettings, load_persistent_settings, save_persistent_settings


def test_p222_old_settings_migrate_to_4d_and_easy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import zesolver.settings_store as store
    import zesolver as pkg

    settings_file = tmp_path / "settings.json"
    settings_file.write_text(json.dumps({"schema_version": 1, "solver_blind_enabled": True}), encoding="utf-8")
    monkeypatch.setattr(store, "SETTINGS_PATH", settings_file)
    monkeypatch.setattr(pkg, "SETTINGS_PATH", settings_file)

    loaded = load_persistent_settings()

    assert loaded.blind_backend_profile == ZEBLIND_4D_EXPERIMENTAL_PROFILE
    assert loaded.blind_4d_manifest_path is None
    assert loaded.interface_mode == "easy"
    assert loaded.solver_blind_enabled is True


def test_p222_profile_and_manifest_path_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import zesolver.settings_store as store
    import zesolver as pkg

    settings_file = tmp_path / "settings.json"
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(store, "SETTINGS_PATH", settings_file)
    monkeypatch.setattr(pkg, "SETTINGS_PATH", settings_file)

    save_persistent_settings(
        PersistentSettings(
            interface_mode="expert",
            solver_blind_enabled=False,
            blind_backend_profile=ZEBLIND_4D_EXPERIMENTAL_PROFILE,
            blind_4d_manifest_path=str(manifest),
        )
    )
    loaded = load_persistent_settings()

    assert loaded.interface_mode == "expert"
    assert loaded.solver_blind_enabled is False
    assert loaded.blind_backend_profile == ZEBLIND_4D_EXPERIMENTAL_PROFILE
    assert loaded.blind_4d_manifest_path == str(manifest)


def test_p222_invalid_profile_and_interface_migrate_to_safe_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import zesolver.settings_store as store
    import zesolver as pkg

    settings_file = tmp_path / "settings.json"
    settings_file.write_text(
        json.dumps({"blind_backend_profile": "surprise_4d", "interface_mode": "other"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(store, "SETTINGS_PATH", settings_file)
    monkeypatch.setattr(pkg, "SETTINGS_PATH", settings_file)

    loaded = load_persistent_settings()

    assert loaded.blind_backend_profile == ZEBLIND_4D_EXPERIMENTAL_PROFILE
    assert loaded.interface_mode == "easy"


def test_p222_manifest_resolver_explicit_env_and_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    explicit = tmp_path / "explicit.json"
    env_path = tmp_path / "env.json"

    assert resolve_default_4d_manifest_path(explicit) == explicit.resolve()

    monkeypatch.setenv(ENV_4D_MANIFEST_PATH, str(env_path))
    assert resolve_default_4d_manifest_path() == env_path.resolve()

    monkeypatch.delenv(ENV_4D_MANIFEST_PATH)
    default = resolve_default_4d_manifest_path()
    assert default.name == "zeblind_4d_experimental_manifest.json"
    assert default.parent.name == "config"


def test_p222_gui_source_contains_required_controls_and_translations() -> None:
    source = (Path(__file__).resolve().parents[1] / "zesolver.py").read_text(encoding="utf-8")
    required = [
        "blind_4d_easy_check",
        "blind_4d_profile_combo",
        "blind_4d_manifest_verify_btn",
        "load_4d_index_manifest",
        "Utiliser ZeBlind 4D",
        "Use ZeBlind 4D",
        "Profil ZeBlind",
        "ZeBlind profile",
        "Couverture limitée aux index 4D installés",
        "Coverage is limited to installed 4D indexes",
        "solver.chain.4d",
        "blind_4d_preflight_failed",
    ]
    missing = [needle for needle in required if needle not in source]
    assert not missing


def test_p222_qt_gui_smoke_is_available_when_dependencies_are_installed() -> None:
    pytest.importorskip("PySide6")
    pytest.importorskip("astroalign")
