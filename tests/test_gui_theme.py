from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from zesolver.gui_theme import ThemeController, ThemeMode, build_dark_palette, build_light_palette, normalize_theme_mode
from zesolver.settings_store import PersistentSettings, load_persistent_settings, save_persistent_settings


def _qapplication():
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is not None:
        if not isinstance(app, QtWidgets.QApplication):
            pytest.skip("QCoreApplication is already active in this pytest process")
        return app
    return QtWidgets.QApplication([])


def _redirect_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import zesolver.settings_store as store

    settings_file = tmp_path / ".zesolver_settings.json"
    monkeypatch.setattr(store, "SETTINGS_PATH", settings_file)
    monkeypatch.setattr(store, "_resolve_settings_path", lambda: settings_file)
    return settings_file


def test_new_profile_defaults_to_system(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _redirect_settings(tmp_path, monkeypatch)

    settings = load_persistent_settings()

    assert settings.ui_theme == "system"


def test_legacy_profile_without_theme_loads_system_and_preserves_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings_file = _redirect_settings(tmp_path, monkeypatch)
    settings_file.write_text(
        json.dumps(
            {
                "schema_version": 14,
                "db_root": "/legacy/db",
                "index_root": "/legacy/index",
                "solver_downsample": 3,
                "interface_mode": "expert",
            }
        ),
        encoding="utf-8",
    )

    settings = load_persistent_settings()

    assert settings.ui_theme == "system"
    assert settings.db_root == "/legacy/db"
    assert settings.index_root == "/legacy/index"
    assert settings.solver_downsample == 3
    assert settings.interface_mode == "expert"


@pytest.mark.parametrize("mode", ["system", "light", "dark"])
def test_theme_setting_roundtrip(mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _redirect_settings(tmp_path, monkeypatch)

    save_persistent_settings(PersistentSettings(ui_theme=mode, solver_workers=4))
    loaded = load_persistent_settings()

    assert loaded.ui_theme == mode
    assert loaded.solver_workers == 4


def test_invalid_theme_falls_back_to_system(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings_file = _redirect_settings(tmp_path, monkeypatch)
    settings_file.write_text(json.dumps({"ui_theme": "sepia", "solver_cache_size": 17}), encoding="utf-8")

    settings = load_persistent_settings()

    assert settings.ui_theme == "system"
    assert settings.solver_cache_size == 17
    assert normalize_theme_mode("broken") == "system"


def test_theme_palettes_have_readable_light_and_dark_contrast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtGui

    app = _qapplication()
    light = build_light_palette()
    dark = build_dark_palette()
    role = QtGui.QPalette.ColorRole

    assert light.color(role.Window).lightness() > light.color(role.WindowText).lightness()
    assert light.color(role.Base).lightness() > light.color(role.Text).lightness()
    assert dark.color(role.Window).lightness() < dark.color(role.WindowText).lightness()
    assert dark.color(role.Base).lightness() < dark.color(role.Text).lightness()
    assert app is not None


def test_theme_controller_applies_modes_persists_and_restores_system(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtGui

    app = _qapplication()
    system_palette = QtGui.QPalette()
    system_palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#abcdef"))
    app.setPalette(system_palette)
    saved: list[str] = []
    controller = ThemeController(app, initial_mode="system", save_callback=saved.append, system_scheme_getter=lambda: "light")

    controller.apply("dark")
    assert app.property("zesolver_theme_mode") == "dark"
    assert saved[-1] == "dark"

    controller.apply("light")
    assert app.property("zesolver_theme_mode") == "light"
    assert saved[-1] == "light"

    controller.apply("system")
    assert app.property("zesolver_theme_mode") == "system"
    assert saved[-1] == "system"
    assert app.property("zesolver_effective_theme") == "light"
    assert app.palette().color(QtGui.QPalette.ColorRole.Window).name().lower() == "#abcdef"


def test_system_theme_changes_only_affect_system_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    app = _qapplication()
    state = {"scheme": "light"}
    controller = ThemeController(app, initial_mode="system", system_scheme_getter=lambda: state["scheme"])

    state["scheme"] = "dark"
    controller.on_system_theme_changed()
    assert app.property("zesolver_effective_theme") == "dark"

    controller.apply("light", persist=False)
    state["scheme"] = "dark"
    controller.on_system_theme_changed()
    assert app.property("zesolver_theme_mode") == "light"
    assert app.property("zesolver_effective_theme") == "light"

    controller.apply("dark", persist=False)
    state["scheme"] = "light"
    controller.on_system_theme_changed()
    assert app.property("zesolver_theme_mode") == "dark"
    assert app.property("zesolver_effective_theme") == "dark"


def test_theme_menu_is_available_in_easy_and_expert_and_persists_without_save_button(tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    script = textwrap.dedent(
        f"""
        import importlib.util
        import json
        import sys
        from pathlib import Path

        from PySide6 import QtGui, QtWidgets
        from zesolver.settings_store import PersistentSettings

        spec = importlib.util.spec_from_file_location("zesolver_app_p3b1h", Path("zesolver.py"))
        appmod = importlib.util.module_from_spec(spec)
        sys.modules["zesolver_app_p3b1h"] = appmod
        spec.loader.exec_module(appmod)

        settings_file = Path({str(tmp_path / ".zesolver_settings.json")!r})
        initial = PersistentSettings(interface_mode="easy", ui_theme="dark")
        appmod.load_persistent_settings = lambda: initial
        appmod.save_persistent_settings = lambda settings: settings_file.write_text(json.dumps({{"ui_theme": settings.ui_theme, "interface_mode": settings.interface_mode}}), encoding="utf-8")
        appmod.args = appmod.build_arg_parser().parse_args(["--gui"])

        def fake_exec(self):
            self.processEvents()
            window = [w for w in self.topLevelWidgets() if w.__class__.__name__ == "ZeSolverWindow"][0]
            assert window.appearance_menu.menuAction().isVisible()
            assert set(window._theme_actions) == {{"system", "light", "dark"}}
            assert window._theme_group.isExclusive()
            assert window._theme_actions["dark"].isChecked()
            visible_easy = window.appearance_menu.menuAction().isVisible()
            window._on_interface_mode_selected("expert")
            visible_expert = window.appearance_menu.menuAction().isVisible()
            window._theme_actions["light"].trigger()
            self.processEvents()
            persisted = json.loads(settings_file.read_text(encoding="utf-8"))
            assert persisted["ui_theme"] == "light"
            assert window._theme_actions["light"].isChecked()
            window._switch_language("en")
            assert window.appearance_menu.title() == "Appearance"
            assert window._theme_actions["light"].isChecked()
            decision = appmod.decide_startup_wizard(window._settings)
            wizard = appmod.ZeSolverStartupWizard(
                settings=window._settings,
                decision=decision,
                save_settings=appmod.save_persistent_settings,
                parent=window,
            )
            wizard.show()
            self.processEvents()
            assert wizard.palette().color(QtGui.QPalette.ColorRole.Window).isValid()
            wizard.close()
            window.close()
            return 0 if visible_easy and visible_expert else 1

        QtWidgets.QApplication.exec = fake_exec
        raise SystemExit(appmod.launch_gui(appmod.args))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr + result.stdout
