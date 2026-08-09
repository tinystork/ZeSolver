from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from zesolver.gui_pipeline.requests import GuiSettingsState
from zesolver.gui_pipeline.settings_adapter import build_gui_solve_request, build_product_settings


def _load_app_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("zesolver_app_s6b2_gui", root / "zesolver/_app.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_move_unresolved_checkbox_and_translations_exist_in_gui_source():
    appmod = _load_app_module()
    source = Path(appmod.__file__).read_text(encoding="utf-8")

    assert "move_unresolved_check" in source
    assert "move_unresolved_files_label" in appmod.GUI_TRANSLATIONS["fr"]
    assert "move_unresolved_files_label" in appmod.GUI_TRANSLATIONS["en"]
    assert "unresolved_by_zesolver" in appmod.GUI_TRANSLATIONS["fr"]["move_unresolved_files_label"]
    assert "unresolved_by_zesolver" in appmod.GUI_TRANSLATIONS["en"]["move_unresolved_files_label"]


def test_gui_settings_transmit_move_unresolved_and_interface_mode(tmp_path):
    state = GuiSettingsState(
        input_dir=tmp_path,
        formats=(".fit",),
        move_unresolved_files=True,
        interface_mode="easy",
    )

    product = build_product_settings(state)
    request = build_gui_solve_request((tmp_path / "a.fit",), state)

    assert product.move_unresolved_files is True
    assert product.interface_mode == "easy"
    assert request.move_unresolved_files is True
    assert request.product_settings.move_unresolved_files is True


def test_expert_mode_remains_explicit():
    product = build_product_settings(GuiSettingsState(interface_mode="expert", move_unresolved_files=False))

    assert product.interface_mode == "expert"
    assert product.move_unresolved_files is False
