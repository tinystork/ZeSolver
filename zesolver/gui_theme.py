"""Application-wide Qt theme handling for ZeSolver."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable


class ThemeMode(str, Enum):
    SYSTEM = "system"
    LIGHT = "light"
    DARK = "dark"


def normalize_theme_mode(value: object) -> str:
    if isinstance(value, ThemeMode):
        return value.value
    if isinstance(value, str):
        candidate = value.strip().lower().replace("_", "-")
        if candidate in {item.value for item in ThemeMode}:
            return candidate
    if value not in (None, ""):
        logging.warning("UI_THEME_SETTING_INVALID value=%r fallback=system", value)
    return ThemeMode.SYSTEM.value


def _qt_modules() -> tuple[Any, Any]:
    from PySide6 import QtCore, QtGui

    return QtCore, QtGui


def _system_color_scheme(app: Any) -> str:
    try:
        style_hints = app.styleHints()
        scheme = style_hints.colorScheme()
        name = str(getattr(scheme, "name", "") or scheme).lower()
        if "dark" in name:
            return ThemeMode.DARK.value
        if "light" in name:
            return ThemeMode.LIGHT.value
    except Exception:
        pass
    try:
        palette = app.palette()
        window = palette.color(palette.ColorRole.Window)
        text = palette.color(palette.ColorRole.WindowText)
        window_luma = (window.red() * 0.299) + (window.green() * 0.587) + (window.blue() * 0.114)
        text_luma = (text.red() * 0.299) + (text.green() * 0.587) + (text.blue() * 0.114)
        return ThemeMode.DARK.value if window_luma < text_luma else ThemeMode.LIGHT.value
    except Exception:
        return ThemeMode.LIGHT.value


def build_light_palette() -> Any:
    _QtCore, QtGui = _qt_modules()
    palette = QtGui.QPalette()
    c = QtGui.QColor
    cr = QtGui.QPalette.ColorRole
    cg = QtGui.QPalette.ColorGroup
    palette.setColor(cr.Window, c("#f5f6f7"))
    palette.setColor(cr.WindowText, c("#202124"))
    palette.setColor(cr.Base, c("#ffffff"))
    palette.setColor(cr.AlternateBase, c("#eef1f4"))
    palette.setColor(cr.ToolTipBase, c("#ffffff"))
    palette.setColor(cr.ToolTipText, c("#202124"))
    palette.setColor(cr.Text, c("#202124"))
    palette.setColor(cr.Button, c("#f0f2f4"))
    palette.setColor(cr.ButtonText, c("#202124"))
    palette.setColor(cr.BrightText, c("#ffffff"))
    palette.setColor(cr.Link, c("#0b57d0"))
    palette.setColor(cr.Highlight, c("#0b57d0"))
    palette.setColor(cr.HighlightedText, c("#ffffff"))
    palette.setColor(cr.PlaceholderText, c("#6c757d"))
    palette.setColor(cg.Disabled, cr.WindowText, c("#8a8f98"))
    palette.setColor(cg.Disabled, cr.Text, c("#8a8f98"))
    palette.setColor(cg.Disabled, cr.ButtonText, c("#8a8f98"))
    palette.setColor(cg.Disabled, cr.Highlight, c("#c5ced8"))
    palette.setColor(cg.Disabled, cr.HighlightedText, c("#5f6670"))
    return palette


def build_dark_palette() -> Any:
    _QtCore, QtGui = _qt_modules()
    palette = QtGui.QPalette()
    c = QtGui.QColor
    cr = QtGui.QPalette.ColorRole
    cg = QtGui.QPalette.ColorGroup
    palette.setColor(cr.Window, c("#202124"))
    palette.setColor(cr.WindowText, c("#f1f3f4"))
    palette.setColor(cr.Base, c("#17181b"))
    palette.setColor(cr.AlternateBase, c("#25272b"))
    palette.setColor(cr.ToolTipBase, c("#2f3136"))
    palette.setColor(cr.ToolTipText, c("#f1f3f4"))
    palette.setColor(cr.Text, c("#f1f3f4"))
    palette.setColor(cr.Button, c("#2b2d31"))
    palette.setColor(cr.ButtonText, c("#f1f3f4"))
    palette.setColor(cr.BrightText, c("#ffffff"))
    palette.setColor(cr.Link, c("#8ab4f8"))
    palette.setColor(cr.Highlight, c("#4c8bf5"))
    palette.setColor(cr.HighlightedText, c("#ffffff"))
    palette.setColor(cr.PlaceholderText, c("#aab0b8"))
    palette.setColor(cg.Disabled, cr.WindowText, c("#9aa0a6"))
    palette.setColor(cg.Disabled, cr.Text, c("#9aa0a6"))
    palette.setColor(cg.Disabled, cr.ButtonText, c("#9aa0a6"))
    palette.setColor(cg.Disabled, cr.Highlight, c("#3b4658"))
    palette.setColor(cg.Disabled, cr.HighlightedText, c("#c8cdd3"))
    return palette


def apply_application_theme(app: Any, mode: object, *, source: str = "user") -> str:
    normalized = normalize_theme_mode(mode)
    if normalized == ThemeMode.SYSTEM.value:
        app.setStyleSheet("")
        system_palette = app.property("zesolver_system_palette")
        if system_palette is not None:
            app.setPalette(system_palette)
        effective = _system_color_scheme(app)
        logging.info("UI_THEME_APPLIED mode=system effective=%s source=%s", effective, source)
    elif normalized == ThemeMode.LIGHT.value:
        app.setStyleSheet("")
        app.setPalette(build_light_palette())
        effective = ThemeMode.LIGHT.value
        logging.info("UI_THEME_APPLIED mode=light effective=light source=%s", source)
    else:
        app.setStyleSheet("")
        app.setPalette(build_dark_palette())
        effective = ThemeMode.DARK.value
        logging.info("UI_THEME_APPLIED mode=dark effective=dark source=%s", source)
    app.setProperty("zesolver_theme_mode", normalized)
    app.setProperty("zesolver_effective_theme", effective)
    _repolish_top_level_widgets(app)
    return effective


def _repolish_top_level_widgets(app: Any) -> None:
    try:
        style = app.style()
        for widget in app.topLevelWidgets():
            style.unpolish(widget)
            style.polish(widget)
            widget.update()
    except Exception:
        pass


class ThemeController:
    def __init__(
        self,
        app: Any,
        *,
        initial_mode: object = ThemeMode.SYSTEM.value,
        save_callback: Callable[[str], None] | None = None,
        system_scheme_getter: Callable[[], str] | None = None,
    ) -> None:
        self.app = app
        self.save_callback = save_callback
        self.system_scheme_getter = system_scheme_getter
        self.mode = normalize_theme_mode(initial_mode)
        self.effective_mode = ThemeMode.LIGHT.value
        self._system_signal_connected = False
        if app.property("zesolver_system_palette") is None:
            app.setProperty("zesolver_system_palette", app.palette())
        self.apply(self.mode, source="startup", persist=False)
        self._connect_system_signal_once()

    def apply(self, mode: object, *, source: str = "user", persist: bool = True) -> str:
        self.mode = normalize_theme_mode(mode)
        if self.mode == ThemeMode.SYSTEM.value and self.system_scheme_getter is not None:
            effective = normalize_theme_mode(self.system_scheme_getter())
            self.app.setStyleSheet("")
            system_palette = self.app.property("zesolver_system_palette")
            if system_palette is not None:
                self.app.setPalette(system_palette)
            if effective not in {ThemeMode.LIGHT.value, ThemeMode.DARK.value}:
                effective = _system_color_scheme(self.app)
            logging.info("UI_THEME_APPLIED mode=system effective=%s source=%s", effective, source)
            self.app.setProperty("zesolver_theme_mode", self.mode)
            self.app.setProperty("zesolver_effective_theme", effective)
            _repolish_top_level_widgets(self.app)
        else:
            effective = apply_application_theme(self.app, self.mode, source=source)
        self.effective_mode = effective
        if persist and self.save_callback is not None:
            try:
                self.save_callback(self.mode)
            except Exception as exc:
                logging.warning("UI_THEME_SAVE_FAILED error=%s", exc)
        return effective

    def on_system_theme_changed(self) -> None:
        if self.mode != ThemeMode.SYSTEM.value:
            return
        effective = self.apply(ThemeMode.SYSTEM.value, source="os", persist=False)
        logging.info("UI_THEME_SYSTEM_CHANGED effective=%s", effective)

    def _connect_system_signal_once(self) -> None:
        if self._system_signal_connected:
            return
        try:
            signal = self.app.styleHints().colorSchemeChanged
        except Exception:
            return
        try:
            signal.connect(self.on_system_theme_changed)
            self._system_signal_connected = True
        except Exception:
            self._system_signal_connected = False


__all__ = [
    "ThemeController",
    "ThemeMode",
    "apply_application_theme",
    "build_dark_palette",
    "build_light_palette",
    "normalize_theme_mode",
]
