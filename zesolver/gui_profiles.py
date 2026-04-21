from __future__ import annotations

from typing import Any


# Solver tab fields considered advanced in "Simple mode".
_SOLVER_ADVANCED_PAIRS: tuple[tuple[str, str], ...] = (
    ("search_scale_label_widget", "search_scale_spin"),
    ("search_attempts_label_widget", "search_attempts_spin"),
    ("max_radius_label_widget", "max_radius_spin"),
    ("ra_hint_label_widget", "ra_hint_spin"),
    ("dec_hint_label_widget", "dec_hint_spin"),
    # Keep instrument/scale hints visible in Simple mode for practical tuning.
    ("formats_label_widget", "formats_edit"),
    ("families_label_widget", "families_combo"),
)

# Settings groups hidden in Easy mode.
_SETTINGS_EXPERT_GROUPS: tuple[str, ...] = (
    "presets_group",
    "fov_group",
    "reco_group",
    "blind_group",
)

# Buttons meant for diagnostics/manual expert runs only.
_SETTINGS_EXPERT_BUTTONS: tuple[str, ...] = (
    "settings_run_blind_btn",
    "settings_run_near_btn",
)


def _set_visible(obj: Any, name: str, visible: bool) -> None:
    widget = getattr(obj, name, None)
    if widget is not None:
        widget.setVisible(bool(visible))


def apply_solver_simple_visibility(window: Any, *, simple: bool) -> None:
    show_advanced = not bool(simple)
    for label_name, widget_name in _SOLVER_ADVANCED_PAIRS:
        _set_visible(window, label_name, show_advanced)
        _set_visible(window, widget_name, show_advanced)


def apply_settings_easy_visibility(window: Any, *, expert: bool) -> None:
    show_expert = bool(expert)
    for name in _SETTINGS_EXPERT_GROUPS:
        _set_visible(window, name, show_expert)
    for name in _SETTINGS_EXPERT_BUTTONS:
        _set_visible(window, name, show_expert)
