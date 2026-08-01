# P3B-1H Theme Selector Report - 2026-08-01

## Status

READY_FOR_P3B1H_THEME_VALIDATION

## Initial dark theme cause

The audit did not find a global forced dark theme in ZeSolver:

- no application-wide `QApplication.setPalette`;
- no application-wide `QApplication.setStyleSheet`;
- no forced `QApplication.setStyle`;
- no forced `Fusion` style.

The dark appearance came from Qt inheriting the operating system or desktop
theme. Local `setStyleSheet()` usage remains limited to semantic/status colors
or compact visual hints. These local styles were not the source of the global
dark appearance.

## Architecture

Theme handling is centralized in `zesolver/gui_theme.py`.

The module provides:

- `ThemeMode`;
- `normalize_theme_mode()`;
- `build_light_palette()`;
- `build_dark_palette()`;
- `apply_application_theme()`;
- `ThemeController`.

`ThemeController` is attached to the `QApplication` as
`zesolver_theme_controller` and is the single source of truth for the active
theme during the GUI session.

## Persistent setting

`PersistentSettings` now has:

```python
ui_theme: str = "system"
```

Accepted values are:

- `system`;
- `light`;
- `dark`.

The settings schema was incremented to `15`.

## Migration policy

New profiles default to `system`.

Old profiles without `ui_theme` load as `system` and preserve all existing
settings. Unknown or corrupted values fall back to `system`, emit
`UI_THEME_SETTING_INVALID`, and do not block startup.

Saving settings normalizes the theme value before writing JSON, so invalid
in-memory values are not persisted.

## System theme detection

System mode uses Qt `styleHints().colorScheme()` when available. If that API is
not available, the controller falls back to palette luminance detection.

At startup the controller is created immediately after `QApplication`, before
the main window or startup wizard are shown. This avoids forcing a dark theme
flash before the selected appearance is applied.

## Dynamic system mode

When Qt exposes `colorSchemeChanged`, the controller connects once and refreshes
the application while `ui_theme == "system"`.

Explicit `light` and `dark` selections ignore later system color-scheme changes.

If the signal is absent on a platform or PySide6 build, startup detection still
works and the application does not fail.

## Palettes

Explicit light and dark modes use Qt palettes, not a broad global QSS.

Light mode provides:

- light window and field backgrounds;
- dark text;
- visible disabled text;
- blue selection and links;
- readable progress bars and menus through standard Qt roles.

Dark mode provides:

- dark window and field backgrounds;
- light text;
- readable disabled text;
- visible selection;
- no black text on dark backgrounds through the application palette.

System mode clears controller-added stylesheet overrides and restores the
captured system palette.

## Local styles

Local styles in `zesolver.py` were audited. The remaining color usage is
semantic:

- success and ready statuses;
- warnings;
- errors;
- progress/highlight states;
- solved and unresolved file states.

No broad local style was added for this mission.

## Menu

The menu path is:

```text
Interface -> Apparence -> Systeme / Clair / Sombre
```

English labels are:

```text
Interface -> Appearance -> System / Light / Dark
```

The three actions are checkable, exclusive through `QActionGroup`, synchronized
with persisted settings, visible in simplified mode, and still visible in
advanced mode.

Changing the action applies the theme immediately, updates
`self._settings.ui_theme`, saves settings automatically, and refreshes menu
checks without requiring restart or a general save button.

## Translations

Added French keys:

- `Apparence`;
- `Systeme`;
- `Clair`;
- `Sombre`.

Added English keys:

- `Appearance`;
- `System`;
- `Light`;
- `Dark`.

Language switching keeps the active theme action checked.

## Telemetry and logs

Added concise logs:

- `UI_THEME_APPLIED mode=system effective=... source=...`;
- `UI_THEME_APPLIED mode=light effective=light source=...`;
- `UI_THEME_APPLIED mode=dark effective=dark source=...`;
- `UI_THEME_SYSTEM_CHANGED effective=...`;
- `UI_THEME_SETTING_INVALID value=... fallback=system`;
- `UI_THEME_SAVE_FAILED error=...`.

No full settings file is logged.

## Automated tests

Added `tests/test_gui_theme.py`.

Covered:

- new profile defaulting to `system`;
- legacy settings without theme;
- round-trip persistence of `system`, `light`, `dark`;
- invalid value fallback;
- no loss of unrelated settings during migration;
- palette contrast sanity checks;
- immediate controller application;
- `Systeme -> Sombre -> Clair -> Systeme`;
- dynamic system changes in system mode;
- dynamic system changes ignored in explicit modes;
- menu presence in simplified and advanced interfaces;
- exclusive/checkable actions;
- initial action synchronization;
- immediate persistence without general save;
- language switching preserving checks;
- startup wizard inheriting the active application palette;
- offscreen Qt execution without a real display server in targeted tests.

Executed:

```text
.venv/bin/python -m py_compile zesolver.py zesolver/gui_theme.py zesolver/settings_store.py
.venv/bin/python -m pytest -q tests/test_gui_theme.py
.venv/bin/python -m pytest -q tests/test_settings_persistence.py tests/test_startup_wizard.py
.venv/bin/python -m pytest -q tests/test_gui_theme.py tests/test_settings_persistence.py tests/test_startup_wizard.py
.venv/bin/python -m pytest -q tests/test_gui_theme.py tests/test_gui_catalog_library_manager.py tests/test_s6b3_easy_expert_settings_gui.py tests/test_gui_development_surface_reorganized.py tests/test_gui_benchmark_removed.py tests/test_startup_wizard.py
```

Results:

- theme targeted: `10 passed`;
- settings + startup wizard: `36 passed`;
- required combined lot: `46 passed`;
- expanded GUI lot: `52 passed`.

Global pytest notes:

- plain `.venv/bin/python -m pytest -q` still aborts during collection because
  `tests/test_catalog_blind4d_manifest_view_cli.py` imports the absent
  `tools.generate_blind4d_manifest_view`;
- the existing P3B-1G report already documents
  `--ignore=tests/test_catalog_blind4d_manifest_view_cli.py`;
- with that ignore, three catalog/path tests remain red in the current clone and
  are unrelated to this theme change;
- excluding those three known catalog tests, the suite reports
  `793 passed, 36 skipped, 3 deselected`.

## Windows validation

Not executed in this Linux workspace. The code is ready for the requested
Windows manual pass:

- new profile follows Windows theme;
- wizard follows active theme;
- `Interface -> Apparence -> Clair` applies immediately and persists;
- `Interface -> Apparence -> Sombre` applies immediately and persists;
- returning to `Systeme` restores OS-driven appearance;
- dynamic system switching follows Qt support when available.

## Linux validation

Automated Qt offscreen validation passed for explicit light, explicit dark, and
system mode logic. A real visual Linux pass was not available in this headless
workspace and remains a manual validation item.

## Modified files

- `zesolver/gui_theme.py`;
- `zesolver/settings_store.py`;
- `zesolver/__init__.py`;
- `zesolver.py`;
- `tests/test_gui_theme.py`;
- `docs/stabilization/p3b1h_theme_selector_report_20260801.md`.

## Remaining limits

- Dynamic system-theme tracking depends on Qt/PySide6 exposing
  `styleHints().colorSchemeChanged` on the platform.
- Native file dialogs and platform dialogs follow Qt/platform theme support.
- Real Windows visual validation is still required before promotion to `main`.

## Recommendation before merge to main

Run the Windows manual checklist, confirm readability of logs, disabled buttons,
progress bars, tables, menus, startup wizard, download wizard, and error
dialogs in all three modes, then promote `test` to `main` in a separate mission.
