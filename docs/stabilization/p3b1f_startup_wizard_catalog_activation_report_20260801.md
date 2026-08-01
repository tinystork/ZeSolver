# P3B-1F Startup Wizard Catalog Activation Report - 2026-08-01

## Status

P3B1F_STARTUP_WIZARD_CATALOG_ACTIVATION_CLOSED

READY_FOR_RELEASE_CANDIDATE_ACCEPTANCE

## Initial Git State

Initial commands were run before modification:

```text
git status --short
git diff --check
git rev-parse HEAD
git rev-parse origin/test
git log --oneline -5
```

The worktree was clean and `git diff --check` was clean.

Local `test` was already ahead of `origin/test` by the P3B-1H theme-selector
commit:

```text
HEAD=43fa8e62d7dcca07d0a85e6b0d33e828f2a5dcd3
origin/test=f42079f5329dd200257f8403a584af8cfae6f520
43fa8e6 Add system light and dark theme selector
f42079f Add automatic catalog download resume controls
a9a7c4c fix passage dossier de travail au gui durant le wizard
2782f10 Fix ASTAP and CatalogLibrary resource composition
d1c4a2e Add multisource parallel catalog downloads
```

No remote update, force operation, or branch promotion was performed.

## Reproduction

The automated regression reproduces the user-visible state:

- `catalog_library_path` points to a READY_FULL library;
- `near_catalog_mode=legacy-index`;
- `blind4d_catalog_mode=external-manifest`;
- `blind_4d_manifest_path` points to an invalid legacy JSON file.

The test then simulates:

```text
Startup wizard -> existing_library -> validated path -> Terminer
```

Before the correction this path could enter `_read_settings_from_ui()` while the
Blind 4D widget still requested `external-manifest`, triggering validation of
the stale manifest.

The corrected test asserts that no external manifest validation is called, the
wizard completes, and the saved settings are:

```text
catalog_library_path=<READY_FULL library>
near_catalog_mode=auto
blind4d_catalog_mode=auto
```

The inactive legacy manifest path is preserved for diagnostics but is not used.

## Root Cause

The failing flow was confirmed:

1. the wizard worker could validate a CatalogLibrary correctly;
2. `ZeSolverStartupWizard.accept()` emitted `librarySelected`;
3. the main window handler called `_on_catalog_library_manager_selected()`;
4. that path called `_read_settings_from_ui()`;
5. hidden advanced controls could still contain
   `blind4d_catalog_mode=external-manifest`;
6. `_read_settings_from_ui()` therefore validated `blind_4d_manifest_path`;
7. the wizard still marked itself complete after signal emission.

This produced the incoherent pair:

```text
BLIND4D_EXTERNAL_MANIFEST_INVALID
Startup wizard completed: existing_library
```

## Persistence Ownership

The main window owns the authoritative `PersistentSettings` instance once the
GUI is running.

The wizard still receives a settings reference for display and standalone test
compatibility, but in the main-window flow it now delegates completion to a
transaction callback:

```python
_complete_startup_wizard_transaction(request) -> StartupWizardCompletionResult
```

Only that callback may mark the wizard completed and save the final settings.
The wizard calls `super().accept()` only when the callback returns success.

The previous double-save risk was real: the main window could replace
`self._settings` after reading the GUI, then the wizard could save its older
`self.settings` reference. That late save has been removed from the main
transactional flow.

## Fix Architecture

The product activation responsibility is centralized in:

```python
_activate_catalog_library_product_mode(path, source, persist=True, show_error=True) -> bool
```

It performs:

- CatalogLibrary validation;
- GUI path synchronization;
- reset of Near and Blind 4D controls to product `auto` modes before settings
  are read;
- `_read_settings_from_ui()` without reintroducing external-manifest rollback;
- coherent `PersistentSettings` update;
- lightweight verification record refresh;
- status/capability summary refresh;
- one save when requested.

On failure it restores the previous settings object and visible path, records
the technical error, and returns `False`.

The startup wizard now sends a structured completion request containing source,
library path, ASTAP path, image directory, and blind-enabled state. Library
sources covered by the same transaction are:

- `existing_library`;
- `official`;
- `local_package`.

All three converge to:

```text
catalog_library_path=<valid library>
near_catalog_mode=auto
blind4d_catalog_mode=auto
```

The ASTAP route is kept distinct:

```text
near_catalog_mode=astap-native
blind4d_catalog_mode=auto
```

Advanced explicit rollback remains available outside the wizard.

## Related Runtime Fix

`resolve_catalog_resources()` now honors a forced legacy Near override only when
the legacy ASTAP root actually exists. This preserves the explicit rollback path
while preventing a missing residual legacy root from masking the intended
CatalogLibrary error.

## Files Modified

- `.gitignore`
- `AGENT.md`
- `CHANGELOG.md`
- `docs/stabilization/p3b1f_startup_wizard_catalog_activation_report_20260801.md`
- `tests/test_gui_catalog_path_confusion.py`
- `tests/test_gui_catalog_rollback_visibility.py`
- `tests/test_startup_wizard.py`
- `tests/test_startup_wizard_catalog_activation.py`
- `tools/generate_blind4d_manifest_view.py`
- `zesolver.py`
- `zesolver/catalog_resources.py`
- `zesolver/gui_startup_wizard.py`

## Automated Tests Added

Added `tests/test_startup_wizard_catalog_activation.py`.

Covered:

- exact stale external-manifest regression on existing-library wizard finish;
- no inactive external manifest validation;
- `auto/auto` persistence for existing, official, and local-package library
  routes;
- preservation of unrelated settings such as `ui_theme`;
- activation failure leaving the wizard open;
- activation failure doing no save and not marking the wizard completed.

Existing source-audit tests were updated to match the corrected save/run
validation distinction.

## Commands Executed

```text
.venv/bin/python -m py_compile zesolver.py zesolver/gui_startup_wizard.py
.venv/bin/python -m pytest -q tests/test_startup_wizard_catalog_activation.py tests/test_startup_wizard.py
.venv/bin/python -m pytest -q tests/test_startup_wizard_catalog_activation.py tests/test_startup_wizard.py tests/test_settings_persistence.py tests/test_gui_catalog_library_control.py tests/test_gui_catalog_library_manager.py tests/test_gui_catalog_library_solve_config.py tests/test_catalog_library_validation.py tests/test_catalog_library_management_service.py tests/test_catalog_distribution.py tests/test_catalog_distribution_multisource.py tests/test_configuration_assembly.py tests/test_gui_catalog_resource_type_validation.py tests/test_catalog_library_blind4d_product_switch.py tests/test_solver_pipeline_near_provider.py
.venv/bin/python -m pytest -q tests/test_catalog_library_no_silent_legacy_fallback.py::test_forced_astap_native_with_missing_resource_does_not_use_residual_legacy tests/test_gui_catalog_path_confusion.py::test_gui_uses_typed_validators_before_save_and_run tests/test_gui_catalog_rollback_visibility.py::test_inactive_legacy_sentinels_do_not_block_catalog_library_auto_mode
.venv/bin/python -m pytest -q tests/test_catalog_blind4d_manifest_view_cli.py tests/test_catalog_blind4d_manifest_view.py
.venv/bin/python -m pytest -q tests/test_startup_wizard_catalog_activation.py tests/test_startup_wizard.py tests/test_catalog_library_no_silent_legacy_fallback.py tests/test_gui_catalog_path_confusion.py tests/test_gui_catalog_rollback_visibility.py
.venv/bin/python -m compileall zesolver zeblindsolver tools
.venv/bin/python -m pytest -q
```

## Automated Results

Targeted results:

```text
37 passed
139 passed
3 passed
20 passed
43 passed
```

Global result after code changes:

```text
801 passed, 36 skipped, 17 warnings in 67.32s
```

`compileall` passed for `zesolver`, `zeblindsolver`, and `tools`.

## Manual Linux Validation

Validated with:

```text
/home/tristan/ZeSolverCatalog/new
```

Prepared state:

```text
near_catalog_mode_ui=legacy-index
blind4d_catalog_mode_ui=external-manifest
blind_4d_manifest_path_ui=/tmp/.../invalid_manifest.json
```

Offscreen GUI wizard path:

```text
Startup wizard -> existing_library -> /home/tristan/ZeSolverCatalog/new -> Terminer
```

Result:

```json
{
  "after": {
    "catalog_library_path": "/home/tristan/ZeSolverCatalog/new",
    "near_catalog_mode": "auto",
    "blind4d_catalog_mode": "auto",
    "blind_4d_manifest_path": "/tmp/.../invalid_manifest.json",
    "startup_wizard_completed": true
  },
  "blind_manifest_validations": [],
  "save_count_before_close": 1,
  "status": "READY_FULL",
  "blind4d_index_count": 47,
  "blind4d_covered_tiles": 1476,
  "blind4d_total_tiles": 1476,
  "blind4d_all_sky": true
}
```

No inactive external manifest validation was observed.

Runtime Blind 4D validation:

```text
.venv/bin/python zesolver.py --headless --catalog-library /home/tristan/ZeSolverCatalog/new --input-dir /tmp/p3b1f-blind-batch-vjcwm4 --max-files 1 --overwrite --blind-only --blind-profile zeblind_4d_experimental --blind4d-catalog-mode auto --log-level INFO
```

Key result:

```text
blind4d_catalog_mode_effective: library-view
blind4d_index_count: 47
blind4d_covered_tiles: 1476
blind4d_total_tiles: 1476
blind4d_all_sky: true
blind4d_external_fallback_used: false
Blind solver success via d50_2822
Done in 71.1s - 1 solved, 0 skipped, 0 failed
```

Runtime Near validation:

```text
.venv/bin/python zesolver.py --headless --catalog-library /home/tristan/ZeSolverCatalog/new --input-dir /tmp/p3b1f-near-batch-pcbi7g --max-files 1 --overwrite --blind-profile zeblind_4d_experimental --blind4d-catalog-mode auto --log-level INFO
```

Key result:

```text
near_catalog_mode_effective: astap-native
near_catalog_provider: astap_native
near_catalog_source: library
near_catalog_fallback_used: false
blind4d_catalog_mode_effective: library-view
blind4d_external_fallback_used: false
Done in 40.1s - 1 solved, 0 skipped, 0 failed
```

The Linux validation confirms READY_FULL runtime resolution without using the
stale external manifest.

## Residual Risks

- The final Windows Release Candidate gate still needs to be performed on the
  exact user-facing package/profile described in the mission.
- This mission did not promote `test` to `main`.
- This report does not declare `PRODUCTION_READY`.

## Verdict

P3B1F_STARTUP_WIZARD_CATALOG_ACTIVATION_CLOSED

READY_FOR_RELEASE_CANDIDATE_ACCEPTANCE
