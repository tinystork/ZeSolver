# P3B-1F Startup Wizard Report - 2026-07-31

Status: `READY_FOR_P3B1F_WIZARD_VALIDATION`

## Initial Behavior

Baseline from the mission context was reproduced conceptually from the current GUI entry points:

- `Interface` already exposed an action wired to `_run_startup_wizard_from_menu`.
- `_run_simple_startup_wizard()` only checked already configured resources.
- No deferred first-launch wizard was scheduled after `window.show()`.
- The legacy "Nouvelles bases detectees" family prompt could still be reached from the DB family scan.

The active branch was `test` at `71743cc` when implementation started. The context reference `09d8606` is the previous commit in the branch history.

## Architecture

Added a dedicated module:

`zesolver/gui_startup_wizard.py`

The module separates:

- startup decision policy: `decide_startup_wizard()`
- explicit persistence helpers: `mark_startup_wizard_completed()`, `clear_invalid_catalog_selection()`
- legacy prompt policy: `should_allow_legacy_family_prompt()`
- catalog operations worker: `StartupCatalogWorker`
- Qt presentation: `ZeSolverStartupWizard`

`zesolver.py` now only owns integration:

- schedules the wizard after `window.show()` with `QTimer.singleShot`
- keeps the menu action relaunchable
- applies selected library/ASTAP paths to the existing main-window settings and status refresh methods
- suppresses the legacy rebuild prompt unless `near_catalog_mode == "legacy-index"`

No ZeNear or ZeBlind solving algorithm was changed.

## State Machine

Decision states:

- `fresh`: no usable catalog configuration; auto-show wizard.
- `ready`: configured ZeSolver library is usable; auto-show only if wizard is not completed for the current wizard version.
- `repair`: configured library path is missing/invalid and must not be trusted silently.
- `astap_near_only`: ASTAP root is configured without a ZeSolver library; proposes ZeNear-only plus full library install.
- `later`: explicit "Configurer plus tard" path, with no automatic relaunch while no broken catalog path is configured.

`READY_FULL` does not force any download. A completed wizard with a still-valid library does not relaunch automatically. A completed wizard whose resources disappeared is not silently trusted.

## Persistence

Settings schema was bumped to `14`.

New fields:

```text
startup_wizard_version
startup_wizard_completed
```

Rules:

- completion is explicit and versioned;
- `catalog_library_path` is only persisted after an operation succeeds and the existing validation path accepts it;
- "Configurer plus tard" clears invalid library selection and marks the wizard completed without creating a fake catalog or relaunch loop;
- closing/cancelling a running worker requests cancellation and does not present a partial library as valid.

## User Flow

Normal wizard pages:

1. Welcome and diagnostic
2. Catalog source
3. Destination and disk preview
4. Installation and validation
5. Image folder and essential settings
6. Summary

Catalog source choices:

- Install official ZeSolver library
- Use existing ZeSolver library
- Use existing ASTAP database - ZeNear only
- Install local package - advanced

The normal wizard does not expose historical Near indexes, S/M/L indexes, external Blind 4D manifests, legacy profiles, or hash reconstruction parameters.

## Service Reuse

The worker delegates to existing services:

- `CatalogDistributionService`
- `CatalogLibraryManagementService`
- `default_library_parent`
- `default_cache_root`
- `resolve_library_destination`
- `validate_library_parent`
- `build_storage_plan`
- `cleanup_distribution_cache` remains in the existing distribution service surface
- `CatalogLibrary` validation and `resolve_catalog_resources`

No parallel download engine was introduced. Official installation still uses the distribution service with resume, staging, hash verification, package assembly and final validation.

## Worker Handling

Heavy work is kept outside the Qt main thread through `StartupCatalogWorker`:

- official distribution discovery/download/install
- package install
- existing library validation/resource resolution
- ASTAP family detection

Cancellation uses a shared event exposed through `request_cancel()`. Closing the wizard requests cancellation and waits briefly for the worker.

## Linux Validation

Fresh profile offscreen validation command shape:

```bash
TEST_HOME=$(mktemp -d)
mkdir -p "$TEST_HOME/.config"
HOME="$TEST_HOME" XDG_CONFIG_HOME="$TEST_HOME/.config" QT_QPA_PLATFORM=offscreen .venv/bin/python <gui-probe>
```

Observed:

```json
{
  "classes_initial": ["ZeSolverWindow", "ZeSolverStartupWizard"],
  "install_parent": "/tmp/tmp.zY2kBhMeQn/ZeSolverCatalog/libraries",
  "legacy_questions": 0,
  "wizard_visible_auto": true,
  "wizard_visible_menu": true
}
```

Real-profile scenario was simulated read-only with:

```text
catalog_library_path=/home/tristan/ZeSolverCatalog/new
db_root=/opt/astap
index_root=/tmp/s6b1-gui-r58n5dd4/index
```

Observed:

```json
{
  "auto_wizard_visible": false,
  "catalog_text": "/home/tristan/ZeSolverCatalog/new",
  "legacy_questions": 0,
  "menu_wizard_visible": true,
  "state": "READY_FULL",
  "window_visible": true
}
```

This confirms priority to the valid ZeSolver library and no automatic reconstruction prompt from the temporary legacy index path.

## Tests

Targeted startup/settings/catalog GUI tests:

```text
.venv/bin/python -m pytest -q tests/test_startup_wizard.py tests/test_settings_persistence.py tests/test_gui_catalog_library_control.py
24 passed in 1.14s
```

Expanded catalog/distribution/GUI smoke:

```text
.venv/bin/python -m pytest -q tests/test_startup_wizard.py tests/test_catalog_distribution.py tests/test_catalog_library_paths.py tests/test_catalog_library_management_service.py tests/test_gui_catalog_library_manager.py tests/test_gui_development_surface_reorganized.py tests/test_gui_catalog_library_control.py tests/test_settings_persistence.py tests/test_settings_migration_v2.py tests/test_solver_profiles.py tests/test_engine_selection.py tests/test_core_import_isolation.py
92 passed in 12.94s
```

Full suite status:

```text
.venv/bin/python -m pytest -q
collection error: tests/test_catalog_blind4d_manifest_view_cli.py imports missing tools.generate_blind4d_manifest_view
```

After installing the missing environment dependency documented by the S6A1C tool:

```text
.venv/bin/python -m pip install "threadpoolctl>=3.6,<4"
.venv/bin/python -m pytest -q tests/test_s6a1c_native_threading.py
6 passed in 2.55s
```

Suite excluding only the known missing-tool collection file:

```text
.venv/bin/python -m pytest -q --ignore=tests/test_catalog_blind4d_manifest_view_cli.py
744 passed, 20 skipped, 17 warnings in 44.30s
```

## Limits

- The full suite still cannot be reported as wholly green because `tools.generate_blind4d_manifest_view` is absent while `tests/test_catalog_blind4d_manifest_view_cli.py` imports it during collection.
- The official catalog download/install path is wired through the existing service and worker, but this run did not download the real remote catalog assets.
- The visual manual test in a real display server remains to be performed by launching without `QT_QPA_PLATFORM=offscreen`.

## Windows Validation Instructions

Use a fresh Windows user profile or temporary config root equivalent, then run:

```powershell
$env:HOME = "$env:TEMP\\zesolver-empty-home"
$env:XDG_CONFIG_HOME = "$env:HOME\\.config"
New-Item -ItemType Directory -Force $env:XDG_CONFIG_HOME | Out-Null
.venv\\Scripts\\python.exe zesolver.py
```

Check:

- wizard appears after the main window is visible;
- no legacy "new databases" rebuild prompt appears;
- default library parent is under `%USERPROFILE%\\ZeSolverCatalog\\libraries`;
- cancel closes cleanly and leaves no invalid `catalog_library_path`;
- menu `Interface` can relaunch the wizard.

For a real Windows profile with an existing valid library, verify:

- no automatic redownload is forced;
- the library appears ready;
- `Interface` still relaunches the wizard manually.
