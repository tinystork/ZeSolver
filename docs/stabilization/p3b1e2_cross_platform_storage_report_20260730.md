# P3B-1E2 - Cross-Platform Catalog Storage

Date: 2026-07-30

## Initial Git State

- Branch: `test`
- Upstream: `origin/test`
- `git status --short`: clean at mission start.
- `git diff --check`: clean at mission start.
- Baseline targeted tests before edits:
  - `.venv/bin/python -m pytest -q tests/test_catalog_distribution.py tests/test_catalog_library_management_service.py tests/test_gui_catalog_library_control.py`
  - Result: `44 passed`.

## Architecture Before

P3B-1E already separated concerns correctly:

- ZIP assets were downloaded into the versioned cache.
- `assemble_distribution()` created a materialized package staging directory beside the destination.
- `CatalogLibraryManagementService.install_materialized_package()` validated package metadata, internal SHA-256, `CatalogLibrary`, and Blind4D view before publishing.
- The GUI did not display the final destination, cache path, or disk preflight before install.

Real file flow before this mission:

```text
cache download ZIPs
  -> package assembly staging beside destination
  -> install_materialized_package moves staging/library to final destination
```

There was no full library copy from the cache volume to the destination volume, but the destination was implicit and no preflight prevented a doomed install.

## Architecture After

New GUI-free module:

- `zesolver/catalog_library/paths.py`

It provides reusable primitives for the Library tab and the future wizard:

- `default_library_parent()`
- `default_cache_root()`
- `resolve_library_destination()`
- `validate_library_parent()`
- `same_filesystem()`
- `volume_key()`
- `build_storage_plan()`
- `format_bytes_binary()`
- `cache_reclaimable_bytes()`
- `cleanup_distribution_cache()`
- `open_in_file_manager()`

`CatalogDistributionService` now consumes these primitives for default cache resolution, destination resolution, preflight checks, staging naming, and cache cleanup.

## Default Paths

Library parent:

- Windows: `%USERPROFILE%\ZeSolverCatalog\libraries`
- Linux: `~/ZeSolverCatalog/libraries`
- macOS: `~/ZeSolverCatalog/libraries`

Download cache:

- Windows: `%LOCALAPPDATA%\ZeSolver\catalogs`
- Windows fallback: `%USERPROFILE%\AppData\Local\ZeSolver\catalogs`
- Linux: `$XDG_CACHE_HOME/ZeSolver/catalogs`
- Linux fallback: `~/.cache/ZeSolver/catalogs`
- macOS: `~/Library/Caches/ZeSolver/catalogs`

The final library directory is always manifest-derived:

```text
<install_parent>/<library_id>-v<version>
```

No path depends on `sys._MEIPASS`, the executable directory, the source repository, the current working directory, or a machine-specific user name.

## Parent Persistence

New setting:

- `PersistentSettings.catalog_library_install_parent`

Meaning:

- `catalog_library_path`: active validated library.
- `catalog_library_install_parent`: preferred parent for future official installs.

Changing the parent in the Library tab saves the parent immediately and does not alter the active library.

Older settings files remain compatible because the new key defaults to `None`.

## Validation Rules

`validate_library_parent()` checks:

- absolute path;
- parent exists or can be created;
- writable parent;
- final destination does not already exist;
- final destination is not inside the cache;
- final destination is not inside application/resource roots;
- macOS `.app` bundle exclusion;
- `.partial` staging path exclusion;
- Windows reserved path components;
- small create/rename/delete probe when requested.

UNC, mounted, or removable-looking destinations are not refused automatically; the result can carry a warning.

## Volume Detection

`same_filesystem()` uses:

- Windows drive identity when available;
- POSIX `st_dev` on the nearest existing parent;
- documented fallback to anchor/volume text if stat resolution fails.

`shutil.disk_usage()` is evaluated on the nearest existing parent so a not-yet-created install parent can still be preflighted.

## Disk Formula

The storage plan distinguishes:

- total download size;
- validated cache bytes;
- partial `.part` bytes;
- remaining download bytes;
- declared installed size;
- safety margin;
- same-volume versus split-volume requirements.

Safety margin:

```text
max(256 MiB, 5% of remaining download + partial bytes + installed size)
```

Same volume:

```text
remaining download + partial bytes + installed size + margin
```

Separate volumes:

```text
cache volume: remaining download + partial bytes + margin
destination volume: installed size + margin
```

The calculation uses manifest `size_bytes` and `installed_size_bytes`; it does not hardcode D50 size or version.

## Staging And Atomicity

The download cache keeps only ZIP assets.

The materialized package staging is created beside the destination:

```text
<install_parent>/.<library_id>-v<version>.partial-<uuid>
```

`install_materialized_package()` then publishes the validated `library/` subdirectory to the final destination on the same filesystem. This avoids an atomic rename attempt from the cache volume to another drive.

The active `catalog_library_path` is updated only after final validation succeeds.

## GUI Changes

The standard `Bibliothèque ZeSolver` tab now shows before install:

- preferred install parent;
- final destination;
- download remaining;
- installed size;
- estimated temporary peak;
- per-volume required and available space;
- cache path.

Added actions:

- `Modifier...` chooses a parent folder and persists it.
- `Ouvrir le cache` opens the version cache/root via the centralized file-manager helper.
- `Vider cette version` removes only validated assets for the current version when no install is active.

The install button is disabled when preflight reports insufficient space or an invalid destination.

## Cache Policy

Validated downloaded assets are kept by default for repair/reinstall.

Explicit cleanup:

- targets only assets listed by the current manifest/plan;
- preserves `.part` files;
- preserves other versions;
- refuses cleanup while an install operation is active.

## Files Modified

- `zesolver/catalog_library/paths.py`
- `zesolver/catalog_library/distribution.py`
- `zesolver/catalog_library/__init__.py`
- `zesolver/settings_store.py`
- `zesolver.py`
- `tests/test_catalog_library_paths.py`
- `tests/test_gui_catalog_library_control.py`
- `docs/stabilization/p3b1e_official_catalog_distribution_report_20260729.md`
- `docs/stabilization/p3b1e2_cross_platform_storage_report_20260730.md`

## Tests

Commands run:

- `.venv/bin/python -m pytest -q tests/test_catalog_library_paths.py`
  - First result: `5 passed, 2 failed`.
  - Cause: test fixtures simulated only 1 KiB/10 KiB free space while product margin is 256 MiB.
  - Fixture corrected.
- `.venv/bin/python -m py_compile zesolver.py zesolver/settings_store.py zesolver/catalog_library/distribution.py zesolver/catalog_library/paths.py`
  - Result: passed.
- `.venv/bin/python -m pytest -q tests/test_catalog_library_paths.py tests/test_catalog_distribution.py tests/test_catalog_library_management_service.py tests/test_gui_catalog_library_control.py`
  - Result: `51 passed`.
- Offscreen GUI launch through `launch_gui()` with real parser defaults and patched `QApplication.exec`.
  - Result: `offscreen_exit=0`.
  - Observed early log remains safe: `GUI settings log before widget ready`.
- `.venv/bin/python -m pytest -q`
  - Result: `742 passed, 10 skipped, 59 warnings`.
  - Skips are external-data/configuration gates: `ZESOLVER_ZN310B_ROOT`, external ASTAP/HNSKY database, S50 frame/index, Blind4D manifest/corpus, and Near/Pipeline corpora.

## Manual Validation

Performed physically in this environment:

- Linux source-tree launch path via offscreen Qt construction.
- Non-GUI path, cache, storage, cleanup tests using injected Windows/Linux/macOS platforms.

Not physically performed here:

- Windows multi-GB installation.
- macOS multi-GB installation.
- Linux real multi-GB GitHub asset install.

## Manual Protocol

Windows:

1. Start from a fresh settings file.
2. Open `Bibliothèque ZeSolver`.
3. Confirm default cache is under `%LOCALAPPDATA%\ZeSolver\catalogs`.
4. Confirm default install parent is `%USERPROFILE%\ZeSolverCatalog\libraries`.
5. Start install, cancel during first component, restart and verify resume.
6. Change parent to a second drive such as `D:\Astronomie\ZeSolverCatalog\libraries`.
7. Confirm C: shows cache requirement and D: shows library requirement.
8. Complete install.
9. Restart ZeSolver and verify `READY_FULL`.
10. Run one Near solve and one Blind4D solve.
11. Use `Vider cette version` and verify only current-version downloaded assets are removed.

Linux:

1. Confirm cache in `$XDG_CACHE_HOME/ZeSolver/catalogs` or `~/.cache/ZeSolver/catalogs`.
2. Confirm install parent in `~/ZeSolverCatalog/libraries`.
3. Repeat cancel/resume/install/cache cleanup.
4. Optionally test a mounted destination under `/mnt` or `/media`.

macOS:

1. Confirm cache in `~/Library/Caches/ZeSolver/catalogs`.
2. Confirm install parent in `~/ZeSolverCatalog/libraries`.
3. Confirm no path is inside `/Applications` or `ZeSolver.app`.
4. Repeat cancel/resume/install/cache cleanup.

## Limits

- `format_bytes_binary()` uses `Gio`/`Tio` consistently as requested; it does not localize decimal punctuation per language yet.
- The GUI currently presents cleanup as an advanced action and a completion detail; it does not show a modal two-choice completion panel.
- The legacy startup wizard was intentionally not modified.

## Wizard Status

The wizard was not modified in this mission.

Recommended next mission:

- Reuse `paths.py` and `CatalogDistributionService.build_storage_plan()` inside the wizard.
- Keep the wizard as a thin consumer: discover release, show storage plan, choose parent, install, then activate only after `install_distribution()` succeeds.
