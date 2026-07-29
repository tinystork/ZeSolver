# P3B-1E - Official Catalog Distribution

Date: 2026-07-29

## Initial State

- Branch: `test`
- Upstream: `origin/test`
- Worktree at mission start: dirty, with pre-existing local edits in `zesolver.py`, settings, pipeline/core modules, GUI tests, and several untracked S6B2/S6B3 files.
- Baseline targeted tests before edits:
  - `pytest -q tests/test_catalog_library_management_service.py tests/test_gui_catalog_library_manager.py tests/test_gui_catalog_library_control.py tests/test_catalog_library_validation.py tests/test_catalog_blind4d_manifest_view.py`
  - Result: `51 passed, 1 skipped`
  - Skip reason: `PySide6` not installed in this environment.

## Located Integration Points

- `CatalogLibraryManagementService`, `LibraryInstallOptions`, `install_package()`, `_materialize_package()`, `_verify_package_hashes()`, `_extract_zip()`, `_validate_archive_member()`: `zesolver/catalog_library/management.py`
- `build_blind4d_manifest_view()`: `zesolver/catalog_library/blind4d_view.py`
- `PersistentSettings.catalog_library_path`, `catalog_library_verification`, `db_root`, immediate save: `zesolver/settings_store.py`
- Graphical library manager and former Database tab construction: `zesolver.py`
- Historical downloads manager and `UrllibBackend`: `zeblindsolver/downloads.py`
- Startup wizard: `_run_simple_startup_wizard()` in `zesolver.py`, intentionally not modified.

## Manifest Contract Observed

Official Release inspected through the GitHub Releases API:

- Repository: `tinystork/ZeSolver-Catalogs`
- Stable tag observed: `d50-v1.1.0`
- Manifest asset: `zesolver-d50-distribution-v1.1.0.json`
- Schema: `zesolver.catalog_distribution.v1`
- Installation model: `merge-assets-into-one-package-root`
- Required top-level fields used: `schema`, `format_version`, `library_id`, `version`, `installation_model`, `catalog_path`, `package_metadata`, `components`, `capabilities`, `installed_size_bytes`
- Component fields used: `id`, `asset`, `required`, `sha256`, `size_bytes`, optional `target`, `installed_size_bytes`, `file_count`

The code does not hardcode asset names, SHA-256, component count, or browser download URLs.

## Architecture

Added GUI-free service:

- `zesolver/catalog_library/distribution.py`

Main pieces:

- `CatalogDistributionService`: release discovery, manifest fetch/parse, install planning, download, assembly, install orchestration.
- `ResumableAssetDownloader`: `.part` downloads with `Range`, `If-Range`, ETag/Last-Modified tracking, 206 validation, 200 restart, 416 restart, cancellation preserving partial files, size and SHA-256 verification.
- Typed frozen models: `DistributionRelease`, `DistributionManifest`, `DistributionComponent`, `DistributionInstallPlan`, `DistributionProgress`, `DistributionInstallResult`.
- Stable error codes via `DistributionErrorCode`.

The GUI only consumes progress events and install results. It does not know HTTP or ZIP extraction details.

## Cache Policy

Default cache:

- Linux/macOS: `~/.cache/ZeSolver/catalogs/<library_id>/<version>/`
- Windows: `%LOCALAPPDATA%/ZeSolver/catalogs/<library_id>/<version>/` when available.

Validated assets are retained after install for repair or reinstall. Existing files are reused only when both size and SHA-256 match.

## HTTP Resume Policy

- Download target is `<asset>.part` until verified.
- Existing partial size becomes `Range: bytes=<offset>-`.
- `If-Range` uses cached ETag first, then Last-Modified.
- `206 Partial Content` is required for a true resume.
- `200 OK` while resuming discards the stale partial and restarts.
- `416` discards the stale partial and retries from zero.
- Changed ETag/Last-Modified discards the stale partial.
- Cancellation leaves `.part` in place.
- Final promotion happens only after exact size and SHA-256 match.

## Assembly Policy

The assembler creates a package staging root directly, without building a complete temporary ZIP.

Implemented rules:

- Metadata component is extracted first.
- Data components are extracted only under their manifest `target`.
- Root `component.json` files from component ZIPs are not published at package root.
- Component descriptors are preserved under `.components/<component_id>.json`.
- `NOTICE.md` and `legal/` collisions are allowed only when content is identical.
- Any other collision with different bytes is refused.
- Absolute paths, `..`, malformed paths, and ZIP symlinks are refused.
- Extraction outside the declared target is refused.

## Management Refactor

`CatalogLibraryManagementService.install_package()` now delegates to:

- `install_materialized_package(package_root, destination)`

This shared primitive handles package metadata checks, disk space, package SHA-256 verification, `CatalogLibrary.open()`, `library.validate()`, Blind4D view validation, atomic publication, and final `LibraryOperationResult`.

Local package install remains compatible.

## GUI

The standard tab formerly titled `Database` / `Base de données` is now:

- FR: `Bibliothèque ZeSolver`
- EN: `ZeSolver Library`

Visible standard path:

- Current library state and location.
- Verify / Repair / Open folder.
- Official distribution discovery.
- Install recommended complete library.
- Cancel while installing.
- Detailed log toggle.
- Advanced options button opening the existing library manager.

The old generic source list, manual URL box, copy URL button, and generic downloads queue are no longer reached by the standard tab builder.

The existing local library manager is preserved for:

- local package install,
- create from ASTAP,
- verify/repair.

## Atomic Activation

Official install flow:

1. Discover Release.
2. Fetch manifest.
3. Build install plan.
4. Download and verify required assets into cache.
5. Assemble into package staging.
6. Delegate final install to `CatalogLibraryManagementService.install_materialized_package()`.
7. Update `PersistentSettings.catalog_library_path`.
8. Update `catalog_library_verification`.
9. Save settings immediately.
10. Refresh GUI validation state.

The active library path is changed only after successful final validation.

## Files Added

- `zesolver/catalog_library/distribution.py`
- `tests/test_catalog_distribution.py`
- `docs/stabilization/p3b1e_official_catalog_distribution_report_20260729.md`

## Files Modified By This Mission

- `zesolver/catalog_library/__init__.py`
- `zesolver/catalog_library/management.py`
- `zesolver.py`

Note: `zesolver.py` already had unrelated local edits before this mission; they were preserved.

## Tests

Commands run:

- `python -m py_compile zesolver/catalog_library/distribution.py zesolver/catalog_library/management.py zesolver/catalog_library/__init__.py`
  - Result: passed.
- `pytest -q tests/test_catalog_distribution.py tests/test_catalog_library_management_service.py`
  - Result: `34 passed`.
- `python -m py_compile zesolver.py zesolver/catalog_library/distribution.py zesolver/catalog_library/management.py && pytest -q tests/test_catalog_distribution.py tests/test_catalog_library_management_service.py tests/test_gui_catalog_library_control.py tests/test_gui_catalog_library_manager.py`
  - Result: `40 passed, 1 skipped`.
  - Skip reason: `PySide6` missing.
- `pytest -q tests/test_catalog_distribution.py tests/test_catalog_library_management_service.py tests/test_catalog_library_validation.py tests/test_catalog_library_blind4d_integration.py tests/test_catalog_library_adopted_runtime.py tests/test_catalog_library_pipeline_integration.py tests/test_catalog_blind4d_manifest_view.py tests/test_downloads.py tests/test_gui_catalog_library_control.py tests/test_gui_catalog_library_manager.py`
  - Result: `71 passed, 1 skipped, 2 failed`.
  - The 2 failures are environment import failures: `astroalign` is not installed, and `tests/test_catalog_library_pipeline_integration.py` imports `zesolver.py` directly.

## Known Limits

- Full real-asset GitHub installation was not executed here to avoid multi-GB downloads during development validation.
- GUI runtime screenshots were not captured because `PySide6` is unavailable in this environment.
- Version comparison is currently equality/difference oriented in the first GUI version; richer semver/newer-than presentation can be added when multiple official versions exist.
- The old database/download code remains in `zesolver.py` after the new tab's return path, but is no longer visible in the standard tab. It can be removed in a cleanup-only mission after GUI validation.

## Follow-Up Launch Fix

After first real launch, the new Library tab restored the CatalogLibrary verification cache before the Settings tab had created `settings_log_view`. `_log_settings()` is now safe during early GUI construction: it writes early messages to the Python logger until the widget exists.

Validation after fix:

- `.venv/bin/python -m py_compile zesolver.py zesolver/catalog_library/distribution.py zesolver/catalog_library/management.py && .venv/bin/python -m pytest -q tests/test_gui_catalog_library_control.py tests/test_catalog_distribution.py tests/test_catalog_library_management_service.py`
  - Result: `41 passed`.
- `timeout 12s env QT_QPA_PLATFORM=offscreen .venv/bin/python zesolver.py`
  - Result: process stayed alive until timeout, no traceback. Expected early log observed: `GUI settings log before widget ready`.

## Follow-Up HTTP 416 Fix

Review found that `urllib` raises `HTTPError(416)` before `_download_once()` can inspect `response.status`. `DistributionError` now preserves the original `http_status`, allowing the downloader to recognize functional 416 responses.

Implemented behavior:

- If a resume request gets HTTP 416 and a `.part` exists, delete both `.part` and `.part.json`.
- Restart exactly once from byte zero.
- If another 416 occurs during that restart path, raise `DISTRIBUTION_DOWNLOAD_RANGE_INVALID`.
- Validate that `Content-Range` on HTTP 206 starts exactly at the requested offset.

Validation after fix:

- `.venv/bin/python -m py_compile zesolver/catalog_library/distribution.py && .venv/bin/python -m pytest -q tests/test_catalog_distribution.py`
  - Result: `16 passed`.
- `.venv/bin/python -m pytest -q tests/test_catalog_distribution.py tests/test_catalog_library_management_service.py tests/test_gui_catalog_library_control.py`
  - Result: `43 passed`.

## Follow-Up Close Event Fix

After the new Library tab replaced the visible legacy Database tab, `_dl_worker` was no longer initialized by the skipped legacy downloads UI path. `closeEvent()` still attempted to stop it on exit, producing `AttributeError: 'ZeSolverWindow' object has no attribute '_dl_worker'`.

Implemented behavior:

- Initialize `self._dl_worker = None` during `ZeSolverWindow.__init__`.
- Use `getattr(self, "_dl_worker", None)` in `closeEvent()` before shutdown.

Validation after fix:

- `.venv/bin/python -m py_compile zesolver.py && .venv/bin/python -m pytest -q tests/test_gui_catalog_library_control.py tests/test_catalog_distribution.py tests/test_catalog_library_management_service.py`
  - Result: `44 passed`.
- Offscreen GUI open + close using the real `zesolver.py` entrypoint and real parser defaults.
  - Result: exit code `0`, no traceback, no Qt thread destruction warning.

## Wizard Preparation

The wizard was not modified.

For the next mission, reuse `CatalogDistributionService` directly from the wizard:

- run `fetch_latest_distribution()` asynchronously;
- present the same single logical library choice;
- call `build_install_plan()` and `install_distribution()`;
- keep `db_root` as advanced/legacy fallback only;
- persist only after `DistributionInstallResult` succeeds.
