# P3A-V3 GUI Live State Report

## Root Causes

WCS cleanup left stale GUI state because `_run_simple_mode_wcs_cleaning()` only
called `zewcscleaner.process_fits()` and logged the summary. It did not refresh
the cached `QTreeWidgetItem` status, `UserRole`, detail, or foreground color.

Legacy progress stayed frozen because `BatchSolver._queue_result()` invoked the
immediate `on_result` callback, but `SolveRunner._legacy_results()` used that
callback only for GC. The GUI saw legacy results later through
`LegacyGuiRunner`, after the phase yield queue was drained.

## WCS Contract

For the main solver list, `WCS present` means:

```text
FITS: PRIMARY HDU has usable WCS cards
Raster: valid sidecar .wcs.json exists
```

If a FITS extension still contains WCS but PRIMARY does not, the main status is
`waiting`; the detail column notes the extension WCS.

## Changes

- Added `EffectiveWcsState` and `inspect_effective_wcs_state()`.
- Reused that helper from `_quick_scan_initial_status()`.
- Refreshed GUI rows immediately after WCS cleanup.
- Added a shared item-status updater for status text, `UserRole`, detail, and
  color.
- Added live-result callback support in `LegacyGuiRunner`.
- Connected `BatchSolver.on_result` to the legacy live callback.
- Deduplicated legacy notifications by resolved path.
- Deduplicated pipeline final replay by resolved path.
- Added GUI progress guards by active `run_id`.
- Added GUI progress deduplication by resolved path.
- Added translated remaining counters.
- Kept the progress timer bounded inside the next file bucket.
- Kept cancellation progress honest instead of forcing 100 percent.

## Automated Tests

Added:

```text
tests/p3av3_helpers.py
tests/test_gui_wcs_cleanup_refresh.py
tests/test_gui_wcs_status_scope.py
tests/test_gui_progress_realtime_legacy.py
tests/test_gui_progress_no_duplicates.py
tests/test_gui_progress_remaining_count.py
tests/test_gui_progress_pipeline.py
tests/test_gui_progress_stop_restart.py
tests/test_gui_progress_stale_callback.py
```

Results:

```text
core boundary check: OK
targeted P3A-V3/P3A-V2: 18 passed
hermetic: PASS (424 passed, 1 skipped, 9 deselected, 44 warnings)
full pytest: 424 passed, 10 skipped, 44 warnings
compileall: OK
git diff --check: OK
```

Warnings remained in known categories:

```text
multiprocessing.popen_fork.DeprecationWarning
astropy.io.fits.card.VerifyWarning
```

## Manual Status

The Wayland GUI session was available, but the scripted manual validation was
not completed. Each harness attempt was overtaken by an automatic large scan
before the scenario could proceed, producing repeated:

```text
N fichier(s) detecte(s)
```

up to more than 100k entries. The harness processes were stopped and no manual
PASS is claimed for P3A-V3.

## Decision

```text
NOT_READY_FOR_P3B_GUI_SIMPLIFICATION
```

Reason: code and automated regressions are green, but the required P3A-V3
Wayland manual validation did not complete.
