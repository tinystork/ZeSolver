# P1D-5A — GUI CatalogLibrary Activation

Decision: `READY_FOR_P1D5B_ADVANCED_COMPATIBILITY_SURFACE`

P1D-5A makes the CatalogLibrary path selectable, validated, persisted and used
from the existing GUI product surface.  It does not remove the legacy fields
or reorganize the settings surface; that remains P1D-5B.

## 1. Objectif

Allow the GUI user to select:

```text
ZeSolver library -> root containing catalog.json
```

and then run with:

```text
Near -> astap_native
Blind 4D -> catalog_library_view
```

without manually editing the settings file and without requiring `db_root`,
`index_root` or `blind_4d_manifest_path` in the normal CatalogLibrary path.

## 2. Etat Git initial

Initial state:

```text
git status --short --branch
## test...origin/test
... P1D-4B and previous P1D changes present and uncommitted ...

git rev-parse HEAD
f8e92b3b6e80de16f04c9f55b5841449c7701446

git diff --check
OK
```

No commit or push was performed.

## 3. Surface GUI avant/apres

Before P1D-5A, the GUI could persist `catalog_library_path` through lower
layers but had no normal visible selector.  Runs still appeared as:

```text
catalog_source = legacy
catalog_library_id = null
near_catalog_provider = legacy_index
blind4d_catalog_source = external_manifest
```

After P1D-5A, the Settings tab has a visible selector and status line:

```text
ZeSolver library [path] [Browse...] [Verify] [Clear]
```

Legacy fields remain visible for compatibility and rollback pending P1D-5B.

## 4. Controle de selection

The selector asks for the library root, not `catalog.json`.  It initializes
from `PersistentSettings.catalog_library_path`, expands `~`, accepts spaces and
non-ASCII paths, and never writes into the library.

`Clear` persists `catalog_library_path = None` and logs that library files are
not deleted.

## 5. Validation et statuts

Validation uses:

```text
CatalogLibrary.open()
CatalogLibrary.validate()
resolve_catalog_resources(catalog_library=...)
```

Stable visible states:

```text
AUCUNE_BIBLIOTHEQUE
VALIDATION_EN_COURS
READY_FULL
READY_PARTIAL
NEAR_ONLY
BLIND4D_ONLY
INVALID
MISSING
```

The current adopted library reports:

```text
READY_PARTIAL
Near ASTAP: D50, 1476 tiles
Blind 4D: 6 indexes / 1476 tiles
Global Blind 4D coverage: no
```

Partial Blind 4D coverage is visible and non-blocking.

## 6. Persistance

`_read_settings_from_ui()` now reads the visible library widget as the source
of truth.  Empty string becomes `None`.  Existing settings without the field
still load, and old fields are preserved.

The real GUI smoke persisted:

```text
catalog_library_path = /tmp/p1d5a_gui_library_DerPSR/library
db_root = null
index_root = null
blind_4d_manifest_path = /tmp/p1d5a_gui_invalid_blind4d_manifest_DO_NOT_USE.json
```

On relaunch the GUI restored the same library path.

## 7. Regles de configuration

With a valid CatalogLibrary, `db_root`, `index_root` and
`blind_4d_manifest_path` are optional for the normal path.

With no library, legacy requirements are preserved.

With an invalid explicit library, the run is blocked and no legacy fallback is
attempted.

## 8. Logique du wizard

The simple wizard checks the library first:

```text
valid library -> ready
invalid library -> verify or select another root
no library -> legacy compatibility checks
```

It does not adopt or create a library.

## 9. Routage Near

In `auto` with the temporary CatalogLibrary:

```text
near_catalog_mode_requested = auto
near_catalog_mode_effective = astap-native
near_catalog_provider = astap_native
near_catalog_source = library
near_catalog_fallback_used = false
```

The invalid legacy index sentinel was not consulted.

## 10. Routage Blind 4D

In `auto` with the temporary CatalogLibrary:

```text
blind4d_catalog_mode_requested = auto
blind4d_catalog_mode_effective = library-view
blind4d_catalog_source = catalog_library_view
blind4d_index_count = 6
blind4d_covered_tiles = 6
blind4d_total_tiles = 1476
blind4d_all_sky = false
blind4d_external_fallback_used = false
```

The invalid external manifest sentinel was not consulted.

## 11. Rollback

Forced rollback remains functional and visible:

```text
near_catalog_mode_effective = legacy-index
near_catalog_provider = legacy_index
blind4d_catalog_mode_effective = external-manifest
blind4d_catalog_source = external_manifest
```

The rollback smoke solved `1/1` copied FITS and logged the forced modes.

## 12. Absence de fallback silencieux

Validated cases:

```text
library + invalid legacy sentinels -> astap_native + catalog_library_view
invalid selected library -> run blocked, worker_started=false
explicit rollback -> legacy/external only because requested
```

No normal library run emitted `legacy_blind4d_manifest_used`.

## 13. Timings de preflight

CatalogLibrary GUI run timings:

```text
catalog_library_open_s = 0.2522
catalog_resource_resolution_s = 0.4530
near_runtime_resolution_s = 0.3288
blind4d_runtime_resolution_s = 0.2595
catalog_preflight_total_s = 1.2961
```

Stop/relaunch second run:

```text
catalog_library_open_s = 0.2101
catalog_resource_resolution_s = 0.4504
near_runtime_resolution_s = 0.0078
blind4d_runtime_resolution_s = 0.2557
catalog_preflight_total_s = 0.9261
```

No optimization or cache policy change was made in this mission.

## 14. Tests cibles

Targeted GUI/catalog tests:

```text
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest \
  tests/test_gui_catalog_library_control.py \
  tests/test_gui_catalog_library_solve_config.py \
  tests/test_catalog_library_blind4d_product_switch.py \
  tests/test_catalog_library_near_runtime_unification.py \
  tests/test_catalog_library_no_silent_legacy_fallback.py \
  tests/test_gui_progress_pipeline.py \
  tests/test_gui_progress_realtime_legacy.py \
  tests/test_gui_progress_no_duplicates.py \
  tests/test_gui_progress_remaining_count.py \
  tests/test_gui_progress_stop_restart.py \
  tests/test_gui_progress_stale_callback.py \
  tests/test_gui_completion_exactly_once.py \
  tests/test_gui_completion_after_stop.py \
  tests/test_gui_completion_after_restart.py \
  tests/test_gui_stale_completion_callback.py \
  tests/test_gui_cancellation.py \
  tests/test_gui_lifecycle.py \
  tests/test_gui_wcs_cleanup_refresh.py \
  tests/test_gui_wcs_status_scope.py \
  -q

34 passed
```

Additional focused set:

```text
27 passed
```

## 15. Barrieres generales

Final barrier results:

```text
tools/check_core_boundaries.py: OK
tools/run_regression_suite.py --hermetic: PASS, 563 passed, 1 skipped, 9 deselected
pytest -q: 563 passed, 10 skipped
compileall: OK
git diff --check: OK
```

## 16. Validation externe

A temporary library was adopted from:

```text
/opt/astap
config/zeblind_4d_experimental_manifest.json
/home/tristan/zesolver_index
```

External automated routing with invalid sentinels returned:

```text
catalog_source = library
catalog_library_status = READY_PARTIAL
near_catalog_provider = astap_native
blind4d_catalog_source = catalog_library_view
blind4d_external_fallback_used = false
coverage = PARTIAL 6 / 1476
```

## 17. Validation graphique reelle

Real Qt graphical validation was run without `QT_QPA_PLATFORM=offscreen`.

CatalogLibrary case:

```text
5/5 SOLVED
progress = 500 / 500
terminal_count = 1
```

Stop and relaunch:

```text
STOP_RUNNER_RECEIVED
STOP_CONTROLLER_RECEIVED
first run terminal_count = 1
relaunch = 5/5 SOLVED
relaunch terminal_count = 1
```

Invalid library:

```text
status = MISSING
message = CATALOG_LIBRARY_MANIFEST_MISSING
worker_started = false
```

Explicit rollback:

```text
1/1 SOLVED
legacy-index + external-manifest visible in log
```

Detailed evidence is in `docs/stabilization/p1d5a_manual_gui_test.md`.

## 18. Integrite

The GUI modified only copied FITS under `/tmp`.  Pixel hashes for copied FITS
were compared against originals:

```text
pixel_hash_pairs_ok = 10
```

Read-only resource snapshot:

```text
/opt/astap: files=1487 size=977092518
/home/tristan/zesolver_index: files=1480 size=371695452
indexes/astrometry_4d: files=6 size=6345747
```

Product strict manifest SHA256:

```text
1847e075b25650ee00664bb6db23f80307f4d89caa548aca4c0d2c09e69e79a5
```

The six product NPZ SHA256 values were re-read and matched the known P1D-4/P1D-5
values.

## 19. Warnings

Observed non-blocking warning:

```text
QLayout::addChildLayout: layout QFormLayout "" already has a parent
```

It appears when opening the existing GUI and is unrelated to CatalogLibrary
routing.  It did not block validation or tests.

The library remains `READY_PARTIAL`, because Blind 4D covers six tiles out of
1476.

## 20. Limites

P1D-5A does not move legacy path fields into a dedicated advanced panel.  The
visual cleanup and clearer compatibility grouping are P1D-5B.

No automatic adoption was added.  Users still need an existing `catalog.json`.

## 21. Etat Git final

Final state:

```text
## test...origin/test
P1D changes present and uncommitted
P1D-5A docs/tests/GUI edits present and uncommitted
```

No commit or push was performed.

## 22. Une seule prochaine etape

P1D-5B: move historical `db_root`, `index_root`, external manifest and rollback
controls into an explicit Advanced/Compatibility surface without changing the
runtime routing validated here.

## 23. Decision de gate

```text
READY_FOR_P1D5B_ADVANCED_COMPATIBILITY_SURFACE
```
