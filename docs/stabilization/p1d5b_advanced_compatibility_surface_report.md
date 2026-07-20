# P1D-5B - Advanced Compatibility Surface

## Objective

P1D-5B removes the main user-facing ambiguity between a ZeSolver
`CatalogLibrary` and the historical Near index directory.  The normal GUI path
now shows a single catalog concept:

```text
Bibliotheque ZeSolver
```

Historical paths remain available for rollback, diagnostics and legacy
installations in an explicit advanced compatibility surface.

## Initial Git State

Initial HEAD:

```text
f8e92b3b6e80de16f04c9f55b5841449c7701446
```

Initial branch:

```text
test...origin/test
```

The worktree already contained the uncommitted P1D changes from P1D-2 through
P1D-5A.  No reset, revert, commit or push was performed.

## User Defect Reproduced

The observed confusion was:

```text
/home/tristan/ZeSolverCatalog
```

entered as a historical Near index path.  The previous technical error was:

```text
LEGACY_NEAR_INDEX_INVALID: /home/tristan/ZeSolverCatalog/manifest.json
```

This was correct internally but poor UX.  P1D-5B adds typed validation and now
reports:

```text
CATALOG_LIBRARY_USED_AS_LEGACY_NEAR_INDEX
Ce dossier est une Bibliotheque ZeSolver, pas un index Near historique.
```

## Surface Before and After

Before P1D-5B, the normal Settings surface still exposed:

```text
Base ASTAP
Dossier d'index
Manifest 4D
Near catalog mode
Blind 4D catalog mode
Build/rebuild/check index actions
```

After P1D-5B, the normal path exposes only:

```text
Bibliotheque ZeSolver
[ path ] [ Parcourir... ] [ Verifier ] [ Effacer ]
```

The historical fields moved into:

```text
Compatibilite historique et diagnostic
```

and the old maintenance actions moved into:

```text
Outils avances et maintenance des catalogues
```

## Normal Path

The normal path expects a directory containing:

```text
catalog.json
```

For the current library, the GUI displays a partial-ready state:

```text
READY_PARTIAL
Near ASTAP: D50 - 1476 tiles
Blind 4D: 6 / 1476 tiles
Global Blind 4D coverage: no
```

The normal modes remain:

```text
near_catalog_mode = auto
blind4d_catalog_mode = auto
```

With a valid library these resolve to:

```text
Near     = astap-native
Blind 4D = catalog_library_view
```

## Advanced Surface

The compatibility surface is checkable and closed by default.  It contains:

```text
Base ASTAP historique
Index Near historique
Manifeste Blind 4D externe
Near mode
Blind 4D mode
Retablir le mode automatique recommande
```

Opening the surface does not activate rollback.

## Typed Validation

Added `zesolver/gui_catalog_validation.py` with validators for:

```text
CatalogLibrary root
Historical ASTAP source
Historical Near index
External Blind 4D manifest
```

Confusion cases are detected explicitly:

```text
CATALOG_LIBRARY_USED_AS_LEGACY_NEAR_INDEX
LEGACY_NEAR_INDEX_USED_AS_CATALOG_LIBRARY
ASTAP_SOURCE_USED_AS_CATALOG_LIBRARY
BLIND4D_MANIFEST_FILE_REQUIRED
```

Inactive legacy fields do not block CatalogLibrary mode.

## Rollback

Explicit rollback remains available:

```text
Near     = legacy-index
Blind 4D = external-manifest
```

When active, the GUI shows:

```text
ROLLBACK HISTORIQUE ACTIF
```

The restore action sets both modes back to `auto` and preserves all historical
paths.

## No Silent Fallback

Validated behavior:

```text
CatalogLibrary valid + modes auto + invalid legacy sentinels
-> astap-native
-> catalog_library_view
-> blind4d_external_fallback_used = false
```

Legacy paths are consulted only when their forced rollback modes are selected.

## Persistence and Migration

The visual move does not delete settings:

```text
catalog_library_path
db_root
index_root
blind_4d_manifest_path
near_catalog_mode
blind4d_catalog_mode
families
```

Old settings files without `catalog_library_path` still load with `None`.
Returning to automatic mode does not clear legacy paths.

## Profiles

Easy mode:

```text
Bibliotheque ZeSolver only
no legacy fields
```

Wizard:

```text
valid library -> ready
missing library -> choose a library
invalid library -> corrective action
legacy compatibility -> secondary advanced path
```

Expert mode:

```text
advanced compatibility surface available, closed by default
```

## Maintenance Tools

Historical build/check tools were moved out of the normal path into the
advanced maintenance section.  They still exist, but selecting a
CatalogLibrary never launches construction, hash rebuild or index validation.

## Qt Warning Fix

Fixed:

```text
QLayout::addChildLayout: layout QFormLayout "" already has a parent
```

Cause: duplicate insertion of form layouts during GUI construction.  The fix
removes the duplicate additions and does not suppress Qt warnings globally.

Real GUI validation reported:

```text
layout_parent_warnings = []
```

## Files Modified

Main P1D-5B files:

```text
zesolver.py
zesolver/gui_profiles.py
zesolver/gui_settings_sections.py
zesolver/gui_catalog_validation.py
tests/test_gui_catalog_compatibility_surface.py
tests/test_gui_catalog_resource_type_validation.py
tests/test_gui_catalog_path_confusion.py
tests/test_gui_catalog_rollback_visibility.py
tests/test_gui_catalog_settings_migration.py
docs/architecture/advanced_catalog_compatibility_surface.md
docs/architecture/gui_catalog_library_selection.md
docs/stabilization/p1d5b_advanced_compatibility_surface_report.md
docs/stabilization/p1d5b_manual_gui_test.md
```

## Targeted Tests

Targeted P1D-5B/P1D-5A GUI and CatalogLibrary tests:

```text
45 passed
```

Focused resource/surface tests before the wider run:

```text
19 passed
```

## General Barriers

Core boundary check:

```text
core boundary check: OK
```

Hermetic regression:

```text
574 passed, 1 skipped, 9 deselected, 58 warnings
status = PASS
```

Full pytest:

```text
574 passed, 10 skipped, 58 warnings
```

Final compile and whitespace barriers:

```text
compileall = OK
git diff --check = OK
```

## External Validation

Real Qt GUI validation used:

```text
/home/tristan/ZeSolverCatalog
/opt/astap
/home/tristan/zesolver_index
config/zeblind_4d_experimental_manifest.json
```

Normal mode:

```text
5 / 5 SOLVED
catalog_source = library
near_catalog_provider = astap_native
blind4d_catalog_source = catalog_library_view
blind4d_external_fallback_used = false
legacy fields hidden in Easy mode
```

Confusion case:

```text
CATALOG_LIBRARY_USED_AS_LEGACY_NEAR_INDEX
```

Rollback:

```text
near_catalog_mode = legacy-index
blind4d_catalog_mode = external-manifest
ROLLBACK HISTORIQUE ACTIF
```

Restore:

```text
near_catalog_mode = auto
blind4d_catalog_mode = auto
legacy paths preserved
```

Stop/relaunch:

```text
stop terminal_count = 1
relaunch 1 / 1 SOLVED
relaunch terminal_count = 1
```

## Integrity

No external resource was modified.  Only temporary copied FITS received WCS
headers; their pixel arrays were unchanged:

```text
pixel_hash_pairs_ok = 5 / 5
```

Current resource checks:

```text
/opt/astap: files=1487 size=977092518
/home/tristan/zesolver_index: files=1480 size=371695452
strict 4D manifest sha256:
1847e075b25650ee00664bb6db23f80307f4d89caa548aca4c0d2c09e69e79a5
CatalogLibrary catalog.json sha256:
d666a62aad8bc619be79bd764b40bd447d336fe919c036a515c545b0016d3c0f
```

The six product NPZ hashes match the previous P1D validations.

## Warnings

The standard test suite still emits the pre-existing warning set already
reported in earlier P1D gates.  No new Qt layout-parent warning appears when
opening the GUI.

## Limits

P1D-5B does not remove legacy compatibility, does not rebuild catalogues or
indexes, and does not start P3B.  It only closes the P1D product surface around
the selected CatalogLibrary.

## Final Git State

Final branch status:

```text
## test...origin/test
```

The worktree remains intentionally dirty with the uncommitted P1D changes,
including the new P1D-5B docs/tests/module.  No commit or push was performed.

## Next Step

Resume P3B GUI simplification.

## Gate Decision

```text
READY_TO_RESUME_P3B_GUI_SIMPLIFICATION
```
