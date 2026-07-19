# P1D-1B ZeNear ASTAP-native Product Switch Report

Decision: `READY_FOR_P1D2_CATALOG_PROVENANCE_AND_ADOPTION`

P1D-1B switches the product ZeNear runtime to ASTAP-native whenever a valid
explicit `CatalogLibrary` provides a Near source. The legacy index provider
remains available as an explicit rollback and as compatibility for old
installations without a library.

## 1. Objectif

Target product policy:

```text
CatalogLibrary with Near
  -> AstapNearCatalogProvider
  -> solve_near(index_root=None, catalog_provider=...)

No CatalogLibrary + legacy index
  -> LegacyIndexNearCatalogProvider

Explicit legacy rollback
  -> LegacyIndexNearCatalogProvider

ASTAP-native error
  -> explicit Near error
  -> no silent provider fallback
  -> normal orchestrator may continue to Blind 4D
```

No Near/Blind algorithm, threshold, profile, catalogue shard, index format,
pixel data, WCS acceptance rule or Blind 4D builder was changed.

## 2. Etat Git initial

Initial state before P1D-1B already included P3A-V3, P1D-0 and P1D-1A
uncommitted work:

```text
## test...origin/test
 M .gitignore
 M AGENT.md
 M zeblindsolver/metadata_solver.py
 M zesolver.py
 M zesolver/catalog_resources.py
 M zesolver/gui_pipeline/legacy_runner.py
 M zesolver/gui_pipeline/pipeline_runner.py
 M zesolver/zeblindsolver.py
?? docs/stabilization/p1d1a_zenear_astap_provider_report.md
?? tests/near_catalog_provider_helpers.py
?? tests/test_near_catalog_provider_*.py
?? tools/compare_zenear_catalog_providers.py
?? zeblindsolver/near_catalog_provider.py
```

P1D-1B preserved the existing worktree and did not revert unrelated changes.

## 3. Politique de selection

Added a central policy in `zesolver.catalog_resources`:

- `NearCatalogMode.AUTO`
- `NearCatalogMode.ASTAP_NATIVE`
- `NearCatalogMode.LEGACY_INDEX`
- `NearCatalogRuntime`
- `resolve_near_catalog_runtime()`

Stable error codes:

- `ASTAP_NEAR_RESOURCE_REQUIRED`
- `ASTAP_NEAR_PROVIDER_INVALID`
- `LEGACY_NEAR_INDEX_REQUIRED`
- `LEGACY_NEAR_INDEX_INVALID`
- `NEAR_CATALOG_MODE_INVALID`
- `NEAR_CATALOG_FAMILY_UNAVAILABLE`

`AUTO` behavior:

- explicit valid library with Near -> ASTAP-native;
- no library + legacy index -> legacy index;
- explicit library without Near -> explicit ASTAP Near unavailable state, no
  legacy fallback;
- explicit invalid library still fails during resource resolution.

## 4. Mode par defaut avant/apres

Before P1D-1B:

```text
CatalogLibrary could replace db_root/families,
but ZeNear product calls still required blind_index_path/index_root.
```

After P1D-1B:

```text
near_catalog_mode=auto
CatalogLibrary Near present -> ASTAP-native provider
No CatalogLibrary + legacy index -> legacy provider
```

Old settings without `near_catalog_mode` migrate logically to `auto`.

## 5. Fichiers modifies

Code:

- `zeblindsolver/near_catalog_provider.py`
- `zeblindsolver/metadata_solver.py`
- `zesolver.py`
- `zesolver/catalog_resources.py`
- `zesolver/core/pipeline.py`
- `zesolver/gui_pipeline/settings_adapter.py`
- `zesolver/settings/assembly.py`
- `zesolver/settings/migration.py`
- `zesolver/settings/product.py`
- `zesolver/settings_store.py`
- `tools/compare_zenear_catalog_providers.py`

Tests:

- `tests/test_near_catalog_runtime_policy.py`
- `tests/test_catalog_library_near_runtime_unification.py`
- `tests/test_catalog_library_no_silent_legacy_fallback.py`
- `tests/test_solver_pipeline_near_provider.py`

Documentation:

- `docs/stabilization/p1d1b_zenear_astap_product_switch_report.md`
- `AGENT.md`

## 6. Integration route PIPELINE

`ExistingNearSolverPort` no longer uses:

```text
resources.legacy_index_root or resources.near.root
```

A Near source root is no longer treated as a historical `index_root`.

The pipeline now resolves `NearCatalogRuntime` and calls:

```text
ASTAP-native: near_solve(index_root=None, catalog_provider=AstapNearCatalogProvider)
Legacy:       near_solve(index_root=legacy_root, catalog_provider=LegacyIndexNearCatalogProvider)
```

For a library without Near, the pipeline records an explicit Near catalogue
unavailable result and can continue to Blind 4D if available.

## 7. Integration route LEGACY

`ImageSolver` builds one `NearCatalogRuntime` during initialization. The provider
is reused by initial Near and rescue attempts for the same solver instance.

The FITS Near branch is now enabled when:

```text
runtime provider available
or runtime has an explicit Near catalogue error
```

It is no longer conditioned only on `blind_index_path`.

In ASTAP-native mode:

- `index_root=None`;
- `catalog_provider=AstapNearCatalogProvider`;
- no inline historical fallback is requested from `near_solve()`;
- a provider/read failure is kept as a Near failure and the normal orchestrator
  may proceed to Blind 4D.

## 8. Rollback explicite

Rollback mode:

```text
near_catalog_mode=legacy-index
```

requires an explicit valid historical Near index. Missing or invalid legacy
index produces a stable error and does not fall back to ASTAP-native.

The CLI exposes:

```bash
--near-catalog-mode auto
--near-catalog-mode astap-native
--near-catalog-mode legacy-index
```

No new normal GUI control was added.

## 9. Absence de fallback silencieux

Tests prove:

- `AUTO + library Near + stale legacy path` still uses ASTAP-native;
- `AUTO + library without Near + legacy path` does not build the legacy provider;
- forced ASTAP-native without Near does not use residual legacy;
- forced legacy without index errors explicitly;
- pipeline ASTAP-native passes `index_root=None`.

The normal fallback chain remains:

```text
ZeNear -> ZeBlind 4D -> Astrometry.net optional
```

Only provider fallback is forbidden.

## 10. Duree de vie et cache du provider

`ImageSolver` builds the provider once per solver instance. Batch workers
construct their own local `ImageSolver`, so provider caches are process-local and
not serialized through `SolveConfig`.

`ExistingNearSolverPort` builds a provider per pipeline instance/request context.
No provider with a mutable cache is stored in persistent settings.

## 11. Telemetrie

Near results now include:

- `near_catalog_mode_requested`
- `near_catalog_mode_effective`
- `near_catalog_provider`
- `near_catalog_source`
- `near_catalog_family`
- `near_catalog_candidate_tiles`
- `near_catalog_loaded_tiles`
- `near_catalog_loaded_stars`
- `near_catalog_fallback_used`
- optional `near_catalog_fallback_reason`

`LegacyIndexNearCatalogProvider` now reports its ASTAP fallback truthfully after
a missing/failed NPZ load. Product telemetry avoids full personal paths.

## 12. Comportement des erreurs

Provider assembly errors are explicit and testable. For product runs, `ImageSolver`
stores the unavailable runtime and reports it during the Near attempt so the
orchestrator can continue to Blind 4D when configured.

Invalid `CatalogLibrary` resolution remains fatal and does not fall back to
legacy paths.

## 13. Tests cibles

Executed:

```bash
.venv/bin/python -m pytest \
  tests/test_metadata_solver.py \
  tests/test_catalog290.py \
  tests/test_catalog_library_adapters.py \
  tests/test_catalog_resource_resolution.py \
  tests/test_catalog_library_near_integration.py \
  tests/test_catalog_library_pipeline_integration.py \
  tests/test_near_catalog_provider_astap.py \
  tests/test_near_catalog_provider_legacy.py \
  tests/test_near_catalog_provider_parity.py \
  tests/test_near_catalog_provider_boundaries.py \
  tests/test_near_catalog_runtime_policy.py \
  tests/test_catalog_library_near_runtime_unification.py \
  tests/test_catalog_library_no_silent_legacy_fallback.py \
  tests/test_solver_pipeline_near_provider.py \
  tests/test_batch_pipeline_zn310b.py \
  tests/test_solver_pipeline_zn310b_production.py \
  tests/test_gui_progress_pipeline.py \
  tests/test_gui_progress_realtime_legacy.py \
  tests/test_gui_progress_no_duplicates.py \
  tests/test_gui_progress_stop_restart.py \
  tests/test_gui_progress_stale_callback.py \
  tests/test_gui_wcs_cleanup_refresh.py \
  tests/test_gui_wcs_status_scope.py \
  tests/test_gui_completion_exactly_once.py \
  tests/test_gui_log_copy_exactly_once.py \
  tests/test_gui_completion_after_stop.py \
  tests/test_gui_completion_after_restart.py \
  tests/test_gui_stale_completion_callback.py \
  tests/test_gui_cancellation.py \
  tests/test_gui_lifecycle.py \
  -q
```

Result:

```text
61 passed, 4 skipped
```

Skips:

- external ASTAP/HNSKY test database absent under repository `database/`;
- `ZESOLVER_ZN310B_ROOT` unset.

## 14. Barrieres generales

Executed:

```bash
.venv/bin/python tools/check_core_boundaries.py
core boundary check: OK

.venv/bin/python tools/run_regression_suite.py --hermetic
449 passed, 1 skipped, 9 deselected, 56 warnings
runner status PASS

.venv/bin/python -m pytest -q
449 passed, 10 skipped, 56 warnings

.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests zesolver.py zewcscleaner.py zeindexcheck.py
OK

git diff --check
OK
```

The requested command includes `zedatabase.py`, but that file is not present in
this checkout. `compileall` reports `Can't list 'zedatabase.py'` and exits 0;
the command was rerun on all existing requested paths.

## 15. Comparaison externe complete

Inputs:

```text
ASTAP:        /opt/astap
Legacy index: /home/tristan/zesolver_index
FITS corpus:  /home/tristan/near_bench100_input
```

Command output:

```text
/tmp/p1d1b_provider_compare/compare.json
/tmp/p1d1b_provider_compare/compare.md
```

30 FITS were compared on separate copies, with direct Near solves only:

```text
legacy_success: 30 / 30
astap_success: 30 / 30
success_intersection: 30
astap_gained: 0
astap_lost: 0
legacy median/p95: 1.327s / 1.474s
astap median/p95: 1.443s / 1.569s
```

Tile candidate cases:

| Case | Legacy | ASTAP-native | Result |
|---|---:|---:|---|
| normal | 4 | 4 | same order |
| RA0 | 6 | 6 | same order |
| high_dec | 6 | 6 | same set, order differs |

The high-declination order difference is the already documented P1D-1A case:
historical manifest centers differ slightly from ASTAP layout-derived centers.
The selected tile set is identical.

For all 30 solve comparisons:

- selected solve tile matched;
- no success was lost;
- source pixels were preserved;
- copy pixels were preserved after WCS writes;
- HDU shapes and data dtypes were preserved.

Observed maxima:

```text
max image-center WCS delta: 0.0003589849877 deg
max scale delta: 0.0004283043 arcsec/pixel
max rotation delta: 0.2124036469 deg
max inlier delta: 14
max RMS delta: 0.4267385428 px
```

The largest image-center delta is on an M106 mosaic frame where both modes use
tile `d50_2823`; ASTAP-native has 54 inliers and RMS 0.327 px versus legacy 42
inliers and RMS 0.754 px. This is not counted as a regression because success,
tile identity, pixel scale, parity and WCS integrity remain valid, with improved
fit diagnostics.

## 16. Integrite FITS et catalogues

External run integrity:

```text
source_pixels_preserved: true
legacy_copy_pixels_preserved: true
astap_copy_pixels_preserved: true
legacy index NPZ count: 1479 -> 1479
ASTAP file count: 1487 -> 1487
```

The ASTAP-native run used `index_root=None`; the old index path is not consulted
in native provider tests, including a sentinel invalid legacy path.

## 17. Warnings

No new warning category was observed.

Known categories:

- `multiprocessing.popen_fork.DeprecationWarning`
- `Astropy VerifyWarning`

The warning count is `56` in the broad suites, higher than earlier P3A runs
because the tree now includes additional process-oriented tests.

## 18. Limites

- P1D-1B does not remove legacy fields from settings or GUI surfaces.
- `CatalogLibrary` manifest provenance/reconstruction fields are still
  insufficient for full repair/adoption; this is P1D-2.
- Blind 4D still uses existing derived indexes and manifests; direct ASTAP to
  Blind 4D building remains P1D-3.
- The pipeline `ProductSettings` does not expose historical index paths as a
  normal product setting; legacy rollback in pipeline requires explicit legacy
  resources from the compatibility layer/tests.

## 19. Etat Git final

Final state remains uncommitted and includes previous P3A/P1D work plus P1D-1B
files. No commit and no push were performed.

Key P1D-1B additions:

```text
tests/test_near_catalog_runtime_policy.py
tests/test_catalog_library_near_runtime_unification.py
tests/test_catalog_library_no_silent_legacy_fallback.py
tests/test_solver_pipeline_near_provider.py
docs/stabilization/p1d1b_zenear_astap_product_switch_report.md
```

## 20. Prochaine etape unique

P1D-2: extend `CatalogLibrary` provenance/adoption/repair so `catalog.json`
can own deterministic source/index reconstruction metadata without moving user
data.

## 21. Decision de gate

`READY_FOR_P1D2_CATALOG_PROVENANCE_AND_ADOPTION`
