# P1D-4B — Basculement produit Blind 4D vers CatalogLibrary

Decision: `READY_FOR_P1D5_PRODUCT_SURFACE_CLEANUP`

P1D-4B makes the `CatalogLibrary` strict Blind 4D view the normal product
runtime source when an explicit valid library provides Blind 4D indexes.  The
external strict manifest remains available only for explicit rollback,
compatibility without a library, diagnostics and tests.  No NPZ, strict product
manifest, ASTAP source, ZeNear behavior, ZeBlind algorithm, threshold, budget or
GUI surface was changed.

## 1. Objectif

Switch product routing from:

```text
blind_4d_manifest_path -> external strict manifest -> Blind 4D runtime
```

to:

```text
CatalogLibrary -> CatalogBlind4DManifestView -> Blind 4D runtime
```

when a library is explicitly configured and contains valid Blind 4D indexes.

## 2. Etat Git initial et HEAD

Initial state for P1D-4B:

```text
git status --short --branch
## test...origin/test
... P1D-4A changes present and uncommitted ...

git rev-parse HEAD
f8e92b3b6e80de16f04c9f55b5841449c7701446

git diff --check
OK
```

The P1D-4A changes were already present in the worktree and were preserved.
No commit or push was performed.

## 3. Politique centrale

Added one product selector:

```python
resolve_blind4d_runtime(resources, mode="auto", external_manifest_path=...)
```

Supported modes:

```text
auto
library-view
external-manifest
```

The result is `Blind4DRuntimeSelection`, exposing requested/effective mode,
source, loaded manifest, library view, external path, index ids/paths, runtime
order, coverage, view fingerprint, warnings and stable error context.

## 4. Mode avant/apres

Before P1D-4B, product routes could only prepare Blind 4D runtime indexes from
an external strict manifest path.

After P1D-4B:

```text
AUTO + library with Blind 4D -> catalog_library_view
AUTO + no library + external manifest -> external_manifest
AUTO + library without Blind 4D -> unavailable, no external fallback
LIBRARY_VIEW forced -> requires valid library view
EXTERNAL_MANIFEST forced -> requires explicit external manifest
```

The external path is not consulted in library mode, including `AUTO` with an
explicit library.

## 5. Loader partage

`zeblindsolver.index_manifest_4d` now has:

```python
load_4d_index_manifest(path)
load_4d_index_manifest_payload(payload, manifest_path=None, index_root=None)
```

Both paths share the same strict validation for schema/version, entries, order,
duplicates, paths, checksums, NPZ metadata, tile keys, counts, sampler and
tolerance.  Relative payload paths require an explicit root and never depend on
the process working directory.

## 6. Integration PIPELINE

`SolverPipeline` resolves Blind 4D runtime through the central selector and
publishes the selection telemetry.  It does not inspect a non-empty external
manifest path independently and does not sort indexes outside
`runtime_order.blind4d`.

The pipeline caches the runtime selection per context.  The cache key includes
the resource object, requested mode, external manifest path, library id,
catalog manifest fingerprint, index ids and index checksums.

## 7. Integration LEGACY

`ImageSolver` and legacy config application resolve Blind 4D runtime through
the same selector.  When a library view is available, the validated loaded
manifest is passed directly to the existing Blind 4D config builder; no strict
JSON materialization is required for the normal solve path.

View errors are reported as Blind 4D catalog/runtime errors, not replaced by a
silent external-manifest fallback.

## 8. Cache et duree de vie

The product does not regenerate the view or reload the six NPZs for each FITS
within the same solver/pipeline context.  The cache is process-local and is
not persisted in settings.  Workers can reconstruct their local runtime from
serializable settings and catalog resources.

## 9. Rollback externe

Rollback remains explicit through:

```text
blind4d_catalog_mode = external-manifest
```

This mode requires a valid external manifest and never attempts a library view.
If the path is absent or invalid, the selector raises a stable external
manifest error.

## 10. Absence de fallback silencieux

Tests cover:

```text
library view ignores a stale external path
Near-only library does not use a residual external manifest
invalid library view does not fall back externally
forced library mode does not fall back externally
forced external mode does not fall back to library
```

Runtime telemetry always reports `blind4d_external_fallback_used = false` in
library mode.

## 11. Reglages et migration

Added additive setting:

```text
blind4d_catalog_mode = auto
```

Accepted values:

```text
auto
library-view
external-manifest
```

Old settings files default to `auto`.  The historical
`blind_4d_manifest_path` is not removed and remains usable for compatibility
and explicit rollback.

## 12. Variables d'environnement

Historical environment manifest variables remain compatibility inputs when no
explicit library is configured.  With a library in `AUTO`, the library view
takes priority and the environment/external manifest is not consulted as a
fallback.

## 13. Telemetrie

Blind 4D runtime telemetry now includes:

```text
blind4d_catalog_mode_requested
blind4d_catalog_mode_effective
blind4d_catalog_source
blind4d_view_fingerprint
blind4d_library_id
blind4d_runtime_order
blind4d_index_ids
blind4d_index_count
blind4d_covered_tiles
blind4d_total_tiles
blind4d_all_sky
blind4d_external_fallback_used
```

Normal telemetry excludes personal full paths.  Diagnostic code can request
paths explicitly.

## 14. Erreurs stables

Added or reused:

```text
BLIND4D_LIBRARY_REQUIRED
BLIND4D_LIBRARY_VIEW_INVALID
BLIND4D_LIBRARY_NO_INDEXES
BLIND4D_EXTERNAL_MANIFEST_REQUIRED
BLIND4D_EXTERNAL_MANIFEST_INVALID
BLIND4D_CATALOG_MODE_INVALID
BLIND4D_RUNTIME_ORDER_INVALID
BLIND4D_RUNTIME_RESOURCE_UNAVAILABLE
```

## 15. Tests cibles

Targeted P1D-4B command:

```text
.venv/bin/python -m pytest \
 tests/test_catalog_blind4d_manifest_view.py \
 tests/test_catalog_library_blind4d_integration.py \
 tests/test_blind4d_runtime_source_policy.py \
 tests/test_blind4d_manifest_payload_loader.py \
 tests/test_catalog_library_blind4d_product_switch.py \
 tests/test_solver_pipeline_routing.py \
 tests/test_blind_production_port.py \
 tests/test_blind_port_config_parity.py \
 tests/test_astap_4d_runtime_validation.py \
 tests/test_batch_pipeline_zn310b.py \
 tests/test_solver_pipeline_zn310b_production.py \
 tests/test_gui_progress_pipeline.py \
 tests/test_gui_progress_realtime_legacy.py \
 tests/test_gui_progress_no_duplicates.py \
 tests/test_gui_progress_stop_restart.py \
 tests/test_gui_progress_stale_callback.py \
 tests/test_gui_completion_exactly_once.py \
 tests/test_gui_cancellation.py \
 tests/test_catalog_library_pipeline_integration.py \
 tests/test_product_settings.py \
 tests/test_settings_persistence.py \
 tests/test_settings_migration_v2.py \
 tests/test_configuration_assembly.py -q
```

Result:

```text
71 passed, 2 skipped
```

The skips are external ZN310B corpus tests with `ZESOLVER_ZN310B_ROOT` unset.

## 16. Barrieres generales

```text
tools/check_core_boundaries.py
core boundary check: OK

tools/run_regression_suite.py --hermetic
PASS, 555 passed, 1 skipped, 9 deselected, 58 warnings

pytest -q
555 passed, 10 skipped, 58 warnings

compileall
OK
```

Warnings are the existing datetime/fork/FITS warning categories already present
in previous runs.

## 17. Validation externe par routes produit

Temporary work root:

```text
/tmp/p1d4b_product_switch_giltejdt
```

Configuration A:

```text
mode = external-manifest
manifest = config/zeblind_4d_experimental_manifest.json
```

Configuration B:

```text
mode = library-view
catalog.json adopted in temporary library
blind_4d_manifest_path = sentinel invalid manifest
```

Both configurations used the same six product NPZ files.  The invalid sentinel
proved the library route did not consult the historical manifest path.

Results:

```text
M106 all30: 30/30 SAME_SUCCESS_EQUIVALENT_WCS
233828: preserved, SAME_SUCCESS_EQUIVALENT_WCS
234013: preserved, SAME_SUCCESS_EQUIVALENT_WCS
negative controls: 0/3 false positives
max center separation: 0.0 arcsec
same index ids: true
same checksums: true
same paths: true
library source telemetry: catalog_library_view
external fallback used in library mode: false
view fingerprint: dc4855672eb3fa3ab2f098228a69c8e52011b52b7a5878aeb22cd9ef03944865
```

Runtime order:

```text
d50_2823_S_q40000
d50_2822_S_q40000
d50_2644_S_q40000
d50_2645_S_q40000
d50_2602_S_q40000
d50_2702_S_q40000
```

Coverage:

```text
PARTIAL, covered_tiles = 6, total_tiles = 1476, all_sky = false
```

## 18. Performance

Measured through the product pipeline on M106 all30:

```text
external median solve: 24.715 s
external p95 solve:    47.496 s
external max solve:    54.558 s

library median solve:  25.513 s
library p95 solve:     42.080 s
library max solve:     48.338 s
```

The small median difference is within run noise for this validation.  P1D-4B
adds a startup/resource-selection cost, then reuses the loaded runtime in the
solver/pipeline context.

## 19. Integrite

Before/after checks remained unchanged:

```text
/opt/astap file count, size and mtime aggregate
/home/tristan/zesolver_index file count, size and mtime aggregate
six product NPZ SHA256 values
config/zeblind_4d_experimental_manifest.json SHA256 and mtime
original FITS pixel hashes
```

No file was added or modified in ASTAP, legacy index, product indexes or the
product strict manifest.  Only temporary output copies received WCS headers.

## 20. Warnings

Library-view telemetry reports:

```text
catalog_library_ready_partial
blind4d_coverage_not_all_sky
```

These are expected and correct for `PARTIAL 6/1476` Blind 4D coverage.

## 21. Limites

The validation focused on the already canonical M106 all30 corpus and available
negative controls through product routes.  External ZN310B tests remained
skipped because `ZESOLVER_ZN310B_ROOT` was not configured.  P1D-4B does not
remove historical settings or GUI surface; that belongs to P1D-5.

## 22. Etat Git final

Final state remained uncommitted on branch `test`; P1D-4A and P1D-4B changes
are present together in the worktree.  No commit or push was performed.

## 23. Prochaine etape unique

P1D-5: product-surface cleanup after the runtime switch, while keeping external
manifest rollback explicit.

## 24. Decision de gate

```text
READY_FOR_P1D5_PRODUCT_SURFACE_CLEANUP
```
