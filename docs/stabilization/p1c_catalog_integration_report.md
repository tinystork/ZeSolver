# P1C CatalogLibrary Adapter Integration Report

Decision: `READY_FOR_SETTINGS_SEPARATION`

P1C integrates the read-only `CatalogLibrary` adapters into the solver
configuration boundary. It does not modify `solve_near()`, `solve_blind()`,
solver thresholds, catalogue shard formats, 4D NPZ formats, source FITS files,
index files, or GUI layout.

## Initial State

Required documents were read first, in order:

- `AGENT.md`
- `docs/stabilization/p0b_regression_foundation_report.md`
- `docs/architecture/catalog_library.md`
- `docs/architecture/catalog_library_implementation.md`
- `docs/architecture/catalog_consumers_inventory.md`
- `docs/architecture/catalog_data_flow.md`
- `docs/architecture/catalog_migration_matrix.md`
- `docs/architecture/astap_family_strategy.md`
- `docs/stabilization/p1a_catalog_architecture_report.md`
- `docs/stabilization/p1b_catalog_library_report.md`
- `tests/corpus/README.md`
- `tests/corpus/manifest.json`
- `tests/corpus/oracles/zenear_reference.json`
- `tests/corpus/oracles/zeblind4d_reference.json`
- `tests/corpus/oracles/pipeline_reference.json`

Initial validation before modification:

```text
.venv/bin/python tools/run_regression_suite.py --hermetic
291 passed, 1 skipped, 6 deselected, 1 warning
runner status PASS

.venv/bin/python -m pytest -q
291 passed, 7 skipped, 1 warning

.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests
OK
```

Initial `git status --short` already contained P0/P1 uncommitted and untracked
files. P1C did not revert or rewrite unrelated existing work.

## P1C-1 Integration Map

Created:

- `docs/architecture/catalog_integration_map.md`

The map records how `db_root`, `families`, `blind_index_path`,
`blind_4d_manifest_path`, `blind_4d_index_paths`, `blind_backend_profile` and
`index_root` are produced and consumed today.

Decision:

- Integration belongs before `ImageSolver` constructs `CatalogDB`.
- Blind 4D integration must still pass through the strict 4D manifest loader.
- `blind_backend_profile` remains a profile selection, not a catalogue resource.
- Legacy direct paths remain compatible when no library is selected.

Validation after the map:

```text
.venv/bin/python tools/run_regression_suite.py --hermetic
291 passed, 1 skipped, 6 deselected, 1 warning
runner status PASS

git diff --check
OK
```

## Resource Context and Resolver

Created:

- `zesolver/catalog_resources.py`

Main API:

```python
resources = resolve_catalog_resources(...)
```

Added immutable resource context:

- `SolverCatalogResources`
- `CatalogResourceResolutionError`

The context contains only resolved catalogue resources:

- optional library path/status/id;
- optional Near descriptor;
- Blind 4D descriptors and runtime paths;
- optional strict 4D manifest path;
- legacy index root;
- source classification: `library`, `legacy`, `environment`, `none`;
- coverage/all-sky status and warnings.

It does not contain matching thresholds, quad parameters, GUI options or solve
results.

Resolution priority:

1. explicitly provided valid `CatalogLibrary`;
2. explicit legacy paths;
3. named environment variables only;
4. explicit absence.

An explicitly corrupt or incompatible library raises a clear error and does not
fall back silently. `READY_PARTIAL` is retained as partial and never becomes
all-sky.

## Pipeline Integration

Modified:

- `zesolver.py`
- `zesolver/settings_store.py`

Added to `SolveConfig`:

```python
catalog_library_path: Optional[Path] = None
```

Added persistent field:

```python
catalog_library_path: Optional[str] = None
```

No legacy field was removed.

Added configuration bridge:

```python
resolve_catalog_resources_for_config(config)
apply_catalog_resources_to_config(config)
```

Behavior:

- With a valid library, `NearCatalogDescriptor.root/families` replace
  `db_root/families` before `CatalogDB` is constructed.
- With a valid library containing a common strict 4D `manifest_path`,
  `blind_4d_manifest_path` is replaced before the current preflight/load path.
- Without a library, legacy fields keep the previous behavior.
- GUI preflight now applies catalogue resources before loading the 4D manifest.
- CLI gained `--catalog-library`; without it, the historical `--db-root` path is
  still required.
- Telemetry logs include source, library status/id, Near families, Blind 4D index
  count, all-sky flag, coverage fraction and warnings without emitting full
  personal paths.

Rollback:

- Stop providing `catalog_library_path` / `--catalog-library`.
- Legacy `db_root`, `families`, `blind_index_path` and
  `blind_4d_manifest_path` remain available.

## Tests Added

Created:

- `tests/catalog_resource_helpers.py`
- `tests/test_catalog_resource_resolution.py`
- `tests/test_catalog_library_near_integration.py`
- `tests/test_catalog_library_blind4d_integration.py`
- `tests/test_catalog_library_pipeline_integration.py`

Coverage includes:

- no library: legacy behavior preserved;
- valid library priority over contradictory legacy paths;
- Near root/families supplied by the library descriptor;
- no source means Near is not announced;
- corrupt source/index is an explicit error;
- Blind 4D strict manifest order preserved;
- bad checksum blocks use;
- six indexes remain partial and `blind4d_all_sky=false`;
- legacy strict manifest still works without a library;
- pipeline assembly uses library Near and Blind 4D resources before legacy;
- partial-library telemetry remains visible.

Targeted result:

```text
.venv/bin/python -m pytest \
  tests/test_catalog_resource_resolution.py \
  tests/test_catalog_library_near_integration.py \
  tests/test_catalog_library_blind4d_integration.py \
  tests/test_catalog_library_pipeline_integration.py \
  -q

14 passed
```

## Validation Before Final Barrier

After code integration:

```text
.venv/bin/python tools/run_regression_suite.py --hermetic
305 passed, 1 skipped, 6 deselected, 1 warning
runner status PASS

.venv/bin/python -m pytest -q
305 passed, 7 skipped, 1 warning

.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests
OK
```

Expected skips remain external-data related:

- external ASTAP/HNSKY database not found under repository `database/`;
- S50 index/frame not configured;
- `ZESOLVER_BLIND4D_MANIFEST` unset;
- P29 source FITS path not mapped;
- `ZESOLVER_CORPUS_ROOT` unset;
- `ZESOLVER_ZN310B_ROOT` unset.

## Final P1C Barrier

Final targeted tests:

```text
.venv/bin/python -m pytest \
  tests/test_catalog_resource_resolution.py \
  tests/test_catalog_library_near_integration.py \
  tests/test_catalog_library_blind4d_integration.py \
  tests/test_catalog_library_pipeline_integration.py \
  -q

14 passed
```

Final hermetic runner:

```text
.venv/bin/python tools/run_regression_suite.py --hermetic

305 passed, 1 skipped, 6 deselected, 1 warning
runner status PASS
```

Final full suite:

```text
.venv/bin/python -m pytest -q

305 passed, 7 skipped, 1 warning
```

Final compile and diff hygiene:

```text
.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests
OK

git diff --check
OK
```

Clean snapshot:

```text
temporary snapshot worktree: /tmp/zesolver-p1c-snapshot-IhYoCf/worktree
temporary snapshot commit: e15e869f461892fe99f19f1430f8bb177c9001c4
git status --short: clean

tools/run_regression_suite.py --hermetic:
295 passed, 11 skipped, 6 deselected, 1 warning
runner status PASS

compileall:
OK
```

The snapshot has additional expected skips because bundled NPZ indexes, ASTAP
source checkout, ZN dump artifacts and selected FITS/report fixtures are not
tracked in the clean snapshot.

## Limits and Risks

- The normal GUI still shows legacy database/index/manifest controls. This is
  intentional; GUI simplification belongs to P3.
- `CatalogLibrary` does not synthesize a strict 4D manifest from raw descriptors.
  A library-provided Blind 4D runtime path is used by the pipeline when the
  library indexes share a strict `manifest_path`; the existing strict loader
  remains the runtime authority.
- Direct legacy paths are still present in `SolveConfig` and `PersistentSettings`
  for rollback and compatibility.
- The repository still contains many pre-existing uncommitted P0/P1 files; clean
  snapshot validation therefore uses the P0B temporary-snapshot method.

## Decision

```text
READY_FOR_SETTINGS_SEPARATION
```

P1C is ready for P2A because `CatalogLibrary` can now feed Near and Blind 4D
resources through a deterministic, additive and reversible boundary; legacy paths
remain compatible; corrupt explicit libraries are not ignored; partial Blind 4D
coverage stays visible and is never promoted to all-sky.
