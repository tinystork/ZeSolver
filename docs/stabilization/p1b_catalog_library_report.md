# P1B CatalogLibrary Read-Only Core Report

Decision: `READY_FOR_CATALOG_LIBRARY_ADAPTER_INTEGRATION`

P1B implements a read-only `CatalogLibrary` core. No solver pipeline, solver threshold, FITS file, catalogue shard, 4D NPZ index, 4D manifest, persistent setting, GUI behavior, or CLI product path was modified.

## Initial State

Required documents were read first:

- `AGENT.md`
- `docs/stabilization/p0b_regression_foundation_report.md`
- `docs/architecture/catalog_consumers_inventory.md`
- `docs/architecture/catalog_data_flow.md`
- `docs/architecture/blind4d_coverage_audit.md`
- `docs/architecture/blind4d_coverage_audit.json`
- `docs/architecture/astap_family_strategy.md`
- `docs/architecture/catalog_library.md`
- `docs/architecture/catalog_manifest_schema.json`
- `docs/architecture/catalog_manifest_example.json`
- `docs/architecture/catalog_migration_matrix.md`
- `docs/stabilization/p1a_catalog_architecture_report.md`
- `tests/corpus/README.md`
- `tests/corpus/manifest.json`
- `tests/corpus/oracles/zenear_reference.json`
- `tests/corpus/oracles/zeblind4d_reference.json`
- `tests/corpus/oracles/pipeline_reference.json`

Initial validation:

```text
.venv/bin/python tools/run_regression_suite.py --hermetic
253 passed, 1 skipped, 6 deselected, 1 warning
runner status PASS

.venv/bin/python -m pytest -q
253 passed, 7 skipped, 1 warning

.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests
OK
```

Initial `git status --short` already contained the P0/P1 uncommitted and untracked baseline files. P1B did not revert or rewrite them.

## Implemented Code

Created:

- `zesolver/catalog_library/__init__.py`
- `zesolver/catalog_library/models.py`
- `zesolver/catalog_library/manifest.py`
- `zesolver/catalog_library/validation.py`
- `zesolver/catalog_library/coverage.py`
- `zesolver/catalog_library/discovery.py`
- `zesolver/catalog_library/adapters.py`

Updated:

- `tools/audit_catalog_library.py`

The implementation is split by responsibility:

- immutable models and enums;
- manifest loading and typed errors;
- path-policy enforcement;
- integrity validation;
- coverage calculation;
- read-only discovery;
- Near and Blind 4D descriptors.

## Manifest Loader

`CatalogLibrary.open(path)` accepts:

- a directory containing `catalog.json`;
- a direct `catalog.json` path.

It rejects:

- missing manifests;
- invalid JSON;
- unsupported schema versions;
- missing required fields;
- relative paths escaping the library;
- implicit environment-variable expansion in JSON;
- relative external references.

Typed exceptions implemented:

- `CatalogLibraryError`
- `CatalogManifestError`
- `CatalogMissingError`
- `CatalogIncompleteError`
- `CatalogCorruptionError`
- `CatalogCompatibilityError`
- `CatalogVersionError`

## Status and Capabilities

Implemented statuses:

- `READY_FULL`
- `READY_PARTIAL`
- `NEAR_ONLY`
- `BLIND4D_ONLY`
- `SOURCE_ONLY`
- `INDEX_BUILD_REQUIRED`
- `INCOMPATIBLE`
- `CORRUPT`
- `MISSING`

Capabilities are calculated from validated data, not declarations alone:

- missing ASTAP source does not enable Near;
- missing or corrupt 4D index does not enable Blind 4D;
- six valid D50 4D indexes enable Blind 4D but do not enable all-sky Blind 4D;
- SHA mismatch forces `CORRUPT`.

## Coverage

Coverage can represent:

- family;
- tile keys;
- covered tile count;
- total tile count;
- fraction;
- declination range;
- RA segments when known;
- scale range placeholders;
- `FULL`, `PARTIAL`, `MISSING`, `INCOMPATIBLE`, `CORRUPT`, `UNKNOWN`.

The current bundled 4D installation remains:

```text
blind4d capability: true
all_sky_blind4d: false
coverage: 6 / 1476 D50 tiles
status without ASTAP source: BLIND4D_ONLY
```

It is never promoted to `READY_FULL`.

## Discovery

`discover_existing()` is non-destructive and returns an in-memory proposal.

It handles:

- no paths;
- ASTAP root only;
- Blind 4D manifest only;
- ASTAP + Blind 4D;
- legacy alias `ZEBLIND_4D_MANIFEST`;
- conflict between `ZESOLVER_BLIND4D_MANIFEST` and `ZEBLIND_4D_MANIFEST`.

When both variables are present and differ, `ZESOLVER_BLIND4D_MANIFEST` wins and a warning is reported.

## Adapters

`library.near_source()` returns a `NearCatalogDescriptor` with:

- root;
- families;
- formats;
- coverage;
- external-reference flag.

`library.blind4d_indexes()` returns validated `Blind4DIndexDescriptor` objects with:

- ID;
- path;
- family;
- tile keys;
- SHA256;
- coverage;
- schema.

`library.blind4d_runtime_paths()` returns validated 4D index paths in manifest order.

No adapter instantiates `CatalogDB`, calls `solve_near()`, loads a 4D runtime index, or modifies the current 4D manifest loader.

## Audit Tool

`tools/audit_catalog_library.py` keeps the P1A output sections:

- `astap`;
- `blind4d`;
- `legacy_index`.

It now also includes:

```text
catalog_library.status
catalog_library.families
catalog_library.blind4d_index_count
catalog_library.issues
```

Real bundled manifest audit:

```text
blind4d status: READY_PARTIAL
enabled tiles: 6
layout tiles: 1476
all_sky: False
catalog_library status: BLIND4D_ONLY
catalog_library blind4d_index_count: 6
```

## Tests Added

Created:

- `tests/catalog_library_fixtures.py`
- `tests/test_catalog_library_manifest.py`
- `tests/test_catalog_library_status.py`
- `tests/test_catalog_library_validation.py`
- `tests/test_catalog_library_discovery.py`
- `tests/test_catalog_library_adapters.py`

Coverage includes:

- valid manifest;
- invalid JSON;
- required field missing;
- supported and unsupported schema versions;
- valid relative path;
- explicit external absolute path;
- escaping relative path rejected;
- stored env-var expansion rejected;
- all nine product statuses;
- SHA correct and incorrect;
- source absent;
- index absent;
- index source unknown;
- coverage contradiction;
- no path discovery;
- ASTAP-only discovery;
- Blind4D-only discovery;
- combined discovery;
- legacy 4D env alias;
- conflicting 4D env vars;
- current bundled 4D manifest is partial and `BLIND4D_ONLY` without ASTAP source;
- Near descriptor;
- Blind 4D descriptor;
- ordered runtime paths;
- exclusion of invalid indexes.

## Commands Executed

Targeted tests:

```bash
.venv/bin/python -m pytest tests/test_catalog_library_manifest.py tests/test_catalog_library_status.py tests/test_catalog_library_validation.py tests/test_catalog_library_discovery.py tests/test_catalog_library_adapters.py -q
```

Result:

```text
38 passed
```

Audit compatibility:

```bash
.venv/bin/python -m pytest tests/test_catalog_library_audit.py -q
```

Result:

```text
5 passed
```

Real audit probe:

```bash
.venv/bin/python tools/audit_catalog_library.py \
  --blind4d-manifest config/zeblind_4d_experimental_manifest.json \
  --output-json /tmp/zesolver_p1b_audit.json \
  --output-md /tmp/zesolver_p1b_audit.md
```

Result:

```text
READY_PARTIAL 6 1476 False
BLIND4D_ONLY 6
```

Final validation is recorded below after completion.

## Solver Boundary

The following behavior remains unintegrated and unchanged:

- `SolveConfig.db_root`
- `SolveConfig.blind_index_path`
- `SolveConfig.blind_4d_manifest_path`
- `PersistentSettings`
- `solve_near()`
- `solve_blind()`
- `ImageSolver`
- GUI
- main CLI

P1C should handle adapter integration.

## Risks and Limits

- NPZ internal metadata is not revalidated by `CatalogLibrary`; the existing strict 4D loader remains the runtime authority.
- The core does not write or migrate manifests.
- Exact all-sky geometry is not inferred from partial tile lists.
- P4 packaging editable install remains unresolved and out of scope.
- Current repository state still includes many uncommitted P0/P1 files; snapshot testing is therefore performed from a temporary commit as in P0B.

## Final Validation

Targeted CatalogLibrary tests:

```text
.venv/bin/python -m pytest \
  tests/test_catalog_library_manifest.py \
  tests/test_catalog_library_status.py \
  tests/test_catalog_library_validation.py \
  tests/test_catalog_library_discovery.py \
  tests/test_catalog_library_adapters.py \
  -q

38 passed
```

Audit compatibility:

```text
.venv/bin/python -m pytest tests/test_catalog_library_audit.py -q

5 passed
```

Hermetic runner:

```text
.venv/bin/python tools/run_regression_suite.py --hermetic

291 passed, 1 skipped, 6 deselected, 1 warning
runner status PASS
```

Full suite:

```text
.venv/bin/python -m pytest -q

291 passed, 7 skipped, 1 warning
```

Compile:

```text
.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests

OK
```

Diff hygiene:

```text
git diff --check

OK
```

Clean snapshot:

```text
temporary snapshot commit: 73a943ce910ec241ab1ec0f03109efb9c4c86ce2
/tmp/zesolver-clean-p1b

tools/run_regression_suite.py --hermetic:
281 passed, 11 skipped, 6 deselected, 1 warning
runner status PASS

compileall:
OK
```

The clean snapshot has additional expected skips because `indexes/astrometry_4d/*.npz`, `ASTAP-main/`, and some `reports/**` FITS/report fixtures are intentionally ignored by Git.

## Decision

```text
READY_FOR_CATALOG_LIBRARY_ADAPTER_INTEGRATION
```
