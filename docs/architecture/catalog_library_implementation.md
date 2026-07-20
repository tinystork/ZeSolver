# CatalogLibrary Read-Only Implementation

P1B implements the first real `CatalogLibrary` core. It is read-only and is not connected to `solve_near()`, `solve_blind()`, the GUI, persistent settings, or the main CLI.

## Modules

`zesolver/catalog_library/models.py`
: Frozen dataclasses and enums for statuses, capabilities, coverage, sources, indexes, issues, validation reports, discovery results, and adapters.

`zesolver/catalog_library/coverage.py`
: Small helpers for parsing and merging coverage claims. `READY_FULL` is never inferred from tile counts alone; `all_sky` must be explicit and consistent.

`zesolver/catalog_library/manifest.py`
: `CatalogLibrary.open()`, JSON loading, schema-version checks, typed exceptions, manifest-to-model conversion, and path policy enforcement.

`zesolver/catalog_library/validation.py`
: File presence, readability, SHA256 integrity, source/index compatibility checks, status calculation, capability calculation, and structured validation reports.

`zesolver/catalog_library/discovery.py`
: Non-destructive discovery of existing ASTAP roots, 4D manifests, legacy index roots, and environment variables. It returns an in-memory proposal only.

`zesolver/catalog_library/adoption.py`
: P1D-2A read-only adoption planner. It inventories explicit existing ASTAP
roots, validates strict Blind 4D manifests, links indexes to source families and
tiles, classifies legacy indexes as compatibility resources, returns repair
prescriptions, and generates an in-memory `catalog.json` preview without
writing it.

`zesolver/catalog_library/atomic_adoption.py`
: P1D-2B explicit atomic writer for a prebuilt adoption plan. It writes only the
plan's `manifest_preview`, uses an exclusive `.catalog-adoption.lock`, writes a
same-directory temporary file, validates before and after replacement, creates
byte-identical backups on replacement, supports rollback, and returns a
structured commit result.

`zesolver/catalog_library/adapters.py`
: Read-only descriptors for future integration with existing Near and Blind 4D inputs. These adapters do not instantiate or call solvers.

`tools/audit_catalog_library.py`
: P1A audit tool still emits the original audit sections, and now also includes a compact `catalog_library` section from the new discovery layer.

## Public API

```python
from zesolver.catalog_library import CatalogLibrary

library = CatalogLibrary.open("/data/ZeSolverCatalog")
report = library.validate()

print(report.status)
print(report.capabilities.near)
print(report.capabilities.blind4d)
print(report.coverage)

near = library.near_source()
indexes = library.blind4d_indexes()
paths = library.blind4d_runtime_paths()
```

P1D-2A adoption planning:

```python
from zesolver.catalog_library import CatalogLibraryAdoptionPlan

plan = CatalogLibraryAdoptionPlan.reference_existing(
    library_root="/data/ZeSolverCatalog",
    astap_roots="/opt/astap",
    blind4d_manifest="/data/blind4d/manifest.json",
    legacy_index_root="/data/zesolver_index",
    fingerprint_policy="fast",
)
preview = plan.manifest_preview
```

The preview is serializable and validable by `CatalogLibrary`, but the planner
does not write it. Atomic writing belongs to P1D-2B.

P1D-2B explicit commit:

```python
from zesolver.catalog_library import CatalogLibraryAdoptionWriter

result = CatalogLibraryAdoptionWriter.commit(
    plan,
    mode="create",
)
```

Replacement must be explicit:

```python
result = CatalogLibraryAdoptionWriter.commit(
    plan,
    mode="replace",
    expected_existing_sha256=current_sha256,
)
```

The advanced CLI is `tools/adopt_catalog_library.py`. Its default mode is
preview-only; writing requires `--write`, and replacing requires
`--replace-existing` plus `--expected-existing-sha256`.

`CatalogLibrary.open(path)` accepts either a library directory containing `catalog.json` or a direct path to `catalog.json`.

## Status Model

Implemented product statuses:

- `READY_FULL`
- `READY_PARTIAL`
- `NEAR_ONLY`
- `BLIND4D_ONLY`
- `SOURCE_ONLY`
- `INDEX_BUILD_REQUIRED`
- `INCOMPATIBLE`
- `CORRUPT`
- `MISSING`

Current deterministic rules:

- SHA mismatch produces `CORRUPT`.
- Incompatible source/index schemas produce `INCOMPATIBLE`.
- valid ASTAP source plus valid partial 4D indexes produces `READY_PARTIAL`.
- valid ASTAP source plus valid all-sky 4D coverage produces `READY_FULL`.
- valid ASTAP source without declared indexes produces `SOURCE_ONLY`.
- valid ASTAP source with unavailable declared indexes produces `NEAR_ONLY`.
- valid 4D indexes without source ASTAP produce `BLIND4D_ONLY`.
- no valid data produces `MISSING`.

## Path Policy

Two path forms are supported:

`relative`
: Relative to the library root. Absolute paths, `~`, and `..` escapes are rejected.

`external_reference`
: Explicit absolute paths used for non-destructive adoption of existing data. Relative external paths are rejected.

The loader does not expand environment variables stored in JSON. A value like `${ZESOLVER_ASTAP_ROOT}` is rejected in a real manifest; environment variables belong to discovery and runners, not persisted library paths.

Symlinks are allowed through normal `Path.resolve()` behavior. The resolved path is preserved in models and reports.

## Validation

`library.validate()` returns:

- status;
- capabilities;
- issues;
- checked source IDs;
- checked index IDs;
- merged coverage.

Issues include:

- `MANIFEST_MISSING`
- `MANIFEST_INVALID_JSON`
- `SCHEMA_UNSUPPORTED`
- `REQUIRED_FIELD_MISSING`
- `SOURCE_PATH_MISSING`
- `SOURCE_UNREADABLE`
- `SOURCE_FAMILY_UNSUPPORTED`
- `INDEX_PATH_MISSING`
- `INDEX_SHA256_MISMATCH`
- `INDEX_SCHEMA_INCOMPATIBLE`
- `INDEX_SOURCE_UNKNOWN`
- `COVERAGE_INCONSISTENT`
- `PATH_ESCAPES_LIBRARY`

An integrity failure is not downgraded to a warning.

## Discovery

```python
from zesolver.catalog_library import discover_existing

proposal = discover_existing(
    astap_root=...,
    blind4d_manifest=...,
    legacy_index_root=...,
)
```

Discovery is read-only. It can also read:

1. explicit arguments;
2. `ZESOLVER_BLIND4D_MANIFEST`;
3. legacy alias `ZEBLIND_4D_MANIFEST`.

If both 4D environment variables are set differently, `ZESOLVER_BLIND4D_MANIFEST` wins and a warning is returned.

## Adapters

`library.near_source()` returns:

- root path;
- family tuple;
- source formats;
- coverage;
- whether any source is external.

This is enough for a future integration to build `CatalogDB(root, families=...)` without modifying `CatalogDB`.

`library.blind4d_indexes()` returns validated index descriptors:

- ID;
- path;
- family;
- tile keys;
- SHA256;
- coverage;
- schema.

Invalid indexes are excluded. The existing 4D manifest loader remains untouched.

## Limits

- The core does not yet write or migrate `catalog.json`.
- It does not validate NPZ internal metadata; current SHA/path validation is enough for P1B and keeps tests lightweight. The existing strict 4D loader remains the runtime authority for NPZ compatibility.
- It does not compute exact all-sky geometry from tile sets.
- It does not scan personal directories.
- It is not wired into solver settings or GUI.

Future integration belongs to `P1C - CatalogLibrary Adapter Integration`.
