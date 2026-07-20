# Library-Owned Blind 4D Manifest

P1D-4A made `catalog.json` capable of generating the strict Blind 4D manifest
view expected by the existing runtime loader.  P1D-4B makes that view the
normal product runtime source when an explicit `CatalogLibrary` provides Blind
4D indexes.

## Authority

The data authority is now:

```text
CatalogLibrary
-> CatalogIndex(engine="blind4d")
-> CatalogBlind4DManifestView
-> load_4d_index_manifest_payload()
-> Blind 4D runtime
```

The current external strict manifest remains a compatibility oracle and a
migration/adoption input.  Normal view generation and normal runtime selection
do not consult it when a library is explicitly configured.

## Runtime Order

Blind 4D search order is a runtime contract.  The additive manifest field is:

```json
{
  "runtime_order": {
    "blind4d": [
      "d50_2823_S_q40000",
      "d50_2822_S_q40000",
      "d50_2644_S_q40000",
      "d50_2645_S_q40000",
      "d50_2602_S_q40000",
      "d50_2702_S_q40000"
    ]
  }
}
```

Old manifests remain readable when the field is absent.  A strict Blind 4D view
cannot be generated from multiple Blind 4D indexes without this explicit order;
the view reports `BLIND4D_VIEW_RUNTIME_ORDER_MISSING` rather than sorting
silently.

## View API

```python
from zesolver.catalog_library import build_blind4d_manifest_view

view = build_blind4d_manifest_view(catalog_library)
```

The returned `CatalogBlind4DManifestView` exposes:

- `payload`: strict manifest JSON payload;
- `schema` and `version`;
- ordered `entries`;
- merged Blind 4D `coverage`;
- deterministic `fingerprint`;
- structured `warnings` and `errors`;
- `source_library_id`;
- `source_manifest_fingerprint`.

Construction is in-memory and read-only by default.

## Strict Entry Fields

Each view entry is generated from a `CatalogIndex` plus the declared NPZ file
itself.  The view validates and emits:

- `id`;
- `path`;
- `filename`;
- `sha256`;
- `quad_schema`;
- `index_version`;
- `level`;
- `tile_keys`;
- `star_count`;
- `quad_count`;
- `sampler_tag`;
- `code_tol_recommended`;
- `catalog_source`;
- `priority`;
- `file_size_bytes`.

The NPZ is loaded only to validate the derived file declared in `catalog.json`
and to supply fields the strict loader checks anyway.  The external strict
manifest is not used to fill missing values.

## Runtime Source Policy

Product routes use one central selector:

```python
resolve_blind4d_runtime(resources, mode="auto", external_manifest_path=...)
```

Supported modes:

```text
auto
library-view
external-manifest
```

Priority is explicit:

```text
forced mode
-> explicit CatalogLibrary
-> external manifest compatibility when no library is configured
```

In `auto`, a valid library with Blind 4D indexes selects
`catalog_library_view` even when a stale external manifest path or environment
manifest exists.  A library without Blind 4D returns Blind 4D unavailable and
does not silently fall back to the external manifest.  Forced
`external-manifest` remains the rollback path and never tries to build a
library view.

Stable runtime errors include:

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

## Payload Loader

`zeblindsolver.index_manifest_4d` keeps the historical file API:

```python
load_4d_index_manifest(path)
```

and exposes an in-memory payload API:

```python
load_4d_index_manifest_payload(payload, manifest_path=None, index_root=None)
```

Both entry points share the same strict validation for schema/version, entry
order, duplicate ids/paths/tiles, paths, SHA256, NPZ metadata, tile keys,
counts, sampler and tolerance.  Relative paths in payload mode require an
explicit `manifest_path` or `index_root`; they never depend on the current
working directory.

## Cache and Lifetime

The product runtime resolves Blind 4D resources once per solver/pipeline
context and reuses the resulting loaded manifest.  The process-local cache is
invalidated by:

```text
catalog library resource
catalog manifest fingerprint
requested mode
external manifest path
runtime order / index ids
index checksums
```

No KD tree or heavy runtime object is written into persisted settings.

## Telemetry

Blind 4D runs expose source-selection telemetry without personal paths in
normal GUI telemetry:

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

For library mode, `blind4d_external_fallback_used` is always `false`.

## Coverage

Source ASTAP coverage and Blind 4D derived-index coverage remain distinct.  A
complete ASTAP D50 source can coexist with:

```text
Blind 4D coverage = PARTIAL, covered_tiles = 6, total_tiles = 1476, all_sky = false
```

The view rejects accidental all-sky promotion through
`BLIND4D_VIEW_COVERAGE_INCONSISTENT`.

## Fingerprint

The view fingerprint covers:

- strict schema and version;
- entry order;
- entry ids;
- path policy/value from `catalog.json`;
- SHA256;
- families and tiles;
- runtime parameters;
- coverage;
- strict-loader counts and schema/version.

It excludes:

- timestamps;
- temporary materialization paths;
- accidental dictionary ordering;
- diagnostic fields such as resolved paths.

Changing order, checksum, tile set, coverage or `code_tol_recommended` changes
the fingerprint.

## Materialization

`view.materialize(path)` writes the strict payload atomically only when called
explicitly:

```text
serialize in memory
-> write temp file in target directory
-> flush + fsync temp
-> validate temp through load_4d_index_manifest()
-> os.replace()
-> fsync directory when supported
-> reload final strict manifest
```

Existing files are refused unless `overwrite=True`.  No materialization happens
at startup and no ASTAP/index resource is modified.

## Stable Errors

Implemented errors include:

```text
BLIND4D_VIEW_NO_INDEXES
BLIND4D_VIEW_INDEX_MISSING
BLIND4D_VIEW_INDEX_CORRUPT
BLIND4D_VIEW_CHECKSUM_MISMATCH
BLIND4D_VIEW_SCHEMA_UNSUPPORTED
BLIND4D_VIEW_RUNTIME_ORDER_MISSING
BLIND4D_VIEW_RUNTIME_ORDER_DUPLICATE
BLIND4D_VIEW_TILE_DUPLICATE
BLIND4D_VIEW_COVERAGE_INCONSISTENT
BLIND4D_VIEW_PATH_INVALID
BLIND4D_VIEW_MATERIALIZATION_FAILED
```

## CLI

Advanced tool:

```bash
.venv/bin/python tools/generate_blind4d_manifest_view.py \
  --catalog-library /tmp/ZeSolverCatalog \
  --report-json /tmp/blind4d_view_preview.json
```

Preview mode is read-only.  Writing requires `--write --output <path>`.

## Product Switch

P1D-4B switches the product routes to this selector while preserving the
external strict manifest as explicit rollback and legacy compatibility.  The
runtime algorithms, thresholds, policies and NPZ indexes remain unchanged.
