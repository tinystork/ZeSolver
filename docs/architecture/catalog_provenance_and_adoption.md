# Catalog Provenance and Adoption

P1D-2A extends the existing `CatalogLibrary` manifest contract without changing
solver behavior.  The schema remains `schema_version = 1`; all provenance and
adoption fields are optional so previously valid manifests keep loading and
produce the same runtime descriptors.

## Scope

The provenance contract describes:

- ASTAP/HNSKY source catalogues adopted by reference or managed inside a
  library;
- physical source shards, families, formats, layout fingerprints and reader
  versions;
- Blind 4D NPZ indexes derived from those sources;
- source-to-index tile relationships;
- deterministic rebuild parameters when they are known;
- legacy historical indexes retained only as compatibility resources;
- repair actions as prescriptions, never execution.

P1D-2A does not write `catalog.json`, move files, rebuild indexes, modify ASTAP
shards, modify NPZ indexes, call builders, import GUI modules, or change
ZeNear/ZeBlind thresholds or runtime acceptance behavior.

## ASTAP Source Provenance

Each ASTAP source keeps the old fields:

```text
id, kind, family, format, path, tile_count, layout, coverage, integrity, status
```

It may also carry:

```text
families
mode
path_policy
shards
layout_fingerprint
reader_version
fingerprint_policy
```

`mode = external-reference` means ZeSolver points at an existing installation
and does not own, move, rename, repair or delete those files.  Persisted paths
must be concrete path references; environment-variable placeholders such as
`${ZESOLVER_ASTAP_ROOT}` are rejected by the loader.

Shard entries are relative to the declared source root and may contain:

```text
path, family, tile_code, size_bytes, sha256, mtime_ns, status
```

`mtime_ns` is informative only and is excluded from canonical provenance
fingerprints.

## Fingerprint Policy

Two verification policies exist:

`fast`
: Verifies existence, file type, file size, deterministic inventory, family
  glob, tile code and layout consistency where available.  It records
  `FAST_VERIFIED` but must not claim cryptographic source verification.

`full`
: Computes or verifies SHA256, complete inventory, layout fingerprint and
  source/index consistency where information exists.  It records
  `FULL_VERIFIED` for files it hashed.

The status vocabulary distinguishes:

```text
UNVERIFIED
FAST_VERIFIED
FULL_VERIFIED
MISMATCH
MISSING
CORRUPT
```

The `CatalogLibrary` runtime validation still hashes only entries that contain
an expected SHA256.  Full source hashing is an explicit adoption/verification
operation, not a side effect of opening a library.

## Blind 4D Provenance

Blind 4D indexes keep the old descriptor fields and may add:

```text
category
families
build_parameters
parameter_status
provenance_fingerprint
reconstruction_status
derived_files
source_file_refs
```

The manifest may also declare top-level runtime order:

```text
runtime_order.blind4d = ordered derived index ids
```

This order is additive in schema version 1 and is required when generating the
strict Blind 4D runtime view.  It prevents `first_accept` and budget-sensitive
runtime behavior from depending on accidental JSON object or filename ordering.

Build parameters may include:

```text
level
mag_cap
max_stars_per_tile
max_quads_per_tile
sampler_tag
code_tol_recommended
dtype
tan_center_policy
star_ordering_policy
star_truncation_policy
projection_implementation
projection_version
quad_schema
quad_version
builder_version
catalog_source
```

Missing historical values are not invented.  `parameter_status` is `KNOWN`,
`PARTIAL` or `UNKNOWN`.  Current tile-NPZ-derived indexes normally remain
`PARTIAL` for deterministic reconstruction until P1D-3 supplies a direct,
fully pinned ASTAP-to-4D builder path.

## Source To Index Link

An index links to source data through:

- `source_ids`;
- `source_tiles`;
- `families`;
- `source_file_refs`;
- `build_parameters`;
- `derived_files`;
- `provenance_fingerprint`.

The link is exact when all referenced source IDs and source tiles exist in the
adopted ASTAP inventory.  Missing source families or tiles create warnings and
repair prescriptions; they do not promote partial coverage to all-sky.

## Canonical Fingerprints

`canonical_provenance_fingerprint()` computes SHA256 over canonical JSON:

- keys sorted recursively;
- compact JSON separators;
- ASCII output;
- no dependency on accidental dictionary order;
- generated timestamps and mtime fields removed;
- relative managed paths represented by their manifest value, not machine
  absolute paths;
- external references remain explicit as external references.

The fingerprint is provenance, not a replacement for file integrity SHA256.
Derived file SHA256 still belongs to integrity metadata and may differ if the
bytes differ.

## Compatibility Resources

Historical indexes are represented outside normal product indexes:

```text
compatibility_resources[].category = compatibility
```

They may reference:

```text
index_root
manifest.json
tiles/
hash_tables/
```

They are not ASTAP astronomical sources, not ZeNear dependencies in
ASTAP-native mode, not required modern library resources, and not evidence of
Blind 4D coverage.

## Adoption Planner

Public API:

```python
from zesolver.catalog_library import CatalogLibraryAdoptionPlan

plan = CatalogLibraryAdoptionPlan.reference_existing(
    library_root=...,
    astap_roots=...,
    blind4d_manifest=...,
    legacy_index_root=...,
    fingerprint_policy="fast",
)
```

The returned plan exposes:

```text
status
sources
indexes
compatibility_resources
coverage
warnings
errors
repair_actions
manifest_preview
telemetry
```

The planner is read-only.  It inspects explicit paths, inventories source
shards, loads the strict Blind 4D manifest with the existing strict loader,
links indexes to source families and tiles, captures the strict manifest entry
order into `runtime_order.blind4d`, preserves partial coverage, and returns a
deterministic in-memory manifest preview.

## Repair Actions

Repair actions are prescriptions only:

```text
VERIFY_SOURCE_SHA256
LOCATE_MISSING_SOURCE
LOCATE_MISSING_INDEX
REBUILD_BLIND4D_INDEX
REGENERATE_STRICT_4D_MANIFEST
REMOVE_STALE_COMPATIBILITY_REFERENCE
RECOMPUTE_COVERAGE
```

Each action records a stable code, severity, resource ID, reason,
preconditions, execution phase and whether it could be automatic.  Rebuilding
Blind 4D indexes is explicitly `P1D-3_OR_LATER`.

## Path Boundaries

Managed relative paths are resolved inside `library_root` and reject traversal,
including Windows-style `..\` traversal.  External references must be absolute
and may live outside `library_root`.  The loader accepts Windows or POSIX
separators for managed relative paths by normalizing them before resolution.

Symlinks follow normal `Path.resolve()` behavior.  A managed path that resolves
outside the library is rejected.  A valid external reference is not rejected
because it is outside the library.
