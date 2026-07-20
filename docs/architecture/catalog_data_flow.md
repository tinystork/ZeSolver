# Catalog Data Flow

P1A maps the current flow as-is. No solver runtime is changed by this document.

## Current Flow

```text
ASTAP/HNSKY source shards
  d05/d20/d50/d80/v50_*.1476
  g05_*.290
        |
        | zewcs290.catalog290.CatalogDB
        | - filename discovery
        | - layout geometry from zewcs290.data.layouts.json
        | - binary record decode
        | - cone/box queries
        |
        +-------------------------> ZeNear / ImageSolver
        |                            - direct catalogue queries
        |                            - strict ASTAP-ISO catalogue selection
        |                            - WCS validation/writing
        |
        | zeblindsolver.db_convert
        | - source tile iteration
        | - TAN projection per tile
        | - magnitude cap
        | - per-tile star cap
        v
historical ZeBlind index root
  manifest.json
  tiles/*.npz
  hash_tables/quads_{level}.npz or .npy
        |
        +-------------------------> Historical ZeBlind runtime
        |                            - legacy diagnostic backend
        |                            - tile cone selection
        |                            - hash lookup
        |
        | zeblindsolver.quad_index_4d.build_experimental_4d_index
        | - selected tile_keys
        | - source tile_npz x/y/ra/dec arrays
        | - 4D AB-code generation
        v
ZeBlind 4D indexes
  indexes/astrometry_4d/*.npz
        |
        | config/zeblind_4d_experimental_manifest.json
        | - index ids and paths
        | - SHA256
        | - tile keys
        | - schema/count compatibility
        v
ZeBlind 4D runtime
  load_4d_index_manifest()
  Quad4DIndex.load()
  cKDTree lookup
```

## Formats

| Layer | Format | Owner | Current validation |
| ----- | ------ | ----- | ------------------ |
| ASTAP/HNSKY shards | `.1476` and `.290` binary tile files | Upstream ASTAP/HNSKY, stored locally by user | `CatalogDB` checks root exists, filenames, header size, record size, and decodes when loaded. |
| ASTAP layout metadata | bundled JSON | ZeSolver package | Loaded by `zewcs290.layouts`; no external version field is linked to an installed catalogue. |
| Historical tile store | NPZ per tile | ZeSolver builder output | `validate_index()` checks manifest, tile files, arrays, empty-tile statistics. |
| Historical quad hash tables | NPZ/NPY | ZeSolver builder output | `QuadIndex.load()` validates level metadata and sampler/hash schema. |
| Historical manifest | `manifest.json` | ZeSolver builder output | JSON with version, tile entries, builder parameters, absolute source `db_root`. |
| 4D index | compressed NPZ | ZeSolver 4D builder output | `load_4d_index_manifest()` opens each NPZ and validates schema, version, tile keys, counts and sampler. |
| 4D manifest | `zeblind.astrometry_4d_index_manifest.v1` JSON | ZeSolver config/runtime | Strict path, SHA256, duplicate and compatibility checks. |
| Regression corpus manifest | test JSON | ZeSolver tests | Test loader validates schema, path resolution and optional SHA. Not a product catalogue manifest. |

## Metadata Preserved and Lost

### Preserved

- ASTAP family key is preserved from source shard into historical tile entries (`family`) and tile keys (`d50_2823`).
- Tile code, ring and rough sky bounds are preserved in historical manifest entries.
- 4D manifest preserves tile keys, index schema, star count, quad count, sampler tag and SHA256.
- 4D NPZ metadata preserves builder parameters such as `max_stars_per_tile`, `max_quads_per_tile`, `sampler_tag`, `dtype`, and `source_index_root`.

### Lost or Weak

- Historical manifest stores the source `db_root` as an absolute path, so provenance is not portable.
- 4D manifest currently stores `source_index_root` strings pointing to historical report directories; they are useful provenance but not portable library references.
- ASTAP/HNSKY source file checksums are not stored in a common manifest.
- Layout version and catalogue family provenance are implicit in code and filenames.
- 4D `supported_scale_range_arcsec` is currently `null`; runtime scale compatibility is therefore inferred from validation, not declared coverage.
- No common manifest states whether Near and Blind indexes were derived from the same ASTAP source snapshot.

## Path Dependencies

| Path | Current source | Portability issue |
| ---- | -------------- | ----------------- |
| `db_root` | CLI/GUI setting, historical index manifest | Exposed directly to users and stored separately from 4D manifest. |
| `index_root` | CLI/GUI setting | Historical derived index path remains separate from source catalogue path. |
| `blind_4d_manifest_path` | CLI/GUI setting or default runtime resource | Product runtime depends on a manifest file rather than a library capability. |
| 4D index relative paths | `config/zeblind_4d_experimental_manifest.json` | Portable inside repository/package, but only for the bundled six-index subset. |
| `source_index_root` in 4D manifest | report/build path | Absolute forensic provenance, not a portable runtime dependency. |

## Validations Present

- `CatalogDB` detects missing roots, unsupported families, bad record sizes and malformed loaded tiles.
- `quad_index_builder.validate_index()` can validate historical index payloads and compare a supplied ASTAP `db_root`.
- `index_manifest_4d.load_4d_index_manifest()` enforces manifest schema, SHA256, index schema, version, tile keys, counts and sampler.
- P0B regression tests validate WCS and FITS pixel integrity for representative cases.

## Validations Absent

- No common manifest asserts that source ASTAP shards and derived indexes share the same source snapshot.
- No product-level status differentiates `NEAR_ONLY`, `BLIND4D_ONLY`, `READY_PARTIAL`, `CORRUPT`, etc.
- No all-sky/partial coverage object is available to GUI or CLI.
- No atomic library manifest update protocol exists for future catalogue installation/build.
- No checksum inventory exists for installed ASTAP source shards.

## CatalogLibrary Boundary

`CatalogLibrary` should sit above these flows and below product configuration:

```text
ProductSettings.catalog_library
        |
        v
CatalogLibrary
  - validates catalog.json
  - exposes Near source root/families
  - exposes Blind 4D manifest/index entries
  - reports coverage and issues
        |
        +--> ZeNear adapter keeps using CatalogDB
        +--> ZeBlind 4D adapter keeps using Loaded4DManifest / NPZ indexes
```

The goal is management unification, not a single binary runtime format.
