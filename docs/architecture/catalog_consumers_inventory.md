# Catalog Consumers Inventory

This inventory was produced for P1A without changing solver behavior. It distinguishes astronomical source data from derived indexes and from settings that merely point to them.

## Summary

ZeSolver currently has three catalogue-related data layers:

1. ASTAP/HNSKY shards are the astronomical source of truth read by `CatalogDB`.
2. The historical ZeBlind index is a derived `tile_npz` and quad-hash store built from ASTAP/HNSKY shards.
3. ZeBlind 4D indexes are derived from the historical `tile_npz` layer and referenced by a strict 4D manifest.

The product-facing weakness is that GUI, CLI, settings, Near, historical Blind, and 4D each still handle separate paths (`db_root`, `index_root`, `blind_4d_manifest_path`). A future `CatalogLibrary` should own these paths and expose capabilities instead of making the user pick internal files.

## Consumers

| Composant | Donnée consommée | Format | Chemin reçu depuis | Usage | Catégorie | Risque |
| --------- | ---------------- | ------ | ------------------ | ----- | --------- | ------ |
| `zewcs290.catalog290.CatalogDB` | ASTAP/HNSKY catalogue shards | `d05/d20/d50/d80/v50_*.1476`, `g05_*.290` | Constructor `db_root`, optional `families` | Discovers tiles, decodes records, serves cone/box queries | `ASTRONOMICAL_SOURCE` | Raises if root missing or no supported tiles; callers expose raw root paths. |
| `zewcs290.layouts` | ASTAP/HNSKY tiling metadata | bundled `layouts.json` | package resource | Computes ring/RA cell geometry for `.1476` and `.290` layouts | `ASTRONOMICAL_SOURCE` | Geometry is embedded in code package; external catalog manifests do not yet state expected layout version. |
| `zeblindsolver.astap_db_reader` | ASTAP/HNSKY shards via `CatalogDB` | decoded `STAR_DTYPE` arrays | `db_root` passed to `iter_tiles()`/`load_tile_stars()` | Supplies tile metadata and stars to index builders | `ASTRONOMICAL_SOURCE` | Uses a process-global cache keyed by resolved root; family filtering is not exposed here. |
| `zeblindsolver.db_convert` | ASTAP/HNSKY source catalogue | `.1476` / `.290` through `astap_db_reader` | CLI `--db-root` and `--index-root` | Builds historical tile NPZ store and quad hash tables | `DERIVED_INDEX` | Writes absolute `db_root` into legacy manifest; builder settings are mixed with product settings. |
| `zeblindsolver.db_convert` output | Historical tile store | `tiles/*.npz` with `ra_deg`, `dec_deg`, `mag`, `x_deg`, `y_deg`, `sweep_rank` | `index_root/tiles` | Runtime catalogue for historical Blind and source for 4D index building | `DERIVED_INDEX` | Derived data can outlive or drift from the source catalog root. |
| `zeblindsolver.db_convert` output | Historical index manifest | `index_root/manifest.json` | `index_root` | Lists tiles, levels, bounds, generation parameters, source `db_root` | `MANIFEST` | Not portable because `db_root` is absolute. |
| `zeblindsolver.quad_index_builder` | Historical manifest, tile NPZs, quad hash tables | JSON + NPZ/NPY | `index_root` | Validates and loads historical quad index; selects tiles for hinted/blind phases | `DERIVED_INDEX` | Still required by legacy paths and by 4D index generation; validation is separate from 4D manifest validation. |
| `zeblindsolver.quad_index_4d.build_experimental_4d_index` | Historical tile NPZs | `index_root/manifest.json` + `tiles/*.npz` | builder arg `index_root`, explicit `tile_keys` | Builds 4D AB-code NPZ indexes | `DERIVED_INDEX` | 4D indexes currently derive from historical tile NPZs, not directly from ASTAP shards; provenance therefore needs both source catalog and intermediate index metadata. |
| `zeblindsolver.quad_index_4d.Quad4DIndex` | ZeBlind 4D index | NPZ with `codes_4d`, quad arrays, catalogue arrays, JSON `metadata` | path from manifest/profile | Runtime 4D candidate lookup | `DERIVED_INDEX` | NPZ format is internal and should not be exposed in normal GUI. |
| `zeblindsolver.index_manifest_4d` | 4D manifest | `zeblind.astrometry_4d_index_manifest.v1` JSON | explicit manifest path or runtime default | Strictly validates enabled 4D indexes, SHA256, schema, counts and tile keys | `MANIFEST` | Reads full indexes for strict runtime validation; partial sky coverage must remain visible. |
| `config/zeblind_4d_experimental_manifest.json` | Product-bundled 4D manifest | JSON | runtime resource | Default 4D manifest for `zeblind_4d_experimental` | `MANIFEST` | Contains only six D50 tiles and absolute historical `source_index_root` provenance strings. |
| `zeblindsolver.metadata_solver.solve_near` | Historical index manifest and ASTAP shards | `index_root/manifest.json`, `CatalogDB(db_root)` | `NearSolveConfig.index_root`; manifest `db_root`; optional family | ZeNear tile selection, strict ASTAP-ISO catalogue queries, WCS writing | `LEGACY_COMPATIBILITY` | Near currently discovers source catalogue through the legacy index manifest, not through a library object. |
| `zeblindsolver.zeblindsolver.solve_blind` | Historical or 4D indexes | historical `index_root` or 4D NPZ paths | `BlindSolveConfig` and profile parameters | Blind solving | `DERIVED_INDEX` | Multiple index configuration surfaces remain active; historical backend must stay diagnostic/legacy. |
| `zeblindsolver.profiles` | 4D profile index paths | profile object generated from loaded 4D manifest | `SolverProfile.to_blind_config()` | Converts profile to `BlindSolveConfig` | `MANIFEST` | Profile name is product-visible while implementation details are still internal. |
| `zesolver.blind4d_runtime` | Default 4D manifest path | JSON path | explicit arg, env `ZEBLIND_4D_MANIFEST`, package config | Resolves product 4D manifest without scanning arbitrary folders | `MANIFEST` | P0B regression env var is `ZESOLVER_BLIND4D_MANIFEST`; runtime uses `ZEBLIND_4D_MANIFEST`. Future library should rationalize this boundary. |
| `zesolver.SolveConfig` | Source catalogue root, families, legacy index path, 4D manifest | `Path` fields and loaded manifest object | CLI, GUI settings, tests | Carries catalogue/index paths into solvers | `USER_SETTING` | One config object mixes product choices with internal runtime details. |
| `zesolver.ImageSolver` | `CatalogDB` instance | decoded ASTAP/HNSKY | `SolveConfig.db_root`, `SolveConfig.families` | ZeSolver application Near route | `ASTRONOMICAL_SOURCE` | Constructs DB directly; future GUI cannot hide catalog internals until this is adapted. |
| `zesolver.build_blind_config_from_app_config` | 4D manifest and profile | `Loaded4DManifest`, profile parameters | app `SolveConfig` | Builds `BlindSolveConfig` for local fallback | `MANIFEST` | Good place for future `CatalogLibrary.blind4d_indexes()` adapter. |
| `zesolver` CLI | ASTAP root, index root, 4D manifest, family | CLI args `--db-root`, `--blind-index`, `--blind-4d-manifest`, `--family` | user arguments | Direct run and GUI launch | `USER_SETTING` | CLI still exposes internal paths needed by current architecture. |
| `zesolver` GUI/settings | ASTAP root, historical index root, 4D manifest, family choices | persistent settings `~/.zesolver_settings.json` | GUI forms and `PersistentSettings` | User configuration, validation, index build workers | `USER_SETTING` | GUI exposes database folder, index folder, manifest file, and dev family selection separately. |
| `zesolver.DbFamilyScanner` / `_scan_astap_families` | ASTAP root filenames | glob patterns from `FAMILY_SPECS` | GUI database tab root | Detects available families for UI | `ASTRONOMICAL_SOURCE` | Scans only filenames; does not validate tile integrity. |
| `zesolver.IndexBuildWorker` | ASTAP root and historical index root | `.1476/.290` to NPZ/hash manifest | GUI settings | Builds historical index from GUI | `DERIVED_INDEX` | Builder remains a developer/maintenance operation; future library should not trigger this during solving. |
| `zesolver.settings_store.PersistentSettings` | Catalogue and index path settings | JSON in home directory | GUI/CLI persistence | Stores `db_root`, `index_root`, `blind_4d_manifest_path`, family cache and many builder/solver parameters | `USER_SETTING` | Legacy settings can point to partial or stale data; migration must be non-destructive. |
| `tools/diagnose_*`, `tools/build_*` | Varies by diagnostic | reports, FITS, ASTAP roots, indexes | CLI args or hard-coded experiment context | Benchmarks and forensic replay | `GENERATED_REPORT` / `LEGACY_COMPATIBILITY` | Some historical tools encode campaign assumptions; they should inform CatalogLibrary but not become product API. |
| `tests/corpus/manifest.json` | External corpus references and oracles | lightweight JSON | env-resolved corpus roots | Regression test data mapping | `GENERATED_REPORT` | Should remain a test fixture, not a runtime catalogue manifest. |

## Current Categories

`ASTRONOMICAL_SOURCE`
: ASTAP/HNSKY shards and their bundled layout definitions.

`DERIVED_INDEX`
: Historical tile NPZs, historical quad hash tables, and 4D NPZ indexes derived from catalogues.

`MANIFEST`
: Metadata files that describe source/index content. Existing manifests are not yet unified.

`CACHE`
: Runtime caches such as `astap_db_reader._TileCache` and `quad_index_builder` manifest cache.

`GENERATED_REPORT`
: Benchmark and diagnostic outputs used as regression evidence.

`USER_SETTING`
: Persisted or CLI-exposed paths and choices.

`LEGACY_COMPATIBILITY`
: Historical index/backend paths retained for reproducibility and diagnostics.

## CatalogLibrary Implications

- `CatalogLibrary` should become the owner of `db_root`, `index_root`, and `blind_4d_manifest_path`.
- Near can still receive a `CatalogDB`/ASTAP source adapter; it does not need to read 4D NPZs.
- Blind 4D can still receive explicit NPZ paths loaded from a manifest; it does not need to decode ASTAP shards at runtime.
- The GUI should see capability/status/coverage, not internal filenames or quad schemas.
- Existing settings should be adopted by reference first; no destructive migration is justified in P1.
