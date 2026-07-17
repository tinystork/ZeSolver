# Catalog Migration Matrix

P1A does not migrate any files. This matrix defines how a future migration should detect and adopt existing installations without moving data by surprise.

## Migration Modes

`REFERENCE_EXISTING`
: Create or update `catalog.json` so it points at existing source/index paths. No data is copied or moved. This is the preferred first migration mode.

`IMPORT_COPY`
: Copy selected data into a new `ZeSolverCatalog/` tree, compute checksums, and leave the original installation untouched. This should require explicit user confirmation.

`FRESH_INSTALL`
: Install or build catalogue data into an empty library. This belongs to a later installer/update phase, not P1A/P1B.

## Existing Installation Matrix

| Installation actuelle | Détectable | Near disponible | Blind disponible | Migration proposée | Destructive |
| --------------------- | ---------: | --------------: | ---------------: | ------------------ | ----------: |
| ASTAP/HNSKY folder only (`db_root`) | Yes, via `FAMILY_SPECS` glob scan and optional `CatalogDB` validation | Yes if supported families are present | No | `REFERENCE_EXISTING` as `SOURCE_ONLY` or `NEAR_ONLY`; offer explicit index install/build later | No |
| Complete ASTAP D50 plus current six bundled 4D indexes | Yes, via source scan and 4D manifest | Yes | Yes, partial 4D | `REFERENCE_EXISTING` as `READY_PARTIAL`; mark Blind 4D coverage limited | No |
| Historical ZeBlind index root only (`index_root/manifest.json`) | Yes, via `quad_index_builder.validate_index` | Maybe, because manifest may contain `db_root`; source root must still be checked | Historical only; 4D only if separate manifest exists | `REFERENCE_EXISTING`; classify historical as compatibility/diagnostic, not product default | No |
| Historical index with valid `db_root` in manifest | Yes | Yes if `db_root` exists and families validate | Historical diagnostic; possible source for future 4D builds | Adopt source and historical derived index by reference | No |
| Historical index with stale/missing `db_root` | Yes | No until source root selected | Historical may still load tile NPZs, but provenance is incomplete | Adopt as `BLIND4D_ONLY`/legacy diagnostic if valid; ask user for source root | No |
| Separate 4D manifest path | Yes, via `load_4d_index_manifest` or read-only audit | No unless ASTAP root also configured | Yes if manifest and NPZs validate | Adopt manifest and indexes by reference; status `BLIND4D_ONLY` or `READY_PARTIAL` depending source | No |
| Multiple 4D manifests | Yes if user supplies paths or settings list exists later | Depends on source | Yes per manifest if valid | Create multiple `derived_indexes` entries; surface union coverage and duplicates | No |
| Partial 4D indexes with missing NPZ | Yes | Depends on source | No for missing entries; maybe partial for remaining enabled entries if manifest policy allows | Report `CORRUPT`/`MISSING`; do not silently ignore enabled missing indexes | No |
| ASTAP source with unsupported families only | Yes | No | No | Report `INCOMPATIBLE` or `MISSING` for current solver capabilities | No |
| Several ASTAP families in one root | Yes | Yes for supported families | No unless indexes exist | Adopt all supported families; product profile chooses default family internally | No |
| GUI persistent settings with `db_root`, `index_root`, `blind_4d_manifest_path` | Yes via `PersistentSettings` | Depends on validation | Depends on validation | Generate candidate library manifest by reference; keep old settings until confirmed | No |
| Old settings selecting historical profile | Yes | Depends | Historical diagnostic only | Migrate product default toward 4D profile; preserve historical only as explicit dev/diagnostic compatibility | No |
| Current bundled `config/zeblind_4d_experimental_manifest.json` | Yes as runtime resource | No source by itself | Yes, partial 4D | Represent as built-in partial derived index set until packaged library exists | No |
| External drive library with relative `catalog.json` | Yes | Depends | Depends | Open directly; prefer relative internal paths for portability | No |
| Empty chosen library directory | Yes | No | No | Initialize `catalog.json` only with explicit user action | No |

## Detection Order

Future code should prefer:

1. explicit `ProductSettings.catalog_library`;
2. explicit CLI/runner override;
3. environment variables used by tests/integration;
4. existing persistent settings (`db_root`, `index_root`, `blind_4d_manifest_path`);
5. package default 4D manifest as a built-in partial resource;
6. clear `MISSING` status.

It should not silently scan the home directory or assume `/home/tristan/...` paths.

## Backward Compatibility

- Old raw path settings remain readable until the migration is complete.
- Migration writes a new library reference only after validation.
- `historical` backend remains available for diagnostics but must not become the normal default.
- If a library manifest is invalid, fall back to old explicit paths only when the user explicitly chooses legacy mode or during tests.
- Adoption by reference must record whether paths are absolute external references.

## Rollback

Because `REFERENCE_EXISTING` does not move data, rollback is simply:

1. stop using the generated `catalog.json`;
2. restore previous explicit settings;
3. leave all source and index files untouched.

`IMPORT_COPY` rollback must never delete original data and should remove only the newly copied library directory with user confirmation.
