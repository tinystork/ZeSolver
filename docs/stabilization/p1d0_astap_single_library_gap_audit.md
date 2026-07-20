# P1D-0 ASTAP Single Library Gap Audit

Decision: `READY_FOR_P1D1_ASTAP_RUNTIME_UNIFICATION`

P1D-0 is an audit/specification mission only. No solver, threshold, profile,
catalogue shard, index, FITS, pipeline or GUI behavior was changed.

## 1. Etat Git initial

Initial command:

```bash
git status --short --branch
```

Initial state:

```text
## test...origin/test
 M AGENT.md
 M zesolver.py
 M zesolver/gui_pipeline/legacy_runner.py
 M zesolver/gui_pipeline/pipeline_runner.py
?? docs/architecture/gui_live_state_flow.md
?? docs/stabilization/original_stabilization_roadmap_20260716.md
?? docs/stabilization/p3av3_gui_live_state_report.md
?? docs/stabilization/p3av3_manual_gui_state_test.md
?? tests/p3av3_helpers.py
?? tests/test_gui_progress_no_duplicates.py
?? tests/test_gui_progress_pipeline.py
?? tests/test_gui_progress_realtime_legacy.py
?? tests/test_gui_progress_remaining_count.py
?? tests/test_gui_progress_stale_callback.py
?? tests/test_gui_progress_stop_restart.py
?? tests/test_gui_wcs_cleanup_refresh.py
?? tests/test_gui_wcs_status_scope.py
```

The working tree already contained P3A-V3 changes. P1D-0 does not revert or
rewrite them.

Note: `docs/stabilization/original_stabilization_mission_20260716.md` was
requested, but is not present in this checkout. The available archived mission
document is `docs/stabilization/original_stabilization_roadmap_20260716.md`.

## 2. Documents lus

Read or inspected for P1D-0:

- `AGENT.md`
- `docs/stabilization/original_stabilization_roadmap_20260716.md`
- `docs/stabilization/p1a_catalog_architecture_report.md`
- `docs/stabilization/p1b_catalog_library_report.md`
- `docs/stabilization/p1c_catalog_integration_report.md`
- `docs/architecture/catalog_consumers_inventory.md`
- `docs/architecture/catalog_data_flow.md`
- `docs/architecture/catalog_library.md`
- `docs/architecture/catalog_library_implementation.md`
- `docs/architecture/catalog_integration_map.md`
- `docs/architecture/catalog_migration_matrix.md`
- `docs/architecture/astap_family_strategy.md`
- `docs/architecture/blind4d_coverage_audit.md`
- `docs/architecture/catalog_manifest_schema.json`
- `docs/architecture/catalog_manifest_example.json`
- P3A/P3A-V2/P3A-V3 architecture and stabilization notes as context only.

Code inspected:

- `zesolver/catalog_library/`
- `zesolver/catalog_resources.py`
- `zesolver/settings_store.py`
- `zesolver/settings/`
- `zesolver/solver_config/`
- `zesolver.py`
- `zeblindsolver/astap_db_reader.py`
- `zeblindsolver/db_convert.py`
- `zeblindsolver/metadata_solver.py`
- `zeblindsolver/quad_index_4d.py`
- `zeblindsolver/index_manifest_4d.py`
- `zeblindsolver/quad_index_builder.py`
- catalogue, ZeNear, Blind 4D, pipeline and settings tests.

## 3. Flux actuel exact ZeNear

Current product path:

```text
Product/GUI/CLI settings
  -> SolveConfig(db_root, families, blind_index_path, catalog_library_path)
  -> apply_catalog_resources_to_config()
     - if CatalogLibrary valid: replaces db_root/families
     - does not remove blind_index_path dependency
  -> BatchSolver / ImageSolver
  -> ImageSolver._run_index_near_solver()
  -> near_solve(fits_path, index_root=self.config.blind_index_path, config=NearSolveConfig)
  -> zeblindsolver.metadata_solver.solve_near()
  -> quad_index_builder.load_manifest(index_root)
  -> manifest["tiles"] for tile candidate metadata
  -> manifest["db_root"] for ASTAP source root
  -> astap_db_reader.iter_tiles(db_root) / load_tile_stars(db_root, TileMeta)
  -> CatalogDB(db_root, family) for raw ASTAP stars
  -> WCS validation/writing
```

### A.1 Donnees lues reellement en mode produit

In the validated strict ASTAP-ISO ZeNear path, the star coordinates used for the
catalogue stack come from ASTAP/HNSKY source shards decoded through
`CatalogDB`/`astap_db_reader.load_tile_stars()`.

However, ZeNear still needs the historical index manifest for:

- tile entry list;
- tile family and `tile_code`;
- tile bounds and centers;
- fallback `tile_file` path;
- historical `db_root` provenance;
- `levels` and general manifest validation side effects in tools/tests.

Thus the product data source is already ASTAP for stars, but the runtime
selection shell is still historical-manifest native.

### A.2 Fonctions lisant directement ASTAP

Direct ASTAP/HNSKY readers:

- `zewcs290.catalog290.CatalogDB`
- `zeblindsolver.astap_db_reader._catalog_for_root()`
- `zeblindsolver.astap_db_reader.iter_tiles()`
- `zeblindsolver.astap_db_reader.load_tile_stars()`
- `zeblindsolver.metadata_solver._get_cached_catalog_db()`
- `zeblindsolver.metadata_solver._get_raw_tile_lookup()`
- strict branches in `zeblindsolver.metadata_solver.solve_near()`
- classic catalogue fallback in `ImageSolver._solve_with_catalog()`

### A.3 Pourquoi `near_solve()` demande encore `index_root`

`near_solve()` has a legacy signature:

```python
solve_near(input_fits, index_root, *, config, cancel_check)
```

It immediately resolves `index_root`, calls `load_manifest(index_path)`, and uses
the manifest as the candidate-tile model. Even when strict ASTAP-ISO later
switches candidate expansion and star loading to ASTAP, the function still gets
its initial manifest, candidate schema and `db_root` through the historical
index root.

### A.4 Informations de l'ancien manifeste encore necessaires

Still used or implied:

- `tiles[].tile_key`
- `tiles[].family`
- `tiles[].tile_code`
- `tiles[].center_ra_deg`
- `tiles[].center_dec_deg`
- `tiles[].bounds.dec_min/dec_max/ra_segments`
- `tiles[].tile_file`
- top-level `db_root`
- `levels` for historical index validation/tools
- `tile_count`, `stars`, `usable_ratio` for diagnostics/health

For strict ASTAP-native ZeNear, only the tile identity and geometry subset is
needed at runtime. `tile_file`, hash tables and quad levels are not
scientifically necessary for the current strict ASTAP-ISO solve.

### A.5 Tuiles NPZ historiques necessaires en ASTAP-ISO strict

No, not as a normal product requirement.

In strict ASTAP-ISO with valid `db_root` and raw tile lookup,
`load_astap_tile_stars(db_root, tm)` supplies the actual stars. Historical
`tiles/*.npz` remains a fallback if ASTAP lookup fails or strict mode cannot use
raw tiles. That fallback is compatibility, not the target product path.

### A.6 Remplacer la selection ancien manifeste par `CatalogDB`

Yes, with characterization tests. The manifest selection can be replaced by a
`CatalogDB`/`TileMeta` candidate provider because `astap_db_reader.iter_tiles()`
already exposes family, tile code, center and bounds. This should not change
astronomical results if:

- tile candidate ordering remains by angular distance;
- the same family filter is used;
- the same `max_tile_candidates` cap is kept;
- the same radius/FOV logic is kept;
- raw star order is preserved;
- fallback legacy mode remains available for comparison.

### A.7 Plus petit changement pour ZeNear ASTAP-native avec rollback

Add a narrow Near catalog-source abstraction without touching thresholds:

```text
NearTileProvider
  - from_legacy_index(index_root)
  - from_astap_catalog(db_root, families)
```

Then add a new optional `NearSolveConfig` field or wrapper input carrying a
prebuilt ASTAP provider. `near_solve()` can keep the old `index_root` signature
temporarily, but product code should call a new ASTAP-native entry point or pass
`index_root=None` only when a provider is supplied. Rollback is selecting the
legacy provider.

The first P1D-1 step should be this provider and parity tests, not direct removal
of `index_root`.

## 4. Flux actuel exact ZeBlind 4D

Current build flow:

```text
ASTAP/HNSKY shards
  -> astap_db_reader.iter_tiles(db_root)
  -> astap_db_reader.load_tile_stars(db_root, tile_meta)
  -> zeblindsolver.db_convert.build_index_from_astap()
     - projects each tile to TAN
     - applies mag_cap
     - caps stars
     - writes historical tiles/*.npz
     - writes historical manifest.json with absolute db_root
     - optionally builds historical quad hash tables
  -> zeblindsolver.quad_index_4d.build_experimental_4d_index(index_root, out_path, tile_keys)
     - reads historical manifest
     - reads historical tile_npz x/y/ra/dec/mag arrays
     - sorts by magnitude
     - samples quads
     - builds AB 4D records
     - writes 4D NPZ
  -> strict 4D manifest
  -> zeblindsolver.index_manifest_4d.load_4d_index_manifest()
  -> ZeBlind 4D runtime
```

### B.1 Chemin exact ASTAP -> index historique -> index 4D

The exact chain is `db_convert.build_index_from_astap()` followed by
`quad_index_4d.build_experimental_4d_index()`.

`db_convert` produces the intermediate `tiles/*.npz` and historical
`manifest.json`. The 4D builder requires that intermediate index root.

### B.2 Donnees `tile_npz` necessaires au builder 4D

Required arrays:

- `x_deg`
- `y_deg`
- `ra_deg`
- `dec_deg`
- `mag`

Used metadata:

- historical manifest tile entries;
- selected `tile_keys`;
- `tile_file` paths;
- builder parameters `level`, `max_stars_per_tile`, `max_quads_per_tile`,
  `sampler_tag`, `code_tol_recommended`, `dtype`.

`sweep_rank` is produced by the historical converter but is not read by the
current 4D builder.

### B.3 Donnees obtenables depuis `CatalogDB` et `astap_db_reader`

Yes. `astap_db_reader.load_tile_stars()` gives `ra_deg`, `dec_deg`, `mag` in raw
ASTAP order. `db_convert` already demonstrates how to derive the remaining
runtime arrays:

- compute tile center with `_cartesian_center()` or layout fallback;
- project with `project_tan()`;
- filter non-finite points;
- apply `mag_cap`;
- cap/truncate stars;
- derive `x_deg/y_deg`.

The missing piece is not data access; it is a single deterministic builder path
and provenance contract.

### B.4 Byte-compatible ou fonctionnellement equivalent depuis ASTAP

Functionally equivalent is realistic for P1D. Byte-compatible is possible only
if all non-scientific bytes are pinned, including:

- generated timestamp;
- JSON metadata ordering;
- NumPy dtype choices;
- compression writer and version behavior;
- ordering of tiles and stars;
- exact TAN center derivation;
- exact truncation mode;
- quad sampler order;
- all builder parameters.

Given the current metadata includes `generated_at`, strict byte identity should
not be the product gate. The P1D gate should require functional equivalence:
same tile keys, same deterministic input star lists, same solve outcomes within
existing tolerances, and stable manifest fingerprints.

### B.5 Parametres a enregistrer pour reconstruction deterministe

Record at least:

- ASTAP family;
- ASTAP source tile keys;
- ASTAP source shard path policy and SHA256/size;
- ASTAP layout key and layout version/fingerprint;
- source reader version;
- `mag_cap`;
- source star ordering/truncation mode;
- `max_stars` used for intermediate tile materialization;
- TAN center policy;
- projection function/version;
- 4D `level`;
- `max_stars_per_tile`;
- `max_quads_per_tile`;
- `sampler_tag`;
- `code_tol_recommended`;
- output dtype;
- quad schema and version;
- builder code version/commit;
- Python/NumPy/SciPy versions if byte reproducibility is required;
- index entry counts and star counts;
- checksum of every derived NPZ.

### B.6 Relier index, familles, tuiles et empreintes ASTAP

Each `CatalogIndex` should map:

```text
index.id
index.engine = blind4d
index.source_ids = astap family source ids
index.source_tiles = d50_2822, ...
index.parameters = deterministic builder parameters
index.integrity.files = derived NPZ hashes
index.provenance.source_files = source shard hashes for those tiles
```

The current manifest has `source_ids` and `source_tiles`, but lacks a compact
per-index source-file fingerprint map and a repair plan.

## 5. Matrice des donnees physiques

| Donnee physique | Role | Source astronomique | Produit normal cible | Etat actuel |
| --- | --- | ---: | ---: | --- |
| ASTAP/HNSKY `.1476` / `.290` | Etoiles de reference | Yes | Yes | Read by `CatalogDB`, but path still exposed as `db_root` |
| `zewcs290` layout data | Geometrie des tuiles | Support metadata | Yes | Bundled code resource, not fingerprinted in library |
| Historical `tiles/*.npz` | Etoiles derivees + projection | No | No in Strategy B | Required today for 4D build, fallback for Near |
| Historical `hash_tables/*` | Backend Blind historique | No | No | Compatibility/diagnostic |
| Historical `manifest.json` | Tile metadata + `db_root` provenance | No | No in Strategy B | Required by current `near_solve()` |
| 4D `*.npz` | Index performance Blind 4D | No, derived | Yes | Product runtime artifact |
| Strict 4D `manifest.json` | Runtime integrity/order | No | Yes, as library-owned descriptor or generated strict view | Still selected separately |
| `catalog.json` | Library management/provenance | No | Yes | Read-only core exists; not complete enough for rebuild/repair |
| Astrometry.net | Web fallback | External service | Optional distinct fallback | Preserved separately |

## 6. CatalogLibrary

### C.1 Criteres P1 deja remplis

Already satisfied:

- `CatalogLibrary` read-only core exists.
- It opens `catalog.json`.
- It validates schema, paths, statuses and checksums.
- It distinguishes `READY_FULL`, `READY_PARTIAL`, `NEAR_ONLY`,
  `BLIND4D_ONLY`, `SOURCE_ONLY`, `CORRUPT`, etc.
- It exposes Near descriptors.
- It exposes Blind 4D descriptors and runtime paths.
- It preserves partial coverage and never promotes six D50 indexes to all-sky.
- It resolves resources into `SolveConfig` additively.
- Legacy direct paths still work when no library is selected.
- Invalid explicit libraries fail clearly.

### C.2 Criteres partiellement remplis

Partial:

- The normal GUI still exposes `db_root`, `index_root`, families and 4D manifest.
- The CLI still exposes internal paths.
- `CatalogLibrary` does not assemble/write/repair a library.
- `CatalogLibrary` does not provide the historical manifest shape needed by
  current ZeNear.
- `CatalogLibrary` does not build 4D indexes from ASTAP.
- Source ASTAP shard checksums are not fully inventoried.
- Strict 4D manifest remains a separate runtime authority.
- Coverage is partial and represented, but not install/update actionable.

### C.3 Pourquoi pas de chemin historique ZeNear

`CatalogLibrary.near_source()` returns only:

```text
root
families
formats
coverage
external_reference
```

It deliberately does not expose `index_root`, `tiles/*.npz`, historical
`manifest.json`, or quad tables. Current `ImageSolver._run_index_near_solver()`
still uses `config.blind_index_path` as `index_root`; therefore a library-only
configuration cannot yet trigger ZeNear unless a legacy index path remains
configured.

### C.4 Ajouter un descripteur Near temporaire ou supprimer la dependance

Recommendation: do not add a product-level "Near historical index" descriptor as
the main path. That would encode Strategy A as permanent architecture.

Use a temporary compatibility descriptor only if needed for rollback/tests:

```text
LegacyNearIndexDescriptor
  root
  manifest_path
  source_db_root
  families
  category = COMPATIBILITY
```

But the P1D-1 implementation should primarily remove the dependency by adding an
ASTAP-native tile provider for ZeNear.

### C.5 Champs de manifeste insuffisants

Insufficient for deterministic reconstruction/repair:

- no full ASTAP source shard checksum inventory;
- no layout-data fingerprint;
- no per-index source-file fingerprint map;
- no builder command line;
- no builder code version/commit per derived index;
- no projection center policy field;
- no `mag_cap` vs 4D `max_stars_per_tile` distinction standardized;
- no star truncation/order policy standardized for 4D direct builds;
- no declared repair action for missing indexes;
- no atomic manifest update state;
- no strict 4D manifest synthesis rules;
- `scale_range_arcsec` is often null;
- example manifest uses `${ZESOLVER_ASTAP_ROOT}`, but real loader rejects `$`
  expansion in persisted paths.

### C.6 API minimale pour assembler sans deplacer les donnees

Add a non-destructive writer/planner:

```python
proposal = CatalogLibraryPlan.reference_existing(
    library_root,
    astap_roots=...,
    blind4d_manifest=...,
    legacy_index_root=...,
)
report = proposal.validate()
proposal.write_catalog_json_atomic()
```

Minimum behavior:

- adopt existing ASTAP roots as explicit external references;
- adopt existing 4D manifest/indexes by reference;
- optionally adopt historical index as compatibility only;
- compute or defer source checksums with explicit status;
- never copy, delete or rebuild data during adoption;
- write `catalog.json.tmp` then replace atomically.

## 7. Inventaire produit et compatibilite

| Surface | Usage actuel | Classement | Sort P1D cible |
| --- | --- | --- | --- |
| `ProductSettings.catalog_library_path` | Product settings v2 | `PRODUIT_NORMAL` | Keep and promote |
| `SolveConfig.catalog_library_path` | Resource resolution | `PRODUIT_NORMAL` | Keep |
| `PersistentSettings.catalog_library_path` | Legacy settings bridge | `PRODUIT_NORMAL` | Keep |
| CLI `--catalog-library` | User/runtime path | `PRODUIT_NORMAL` | Keep |
| `settings_store.db_root` | Raw ASTAP path | `COMPATIBILITE` | Deprecate from normal UI |
| `SolveConfig.db_root` | Existing `CatalogDB` input | `COMPATIBILITE` | Internal after library resolution |
| CLI `--db-root` | Direct legacy run | `COMPATIBILITE` / `OUTIL_AVANCE` | Keep temporarily |
| GUI `settings_db_edit` / database tab | Raw source picker | `DEPRECATION_POSSIBLE` | Move to advanced/adoption |
| env `ZESOLVER_ASTAP_ROOT` | Discovery/tests | `TEST` / `DIAGNOSTIC` | Keep for tests |
| `settings_store.index_root` | Historical index root | `COMPATIBILITE` | Deprecate from normal UI |
| `SolveConfig.blind_index_path` | Near index root, raster blind bridge, diagnostics | `COMPATIBILITE` | Remove from product ZeNear via P1D |
| CLI `--blind-index` | Historical index root | `OUTIL_AVANCE` / `DIAGNOSTIC` | Keep legacy-only |
| GUI `settings_index_edit` | Historical index root | `DEPRECATION_POSSIBLE` | Move to advanced |
| `benchmark_index_root` | Benchmark path | `DIAGNOSTIC` | Keep diagnostic |
| `blind_4d_manifest_path` | Strict 4D runtime manifest | `COMPATIBILITE` | Library-owned or synthesized |
| CLI `--blind-4d-manifest` | Direct strict manifest | `OUTIL_AVANCE` / `COMPATIBILITE` | Keep advanced override |
| GUI 4D manifest edit | Direct manifest file | `DEPRECATION_POSSIBLE` | Hide from normal UI after P1D |
| env `ZESOLVER_BLIND4D_MANIFEST` | Tests/regression | `TEST` | Keep |
| env `ZEBLIND_4D_MANIFEST` | Legacy alias/runtime | `COMPATIBILITE` | Deprecate alias later |
| `families` / CLI `--family` | Family filter | `OUTIL_AVANCE` / `COMPATIBILITE` | Product profile should own |
| GUI family combo | Family filter from index manifest | `DEPRECATION_POSSIBLE` | Move to advanced |
| `dev_family_auto/selection` | Developer family override | `OUTIL_AVANCE` | Keep advanced |
| `db_family_cache` | GUI detection cache | `DIAGNOSTIC` | Replace with library status later |

## 8. Comparaison strategie A / B

### Strategie A - pont de compatibilite

Description:

```text
CatalogLibrary
  -> ASTAP raw source
  -> historical Near/index root
  -> strict Blind 4D manifest and NPZ
```

Advantages:

- low implementation risk;
- preserves current `near_solve(index_root)` behavior;
- quick GUI simplification possible;
- easy rollback through existing paths;
- can represent current installations by reference.

Disadvantages:

- keeps duplicated star materialization in `tiles/*.npz`;
- keeps old `db_root` provenance as a runtime dependency;
- does not fully meet "unique ASTAP source" product intent;
- makes repair/rebuild provenance more complex;
- risks normalizing the historical index as a second library layer.

### Strategie B - ASTAP-native

Description:

```text
CatalogLibrary
  -> ASTAP raw source
     -> ZeNear tile provider
     -> direct 4D index builder
  -> library-owned derived Blind 4D indexes
```

Advantages:

- matches the product goal;
- removes historical `index_root` from normal solving;
- one astronomical source of truth;
- simpler provenance: ASTAP source shard -> derived 4D index;
- normal GUI can hide families/manifests/index roots honestly.

Disadvantages:

- requires Near parity characterization;
- requires direct 4D builder characterization;
- byte identity may be unrealistic unless metadata/time/compression are pinned;
- needs richer manifest fields and deterministic build records;
- migration must preserve rollback.

### Recommendation

Use a short A-to-B transition, but make B the target and P1D-1 the start of B.

Do not deepen Strategy A except as a rollback adapter. The first useful slice is
ASTAP-native ZeNear candidate selection because it removes the most visible
runtime contradiction: product library selected, but product ZeNear still
requires `blind_index_path`.

## 9. Architecture cible recommandee

Target:

```text
ZeSolverCatalog/
├── catalog.json
├── astap/
│   └── raw/
│       ├── d50_....1476
│       └── ...
└── indexes/
    └── blind4d/
        ├── manifest.json
        └── *.npz
```

Runtime:

```text
ProductSettings.catalog_library_path
  -> CatalogLibrary.open()
  -> Near ASTAP tile provider
     -> CatalogDB / astap_db_reader
     -> ZeNear
  -> Blind4D strict manifest or library-synthesized strict view
     -> existing 4D runtime
  -> Astrometry.net optional web fallback
```

Important boundary:

- `CatalogLibrary` owns paths, status, coverage, provenance and repair plans.
- ZeNear owns solving logic only.
- ZeBlind 4D owns matching logic only.
- Historical indexes are compatibility/diagnostic artifacts, not normal product
  source data.

## 10. Sequence de migration en petites etapes

1. **P1D-1 ASTAP runtime unification for ZeNear**
   - add ASTAP-native tile provider;
   - keep legacy provider rollback;
   - prove parity on existing ZeNear tests/corpus.
2. **P1D-2 Library manifest repair/provenance expansion**
   - add source shard fingerprint policy;
   - add deterministic builder parameter schema;
   - add non-destructive `REFERENCE_EXISTING` writer.
3. **P1D-3 Direct ASTAP -> 4D builder prototype**
   - produce functionally equivalent 4D indexes from ASTAP;
   - compare against historical tile_npz-derived indexes.
4. **P1D-4 Library-owned 4D runtime manifest**
   - synthesize or validate strict 4D manifest from `catalog.json`;
   - direct `blind_4d_manifest_path` becomes advanced override only.
5. **P1D-5 Product surface cleanup**
   - normal settings show only `catalog_library_path`;
   - raw `db_root`, `index_root`, families and manifest path move to advanced
     compatibility/diagnostic.

## 11. Risques de non-regression

- Near tile ordering or bounds differ between historical manifest and
  `CatalogDB` tiles.
- ASTAP raw row order vs historical `native_prefix` truncation differs.
- Family filtering changes candidate coverage.
- Direct 4D builder changes TAN center policy or magnitude filtering.
- Existing six-index 4D partial coverage is accidentally presented as full.
- Strict 4D manifest order changes solve outcomes/performance.
- GUI/CLI compatibility paths break existing local workflows.
- Tests accidentally use synthetic fixtures that hide real ASTAP layout issues.

## 12. Tests necessaires pour P1D-1

Required:

- `tests/test_zenear_astap_native_tile_provider.py`
- `tests/test_zenear_astap_native_legacy_parity.py`
- `tests/test_zenear_astap_native_no_index_root.py`
- `tests/test_catalog_library_near_runtime_unification.py`
- `tests/test_catalog_library_no_silent_legacy_fallback.py`
- `tests/test_near_tile_selection_from_catalogdb.py`
- regression Near/corpus checks where external data are configured;
- existing CatalogLibrary, pipeline, Blind 4D and GUI lifecycle tests.

Key assertions:

- library-only Near can run without `blind_index_path`;
- old `index_root` path still works in explicit legacy mode;
- selected tile keys match or acceptable deltas are documented;
- solved count/RMS/inliers do not regress on the reference corpus;
- missing/corrupt library does not fall back silently;
- partial 4D coverage remains visible.

## 13. Reponses synthetiques aux questions obligatoires

### A. ZeNear

1. ZeNear reads ASTAP stars through `CatalogDB` in strict product mode, but uses
   the historical manifest for candidate metadata and `db_root`.
2. Direct ASTAP readers are `CatalogDB`, `astap_db_reader.iter_tiles()`,
   `load_tile_stars()`, and strict branches in `metadata_solver.solve_near()`.
3. `near_solve()` asks for `index_root` because its current tile provider is
   `quad_index_builder.load_manifest(index_root)`.
4. It still needs tile identity, family, tile code, bounds, centers and `db_root`.
5. Historical tile NPZs are not required in strict ASTAP-ISO product mode except
   as fallback/compatibility.
6. Yes, selection can be replaced by `CatalogDB`/`TileMeta` if ordering, caps and
   family filters are characterized.
7. Smallest change: add an ASTAP-native `NearTileProvider` with legacy provider
   rollback.

### B. ZeBlind 4D

1. ASTAP -> `db_convert` historical `tiles/*.npz` -> `quad_index_4d` -> strict
   manifest -> runtime.
2. The 4D builder needs `x_deg`, `y_deg`, `ra_deg`, `dec_deg`, `mag`, tile keys
   and builder parameters.
3. Yes, all star data can be obtained from `CatalogDB`/`astap_db_reader`; `x/y`
   are reproducible projections.
4. Functionally equivalent direct indexes are realistic; byte-compatible indexes
   require pinning timestamp/compression/dtype/order metadata.
5. Record source hashes, layout fingerprint, projection/truncation policies,
   builder params, code versions and derived checksums.
6. Link each index through `source_ids`, `source_tiles`, per-source fingerprints,
   builder parameters and derived NPZ hashes.

### C. CatalogLibrary

1. Read-only manifest, validation, status, coverage and adapters are implemented.
2. Writing, repair, full source fingerprints, ASTAP-native runtime and normal GUI
   simplification are partial/missing.
3. It does not provide historical Near paths because its descriptor intentionally
   models ASTAP source, not old `index_root`.
4. Add only a temporary compatibility descriptor if necessary; preferred path is
   removing the dependency.
5. Manifest lacks full source fingerprint and deterministic rebuild metadata.
6. Add a non-destructive `REFERENCE_EXISTING` planner/writer API.

### D. Produit et compatibilite

The normal product path should retain only `catalog_library_path`. Raw
`db_root`, `index_root`, `blind_index_path`, `blind_4d_manifest_path` and
families should move to compatibility, advanced tools, tests or diagnostics.

## 14. Une seule prochaine etape

P1D-1 should implement and test an ASTAP-native ZeNear tile provider, with the
legacy historical manifest provider retained as explicit rollback.

## 15. Gate

P1D-0 provides enough evidence to start P1D-1 because:

- the current gap is precisely located;
- no algorithm change is required to start;
- the first migration slice is narrow and testable;
- rollback is obvious;
- the target preserves P0, P2 and P3A invariants.

Decision:

```text
READY_FOR_P1D1_ASTAP_RUNTIME_UNIFICATION
```
