# Catalog Integration Map

P1C-1 maps the current catalogue resource construction before integration. No
runtime behavior is changed by this document.

## Current Construction Summary

The current product path still builds catalogue resources directly from legacy
fields on `SolveConfig`:

- `SolveConfig.db_root` and `SolveConfig.families` are produced by CLI arguments
  or GUI persistent settings and are consumed by `ImageSolver`, which constructs
  `CatalogDB(config.db_root, families=config.families, ...)`.
- `SolveConfig.blind_index_path` is the historical index root. It remains used
  by diagnostics, historical fallback compatibility, Astrometry.net local
  fallback metadata, and index preflight paths.
- `SolveConfig.blind_backend_profile` selects either `historical` or
  `zeblind_4d_experimental`. The product default is the 4D experimental profile.
- `SolveConfig.blind_4d_manifest_path` points at the strict 4D manifest.
  `load_4d_index_manifest()` remains the runtime authority for NPZ compatibility.
- `SolveConfig.blind_4d_loaded_manifest` caches a loaded strict manifest after
  CLI/GUI preflight.
- 4D runtime index paths are not stored directly on `SolveConfig`; they are
  produced by `Loaded4DManifest.enabled_index_paths` and then copied into
  `BlindSolveConfig.blind_astrometry_4d_index_paths` by
  `SolverProfile.apply_to_config()`.

`CatalogLibrary` currently exists only as a read-only core and adapter source.
The solver pipeline, CLI, GUI and persistent settings do not depend on it yet.

## Parameter Flow

| Paramètre actuel | Produit par | Consommé par | Remplaçable par CatalogLibrary | Compatibilité nécessaire |
| ---------------- | ----------- | ------------ | -----------------------------: | ------------------------ |
| `db_root` | CLI `--db-root`; GUI settings tab `settings_db_edit`; `PersistentSettings.db_root`; tests constructing `SolveConfig`; historical index manifests for Near wrapper paths | `SolveConfig`; `ImageSolver.__init__`; `CatalogDB`; GUI startup wizard and family scanner; index builder; tools and benchmarks | Yes, via `NearCatalogDescriptor.root` | Keep explicit `db_root` working when no library is provided. If an explicit library is invalid, do not silently fall back to `db_root` unless a documented compatibility mode asks for legacy. |
| `families` | CLI `--family`; GUI solver family combo; GUI developer family selection/cache; `PersistentSettings.solver_family`, `dev_family_auto`, `dev_family_selection`, `db_family_cache`; tests | `SolveConfig.__post_init__`; `CatalogDB`; `ImageSolver._init_family_candidates()` and Near attempts; `solve_near()` config conversion | Yes, via `NearCatalogDescriptor.families` | Preserve explicit family filtering for legacy mode. Library families must only include validated ASTAP sources; missing library source means Near unavailable, not silently all-family. |
| `blind_index_path` | CLI `--blind-index`; GUI `settings_index_edit`; `PersistentSettings.index_root`; tests and diagnostics | `SolveConfig`; historical blind diagnostics; GUI index health/build; Astrometry.net fallback metadata; `ImageSolver` preflight for historical path; `solve_near()` wrapper/tests where legacy index root is still passed | Partly. Blind 4D should use library descriptors; historical index root remains legacy/diagnostic | Do not remove the field in P1C. Preserve historical/diagnostic paths and batch tests. When 4D library resources are used, `blind_index_path` may remain present but is not the source of 4D runtime NPZ paths. |
| `index_root` | Name used by historical ZeBlind modules, GUI index builder, tools, tests; usually same value as `blind_index_path` or `PersistentSettings.index_root` | `zeblindsolver.db_convert`; `quad_index_builder`; historical `solve_blind()` calls; `solve_near(fits_path, index_root, ...)`; GUI build/test workers | Partly. Library can own a legacy descriptor later, but P1C only needs Near and Blind 4D resources | Keep builder and diagnostic use unchanged. Do not move or rebuild indexes. Do not infer source catalogue from a legacy index when an explicit library was selected and failed. |
| `blind_backend_profile` | Default `ZEBLIND_4D_EXPERIMENTAL_PROFILE`; CLI `--blind-profile`; GUI profile combo; `PersistentSettings.blind_backend_profile`; settings migration normalizes historical to 4D default | `SolveConfig.__post_init__`; `ensure_loaded_4d_manifest()`; `build_blind_solve_config()`; GUI preflight; `BatchSolver` worker strategy; logs/reports | No. This is an internal/profile selection, not a catalogue resource | Preserve current values and default. CatalogLibrary integration must feed resources into the selected profile without changing thresholds or profile parameters. |
| `blind_4d_manifest_path` | CLI `--blind-4d-manifest`, defaulting through `resolve_default_4d_manifest_path()`; GUI manifest edit/default; `PersistentSettings.blind_4d_manifest_path`; tests | `load_4d_index_manifest()` in CLI, GUI preflight, `ensure_loaded_4d_manifest()`, `build_blind_solve_config()` | Yes, via a library-provided common runtime manifest or validated runtime path set | Keep direct manifest path working when no library is provided. If a library is provided, library Blind 4D resources take priority over the legacy manifest path. Runtime strict loader remains final authority. |
| `blind_4d_loaded_manifest` | CLI preflight; GUI `_preflight_4d_manifest_for_run()` followed by `dataclasses.replace()`; tests | `ensure_loaded_4d_manifest()`; `build_blind_solve_config()`; `ImageSolver._run_blind_solver()` logging and `index_root` derivation | Yes, as a loaded strict manifest built from library-selected resources when available | Preserve cache behavior. Do not bypass strict manifest validation. If resource resolution yields invalid paths/checksums, fail clearly before solving. |
| `blind_4d_index_paths` / `BlindSolveConfig.blind_astrometry_4d_index_paths` | `Loaded4DManifest.enabled_index_paths`; `SolverProfile.apply_to_config()` | `zeblindsolver.solve_blind()` 4D route; logs in `ImageSolver._run_blind_solver()` | Yes, via `Blind4DIndexDescriptor.path` converted through the existing strict manifest/profile path | Only validated indexes may be passed. Keep manifest order. Six D50 indexes remain partial coverage and must not become all-sky. |
| `catalog_library_path` | Not present yet in `SolveConfig`, CLI, GUI, or `PersistentSettings` | Not consumed yet | New optional path field for P1C/P2A | Add without deleting legacy fields. Persist only paths later; runtime may load a `CatalogLibrary` object internally. |

## Current Entry Points

### CLI

`run_cli()` currently requires `--db-root` and `--input-dir` in headless mode.
When `--blind-profile zeblind_4d_experimental` is active, it defaults a missing
`--blind-4d-manifest` through `resolve_default_4d_manifest_path()` and loads it
with `load_4d_index_manifest()` before constructing `SolveConfig`.

### GUI

The settings panel persists `db_root`, `index_root`, `blind_backend_profile` and
`blind_4d_manifest_path`. `_build_config()` copies those settings into
`SolveConfig`. `_start_solving()` preflights the 4D manifest and replaces
`blind_4d_manifest_path` / `blind_4d_loaded_manifest` before launching the
worker.

The GUI still exposes raw database, index, family and 4D manifest controls. P1C
must not redesign or remove them.

### Batch

`BatchSolver` owns phase orchestration. It builds one `ImageSolver(config)`, runs
Near first, then phase-2 Blind for unresolved files. It passes the same
`SolveConfig` into process-pool Near workers.

### Near Pipeline

`ImageSolver.__init__()` constructs `CatalogDB` directly unless `blind_only` is
set. Near attempts use the family order exposed by that database. This makes the
caller the clean integration point: convert a validated `NearCatalogDescriptor`
to `db_root` and `families` before `ImageSolver` is constructed.

### Blind 4D Runtime

`ImageSolver._run_blind_solver()` calls `ensure_loaded_4d_manifest()` for the 4D
profile. `build_blind_solve_config()` applies the `zeblind_4d_experimental`
profile using `Loaded4DManifest.enabled_index_paths`. The existing strict
manifest loader remains the compatibility authority for NPZ files.

### Presets and Diagnostics

`zeblindsolver.profiles` currently contains the 4D internal parameters. GUI FOV
presets and tools still build historical indexes and benchmark configurations
from direct `db_root`/`index_root` values. These are not product-library
resolution points in P1C.

## P1C Integration Decision

The P1C integration point should be a small resource-resolution layer before
`ImageSolver`, `BatchSolver`, CLI and GUI preflight consume catalogue paths.

Required behavior:

1. explicit valid `CatalogLibrary` wins for Near and Blind 4D resources;
2. absent library preserves current legacy behavior;
3. explicitly invalid library fails clearly;
4. environment discovery is opt-in and controlled by named variables only;
5. `READY_PARTIAL` remains visible and never implies all-sky Blind 4D;
6. rollback is simply disabling the new library path and using legacy fields.
