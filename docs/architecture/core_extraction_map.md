# Core Extraction Map

P2B-1 map before code extraction. No solver algorithm is moved by this
document.

## Sources Inspected

- `zesolver.py`
- `ImageSolver`
- `BatchSolver`
- CLI entry points in `zesolver.py`
- GUI launch path in `launch_gui()`
- P1C catalogue resource resolver
- P2A settings/profile assembly layer
- current FITS/WCS write helpers

## Size Signals

| Item | Location | Lines | Classification | Extraction Signal |
| ---- | -------- | ----: | -------------- | ----------------- |
| `ImageSolver` | `zesolver.py:1926` | 2118 | ORCHESTRATION / IO / COMPATIBILITY | High |
| `ImageSolver._run_blind_solver` | `zesolver.py:3024` | 566 | ORCHESTRATION around SOLVER_ALGORITHM | High |
| `ImageSolver._run_index_near_solver` | `zesolver.py:3718` | 293 | ORCHESTRATION around SOLVER_ALGORITHM | High |
| `BatchSolver.run` | `zesolver.py:4058` | 608 | ORCHESTRATION / CONCURRENCY | Later P2B-2 |
| `run_cli` | `zesolver.py:4951` | 152 | CLI / COMPATIBILITY | Medium |
| `launch_gui()` | `zesolver.py:5105` | 5155 | GUI | Later P3 |
| `solve_near()` wrapper | `zesolver/zeblindsolver.py:327` | 86 | SOLVER_ALGORITHM adapter | Do not rewrite |

## Responsibility Table

| Responsabilité | Emplacement actuel | Entrées | Sorties | Effets de bord | Cible proposée | Risque |
| -------------- | ------------------ | ------- | ------- | -------------- | -------------- | ------ |
| Parse CLI options | `build_arg_parser`, `run_cli` | `argv`, env | `SolveConfig`, exit code | Logging, filesystem writes | Keep compatibility; later adapter to `ProductSettings` | COMPATIBILITY |
| Construct legacy config | `SolveConfig`, GUI builders, CLI | persisted settings, CLI args | `SolveConfig` | none | P2A assembly bridge | COMPATIBILITY |
| Resolve catalog resources | `resolve_catalog_resources_for_config`, `apply_catalog_resources_to_config` | `SolveConfig`, env, `CatalogLibrary` | legacy resource fields | may load manifests | `SolverPipeline` pre-solve resource step | ORCHESTRATION |
| Load 4D manifest | `ensure_loaded_4d_manifest`, `build_blind_solve_config` | manifest path | `Loaded4DManifest`, blind config | file reads | resource/preflight boundary; runtime remains authority | ORCHESTRATION |
| Batch file collection | `BatchSolver._collect_files`, `_iter_image_files` | input dir, formats | paths | filesystem reads | P2B-2 batch extraction | IO |
| FITS/raster metadata load | `ImageSolver._load_image`, `_load_fits`, `_load_raster`, metadata helpers | input path | `ImageMetadata`, image array | file reads | `preflight.py` plus compatibility loaders | IO |
| Existing WCS detection | `_header_has_wcs`, `_quick_scan_initial_status`, `_load_fits` | FITS headers | status/metadata | file reads | `preflight.py` | IO |
| Overwrite/copy policy | `SolveConfig.overwrite`, `ImageSolver.solve_path`, `_write_solution` | output policy, paths | target path | file writes | `wcs_io.py` | IO |
| Near solve orchestration | `ImageSolver._run_index_near_solver` | path, metadata, config | `ImageSolveResult` or `None` | WCS write through Near wrapper | `NearSolverPort` adapter, `pipeline.py` routing | ORCHESTRATION |
| Near algorithm | `zeblindsolver.metadata_solver.solve_near`, `near_solve()` wrapper | FITS, index root, `NearSolveConfig` | solve dict/result, WCS cards | writes WCS to supplied FITS | production port calls existing function | SOLVER_ALGORITHM |
| Blind solve orchestration | `ImageSolver._run_blind_solver`, `_resolve_with_blind_after_failure` | path, metadata, config | `ImageSolveResult` or `None` | WCS writes, logs | `BlindSolverPort` adapter, `pipeline.py` routing | ORCHESTRATION |
| Blind algorithm | `zeblindsolver` backend and 4D runtime | FITS/image, indexes, profile | solve result | may read indexes/write WCS | production port calls existing path | SOLVER_ALGORITHM |
| Near then Blind policy | `solve_path`, `_run_index_near_solver`, `_resolve_with_blind_after_failure` | metadata, config, flags | final result | logs, WCS writes | `pipeline-v1` in `pipeline.py` | ORCHESTRATION |
| Astrometry.net fallback | `ImageSolver` web fallback hooks | API settings, hints | result | network | keep legacy; future port | IO |
| Result normalization | `ImageSolveResult`, `_result_to_payload`, `_payload_to_result`, `_build_blind_result` | engine dicts/WCS | app result payloads | none | `result_adapter.py` | PURE_LOGIC |
| WCS pixel scale/orientation extraction | `_pixel_scale_arcsec`, `_build_solution`, corpus helpers | WCS/header | typed fields | none | `result_adapter.py` | PURE_LOGIC |
| FITS WCS writing | `_write_solution`, `_write_fits_solution`, `near_solve` wrapper side effect | result/WCS/header | updated FITS | file writes | `wcs_io.py`; production port may still delegate existing writer initially | IO |
| Pixel integrity checks | regression tests, WCS helpers | FITS before/after | hash equality | file reads | `wcs_io.py` utilities | IO |
| Telemetry/log report | scattered logging, `ZN310B_EVENT`, P1C telemetry | profiles, catalog, results | logs/report dicts | logging | `telemetry.py` | ORCHESTRATION |
| Cancellation | `ImageSolver._cancelled`, `_cancel_event`, callbacks | cancel event | stopped run | cooperative stop | `RuntimeOptions.cancel_token`, `preflight.py`, routing checks | RUNTIME |
| Progress callbacks | `BatchSolver.run`, GUI callbacks | result/status | UI updates | callback calls | `RuntimeOptions.progress_callback` | GUI / RUNTIME |
| GUI controls | `launch_gui()` | user interaction | persistent settings/config | GUI state/files | no P2B-1 extraction except compatibility | GUI |
| Batch concurrency | `BatchSolver.run`, worker helpers | file list, workers | batch summary | process/thread pools | P2B-2 | ORCHESTRATION |
| Developer diagnostics | `tools/diagnose_*`, dev overrides | CLI/env | reports | file writes | stay outside product pipeline | DEVELOPER_TOOL |

## P2B-1 Boundary

Extract now:

- public request/result contracts;
- `SolverPipeline` façade;
- narrow Near/Blind ports for dependency injection;
- preflight checks that do not modify FITS;
- Near then Blind routing policy from `pipeline-v1`;
- result normalization helpers;
- WCS output helper utilities;
- telemetry snapshot.

Keep in place:

- `solve_near()` internals;
- `solve_blind()` / Blind 4D internals;
- `ImageSolver` and GUI behavior unless accessed through a compatibility test;
- batch concurrency.

## Risk Controls

- New pipeline tests use deterministic fake ports first.
- Production adapters remain thin wrappers over existing behavior.
- Legacy `ImageSolver` path is kept available while parity tests compare status
  and routing behavior.
- No profile `v1` value is changed.
- No catalogue/index/FITS format is changed.
