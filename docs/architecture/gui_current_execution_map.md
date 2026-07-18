# GUI Current Execution Map

P3A starts from the legacy Qt flow in `zesolver.py`. This map records the
observable path before the progressive pipeline integration.

| Step | Class/function | Thread | Input | Output | Side effects | Future target |
| ---- | -------------- | ------ | ----- | ------ | ------------ | ------------- |
| User starts scan | `ZeSolverWindow.scan_files()` | Qt main | input folder, format text, max files | starts `FileScanner` | clears tree, pending files, scan buffer | GUI_ONLY |
| File discovery | `FileScanner.run()` | QThread | root, extensions, limit | `file_found`, `finished` signals | reads FITS headers for WCS status | LEGACY_COMPATIBILITY |
| Start button | `ZeSolverWindow._start_solving()` | Qt main | `_pending_files`, widgets, persistent settings | runner instance | message boxes, progress reset, running state | RUNNER_CREATION |
| Simple mode wizard | `_run_simple_mode_assistant()` | Qt main | widgets and persistent settings | bool | message boxes, optional tab activation | GUI_ONLY |
| Optional WCS cleaning | `_run_simple_mode_wcs_cleaning()` | Qt main | FITS paths | bool | may remove WCS from source FITS when explicitly requested | LEGACY_COMPATIBILITY |
| Legacy config build | `_build_config()` | Qt main | widget snapshot plus `PersistentSettings` | `SolveConfig` | persists settings to disk | SETTINGS_MAPPING |
| Catalog resolution | `apply_catalog_resources_to_config()` | Qt main | `SolveConfig` | updated config + `SolverCatalogResources` | diagnostic log line | SETTINGS_MAPPING |
| Blind 4D preflight | `_preflight_4d_manifest_for_run()` | Qt main | config/profile | loaded manifest | warning dialogs on failure | SETTINGS_MAPPING |
| Backend switch | `_start_solving()` | Qt main | backend combo | `AstrometryRunner` or `SolveRunner` | log route choice | ENGINE_SELECTION |
| Local legacy runner | `SolveRunner.run()` | QThread | `SolveConfig`, files | `ImageSolveResult` signals | instantiates `BatchSolver`, owns cancel event | WORKER_EXECUTION |
| Legacy batch | `BatchSolver.run()` | worker thread plus pools | `SolveConfig`, files | result iterator | thread/process pools, memory logs, optional web fallback | LEGACY_COMPATIBILITY |
| Legacy per-image solve | `ImageSolver.solve_path()` | worker pool | FITS/raster path | `ImageSolveResult` | reads/writes FITS/WCS, Near then optional Blind | LEGACY_COMPATIBILITY |
| Legacy blind fallback | `ImageSolver.solve_path_blind_only()` | worker pool | unresolved path | `ImageSolveResult` | Blind 4D/historical solve, possible WCS writes | LEGACY_COMPATIBILITY |
| Web route | `AstrometryRunner.run()` | QThread | files, API config | `ImageSolveResult` signals | network calls, optional local fallback | LEGACY_COMPATIBILITY |
| Progress mapping | `_on_worker_progress()` | Qt main | `ImageSolveResult` | UI updates | tree item update, progress bar, log | PROGRESS_MAPPING |
| Error mapping | `_on_worker_error()` | Qt main | error text | dialog/log | modal error dialog | RESULT_MAPPING |
| Finish mapping | `_on_worker_finished()` | Qt main | signal | UI idle state | stops timer, copies runtime log, deletes runner | LIFECYCLE |
| Stop button | `_stop_solving()` | Qt main | active runner | cancel request | sets runner cancel event, log | CANCELLATION |
| Window close | `closeEvent()` | Qt main | active threads | shutdown attempts | requests cancellation and waits bounded time | LIFECYCLE |

## Classification

- `GUI_ONLY`: widget state, dialogs, file tree, log pane, progress bar.
- `SETTINGS_MAPPING`: conversion from widgets/settings to solver configuration.
- `ENGINE_SELECTION`: currently an implicit local-vs-Astrometry branch.
- `RUNNER_CREATION`: creates a concrete QThread runner.
- `WORKER_EXECUTION`: runs off the Qt main thread.
- `PROGRESS_MAPPING`: maps backend-specific progress/results into widgets.
- `RESULT_MAPPING`: normalizes result text and status colors.
- `CANCELLATION`: cooperative cancellation via thread-local `threading.Event`.
- `LIFECYCLE`: bounded shutdown and single finalization.
- `LEGACY_COMPATIBILITY`: `BatchSolver`, `ImageSolver`, raster, web, process/hybrid
  scheduling, adaptive hints, and warm-start behavior.

## P3A Decision

P3A inserts `GuiSolveController` between `_start_solving()` and concrete runners.
The controller uses the P2B-2C `EngineSelector` and routes validated local FITS
requests to `PipelineGuiRunner`. Raster, web, process/hybrid, adaptive hints,
and unknown capabilities remain on the legacy runners with an explicit reason.

The Qt threading model remains QThread-based in P3A. The new controller and
request/result/progress models are independent of widgets; `zesolver.py` keeps
the minimal signal wiring needed to update the existing GUI.
