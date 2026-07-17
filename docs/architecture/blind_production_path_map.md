# Blind Production Path Map

Phase: P2B-2A - Production Blind Adapter

This document maps the current real Blind path before extraction. No solver
algorithm is moved or changed by this mapping.

## Current Call Chain

The production Blind 4D route is currently entered through `ImageSolver`:

```text
ImageSolver.solve_path()
  -> ImageSolver._resolve_with_blind_after_failure()
     -> ImageSolver._run_blind_solver()
        -> ensure_loaded_4d_manifest()
        -> build_blind_solve_config()
        -> zesolver.zeblindsolver.blind_solve()
           -> zeblindsolver.zeblindsolver.solve_blind()
        -> ImageSolver._build_blind_result()

ImageSolver.solve_path_blind_only()
  -> ImageSolver._run_blind_solver()
     -> same Blind 4D route
```

The lower-level `zesolver.zeblindsolver.blind_solve()` wrapper writes WCS cards
to the FITS file it receives. A production `SolverPipeline` port therefore must
run it on a temporary copy when the pipeline owns final WCS output.

## Responsibility Table

| Etape | Entrees | Sorties | Effets de bord | Etat partage | Dependance ImageSolver | Cible |
| ----- | ------- | ------- | -------------- | ------------ | ---------------------- | ----- |
| `solve_path()` input dispatch | input path, `SolveConfig`, overwrite/blind flags | Near attempt, Blind fallback request, `ImageSolveResult` | reads FITS/raster, may copy/write WCS through subroutes | config, cancel event, IO semaphore | strong | COMPATIBILITY |
| `_load_image()` / `_load_fits()` | FITS/raster path | float32 image, `ImageMetadata` | reads source, raster sidecar opportunistic write for rasters only | IO semaphore | strong | INPUT_LOADING |
| Existing WCS policy | FITS header, overwrite flag | skip/error or continue | none beyond reads | config | strong | INPUT_LOADING |
| `_resolve_with_blind_after_failure()` | failed Near state, metadata hints, cached Blind result | optional `ImageSolveResult` | calls Blind route; no direct FITS write itself | config, run_info | medium | FALLBACK |
| `_should_try_blind()` | path, `blind_enabled` | boolean | none | config | low | CONFIG_ASSEMBLY |
| 4D profile gate | `blind_backend_profile` | continue or `BLIND4D_CONFIGURATION_REQUIRED` result | logs warning | config | low | MANIFEST_PREFLIGHT |
| `ensure_loaded_4d_manifest()` | `SolveConfig.blind_4d_manifest_path`, cached loaded manifest | `Loaded4DManifest` | reads manifest and NPZ headers; validates SHA and compatibility | optional cached loaded manifest | none | MANIFEST_PREFLIGHT |
| Manifest diagnostics | loaded manifest | run info/log entries | local diagnostic logs only | `_blind_index_checked` cache | strong | TELEMETRY |
| Hint resolution | explicit config hints, metadata hints, adaptive history | final RA/Dec hints | logs adaptive hint decisions | adaptive hint memory | strong for adaptive branch | HINT_RESOLUTION |
| 4D hint policy | final RA/Dec hints, profile | RA/Dec cleared for 4D | none | profile decision | low | HINT_RESOLUTION |
| `build_blind_solve_config()` | app `SolveConfig`, final hints, loaded manifest | low-level `BlindSolveConfig` | none | profile registry | none | CONFIG_ASSEMBLY |
| Blind quality profile | metadata, peaks, frame data | profile label, metrics | logs metrics | none | strong for degraded historical branch | CONFIG_ASSEMBLY |
| 4D quality contract | Blind profile | fixed `p220_contract` diagnostics | logs metrics | profile decision | low | CONFIG_ASSEMBLY |
| `blind_solve()` wrapper | FITS path, index root, `BlindSolveConfig` | `BlindSolveResult` dict | writes WCS to the passed FITS on success; reads FITS; may use prep cache | prep cache | none | ENGINE_CALL |
| `zeblindsolver.solve_blind()` | FITS path, index root, low-level config | `WcsSolution` | internal caches/globals, reads indexes, computes solution | module-level blind caches | none | SOLVER_ALGORITHM |
| Rescue loop | failed Blind result, quality profile | optional retried result | additional solver calls/logs | prep cache | strong | FALLBACK |
| Run info normalization | `BlindSolveResult.stats` | `run_info_blind_*` entries | logs diagnostics | run_info list | strong | TELEMETRY |
| Hint memory update | successful result keywords/WCS | remembered hints | updates neighbor/adaptive hint memory | ImageSolver hint caches | strong | TELEMETRY |
| ZN310B event | result stats/profile | JSON log event | local log | none | low | TELEMETRY |
| `_build_blind_result()` | `BlindSolveResult`, run_info | `ImageSolveResult` | none | none | low | RESULT_VALIDATION |
| Raster Blind bridge | raster data/path | sidecar JSON result | writes temp FITS and JSON sidecar | IO semaphore | strong | COMPATIBILITY |

## Extraction Decisions

- The first production port will target the 4D product route only. The legacy
  historical backend remains available through `ImageSolver` as rollback and
  is not promoted into `SolverPipeline`.
- The port receives already resolved hints. Adaptive neighbor hints remain in
  `ImageSolver` during P2B-2A because they depend on ImageSolver state and are
  not needed for the 4D route, which clears RA/Dec hints by contract.
- The port will use strict manifest loading through `load_4d_index_manifest()`
  or the existing `ensure_loaded_4d_manifest()` behavior. A corrupt manifest or
  SHA mismatch remains a hard error, not a silent fallback.
- Because the existing wrapper writes WCS to its input path, the port will call
  it on a temporary copy, read the resulting WCS and keyword updates, then
  return an `EngineSolveResult` with `wcs_written=False`. `wcs_io.py` remains
  responsible for final output writes.
- Quality/rescue paths that only apply to the historical backend are not
  reimplemented in the new port. The 4D path already disables rescue plans and
  fixes the quality profile to the P2.20 contract.

## Temporary Dependencies

- `build_blind_solve_config()` remains the authority for low-level config
  parity during P2B-2A.
- `ImageSolver` remains the owner of raster bridging, adaptive neighbor hints,
  legacy Blind rollback, GUI compatibility and historical run-info formatting.
- Batch routing remains untouched until P2B-2A reaches
  `READY_FOR_BATCH_EXTRACTION`.
