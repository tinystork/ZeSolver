# GUI Pipeline Controller

P3A adds `zesolver/gui_pipeline/` as an application boundary between the
existing Qt window and the solver engines.

```text
ZeSolverWindow
  -> SolveRunner(QThread)
    -> GuiSolveController
      -> EngineSelector
        -> PipelineGuiRunner -> BatchSolverPipeline -> SolverPipeline
        -> LegacyGuiRunner   -> BatchSolver/ImageSolver
```

## Request Boundary

`GuiSolveRequest` is immutable and contains paths, requested engine mode,
backend, overwrite policy, worker count, pipeline product/runtime settings,
legacy compatibility config, and resolved catalog resources. It contains no Qt
widgets, models, signals, open FITS handles, `ImageSolver`, or `BatchSolver`.

The GUI builds this request from a frozen snapshot of the legacy `SolveConfig`.
Widget changes during a run do not affect the active request.

## Engine Selection

The controller delegates policy to `zesolver.engine_selection`.

- `AUTO + local FITS` selects `PIPELINE`.
- `AUTO + raster/web/process/hybrid/unknown` selects `LEGACY` with a reason.
- explicit `PIPELINE` rejects unsupported requests with no silent fallback.
- explicit `LEGACY` preserves rollback.

The temporary activation source is:

```text
ZESOLVER_GUI_ENGINE
  -> auto | pipeline | legacy
  -> AUTO when unset
```

The selected route, reason, support flag, and warnings are logged at run start.

## Runners

`PipelineGuiRunner` uses the validated core batch and per-file pipeline:

```text
BatchSolverPipeline(phase=near)
  -> SolverPipeline(blind_only=false)
BatchSolverPipeline(phase=blind)
  -> SolverPipeline(blind_only=true)
```

`LegacyGuiRunner` receives an injected callable from `zesolver.py`, so the new
GUI adapter package does not import the monolithic root file. The callable wraps
the existing `BatchSolver` behavior.

## Result Boundary

Both routes emit `GuiFileResult`. The legacy GUI table still consumes the
existing simple status labels through `legacy_status`, while the richer status
and details remain available for future P3B simplification.

## Rollback

Rollback remains explicit:

```bash
ZESOLVER_GUI_ENGINE=legacy .venv/bin/python zesolver.py
```

No P3A code removes `ImageSolver`, `BatchSolver`, `SolveRunner`, the web
fallback, raster handling, or legacy process/hybrid scheduling.
