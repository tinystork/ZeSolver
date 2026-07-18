# GUI Pipeline Threading

P3A keeps the existing Qt threading shape and changes only the worker body for
local solves.

## Model

```text
Qt main thread
  - collects widget values
  - builds SolveConfig and GuiSolveRequest snapshot
  - creates SolveRunner
  - receives signals
  - updates widgets

SolveRunner QThread
  - owns cancellation event
  - runs GuiSolveController
  - creates PipelineGuiRunner or LegacyGuiRunner
  - emits progress/info/error/finished signals

Worker pools
  - owned by BatchSolverPipeline or legacy BatchSolver
```

## Rules

- Solver work never runs in the Qt main thread.
- Widgets are updated only by existing Qt signal handlers.
- `GuiSolveController` and runners contain no widget references.
- `finished` is emitted once by `SolveRunner.finally`.
- Stop sets a `threading.Event`; pipeline and legacy routes receive that event.
- Window close keeps the existing bounded shutdown path.
- Exceptions in the worker are converted to `error` signals.

## P3A Limits

The thread model is intentionally conservative:

- the Astrometry.net route remains the legacy `AstrometryRunner`;
- raster sidecars remain legacy;
- process/hybrid Near scheduling remains legacy;
- Blind cancellation is still best effort, matching the P2B batch limitation.

P3B can simplify visible UX only after manual GUI validation confirms this
functional bridge behaves correctly.
