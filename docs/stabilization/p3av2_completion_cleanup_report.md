# P3A-V2 Completion Cleanup Report

## Objective

Guarantee exactly-once GUI run completion before declaring the P3A manual
validation ready for P3B simplification.

## Root Cause

The duplicate final log lines came from real duplicate terminal events, not only
from two independent log statements.

`SolveRunner` and `AstrometryRunner` both subclassed `QThread`, declared
`finished = QtCore.Signal()`, and manually emitted `self.finished.emit()` from
`run()`. Qt also emits `QThread.finished` automatically when `run()` returns.
The GUI connected `worker.finished` directly to `_on_worker_finished()`, whose
side effects include terminal logging, log copy, technical cleanup, and state
reset.

## Chain Before Correction

```text
manual runner finished emit
-> _on_worker_finished()
-> native QThread.finished
-> _on_worker_finished()
```

## Chain After Correction

```text
QThread.run returns
-> native QThread.finished
-> _on_worker_finished()
-> RunLifecycle.finish_once(run_id)
-> terminal message
-> RunLifecycle.mark_log_copy_once(run_id)
-> log copy attempt
-> technical cleanup
-> RunLifecycle.transition_idle_once(run_id)
```

## Changes

- Removed manual `finished.emit()` from `SolveRunner`.
- Removed manual `finished.emit()` from `AstrometryRunner`.
- Added `RunLifecycle` run IDs and exactly-once guards.
- Added per-run lifecycle counters for completion telemetry.
- Kept cumulative counters for aggregate lifecycle tests.
- Added GUI-side `run_id` assignment to each worker.
- Added exactly-once protection to `_on_worker_finished()`.
- Added exactly-once protection to `_copy_runtime_log_to_output()`.
- Distinguished terminal logs for success, cancellation, and error.
- Made `closeEvent()` reuse the same guarded completion path when a worker has
  completed during bounded shutdown.

## Manual Validation

Graphical environment:

```text
DISPLAY=:1
WAYLAND_DISPLAY=wayland-0
XDG_SESSION_TYPE=wayland
Qt platform=wayland
```

Manual test root:

```text
/tmp/zesolver_p3av2_gui_v2_4mLn9q
```

Structured summary:

```text
/tmp/p3av2_final_manual.json
```

| Test | Engine demande | Engine selectionne | run_id | Fin terminale | Copies log | Retour IDLE | Callback tardif | Resultat |
| ---- | -------------- | ------------------ | -----: | ------------: | ---------: | ----------: | --------------: | -------- |
| A FITS pipeline | pipeline | pipeline | 1 | 1 | 1 | 1 | 0 | PASS |
| B raster AUTO legacy | auto | legacy | 2 | 1 | 1 | 1 | 0 | PASS |
| C Stop/restart stop | auto | legacy | 3 | 1 | 1 | 1 | 0 | PASS |
| C Stop/restart relance | auto | legacy | 4 | 1 | 1 | 1 | 0 | PASS |

Observed terminal traces:

```text
GUI_COMPLETION_TRACE run_id=1 handler=_on_worker_finished call_index=1
GUI_COMPLETION_TRACE run_id=2 handler=_on_worker_finished call_index=1
GUI_COMPLETION_TRACE run_id=3 handler=_on_worker_finished call_index=1
GUI_COMPLETION_TRACE run_id=4 handler=_on_worker_finished call_index=1
```

Log copies:

```text
/tmp/zesolver_p3av2_gui_v2_4mLn9q/test_a_pipeline_fits/zesolver_run_20260718_171316.log
/tmp/zesolver_p3av2_gui_v2_4mLn9q/test_b_raster_auto/zesolver_run_20260718_171343.log
/tmp/zesolver_p3av2_gui_v2_4mLn9q/test_c_stop_restart/zesolver_run_20260718_171347.log
/tmp/zesolver_p3av2_gui_v2_4mLn9q/test_c_stop_restart/zesolver_run_20260718_171423.log
```

Process checks after Stop/restart:

```text
ps_after_run=
zombies_after_run=
ps_after_all=
zombies_after_all=
```

No matching orphan ZeSolver/ProcessPoolExecutor process and no zombie process
were reported.

## Automated Validation

Pre-final-patch P3A-V2 automated baseline:

```text
core boundary check: OK
P3A-V2 targeted: 10 passed
Hermetic: 416 passed, 1 skipped, 9 deselected
Full pytest: 416 passed, 10 skipped
compileall: OK
git diff --check: OK
```

Post-final-patch targeted lifecycle check:

```text
5 passed
```

Post-final-patch release-gate checks:

```text
core boundary check: OK
targeted Qt/lifecycle: 10 passed
hermetic: PASS (416 passed, 1 skipped, 9 deselected, 44 warnings)
full pytest: 416 passed, 10 skipped, 44 warnings
compileall: OK
git diff --check: OK
```

## Warning Categories

Known categories only:

```text
multiprocessing.popen_fork.DeprecationWarning
astropy.io.fits.card.VerifyWarning
```

No new warning category was introduced during P3A-V2.

## Decision

```text
READY_FOR_P3B_GUI_SIMPLIFICATION
```
