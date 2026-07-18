# GUI Completion Signal Map

Mission: P3A-V2 exactly-once GUI completion cleanup.

## Root Cause

The duplicate `Traitement terminé.` / `Log copied to output folder:` lines came
from two terminal events reaching the same GUI handler.

`SolveRunner` and `AstrometryRunner` subclassed `QThread`, declared a signal
named `finished`, and manually called `self.finished.emit()` in `run()`.
`QThread` already owns a native `finished` signal emitted automatically when the
thread returns. The GUI connected:

```python
self._worker.finished.connect(self._on_worker_finished)
```

so the manual signal and the native thread completion could both invoke the same
functional completion handler.

## Chain Before P3A-V2

```text
SolveRunner.run() finally
-> self.finished.emit()
-> ZeSolverWindow._on_worker_finished()
-> "Traitement terminé."
-> _copy_runtime_log_to_output()

QThread run returns
-> QThread.finished
-> ZeSolverWindow._on_worker_finished()
-> "Traitement terminé."
-> _copy_runtime_log_to_output()
```

The same pattern existed on the Astrometry.net legacy web runner.

## Chain After P3A-V2

```text
worker/runner run returns
-> native QThread.finished
-> ZeSolverWindow._on_worker_finished()
-> RunLifecycle.finish_once(run_id)
-> one terminal log line
-> one runtime log-copy attempt
-> worker technical cleanup / deleteLater()
-> RunLifecycle.transition_idle_once(run_id)
```

The runner no longer emits `finished` manually. `QThread.finished` is now a
technical thread-completion signal consumed once by the GUI. The functional
terminal event is guarded by `RunLifecycle`.

## Signal Table

| Source | Signal/callback | Connecte dans | Destinataire | Nombre attendu | Risque de double appel |
| ------ | --------------- | ------------- | ------------ | -------------: | ---------------------- |
| `SolveRunner` local GUI thread | native `QThread.finished` | `ZeSolverWindow._start_solving()` | `_on_worker_finished()` | 1 | Low after manual emit removal; guarded by `run_id` |
| `AstrometryRunner` web thread | native `QThread.finished` | `ZeSolverWindow._start_solving()` | `_on_worker_finished()` | 1 | Low after manual emit removal; guarded by `run_id` |
| `SolveRunner.progress` | Qt signal | `_start_solving()` | `_on_worker_progress()` | per file | Does not trigger terminal cleanup |
| `AstrometryRunner.progress` | Qt signal | `_start_solving()` | `_on_worker_progress()` | per file | Does not trigger terminal cleanup |
| `SolveRunner.error` / `AstrometryRunner.error` | Qt signal | `_start_solving()` | `_on_worker_error()` | per error | Marks failed state; does not copy log |
| `GuiSolveController.run()` return | direct call inside `SolveRunner.run()` | `SolveRunner.run()` | local stack only | 1 | No GUI handler connection |
| `LegacyGuiRunner.run()` return | direct call inside controller | controller | local stack only | 1 | No GUI handler connection |
| `PipelineGuiRunner.run()` return | direct call inside controller | controller | local stack only | 1 | No GUI handler connection |
| `closeEvent()` bounded wait | direct call | `ZeSolverWindow.closeEvent()` | `_shutdown_thread()` / optional finalization | 0-1 | Uses same `run_id` guard |

## Exactly-once Owner

The unique owner of functional completion is:

```text
ZeSolverWindow._on_worker_finished()
  -> RunLifecycle.finish_once(run_id)
```

`thread.finished` is technical. The controller and runners do not perform GUI
final logging or log-copy side effects.

## Run ID Rules

- every GUI run receives a new integer `run_id`;
- the active worker stores `_gui_run_id`;
- a completion callback whose `run_id` is not active is ignored;
- a second completion for the same `run_id` is ignored;
- log copying uses `RunLifecycle.mark_log_copy_once(run_id)`;
- IDLE transition uses `RunLifecycle.transition_idle_once(run_id)`.
