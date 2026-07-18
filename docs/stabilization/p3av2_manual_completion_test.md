# P3A-V2 Manual Completion Test

Status: PASS in a real Wayland GUI session.

Graphical environment:

```text
DISPLAY=:1
WAYLAND_DISPLAY=wayland-0
XDG_SESSION_TYPE=wayland
Qt platform=wayland
```

Test root:

```text
/tmp/zesolver_p3av2_gui_v2_4mLn9q
```

Structured run summary:

```text
/tmp/p3av2_final_manual.json
```

Runner stdout/stderr:

```text
/tmp/p3av2_final_manual_runner.out
```

## Results

| Test | Engine demande | Engine selectionne | run_id | Fin terminale | Copies log | Retour IDLE | Callback tardif | Resultat |
| ---- | -------------- | ------------------ | -----: | ------------: | ---------: | ----------: | --------------: | -------- |
| A FITS pipeline | pipeline | pipeline | 1 | 1 | 1 | 1 | 0 | PASS |
| B raster AUTO legacy | auto | legacy | 2 | 1 | 1 | 1 | 0 | PASS |
| C Stop/restart stop | auto | legacy | 3 | 1 | 1 | 1 | 0 | PASS |
| C Stop/restart relance | auto | legacy | 4 | 1 | 1 | 1 | 0 | PASS |

## Test A - FITS Pipeline

Command shape:

```bash
ZESOLVER_GUI_ENGINE=pipeline ZE_NEAR_PARALLEL_MODE=threads .venv/bin/python zesolver.py
```

Files:

```text
/tmp/zesolver_p3av2_gui_v2_4mLn9q/test_a_pipeline_fits/001_Light_mosaic_M 106_20.0s_IRCUT_20250518-233459.fit
/tmp/zesolver_p3av2_gui_v2_4mLn9q/test_a_pipeline_fits/002_Light_M 31_11_30.0s_IRCUT_20250922-230409.fit
```

Observed:

```text
GUI_RUN_BEGIN run_id=1
Engine selection: requested=pipeline selected=pipeline supported=True reason=pipeline_requested_supported
GUI_COMPLETION_TRACE run_id=1 handler=_on_worker_finished call_index=1
terminal_message_count=1
log_copy_attempt_count=1
run_idle_transition_count=1
stale_callback_count=0
```

Runtime log copy:

```text
/tmp/zesolver_p3av2_gui_v2_4mLn9q/test_a_pipeline_fits/zesolver_run_20260718_171316.log
```

## Test B - Raster AUTO

Command shape:

```bash
ZESOLVER_GUI_ENGINE=auto ZE_NEAR_PARALLEL_MODE=threads .venv/bin/python zesolver.py
```

File:

```text
/tmp/zesolver_p3av2_gui_v2_4mLn9q/test_b_raster_auto/021_Light_NGC 6888_30.0s_LP_20250619-020524.png
```

Observed:

```text
GUI_RUN_BEGIN run_id=2
Engine selection: requested=auto selected=legacy supported=True reason=auto_legacy: raster_not_supported_by_pipeline:.png; raster_sidecar_requires_legacy
GUI_COMPLETION_TRACE run_id=2 handler=_on_worker_finished call_index=1
terminal_message_count=1
log_copy_attempt_count=1
run_idle_transition_count=1
stale_callback_count=0
```

Runtime log copy:

```text
/tmp/zesolver_p3av2_gui_v2_4mLn9q/test_b_raster_auto/zesolver_run_20260718_171343.log
```

Raster sidecar:

```text
/tmp/zesolver_p3av2_gui_v2_4mLn9q/test_b_raster_auto/021_Light_NGC 6888_30.0s_LP_20250619-020524.png.wcs.json
```

## Test C - Stop Then Restart

Command shape:

```bash
ZESOLVER_GUI_ENGINE=auto ZE_NEAR_PARALLEL_MODE=process .venv/bin/python zesolver.py
```

Files:

```text
24 FITS copies from /home/tristan/near_bench100_input
```

Run 1 observed:

```text
GUI_RUN_BEGIN run_id=3
STOP_UI_CLICKED=1
STOP_CONTROLLER_RECEIVED=1
STOP_TOKEN_SET=1
EXECUTOR_SHUTDOWN_FINISHED=1
Engine selection: requested=auto selected=legacy supported=True reason=auto_legacy: batch_worker_strategy_requires_legacy:process
GUI_COMPLETION_TRACE run_id=3 handler=_on_worker_finished call_index=1
terminal_message_count=1
log_copy_attempt_count=1
run_idle_transition_count=1
stale_callback_count=0
```

Run 2 observed:

```text
GUI_RUN_BEGIN run_id=4
Engine selection: requested=auto selected=legacy supported=True reason=auto_legacy: batch_worker_strategy_requires_legacy:process
GUI_COMPLETION_TRACE run_id=4 handler=_on_worker_finished call_index=1
terminal_message_count=1
log_copy_attempt_count=1
run_idle_transition_count=1
stale_callback_count=0
```

Runtime log copies:

```text
/tmp/zesolver_p3av2_gui_v2_4mLn9q/test_c_stop_restart/zesolver_run_20260718_171347.log
/tmp/zesolver_p3av2_gui_v2_4mLn9q/test_c_stop_restart/zesolver_run_20260718_171423.log
```

Process checks after each run:

```text
ps_after_run=
zombies_after_run=
ps_after_all=
zombies_after_all=
```

No matching orphan ZeSolver/ProcessPoolExecutor process and no zombie process
were reported.

## Notes

The first Stop/restart attempt before the final patch confirmed correct
functional cancellation but exposed a diagnostic drift: the second run emitted
`call_index=2` because the displayed call index used the cumulative lifecycle
counter. The final patch keeps cumulative counters for aggregate tests and uses
per-run counters for GUI completion telemetry. The replay above confirms
`call_index=1` for each run.
