# GUI Live State Flow

## WCS State

The main solver list now uses the WCS state effectively consumed by the solver.

```text
FITS PRIMARY has WCS -> status=wcs
FITS PRIMARY has no WCS -> status=waiting
FITS extension-only WCS -> status=waiting, detail notes extension WCS
Raster valid sidecar .wcs.json -> status=wcs
```

This keeps the GUI aligned with `ImageSolver._load_fits`, which uses the primary
HDU for the main FITS solve path. Extension WCS is diagnostic detail, not a
green solved state for the main list.

## WCS Cleanup Refresh

Before:

```text
_run_simple_mode_wcs_cleaning
-> zewcscleaner.process_fits(all_hdus=False)
-> log summary only
-> QTreeWidgetItem remains green/stale
```

After:

```text
_run_simple_mode_wcs_cleaning
-> zewcscleaner.process_fits(all_hdus=False)
-> inspect_effective_wcs_state(path)
-> update item status text, UserRole, detail, and foreground
```

Large batches update items in place and periodically yield to Qt in batches.

## Legacy Live Results

Before:

```text
BatchSolver._queue_result
-> on_result callback
-> GC only
-> result stored in yield_queue
-> LegacyGuiRunner sees it only when the phase yields
-> GUI progress can stay at 0 for a long phase
```

After:

```text
BatchSolver._queue_result
-> on_result callback
-> LegacyGuiRunner live_result_callback
-> QThread progress signal
-> ZeSolverWindow._on_worker_progress
-> item update, completed counter, remaining counter, progress bar
```

The later iterator yield is still used for the run summary, but the same path is
not notified twice.

## Pipeline Results

`PipelineGuiRunner` already receives per-result progress from
`BatchSolverPipeline`. It now also guards final-result replay so a path emitted
through the progress sink is not emitted again during final summary handling.

## Progress State

Per run, the GUI keeps:

```text
run_id
total
completed
seen_paths
remaining = max(0, total - completed)
```

Each terminal result path can increment `completed` only once for the active
`run_id`. The status label is translated:

```text
{done} / {total} - {remaining} remaining
```

The progress bar keeps the existing 100-units-per-file granularity. Terminal
results snap the bar to the completed-file boundary. The timer can only advance
within the next file bucket and never rewinds completed progress.

## Stop And Restart

Stop does not force the bar to 100 percent. Cancelled runs keep an honest status:

```text
Stopped: {done} processed, {remaining} remaining
```

Starting a new run resets progress counters and the seen-path set. Stale progress
signals whose `run_id` does not match the active lifecycle are ignored.
