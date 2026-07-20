# P3A-V1 Stop Hardening Report

## Objective

Make GUI Stop responsive again on the AUTO -> LEGACY process Near route without
changing Near/Blind algorithms, thresholds, profiles, catalogs, indexes, worker
defaults, or the legacy scheduler.

## Confirmed Cause

Two cancellation gaps were present:

1. `SolveRunner.request_cancel()` set only the QThread-local event. The selected
   `LegacyGuiRunner` / `PipelineGuiRunner` owned its own event and did not
   reliably receive Stop through the controller.
2. Near process workers initialized `ImageSolver` with `set_cancel_event(None)`,
   so active workers could not observe the GUI stop token.

The legacy process scheduler also converted some forced worker termination
exceptions into failed results.

## Changes

- Added `zesolver/cancellation.py`.
- Forwarded GUI Stop through `GuiSolveController.cancel()`.
- Passed a Manager-backed process token to Near process/hybrid workers.
- Stopped new process submissions after Stop.
- Cancelled pending futures and emitted explicit `cancelled` results.
- Added bounded process shutdown with terminate fallback.
- Protected FITS WCS writes with a `wcs_write` worker state.
- Mapped legacy `cancelled` to GUI `CANCELLED`.
- Added immediate GUI Stop feedback: `Arrêt en cours...`, Stop disabled,
  Resolve remains disabled until cleanup.

## Validation

Pre-change targeted baseline:

```text
core boundary check: OK
8 passed in 7.18s
```

P3A-V1 targeted tests:

```text
9 passed in 8.87s
```

Required targeted command:

```text
core boundary check: OK
11 passed in 10.77s
retest after hybrid cancellation-result alignment: 11 passed in 9.36s
```

Corpus runner:

```text
status: PASS
9 skipped, 409 deselected
```

External corpus environment variables were unset in this session, so corpus data
cases were skipped rather than executed against the local reference corpus.

Hermetic runner:

```text
status: PASS
408 passed, 1 skipped, 9 deselected, 44 warnings
```

Full pytest:

```text
408 passed, 10 skipped, 44 warnings
```

Additional checks:

```text
compileall: OK
git diff --check: OK
```

Warnings in the targeted multiprocessing tests:

```text
18 multiprocessing.popen_fork.DeprecationWarning
```

The warning categories remain the known multiprocessing fork deprecation and
Astropy FITS verify warning. The multiprocessing warning count increases because
P3A-V1 adds process-cancellation coverage.

## Open Risk

The real visual GUI run with 30 FITS / 6 workers has not been executed in this
headless session. The code is ready for that manual revalidation gate once the
full automated regression commands pass.
