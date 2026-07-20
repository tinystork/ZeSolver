# P2B-2B Batch Extraction Report

Phase: P2B-2B - Batch & Concurrency Extraction

## Summary

Added an additive batch runner under:

```text
zesolver/core/batch/
```

The new runner uses `SolverPipeline` through a phase-aware factory and preserves
the product route:

```text
Near first on all inputs.
Blind 4D only on unresolved Near failures.
```

The legacy monolithic `BatchSolver` remains available and unchanged as rollback.
The GUI was not rebuilt or rewired.

## Files Added

Code:

```text
zesolver/core/batch/__init__.py
zesolver/core/batch/models.py
zesolver/core/batch/runner.py
```

Tests:

```text
tests/batch_pipeline_fixtures.py
tests/test_batch_pipeline_models.py
tests/test_batch_pipeline_scheduling.py
tests/test_batch_pipeline_routing.py
tests/test_batch_pipeline_concurrency.py
tests/test_batch_pipeline_cancellation.py
tests/test_batch_pipeline_failures.py
tests/test_batch_pipeline_compatibility.py
tests/test_batch_pipeline_zn310b.py
```

Docs:

```text
docs/architecture/batch_concurrency_map.md
docs/architecture/batch_pipeline.md
```

## Behavior Preserved

Validated invariants:

```text
batch empty returns no results
single/multiple requests complete
Near successes do not enter Blind
Near failures enter Blind
Blind failures remain failures
no image is lost
no result is duplicated
preserve_order=true keeps input order
worker count 1 and >1 produce equivalent final statuses
cancellation before run returns CANCELLED results
worker/engine exceptions become FAILED results
progress callbacks receive final results
legacy BatchSolver remains available
```

## ZN3.10B Batch Validation

Command:

```bash
.venv/bin/python -m pytest tests/test_batch_pipeline_zn310b.py -q
```

Result:

```text
1 passed in 145.97s
```

Observed:

```text
8 inputs
8 results
8 solved
0 failed
0 skipped
0 cancelled
CONTROL_NEAR_CORRECT=3
NOHINT_4D_CORRECT=3
BADHINT_4D_CORRECT=2
```

The runner used worker count `1` for this real-data validation to preserve a
conservative baseline during extraction.

## Validation

P2B-2B targeted:

```text
12 passed in 145.36s
```

Corpus:

```text
6 passed, 2 skipped, 363 deselected in 293.28s
runner PASS
```

Expected corpus skips:

```text
external ASTAP/HNSKY family 'g05' not found under /opt/astap
blind4d source FITS paths not mapped yet: blind4d_p29_232329
```

Hermetic:

```text
362 passed, 1 skipped, 8 deselected, 1 warning
runner PASS
```

Full pytest:

```text
368 passed, 3 skipped, 37 warnings
```

Compile and diff:

```text
compileall: OK
git diff --check: OK
```

## Open Risks

- The first extracted runner uses threads only. The legacy process/hybrid Near
  scheduler remains mapped but not moved.
- `io_concurrency` is modeled in the request contract but not yet used by the
  extracted runner.
- Astrometry.net batch fallback remains in the legacy path until a later phase.
- P29 remains explicitly skipped because no clean, reliable fixture was mapped.

These are compatible with the progressive extraction plan because the legacy
batch remains available and the new runner is not yet forced into the GUI.

## Decision

```text
READY_FOR_GUI_PIPELINE_INTEGRATION
```
