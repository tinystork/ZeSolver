# P2B-1 SolverPipeline Facade & Orchestration Extraction Report

Date: 2026-07-17

Decision: `READY_FOR_FURTHER_MONOLITH_EXTRACTION`

## Preconditions

P2B-1 started only after P2B-0.2 concluded:

```text
READY_FOR_CORE_EXTRACTION
```

Local milestone state:

```text
P0-P2A freeze commit: 46208a9
P2B-0 external gate docs: 9b08fbf
P2B-0.1 failed repair docs: 8c5913e
P2B-0.2 canonical fixture repair: 140d9ad
```

The external corpus gate passed before this extraction:

```text
4 passed, 2 skipped, 322 deselected; compileall OK; runner PASS
```

## Scope

This phase introduced a stable pipeline façade and extracted orchestration
contracts around the existing engines.

No solver algorithm was rewritten:

- `solve_near()` internals unchanged;
- Blind 4D runtime unchanged;
- no astrometric thresholds changed;
- no profile `v1` values changed;
- no catalogue/index/FITS formats changed;
- GUI not redesigned.

## Files Added

```text
zesolver/core/__init__.py
zesolver/core/models.py
zesolver/core/pipeline.py
zesolver/core/preflight.py
zesolver/core/result_adapter.py
zesolver/core/telemetry.py
zesolver/core/wcs_io.py
```

Tests:

```text
tests/solver_pipeline_fixtures.py
tests/test_solver_pipeline_models.py
tests/test_solver_pipeline_preflight.py
tests/test_solver_pipeline_routing.py
tests/test_solver_pipeline_result_adapter.py
tests/test_solver_pipeline_wcs_io.py
tests/test_solver_pipeline_compatibility.py
```

Documentation:

```text
docs/architecture/core_extraction_map.md
docs/architecture/solver_pipeline.md
docs/stabilization/p2b1_core_extraction_report.md
```

## Public Contracts

Added:

```text
SolveRequest
SolveResult
SolveStatus
EngineSolveResult
SolverPipeline
NearSolverPort
BlindSolverPort
```

Statuses:

```text
SOLVED
UNSOLVED
REJECTED_FALSE_SOLUTION
INVALID_INPUT
CATALOG_UNAVAILABLE
CANCELLED
FAILED
```

The pipeline constructor accepts:

```text
ProductSettings
RuntimeOptions
near_profile
blind_profile
pipeline_profile
catalog_resources
NearSolverPort
BlindSolverPort
```

## Extracted Responsibilities

Preflight:

- FITS exists/readable;
- image data present;
- dimensions valid;
- existing WCS rejected when overwrite is forbidden;
- catalogue resource availability checked.

Routing:

- Near attempted first when available;
- Blind attempted after Near failure/rejection when available and enabled;
- Blind is skipped when Near succeeds;
- cancellation before Near and between Near/Blind returns `CANCELLED`;
- no image is dropped between failed Near and Blind attempt.

Result normalization:

- backend;
- status;
- WCS write state;
- center/scale/orientation/parity where available;
- inliers and RMS;
- profile ids;
- catalogue status;
- warnings and errors.

WCS IO:

- copy output support;
- source pixels unchanged when output path differs;
- pixel fingerprint before/after;
- WCS read-back validation;
- explicit write failure result.

Telemetry:

- request id;
- profile ids;
- catalogue source/status/coverage;
- Near/Blind attempted/result;
- final status;
- WCS write state;
- duration;
- warnings.

## Compatibility

The legacy `ImageSolver` path remains present and unmodified. P2B-1 tests verify
that these entrypoints still exist:

```text
ImageSolver._run_index_near_solver
ImageSolver._run_blind_solver
ImageSolver._write_fits_solution
```

The production Near port is a thin wrapper over the existing `near_solve()`.
The production Blind port remains intentionally unconfigured in this phase; the
full legacy Blind orchestration stays in `ImageSolver` until a dedicated later
extraction wires it behind a port. Tests use deterministic Blind doubles to
protect the routing policy now.

## Targeted Validation

```text
.venv/bin/python -m pytest \
  tests/test_solver_pipeline_models.py \
  tests/test_solver_pipeline_preflight.py \
  tests/test_solver_pipeline_routing.py \
  tests/test_solver_pipeline_result_adapter.py \
  tests/test_solver_pipeline_wcs_io.py \
  tests/test_solver_pipeline_compatibility.py \
  -q

19 passed
```

Covered cases include:

- invalid FITS;
- WCS existing and overwrite forbidden;
- resources Near only / Near + partial Blind;
- no catalogue resources;
- Near success;
- Near failure then Blind success;
- Near failure then Blind failure;
- rejected Near solution followed by Blind;
- cancellation before Near;
- cancellation between Near and Blind;
- engine exception normalization;
- WCS write success;
- WCS write failure;
- pixel integrity;
- v1 profile ids in result/telemetry;
- legacy path still available.

## Final Validation

```text
.venv/bin/python -m pytest -q
344 passed, 3 skipped, 1 warning

.venv/bin/python tools/run_regression_suite.py --hermetic
340 passed, 1 skipped, 6 deselected, 1 warning
compileall: OK
runner status: PASS

.venv/bin/python tools/run_regression_suite.py --corpus
4 passed, 2 skipped, 341 deselected
compileall: OK
runner status: PASS

.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests
OK

git diff --check
OK
```

## Open Risks

- `BatchSolver` and batch concurrency remain in the monolith; planned for
  P2B-2.
- Full Blind production adapter is deliberately deferred to avoid moving the
  566-line legacy Blind orchestration without deeper parity coverage.
- GUI still calls the legacy path; P3 remains the GUI simplification phase.

## Conclusion

```text
READY_FOR_FURTHER_MONOLITH_EXTRACTION
```
