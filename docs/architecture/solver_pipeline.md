# SolverPipeline

`SolverPipeline` is the P2B-1 façade around the existing solver engines. It is
an orchestration boundary, not a replacement for ZeNear or ZeBlind.

## Public Contracts

```python
from zesolver.core import SolveRequest, SolverPipeline

request = SolveRequest(
    input_path=Path("input.fit"),
    output_path=Path("output.fit"),
    overwrite_wcs=True,
    metadata_overrides={},
    request_id="optional-id",
)
```

`SolveResult` normalizes the observable result:

- status;
- backend;
- WCS write state;
- center/scale/orientation/parity when available;
- inliers and RMS when available;
- profile ids;
- catalogue status;
- warnings and error.

Statuses are explicit:

```text
SOLVED
UNSOLVED
REJECTED_FALSE_SOLUTION
INVALID_INPUT
CATALOG_UNAVAILABLE
CANCELLED
FAILED
```

## Construction

```python
pipeline = SolverPipeline(
    product_settings=product_settings,
    runtime_options=runtime_options,
    near_profile="zenear-v1",
    blind_profile="zeblind4d-v1",
    pipeline_profile="pipeline-v1",
)
```

Internally, the façade:

1. builds the P2A resolved configuration;
2. resolves P1C catalogue resources;
3. runs preflight checks;
4. attempts Near when available;
5. follows `pipeline-v1` and attempts Blind 4D after Near failure;
6. normalizes the final result;
7. isolates WCS writing when a port returns a WCS that has not already been
   written;
8. records telemetry.

## Ports

The façade uses narrow ports:

```python
class NearSolverPort(Protocol):
    def solve(self, request, *, resources, configuration) -> EngineSolveResult: ...

class BlindSolverPort(Protocol):
    def solve(self, request, *, resources, configuration) -> EngineSolveResult: ...
```

Tests inject deterministic doubles. The production Near adapter is a thin
wrapper around the existing `near_solve()`. Blind production routing remains
owned by the legacy path until a later extraction phase wires the full Blind
adapter.

## Preflight

`preflight.py` checks without modifying FITS:

- file exists and is a regular file;
- FITS is readable;
- image data is present and dimensions are valid;
- existing celestial WCS is rejected when overwrite is forbidden;
- catalogue resources are available.

## WCS IO

`wcs_io.py` provides safe output helpers:

- copy output mode support;
- no source modification when writing to a separate output;
- pixel fingerprint before/after;
- WCS write and read-back validation;
- explicit failure result.

The current production Near wrapper may still write WCS itself. If a port
returns `wcs_written=True`, the façade does not rewrite it.

## Compatibility

P2B-1 does not remove or rewrite:

- `ImageSolver`;
- `BatchSolver`;
- `solve_near()`;
- Blind 4D runtime;
- GUI construction.

The old path remains available while new tests prove that the façade can carry
P2A profiles and P1C catalogue resources.
