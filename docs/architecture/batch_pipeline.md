# Batch Pipeline

Phase: P2B-2B - Batch & Concurrency Extraction

## Purpose

`zesolver/core/batch/` introduces a batch runner that depends on the stable
`SolverPipeline` facade instead of directly importing Near, Blind, CatalogDB or
GUI objects.

The legacy `BatchSolver` in `zesolver.py` remains available and is not removed
or rewired during this phase.

## Public Contracts

```python
BatchSolveRequest(
    requests: tuple[SolveRequest, ...],
    workers: int,
    io_concurrency: int,
    preserve_order: bool,
    stop_on_error: bool,
    cancel_token: object | None,
)
```

```python
BatchProgress(
    total: int,
    queued: int,
    running: int,
    solved: int,
    failed: int,
    skipped: int,
    cancelled: int,
)
```

```python
BatchSolveResult(
    results: tuple[SolveResult, ...],
    progress: BatchProgress,
    cancelled: bool,
    duration_s: float,
)
```

The runner accepts a phase-aware factory:

```python
BatchSolverPipeline(
    solver_pipeline_factory=lambda phase: SolverPipeline(...),
)
```

The current phases are:

```text
near
blind
```

The expected production factory creates:

```text
phase=near  -> SolverPipeline(..., ProductSettings(blind_enabled=False))
phase=blind -> SolverPipeline(..., ProductSettings(blind_enabled=True, blind_only=True))
```

This preserves the historical global route:

```text
Near on every input first.
Blind only on unresolved Near failures.
```

## Scheduling

The first extracted runner uses `ThreadPoolExecutor` for phase work. It does not
move the legacy process/hybrid Near scheduler yet. Those heuristics remain
mapped in `docs/architecture/batch_concurrency_map.md` for later extraction.

Observable invariants:

```text
input_count = solved + failed + skipped + cancelled
no duplicate final results
no missing final results
preserve_order=True returns input order
preserve_order=False returns completion/emission order
```

## Cancellation

Cancellation is read from `BatchSolveRequest.cancel_token`. Supported token
shapes:

```text
callable -> bool
object with is_set() -> bool
truthy object
```

Cancellation is checked:

```text
before the run
before each worker task
between Near and Blind
while collecting futures
```

Pending futures are cancelled where possible. Running solver calls still rely on
the underlying `SolverPipeline`/engine cancellation semantics.

## Error Normalization

Worker or engine exceptions become `SolveResult(status=FAILED)` with an error
prefix:

```text
ENGINE_FAILED
WORKER_CRASHED
WORKER_FAILED_TO_RETURN_RESULT
CANCELLED_BEFORE_NEAR
CANCELLED_BEFORE_BLIND
CANCELLED_AFTER_ERROR
```

Exceptions are not converted to success.

## Current Limits

The following remain in the legacy batch path for later phases:

```text
process-pool Near worker initialization
hybrid CPU/GPU Near scheduling
near_detect_gpu_slots heuristics
I/O autotune semaphores
Astrometry.net batch fallback
GUI SolveRunner bridge
periodic GC hook
```

This is intentional for P2B-2B: the new batch runner is additive and reversible.
