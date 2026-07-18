# Cancellation Contract

P3A-V1 introduces a small common cancellation contract in
`zesolver/cancellation.py`.

```python
class CancellationToken(Protocol):
    def cancel(self) -> None: ...
    def is_cancelled(self) -> bool: ...
    def is_set(self) -> bool: ...
```

## Implementations

- `ThreadCancellationToken`: wraps a normal `threading.Event`.
- `ProcessCancellationToken`: wraps a `multiprocessing.Manager().Event` and an
  optional shared worker-state dict.
- `CompositeCancellationToken`: reports cancelled when any child token is set.
- `ProcessCancellationController`: owns the Manager lifecycle for one batch run.

## Worker Contract

Process workers receive the process token in the `ProcessPoolExecutor`
initializer. The worker-global token is used by `ImageSolver` exactly like the
thread event: `ImageSolver._cancelled()` accepts `is_cancelled()` or `is_set()`.

Worker states are diagnostic and safety-oriented:

```text
initializing
idle
active
wcs_write
```

Forced termination skips workers whose state is `wcs_write`.

## Result Contract

An intentional Stop is not an astrometric failure.

Legacy cancellation returns:

```text
ImageSolveResult.status = "cancelled"
ImageSolveResult.message = "cancelled"
```

GUI adapters expose this as:

```text
GuiFileResult.status = "CANCELLED"
```

The final batch result keeps one entry per input path, with no duplicates.

## Shutdown Contract

On Stop:

1. set the shared token;
2. stop accepting new submissions;
3. cancel futures that have not started;
4. give active workers a bounded grace period;
5. terminate non-cooperative workers outside WCS critical sections;
6. never reuse the closed executor on a later run.

`shutdown_process_executor()` encapsulates the compatibility fallback that uses
`ProcessPoolExecutor._processes` when public terminate APIs are unavailable in
the runtime Python.
