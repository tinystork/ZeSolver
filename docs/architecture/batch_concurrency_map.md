# Batch Concurrency Map

Phase: P2B-2B - Batch & Concurrency Extraction

This document maps the current monolithic batch behavior before extraction.

## Current Shape

`BatchSolver` lives in `zesolver.py` and owns:

```text
file collection
catalog resource application
ImageSolver construction
Near phase scheduling
Blind phase scheduling
optional Astrometry.net fallback
progress callbacks
cooperative cancellation
worker pool lifecycle
memory telemetry
failure emission
```

The current observable route is:

```text
Phase 1: run Near on all inputs with per-file Blind fallback disabled.
Phase 2: run Blind only on unresolved Near failures when Blind is enabled.
Phase 3: optional Astrometry.net fallback only when explicitly configured.
Final: emit remaining unresolved failures in input order.
```

## Mechanism Table

| Mecanisme | Emplacement | Etat partage | Thread-safe | Process-safe | Effet observable | Cible |
| --------- | ----------- | ------------ | ----------: | -----------: | ---------------- | ----- |
| File collection | `BatchSolver._collect_files()` | input directory, formats, max_files | yes | n/a | input order follows `_iter_image_files()` then `max_files` truncation | `core/batch/scheduling.py` |
| Catalog application | `BatchSolver.__init__()` | `SolveConfig`, `SolverCatalogResources` | n/a | n/a | config may be updated from CatalogLibrary/environment resources | pipeline factory / compatibility adapter |
| Shared `ImageSolver` | `BatchSolver.__init__()` | config, catalog DB, hint caches, IO semaphore | partially | no | thread phase shares one solver; process phase uses worker-local solvers | legacy compatibility only |
| Cancellation event propagation | `BatchSolver.run()` and `ImageSolver.set_cancel_event()` | `threading.Event` | yes | not propagated into process workers after init | cancels pending futures; running process Near task finishes cooperatively only through result boundary | `core/batch/cancellation.py` |
| Generic phase runner | nested `_run_phase()` | thread pool, inflight futures, callback | yes | n/a | submits up to worker count, replenishes on completion, cancels pending on stop | `core/batch/runner.py` |
| Near process pool | nested `_run_phase_near_process()` | process pool, worker global `_PROC_NEAR_SOLVER` | n/a | yes for config/path payloads | worker exceptions become failed results | future worker adapter |
| Near hybrid pool | nested `_run_phase_near_hybrid()` | separate CPU/GPU process pools | n/a | yes for config/path payloads | preserves pool affinity when replenishing work | future worker adapter |
| Worker init | `_near_worker_init()` / `_near_worker_init_with_backend()` | worker global solver/backend | n/a | yes | forces CPU when CUDA unavailable, creates worker-local `ImageSolver` | future worker lifecycle |
| Worker payload conversion | `_result_to_payload()` / `_payload_to_result()` | serializable dict | yes | yes | prevents `ImageSolveResult` object transport issues | future compatibility adapter |
| Near auto strategy | nested `_auto_near_strategy()` | env, CPU/RAM/CUDA probes, optional autotune file | read-only mostly | n/a | chooses `thread`, `process`, or `hybrid`; scales workers conservatively | future scheduling policy |
| Memory telemetry | `_log_runtime_memory()` | process/GPU probes, timer | yes | n/a | diagnostic logs at interval; no result behavior | `core/batch/telemetry.py` |
| Phase 1 emit | nested `_emit_phase1()` | `unresolved`, `yield_queue` | single coordinator thread | n/a | Near solved/skipped emitted; failures held for Blind/Astrometry when eligible | `core/batch/scheduling.py` |
| Blind worker count | `_auto_blind_worker_count()` | workers, RAM, `ZE_BLIND_WORKERS`, profile | yes | n/a | 4D profile normally limits Blind concurrency unless overridden | `core/batch/scheduling.py` |
| Phase 2 Blind | `self.solver.solve_path_blind_only()` via `_run_phase()` | shared `ImageSolver` | partially | n/a | only unresolved Near failures enter Blind; solved Blind emitted | `BatchSolverPipeline` using `SolverPipeline` |
| Astrometry fallback | `zeblindsolver.astrometry_backend.solve_batch()` | API key, timeout, final_unresolved | external service | n/a | last resort only with API key and fallback enabled | kept out of first extracted batch or explicit optional hook |
| Progress callback | `_queue_result()` and GUI `SolveRunner._on_result()` | callback, GUI signal bridge | callback-defined | n/a | exceptions swallowed; GUI progress per emitted result | `ProgressSink` protocol |
| GC hook | GUI `SolveRunner._on_result()` | `gc_interval` | yes | n/a | optional collection every N results | compatibility layer |
| Final unresolved emission | final loop over `self.files` | `final_unresolved` | coordinator only | n/a | failures emitted in input order | `core/batch/runner.py` |
| GUI runner | nested `SolveRunner` | Qt thread, signals, cancel event | Qt-owned | no | starts BatchSolver, forwards progress, handles cancel | compatibility adapter; GUI unchanged |

## Extraction Decision

P2B-2B will introduce a new `BatchSolverPipeline` that uses
`SolverPipeline.solve()` per request. The first extracted runner will focus on
the stable product route:

```text
request identity
preserve_order
worker count
progress callbacks
cancellation before/during scheduling
exception normalization
no duplicate results
no lost inputs
```

The legacy `BatchSolver` remains in `zesolver.py` during this phase. GUI/CLI
integration may continue using the legacy batch until parity is demonstrated.

## Preserved Constraints

- Near/Blind solver algorithms are not changed.
- `pipeline-v1` remains Near first, then Blind 4D after Near failure.
- Blind 4D concurrency is bounded by profile/runtime policy; no default worker
  count change is introduced.
- Worker exceptions are normalized as failed/cancelled results, never success.
- Partial Blind 4D coverage remains telemetry; it is not promoted to all-sky.
