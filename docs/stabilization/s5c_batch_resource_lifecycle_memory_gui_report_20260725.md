# S5C - Batch Resource Lifecycle, Memory, and GUI Preflight Stabilization - 2026-07-25

## 1. Etat Git initial

Initial command:

```text
git status --short --branch
git diff --check
```

Initial status:

```text
## test...origin/test
 M tests/test_astap_4d_builder_cli.py
 M tests/test_astap_4d_runtime_validation.py
 M tests/test_catalog_blind4d_manifest_view.py
 M tests/test_catalog_library_management_service.py
 M zeblindsolver/astap_4d_builder.py
 M zesolver/catalog_library/blind4d_view.py
?? docs/stabilization/s5_blind4d_library_view_validation_report_20260723.md
?? docs/stabilization/s5b_blind4d_generated_index_runtime_regression_report_20260724.md
?? tools/diagnose_s5b_blind4d_generated_index_regression.py
```

`git diff --check` was clean. No commit, no push, no D50 rebuild, no catalog deletion.

## 2. Reproduction

The pre-change runner source from `git show HEAD:zesolver/core/batch/runner.py` was executed with a scripted four-file mixed batch. It constructed:

```text
['near', 'near', 'near', 'near', 'blind', 'blind']
```

That confirms pipeline construction was task/file-local for the old runner shape. Static inspection of the initial `SolverPipeline.solve()` also showed Blind runtime resolution before Near.

## 3. Cycle De Vie Avant

Before S5C:

```text
GUI main thread
  -> config
  -> catalog resource resolution
  -> Near runtime resolution
  -> Blind 4D runtime resolution / manifest payload / index load
  -> GUI_RUN_BEGIN
  -> Batch runner
       near task per file -> new SolverPipeline
       blind task per unresolved file -> new SolverPipeline
```

Blind could be resolved even if Near would solve every file.

## 4. Compteurs Avant

Old runner reproduction:

```text
near pipeline constructions = 4
blind pipeline constructions = 2
files = 4
workers = 6
```

The old runner did not propagate telemetry context into worker threads, which is now fixed.

## 5. Memoire Avant

User-observed full D50 run:

```text
catalog_resource_resolution = 341.18 s
blind4d_runtime_resolution = 284.17 s
first Near delayed about 78 min after GUI_RUN_BEGIN
RSS > 4 GiB, swap active
```

## 6. Cause Racine

Root causes:

- GUI main-thread preflight performed heavy catalog and Blind runtime work.
- `SolverPipeline.solve()` resolved Blind before Near.
- Batch tasks created a fresh pipeline per file/phase.
- Runtime ownership was split between pipeline and Blind port.
- Strict manifest validation loaded `.npz` payloads and built KD-trees during phases that only needed manifest structure.

## 7. Architecture Retenue

Chosen design:

- Batch telemetry via `zesolver/resource_telemetry.py`.
- Resources resolved once in `PipelineGuiRunner` before worker submission.
- Pipeline cached per worker thread, not per file.
- Blind phase uses one worker and one shared `ProductionBlindSolverPort`.
- Blind port is the sole owner of full Blind runtime resolution/loading.
- Manifest loader supports `validate_indexes=False` for lightweight validation.
- GUI preflight moved into the existing `SolveRunner` worker thread path.

## 8. Cycle De Vie Apres

```text
GUI main thread
  -> build config only
  -> start SolveRunner QThread
SolveRunner / PipelineGuiRunner
  -> preparation phase signal
  -> resolve catalog resources once
  -> create shared Blind port
  -> Near phase, pipeline per worker
       emit Near successes progressively
  -> if unresolved exists:
       Blind phase, one worker, shared runtime
  -> final telemetry snapshot + diagnostic gc
```

## 9. Compteurs Apres

Synthetic all-Near with runner-owned resources:

```text
catalog_resource_resolution_count = 1
blind_runtime_resolution_count = 0
blind_index_payload_load_count = 0
blind_kdtree_build_count = 0
solver_pipeline_constructor_count = 2
worker_thread_count = 2
```

Synthetic mixed batch, resources injected:

```text
blind_runtime_resolution_count = 1
blind_index_payload_load_count = 1
blind_kdtree_build_count = 1
solver_pipeline_constructor_count = 5
blind_port_constructor_count = 1
worker_thread_count = 4
```

## 10. Memoire Apres

Measured synthetic RSS:

```text
all-Near: after_preflight 129336 KiB -> after_diagnostic_gc 130752 KiB
mixed: after_preflight 131260 KiB -> after_diagnostic_gc 131588 KiB
```

The monolithic full D50 RSS was not re-measured here because S5C explicitly forbids rebuilding/deleting libraries and S5D owns partitioning. The important verified property is no per-file Blind payload duplication.

## 11. Comportement All-Near

Verified:

```text
4/4 SOLVED via NEAR
blind_runtime_resolution_count = 0
blind_index_payload_load_count = 0
blind_kdtree_build_count = 0
```

## 12. Comportement Mixte

Verified:

```text
2 SOLVED via NEAR
2 SOLVED via BLIND4D
blind runtime resolved once
terminal result count = file count
no duplicate emitted path
```

## 13. Enchainement WCS Cleaner

WCS Cleaner was not changed. The solve launch no longer performs catalog/Blind runtime preflight on the main Qt thread after cleanup; `_start_solving()` builds config, sets the preparation status, starts `SolveRunner`, and returns to the event loop.

## 14. Reactivite GUI

Added Qt offscreen test:

```text
test_s5c_qt_event_loop_stays_responsive_during_slow_preflight
```

It injects a slow preflight and verifies a `QTimer` continues ticking while `PipelineGuiRunner` runs in a worker thread.

## 15. Deuxieme Run

Added same-process two-run test:

```text
test_s5c_two_runs_same_process_keep_counters_bounded
```

Observed all-Near Blind counters remain zero on run 1 and run 2.

## 16. Annulation

The existing cancellation token is still passed from GUI to `BatchSolveRequest` and to solver runtime options. Batch cancellation before Near, between Near and Blind, and stop-on-error behavior remain covered by existing batch tests. Preflight cancellation is cooperative after resource resolution returns; no uninterruptible GUI-thread block remains.

## 17. Resultats Progressifs

Near successes are emitted from `as_completed()` during the Near phase. Files solved by Near no longer wait for all Near tasks, Blind preparation, or Blind completion.

## 18. Fichiers Modifies

S5C files:

```text
zeblindsolver/index_manifest_4d.py
zesolver.py
zesolver/catalog_resources.py
zesolver/resource_telemetry.py
zesolver/core/batch/models.py
zesolver/core/batch/runner.py
zesolver/core/blind_port.py
zesolver/core/pipeline.py
zesolver/gui_pipeline/controller.py
zesolver/gui_pipeline/pipeline_runner.py
zesolver/gui_pipeline/requests.py
tests/test_s5c_batch_resource_lifecycle.py
tests/test_catalog_library_blind4d_product_switch.py
tests/test_gui_catalog_library_control.py
```

Other S5/S5B files were already dirty at the start and were not reverted.

## 19. Tests

Added/updated coverage for:

- one resource resolution per batch;
- no Blind runtime for all-Near;
- one Blind runtime for mixed;
- pipeline per worker;
- shared Blind runtime;
- progressive emission;
- final order preservation;
- Qt event loop responsiveness;
- second run in same process;
- updated ownership contract for Blind runtime selection.

## 20. Barrieres

Passed:

```text
.venv/bin/python tools/check_core_boundaries.py
core boundary check: OK

.venv/bin/python tools/run_regression_suite.py --hermetic
622 passed, 1 skipped, 9 deselected
status: PASS

QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
622 passed, 10 skipped

.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests zesolver.py zewcscleaner.py zeindexcheck.py
PASS

git diff --check
PASS
```

## 21. Etat Git Final

Final `git status --short --branch` includes pre-existing S5/S5B dirty files plus S5C changes. No commit and no push were made.

## 22. Limites

- Full dense `direct-d50` scientific candidate topology remains untouched for S5D.
- Full D50 RSS target is not promised while the index is monolithic.
- Preflight cancellation is cooperative at phase boundaries, not inside every filesystem/hash primitive.
- Catalog runtime may still intentionally cache lightweight descriptors; large Blind payload ownership is bounded to the shared Blind port.

## 23. Prochaine Etape

S5D should partition/diversify the full D50 Blind 4D index and fix the `hits=2000/tested=0` scientific candidate topology without reintroducing per-file runtime duplication.

## 24. Decision De Gate

```text
READY_FOR_S5D_BLIND4D_FULL_D50_PARTITIONING
```
