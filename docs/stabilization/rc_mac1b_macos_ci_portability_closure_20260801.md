# RC-MAC-1B - macOS CI Portability Closure

Date: 2026-08-01
Branch: `test`

Status after GitHub Actions validation:

```text
MACOS_COMPATIBILITY_AUDIT_PASSED
MACOS_CI_VALIDATED
MACOS_RUNTIME_VALIDATION_PENDING
```

This does not claim `MACOS_RUNTIME_VALIDATED`, `MACOS_PRODUCTION_READY`, or
`PRODUCTION_READY_FOR_MACOS`.

## 1. Initial Git State

Initial checks:

```text
git status --short
  clean
git branch -vv
  * test 386d2b1 [origin/test] Fix Python 3.11 checksum compatibility
git log --oneline --decorate -10
  386d2b1 (HEAD -> test, origin/test) Fix Python 3.11 checksum compatibility
  98a8de3 Improve macOS CI compile diagnostics
  4ec11ff update requirements.txt
  d87f771 Add macOS compatibility audit and CI
  5ec7fe6 Align Blind4D coverage warning semantics
  57caf7f agent mis a jour
  5858b51 Close startup wizard catalog activation transaction
  43fa8e6 Add system light and dark theme selector
  f42079f Add automatic catalog download resume controls
  a9a7c4c fix passage dossier de travail au gui durant le wizard
git rev-parse HEAD
  386d2b132d92329585e8e9c2dd10165ff911562c
git rev-parse origin/test
  386d2b132d92329585e8e9c2dd10165ff911562c
git diff --check
  clean
```

## 2. Scope

Closed the five macOS CI portability failures reported after RC-MAC-1:

- deterministic worker-control GUI test;
- spawn-compatible legacy executor recreation;
- WCS writer cancellation safety without timing sleeps;
- Darwin-safe S6A1C native-thread telemetry;
- deterministic Blind 4D ring-coverage sampling on tied distances.

No solver thresholds, catalogue formats, downloader behavior, themes, or
wizard transactions were changed.

## 3. Root Causes and Fixes

### 3.1 GUI Worker Cap

Cause:

- the test expected `solver_workers=3` to survive, but the GUI worker spinbox
  clamps against `os.cpu_count()`;
- a macOS runner with fewer visible CPUs could reduce the value before
  persistence.

Fix:

- `tests/test_gui_development_surface_reorganized.py` injects
  `os.cpu_count() -> 8` inside the isolated GUI subprocess;
- the persisted worker assertion remains exactly `3`.

### 3.2 Legacy Executor Recreation

Cause:

- under `spawn`, a process-pool failure/fallback path could leave a stale
  unresolved entry for a path that later produced a solved fallback result;
- final unresolved emission could then produce more terminal results than input
  paths.

Fix:

- `BatchSolver` now uses a single process context for the process cancellation
  manager and every `ProcessPoolExecutor`;
- `ZE_PROCESS_START_METHOD=spawn` can force spawn on Linux tests;
- terminal result queuing now rejects duplicate paths;
- solved/skipped phase-1 results clear stale `unresolved` entries before
  queuing;
- the regression test asserts exactly one terminal result per path and runs
  with `spawn`.

### 3.3 WCS Writer Stop Safety

Cause:

- the test used `time.sleep(0.1)` to guess that the worker was inside
  `wcs_write`;
- that was brittle with macOS `spawn` startup latency and could test the wrong
  state.

Fix:

- the test creates the cancellation manager and process pool from the same
  `multiprocessing.get_context("spawn")`;
- it waits until the worker publishes `wcs_write` through the shared token;
- if the future fails before reaching `wcs_write`, the worker exception is
  surfaced;
- `protected_wcs_writers == 1` and the "do not kill WCS writer" invariant are
  unchanged.

### 3.4 Native Thread Sampling

Cause:

- `tools/measure_s6a1c_native_threading.py` sampled process thread counts from
  `/proc/self/status`, which is Linux-only;
- macOS returned `None`, while the test required a numeric
  `threads_process_peak`.

Fix:

- the tool now emits `process_thread_sampling_supported`;
- Linux keeps numeric thread sampling through `/proc`;
- Darwin reports unsupported explicitly instead of fabricating a value;
- the test still validates all other telemetry and only relaxes the process
  thread count when the capability is false.

### 3.5 Blind 4D Ring Coverage

Cause:

- ring coverage used `np.argpartition` before sorting nearest neighbors;
- tied distances on grid-like geometry can be partitioned differently between
  platforms, causing the mid-rank local-geometry fixture to disappear on macOS.

Fix:

- nearest-neighbor ordering now uses deterministic distance then index
  ordering;
- this is a tie-break only and does not change the scientific scoring or
  thresholds.

## 4. Workflow Update

Updated:

```text
.github/workflows/macos-ci.yml
  actions/checkout@v6
  actions/setup-python@v6
```

The workflow still runs on `macos-latest`, Python 3.11, and keeps the existing
CI-compatible suite shape. No new test-file exclusion was added.

## 5. Files Modified

```text
.github/workflows/macos-ci.yml
CHANGELOG.md
docs/stabilization/rc_mac1b_macos_ci_portability_closure_20260801.md
tests/test_gui_development_surface_reorganized.py
tests/test_legacy_executor_recreation.py
tests/test_legacy_process_stop_wcs_safety.py
tests/test_s6a1c_native_threading.py
tools/measure_s6a1c_native_threading.py
zeblindsolver/quad_sampling.py
zesolver.py
```

## 6. Local Validation

Commands executed locally:

```text
.venv/bin/python -m compileall zesolver zeblindsolver tools
  OK

.venv/bin/python -m pytest -q \
  tests/test_gui_development_surface_reorganized.py \
  tests/test_legacy_executor_recreation.py \
  tests/test_legacy_process_stop_wcs_safety.py \
  tests/test_s6a1c_native_threading.py \
  tests/test_synthetic.py::test_catalog_ring_coverage_reaches_mid_rank_local_geometry \
  tests/test_macos_compatibility.py \
  tests/test_process_cancellation_token.py
  20 passed

.venv/bin/python -m pytest -q
  2 failed, 813 passed, 37 skipped

.venv/bin/python -m pytest -q --ignore tests/test_zn310b_gui_fallback_dataset.py
  799 passed, 37 skipped

git diff --check
  clean
```

The two raw-suite failures are the pre-existing external ZN310B local corpus
state:

```text
tests/test_zn310b_gui_fallback_dataset.py::test_zn310b_originals_remain_unmodified_by_source_sha
tests/test_zn310b_gui_fallback_dataset.py::test_zn310b_all_generated_copies_have_no_old_wcs
```

They concern `/home/tristan/near_bench_cmp30/...` and are not introduced by
RC-MAC-1B. The macOS workflow already ignores that file from RC-MAC-1; this
mission added no new exclusion.

## 7. GitHub Actions Result

GitHub Actions run:

```text
run_url: https://github.com/tinystork/ZeSolver/actions/runs/30717490880
run_id: 30717490880
job_id: 91415505296
commit: caed8f21f0eff9e7589098e1638df6c53b7f16e1
runner: macos-latest -> macos-26-arm64
runner_version: 2.336.0
macOS: 26.5.2 / Darwin 25.5.0
architecture: arm64
python: CPython 3.11.9
duration: 3m15s
targeted_macos_tests: 79 passed, 1 warning
full_ci_compatible_suite: 794 passed, 42 skipped, 2 warnings
result: success
```

The workflow used `actions/checkout@v6` and `actions/setup-python@v6`.

## 8. Residual Risks

- macOS physical runtime validation remains pending.
- The public macOS `.app`, signing, notarization, and DMG work remain outside
  this mission.
- The raw local suite remains sensitive to the mutable external ZN310B corpus.

## 9. Verdict

Final status:

```text
MACOS_COMPATIBILITY_AUDIT_PASSED
MACOS_CI_VALIDATED
MACOS_RUNTIME_VALIDATION_PENDING
```
