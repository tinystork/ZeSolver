# RC-GPU-1B GPU Provisioning Thread Lifecycle

## Git Initial State

- Branch: `test`
- Initial HEAD: `4b73cc8cc53095c42bb612540b14e819c77ed93a`
- Initial `origin/test`: `c58f158113be17ff28d03f93cd8015a7d2dbdb37`
- Initial worktree before RC-GPU-1B edits: clean except the active RC-GPU-1 local commit.
- Initial `git diff --check`: clean.

## Reproduction

Manual validation reported that clicking `Installer l'acceleration GPU` could
finish the pip install successfully, then crash the process with:

```text
QThread: Destroyed while thread '' is still running
Abandon (core dumped)
```

The pip plan was valid, no orphan pip process remained, and the installed
packages included:

```text
cupy-cuda12x 14.1.1
nvidia-cublas-cu12 12.9.2.10
nvidia-cuda-nvrtc-cu12 12.9.86
nvidia-cuda-runtime-cu12 12.9.79
```

This confirms a Qt lifecycle race after successful provisioning, not a pip
failure.

## Root Cause

`StartupGpuProvisionWorker` defined a custom signal named `finished`. That name
hid the native `QThread.finished` lifecycle signal. The wizard therefore treated
the provisioning result as if the Qt thread itself had already stopped, cleared
its worker reference too early, and could destroy the wizard while the QThread
was still running.

The provisioner also used `PIPE` for subprocess output but only consumed output
after the process had ended. A verbose pip run could therefore block on a full
pipe.

## Fix

- Renamed the custom signal to `resultReady`.
- Connected `resultReady` only to result handling.
- Connected native `QThread.finished` to lifecycle cleanup and `deleteLater`.
- Kept a strong `self._gpu_worker` reference until native `QThread.finished`.
- Added `_wait_for_gpu_provisioning_stop()` for `reject()` and `closeEvent()`.
- Disabled wizard navigation buttons during provisioning while keeping an
  explicit cancel path through the GPU page.
- Refused close when the worker cannot stop within the wait budget.
- Appended stdout/stderr tails to the wizard diagnostic panel.
- Replaced delayed subprocess output collection with a continuous reader thread
  using `stderr=STDOUT`, `shell=False`, command allowlist validation, timeout,
  cancellation, PID/returncode logs, and tail preservation.

## Tests Added

- `tests/test_gpu_provisioning_qt.py`
  - native `QThread.finished` is no longer masked;
  - `resultReady` does not clear the worker while `isRunning()` can still be
    true;
  - success displays restart guidance and cleans the worker after native
    finish;
  - pip/provisioning errors are displayed without closing the application;
  - cancellation stops the worker before `reject()` completes;
  - `closeEvent()` is ignored when a running worker does not stop.
- `tests/test_gpu_support.py`
  - real subprocess success requires restart;
  - more than 1 MiB of stdout/stderr is drained without deadlock;
  - non-zero pip exit reports `INSTALL_FAILED` with captured output.
- `tests/test_gpu_wizard_contract.py`
  - static guard that `StartupGpuProvisionWorker` exposes `resultReady` and
    does not define a custom `finished` signal.

## Validation

Commands run before commit:

```bash
.venv/bin/python -m py_compile zesolver/gui_startup_wizard.py zesolver/gpu_support/provisioning.py tests/test_gpu_support.py tests/test_gpu_provisioning_qt.py tests/test_gpu_wizard_contract.py
.venv/bin/python -m pytest -q tests/test_gpu_support.py tests/test_gpu_wizard_contract.py tests/test_gpu_provisioning_qt.py
.venv/bin/python -m pytest -q tests/test_gpu_support.py tests/test_gpu_wizard_contract.py tests/test_gpu_provisioning_qt.py tests/test_settings_persistence.py tests/test_startup_wizard.py tests/test_s6a2_zenear_gpu_fallback.py tests/test_s6a2b_gpu_telemetry.py
.venv/bin/python -m compileall zesolver zeblindsolver tools
.venv/bin/python -m pytest -q --deselect tests/test_zn310b_gui_fallback_dataset.py::test_zn310b_originals_remain_unmodified_by_source_sha --deselect tests/test_zn310b_gui_fallback_dataset.py::test_zn310b_all_generated_copies_have_no_old_wcs
.venv/bin/python -m pytest -q
```

Observed results:

- GPU targeted: `20 passed`
- GPU + wizard/settings fallback bundle: `65 passed`
- compileall: OK
- compatible global excluding only the two known external ZN310B local-corpus
  failures: `838 passed, 33 skipped, 2 deselected`
- raw global: `2 failed, 838 passed, 33 skipped`
- `git diff --check`: clean

The raw global failures are the pre-existing external ZN310B local-corpus
issues already tracked on this workstation:

- `test_zn310b_originals_remain_unmodified_by_source_sha`: source SHA mismatch
  under `/home/tristan/near_bench_cmp30/...`;
- `test_zn310b_all_generated_copies_have_no_old_wcs`: one generated
  `wrong_hints` copy still contains WCS.

The final global and `git diff --check` results are recorded in the mission
close-out.

## Manual Validation Status

Requested manual environment:

```bash
ZESOLVER_ALLOW_GPU_PROVISIONING=1
TMPDIR=$HOME/.cache/zesolver-gpu-tmp
```

The code path now keeps the wizard alive until native thread completion,
streams pip output to the GPU diagnostic panel, displays `INSTALL_FAILED` for
non-zero exits, and persists restart-required success without closing ZeSolver.

The final real click validation remains intentionally unpushed until Tristan
can re-run it on the same NVIDIA environment.

## Files Modified

- `zesolver/gui_startup_wizard.py`
- `zesolver/gpu_support/provisioning.py`
- `tests/test_gpu_support.py`
- `tests/test_gpu_wizard_contract.py`
- `tests/test_gpu_provisioning_qt.py`
- `CHANGELOG.md`
- `docs/stabilization/rc_gpu1b_gpu_provisioning_thread_lifecycle_20260801.md`

## Verdict

RC-GPU-1B closes the Qt lifecycle crash class for successful and failed guided
GPU provisioning while leaving CPU operation, CuPy optionality and future
packaging decisions unchanged.
