# RC-GPU-1B Addendum - GPU Installation Feedback

Date: 2026-08-02

## Scope

This addendum closes the remaining usability gap in the guided GPU provisioning
flow: during a long `cupy-cuda12x[ctk]` installation, the startup wizard must
show visible, progressive feedback instead of leaving the user in the dark.

## Confirmed Manual Context

The manually validated source-managed Linux environment now has working CUDA
acceleration:

- CuPy: `14.1.1`
- GPU: `NVIDIA GeForce MX150`
- `device_count=1`
- CUDA allocation, calculation and synchronization: OK
- Result: `CUDA_SELF_TEST_OK`
- Platform: Linux `x86_64`
- Python: `3.13.5`
- NVIDIA driver: `550.163.01`

## Root Issue

The previous fix made the Qt thread lifecycle safe and drained pip output, but
the UI still exposed only coarse status. A large package install could appear
idle because pip output was not structured into user-visible phases, and the
post-install CUDA verification was not run as an explicit fresh process stage.

## Fix

- Added `GpuProvisioningProgress` as a GUI-free structured progress event.
- Kept compatibility with string progress callbacks.
- Streamed subprocess output continuously with `stderr=STDOUT`, `shell=False`,
  and a dedicated reader thread using chunked pipe reads.
- Added explicit phases:
  - `preparation`
  - `download`
  - `install`
  - `pip_check`
  - `self_test`
  - `restart_required`
- Ran `python -m pip check` after a successful pip install.
- Ran a fresh subprocess after pip:

  ```bash
  python -m zesolver.gpu_diagnostic --json --self-test
  ```

- Parsed the JSON self-test report and displayed GPU name, CuPy version, device
  count and self-test result in the wizard.
- Kept the wizard open on `INSTALL_FAILED` and appended stdout/stderr tails.
- Added an indeterminate progress bar during provisioning.
- Kept the existing worker lifecycle guard: no cleanup until native
  `QThread.finished`.

## Files Modified

- `zesolver/gpu_support/models.py`
- `zesolver/gpu_support/provisioning.py`
- `zesolver/gpu_support/__init__.py`
- `zesolver/gui_startup_wizard.py`
- `tests/test_gpu_support.py`
- `tests/test_gpu_provisioning_qt.py`
- `CHANGELOG.md`

## Tests

Added or strengthened tests covering:

- streaming more than 1 MiB of subprocess stdout/stderr without deadlock;
- progressive provisioning messages reaching the Qt wizard log;
- successful provisioning showing GPU/CuPy/self-test details;
- self-test failure returning `INSTALL_FAILED` with diagnostic tail;
- pip error shown without closing ZeSolver;
- worker cleanup remaining tied to native `QThread.finished`.

## Local Validation

Commands executed:

```bash
python -m py_compile zesolver/gpu_support/models.py zesolver/gpu_support/provisioning.py zesolver/gui_startup_wizard.py
.venv/bin/python -m pytest -q tests/test_gpu_support.py tests/test_gpu_provisioning_qt.py tests/test_gpu_wizard_contract.py
.venv/bin/python -m pytest -q tests/test_gpu_support.py tests/test_gpu_provisioning_qt.py tests/test_gpu_wizard_contract.py tests/test_startup_wizard.py tests/test_settings_persistence.py tests/test_s6a2_zenear_gpu_fallback.py
.venv/bin/python -m zesolver.gpu_diagnostic --json --self-test
python -m compileall zesolver zeblindsolver tools
.venv/bin/python -m pytest -q --deselect tests/test_zn310b_gui_fallback_dataset.py::test_zn310b_originals_remain_unmodified_by_source_sha --deselect tests/test_zn310b_gui_fallback_dataset.py::test_zn310b_all_generated_copies_have_no_old_wcs
```

Results:

- GPU Qt/support targeted: `22 passed`
- GPU + wizard/settings/fallback targeted: `63 passed`
- Compileall: OK
- Compatible suite excluding only the two known external ZN310B corpus checks:
  `840 passed, 33 skipped, 2 deselected`
- `git diff --check`: OK

Observed diagnostic:

- `effective_backend=cuda`
- `reason_code=GPU_READY`
- `cupy_package_name=cupy-cuda12x`
- `cupy_version=14.1.1`
- `device_names=["NVIDIA GeForce MX150"]`
- `device_count=1`

## Residual Risks

- The public GPU packaging story remains pending for frozen executables.
- Windows GPU source-managed validation is still required before calling the
  package profile officially supported for public beta.
- The wizard does not attempt automatic application restart; it asks the user
  to relaunch ZeSolver after successful provisioning.

## Verdict

`RC_GPU1B_INSTALL_FEEDBACK_CLOSED`
