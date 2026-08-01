# RC-GPU-1C - Source-Managed GPU Provisioning Promotion

Date: 2026-08-02

## Scope

Promote guided GPU provisioning so a safe source-managed ZeSolver virtual
environment can offer the CuPy install button without requiring users to set
`ZESOLVER_ALLOW_GPU_PROVISIONING=1` manually.

## Confirmed Manual Validation

Linux NVIDIA validation completed before this change:

- Wizard installation of `cupy-cuda12x[ctk]`: succeeded
- `pip check`: succeeded
- Fresh CUDA self-test: succeeded
- GPU: `NVIDIA GeForce MX150`
- CuPy: `14.1.1`
- Batch of 10 images:
  - `requested=auto`
  - `selected=cuda`
  - `used=cuda`
  - `cuda_images=10`
  - `cpu_images=0`
  - `fallbacks=0`
  - `gpu_errors=0`

## Runtime Detection

`detect_gpu_runtime_context()` now marks a context as `SOURCE_MANAGED` and
mutable only when all of these are true:

- the application is not frozen;
- the application is not running as an embedded host;
- `sys.prefix != sys.base_prefix`;
- `sys.executable` is inside the active virtual environment;
- the provisioning command targets exactly the active `sys.executable`.

Python system installs and unproven interpreters remain non-mutable.

## Environment Overrides

- `ZESOLVER_DISABLE_GPU_PROVISIONING=1` forces diagnostic-only behavior and wins
  over every other signal.
- `ZESOLVER_ALLOW_GPU_PROVISIONING=1` remains available for tests and advanced
  diagnostics, but it does not make a system Python mutable.

## Temporary Directory

Pip subprocesses receive a ZeSolver-owned GPU temp directory through `TMPDIR`,
`TMP` and `TEMP`.

- Linux: `~/.cache/zesolver/gpu-tmp`
- Windows: user-local ZeSolver cache under `LOCALAPPDATA`
- macOS: user cache under `~/Library/Caches/ZeSolver`

The parent process environment is not modified.

## UI Behavior

When provisioning is disabled, the wizard now shows the concrete reason:

- Python system or unproven environment;
- frozen standalone executable;
- embedded host runtime;
- unsupported platform.

The explicit confirmation dialog is unchanged and still shows the interpreter
and package before installation. No install starts without consent.

## Tests Added

- repository `.venv` auto-detects as source-managed;
- another safe user venv auto-detects as source-managed;
- Python system remains non-mutable even with allow override;
- disable override wins over allow;
- frozen and embedded contexts remain non-mutable;
- mismatched Python executable is refused;
- pip command targets active `sys.executable`;
- GPU temp directory is passed to pip subprocesses;
- declining the GUI confirmation starts no worker/provisioner.

## Local Results

Commands:

```bash
python -m py_compile zesolver/gpu_support/models.py zesolver/gpu_support/runtime.py zesolver/gpu_support/policy.py zesolver/gpu_support/provisioning.py zesolver/gpu_diagnostic.py zesolver/gui_startup_wizard.py
.venv/bin/python -m pytest -q tests/test_gpu_support.py tests/test_gpu_provisioning_qt.py tests/test_gpu_wizard_contract.py
.venv/bin/python -m pytest -q tests/test_gpu_support.py tests/test_gpu_provisioning_qt.py tests/test_gpu_wizard_contract.py tests/test_startup_wizard.py tests/test_settings_persistence.py tests/test_s6a2_zenear_gpu_fallback.py
.venv/bin/python -m zesolver.gpu_diagnostic --json --self-test --show-install-plan
.venv/bin/python -m pytest -q --deselect tests/test_zn310b_gui_fallback_dataset.py::test_zn310b_originals_remain_unmodified_by_source_sha --deselect tests/test_zn310b_gui_fallback_dataset.py::test_zn310b_all_generated_copies_have_no_old_wcs
.venv/bin/python -m pytest -q
```

Results:

- GPU targeted: `31 passed`
- GPU + wizard/settings/fallback targeted: `72 passed`
- Compatible suite excluding only the two known external ZN310B corpus checks:
  `849 passed, 33 skipped, 2 deselected`
- Raw local suite: `2 failed, 849 passed, 33 skipped`; the failures are the
  pre-existing mutable external ZN310B corpus checks:
  `test_zn310b_originals_remain_unmodified_by_source_sha` and
  `test_zn310b_all_generated_copies_have_no_old_wcs`.
- CLI diagnostic in current `.venv`: `distribution_kind=source_managed`,
  `effective_backend=cuda`, `reason_code=GPU_READY`, `provisioning_plan=ALREADY_AVAILABLE`

## Residual Risk

Frozen GPU packaging remains pending. Embedded ZeMosaic/ZeSeestarStacker
integration remains contract-ready but not implemented in this mission.

## Verdict

`RC_GPU1C_SOURCE_MANAGED_PROVISIONING_PROMOTED`
