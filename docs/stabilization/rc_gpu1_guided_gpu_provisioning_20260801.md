# RC-GPU-1 Guided GPU Provisioning

## 1. Git Initial

- Branch: `test`
- Initial HEAD: `c58f158113be17ff28d03f93cd8015a7d2dbdb37`
- Initial `origin/test`: `c58f158113be17ff28d03f93cd8015a7d2dbdb37`
- Initial worktree: clean
- Initial `git diff --check`: clean

## 2. Initial Behavior

The existing ZeNear CUDA path imported/probed CuPy during image detection. In
`auto`, a permanent missing module selected CPU but did not disable GPU for the
batch, so every image repeated:

```text
ZeNear CUDA fallback: reason=No module named 'cupy' continuing_on=cpu
```

## 3. Architecture

Added `zesolver.gpu_support`, a GUI-free package split into:

- `models.py`: structured reports, runtime contexts and result enums;
- `probe.py`: platform/CuPy/NVIDIA detection;
- `self_test.py`: real CuPy allocation/calculation test;
- `policy.py`: package/profile decision;
- `provisioning.py`: Python, guidance-only and host-adapter provisioners.

The GUI only consumes this service. It no longer owns direct CuPy detection
logic for the Performance combo.

## 4. Runtime Contexts

- `SOURCE_MANAGED`: install plan only when `allow_environment_mutation=true`.
- `FROZEN_STANDALONE`: diagnostic/guidance only; no pip command.
- `EMBEDDED_HOST`: report and plan can be passed to a host callback.
- `UNKNOWN`: diagnostic only.

The startup wizard enables source provisioning only when
`ZESOLVER_ALLOW_GPU_PROVISIONING=1` is present.

## 5. CuPy Profile

Candidate profile: `cupy-cuda12x[ctk]`.

It is allowlisted and shown with consent, target interpreter and restart
requirement. It is not yet declared an official public GPU packaging profile
because Windows installation validation is still pending.

Local diagnostic found:

```text
platform=linux
architecture=x86_64
GPU=NVIDIA GeForce MX150
driver=550.163.01
CuPy=absent
effective_backend=cpu
reason=CUPY_NOT_INSTALLED
```

With explicit source mutation enabled, the generated plan is:

```text
/home/tristan/.openclaw/workspace/projects/ZeSolver/.venv/bin/python -m pip install cupy-cuda12x[ctk]
```

No installation was run against the main project environment.

A throwaway venv validation installed `cupy-cuda12x==14.1.1` successfully, then
the real self-test failed with:

```text
Failed to find CUDA headers. Please install CUDA toolkit headers
(e.g., pip install cupy-cuda12x[ctk]) or specify CUDA_PATH environment variable.
```

The follow-up attempt to install `cupy-cuda12x[ctk]` in the throwaway venv was
aborted by temporary storage exhaustion (`No space left on device`) and the venv
was deleted. This is why the profile is documented as a candidate, not as an
officially supported public GPU profile yet.

## 6. Security

- `shell=False`;
- closed package allowlist;
- no user-supplied pip package names;
- no driver installer;
- no CUDA Toolkit install;
- no package manager commands;
- no admin escalation;
- timeout/cancel support in the provisioner;
- CPU remains available in every failure mode.

## 7. API

CLI:

```bash
python -m zesolver.gpu_diagnostic --json --show-install-plan
```

Host API:

```python
report = probe_gpu_capability(runtime_context)
plan = build_gpu_provisioning_plan(report, runtime_context)
```

## 8. Wizard And Settings

The startup wizard now has a non-blocking optional page:

- GPU already ready: report device/CuPy/self-test status;
- NVIDIA detected and source mutation allowed: offer optional install;
- frozen/embedded/unknown: explain CPU continuation;
- macOS: report CUDA unsupported;
- broken/conflicting CuPy: show diagnostic and continue CPU.

Persistent fields record only behaviorally useful state:

- diagnostic executed;
- GPU available;
- CPU chosen by user;
- restart required;
- last reason code;
- schema version.

The Performance tab exposes `Diagnostic acceleration GPU` for manual reruns.

## 9. Batch Fallback

`CUPY_NOT_INSTALLED` is now a permanent batch condition:

```text
ZeNear GPU unavailable: reason=CUPY_NOT_INSTALLED CPU selected for the complete batch
```

The first image records one fallback. Later images in the same batch run CPU
directly and do not increment fallback telemetry again.

## 10. Tests Added

- `tests/test_gpu_support.py`
- `tests/test_gpu_wizard_contract.py`
- settings persistence coverage for GPU diagnostic fields
- updated ZeNear fallback expectations for one-time batch disable

Covered cases include macOS, no NVIDIA, source-managed plan, frozen guidance,
embedded host delegation, fake CuPy success, broken runtime, package conflict,
pip success simulation, CLI JSON, wizard contract, explicit CPU choice contract,
and batch Auto without CuPy.

## 11. Validation

Commands run:

```bash
.venv/bin/python -m py_compile ...
.venv/bin/python -m pytest -q tests/test_gpu_support.py tests/test_gpu_wizard_contract.py tests/test_s6a2_zenear_gpu_fallback.py tests/test_s6a2b_gpu_telemetry.py
.venv/bin/python -m pytest -q tests/test_settings_persistence.py tests/test_startup_wizard.py tests/test_gpu_support.py tests/test_gpu_wizard_contract.py
.venv/bin/python -m compileall zesolver zeblindsolver tools
.venv/bin/python -m zesolver.gpu_diagnostic --json --show-install-plan
.venv/bin/python -m zesolver.gpu_diagnostic --json --show-install-plan --distribution-kind source_managed --allow-environment-mutation
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q --deselect tests/test_zn310b_gui_fallback_dataset.py::test_zn310b_originals_remain_unmodified_by_source_sha --deselect tests/test_zn310b_gui_fallback_dataset.py::test_zn310b_all_generated_copies_have_no_old_wcs
```

Intermediate results:

- GPU targeted: `20 passed`
- settings/wizard/GPU: `48 passed`
- final targeted bundle: `57 passed`
- compileall: OK
- raw global: `826 passed, 37 skipped, 2 failed`
- compatible global excluding only the two known external ZN310B local-corpus
  failures: `826 passed, 37 skipped, 2 deselected`

The raw global failures are unchanged local external-corpus issues:

- `test_zn310b_originals_remain_unmodified_by_source_sha`: source SHA mismatch
  under `/home/tristan/near_bench_cmp30/...`;
- `test_zn310b_all_generated_copies_have_no_old_wcs`: one generated copy still
  contains WCS.

They are not introduced by RC-GPU-1 and no repository skip was added.

## 12. Limits

- The candidate profile still needs a successful full install/self-test on a
  machine with enough temporary storage, then Windows validation.
- No public frozen GPU package is produced.
- No NVIDIA driver or CUDA Toolkit installation is attempted.
- No macOS CUDA support is claimed.
- No ZeMosaic or ZeSeestarStacker integration is implemented here.
- Windows GPU installation validation remains required before marking the
  `cupy-cuda12x` profile officially supported for public beta.

## 13. Files Modified

- `zesolver/gpu_support/*`
- `zesolver/gpu_diagnostic.py`
- `zesolver/gui_startup_wizard.py`
- `zesolver/settings_store.py`
- `zesolver.py`
- `zeblindsolver/metadata_solver.py`
- `tests/test_gpu_support.py`
- `tests/test_gpu_wizard_contract.py`
- `tests/test_settings_persistence.py`
- `README.md`
- `CHANGELOG.md`
- `AGENT.md`
- `docs/architecture/gpu_provisioning_strategy.md`

## 14. Verdict

The final commit hash is reported in the mission close-out. A commit cannot
contain its own stable hash without changing that hash.

```text
RC_GPU1_DIAGNOSTIC_AND_PROVISIONING_ARCHITECTURE_CLOSED
GPU_SOURCE_INSTALLATION_GUIDED
FROZEN_GPU_PACKAGING_PENDING
EMBEDDED_HOST_INTEGRATION_READY
```
