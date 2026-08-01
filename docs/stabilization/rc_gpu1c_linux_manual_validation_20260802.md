# RC-GPU-1C — Linux Manual GPU Validation

Date: 2026-08-02  
Branch: `test`  
Validated commit: `618ef00 Promote safe source-managed GPU provisioning`

## 1. Objective

Validate the complete source-managed GPU provisioning flow on a real Linux
NVIDIA workstation, without using:

```text
ZESOLVER_ALLOW_GPU_PROVISIONING
ZESOLVER_DISABLE_GPU_PROVISIONING
````

The validation covers:

* automatic detection of a safe virtual environment;
* guided CuPy installation from the startup wizard;
* pip integrity check;
* fresh-process CUDA self-test;
* real ZeNear batch execution on CUDA;
* CPU fallback preservation.

## 2. Test Environment

```text
OS: Linux x86_64
Python: 3.13.5
Virtual environment: ZeSolver/.venv
GPU: NVIDIA GeForce MX150
NVIDIA driver: 550.163.01
CuPy installed profile: cupy-cuda12x[ctk]
CuPy version: 14.1.1
```

## 3. Initial Clean State

All CuPy and Python CUDA runtime packages were removed from the ZeSolver
virtual environment.

Verification:

```text
pip GPU package inventory: empty
pip check: No broken requirements found.
```

No provisioning override was active:

```bash
unset ZESOLVER_ALLOW_GPU_PROVISIONING
unset ZESOLVER_DISABLE_GPU_PROVISIONING
```

## 4. Automatic Runtime Detection

Command:

```bash
.venv/bin/python -m zesolver.gpu_diagnostic \
  --json \
  --self-test \
  --show-install-plan
```

Observed result before installation:

```text
distribution_kind=source_managed
effective_backend=cpu
reason_code=CUPY_NOT_INSTALLED
provisioning_available=true
environment_reason=Source-managed virtual environment detected.
gpu_temp_dir=/home/tristan/.cache/zesolver/gpu-tmp
```

The proposed command targeted the active interpreter exactly:

```text
/home/tristan/.openclaw/workspace/projects/ZeSolver/.venv/bin/python
-m pip install
cupy-cuda12x[ctk]
```

No manual environment override was required.

## 5. Wizard Installation

ZeSolver was launched normally from the project virtual environment.

The startup wizard:

* enabled the GPU installation button automatically;
* requested explicit user confirmation;
* displayed progressive installation feedback;
* used `shell=false`;
* installed `cupy-cuda12x[ctk]`;
* ran `pip check`;
* ran a fresh-process CUDA self-test.

Observed process results:

```text
GPU provisioning pip install: returncode=0
pip check: returncode=0
fresh CUDA self-test: returncode=0
```

No Qt crash occurred.

The previous failure was not reproduced:

```text
QThread: Destroyed while thread is still running
Abandon (core dumped)
```

## 6. Installed Runtime

Installed packages included:

```text
cupy-cuda12x==14.1.1
nvidia-cublas-cu12==12.9.2.10
nvidia-cuda-nvrtc-cu12==12.9.86
nvidia-cuda-runtime-cu12==12.9.79
nvidia-cufft-cu12==11.4.1.4
nvidia-curand-cu12==10.3.10.19
nvidia-cusolver-cu12==11.7.5.82
nvidia-cusparse-cu12==12.5.10.65
nvidia-nvjitlink-cu12==12.9.86
```

## 7. Real CUDA Batch Validation

A four-image ZeNear batch was executed with detection backend set to `auto`.

Observed activation:

```text
ZeNear detection active:
requested=auto
selected=cuda
used=cuda
device=0
```

Final telemetry:

```text
cuda_images=4
cpu_images=0
fallbacks=0
gpu_errors=0
gpu_oom=0
device=0
terminal=completed
```

Results:

```text
4 images processed
4 images solved
0 CPU detections
0 GPU fallbacks
0 GPU errors
0 GPU out-of-memory events
```

## 8. Validated Product Behavior

The following behavior is now manually confirmed:

* a safe ZeSolver virtual environment is detected automatically;
* no hidden provisioning variable is required;
* installation remains opt-in;
* pip targets only the active virtual environment;
* the ZeSolver-owned temporary directory is used;
* installation feedback remains visible;
* the application stays alive during and after provisioning;
* a fresh CUDA self-test succeeds;
* ZeNear automatically selects and uses CUDA;
* CPU fallback remains available.

## 9. Residual Scope

Not validated by this Linux test:

* Windows source-managed GPU installation;
* frozen executable GPU packaging;
* macOS CUDA support;
* embedded ZeMosaic or ZeSeestarStacker provisioning.

## 10. Verdict

```text
RC_GPU1C_SOURCE_MANAGED_PROVISIONING_PROMOTED
GPU_SOURCE_INSTALLATION_VALIDATED_LINUX
GPU_RUNTIME_VALIDATED_LINUX_NVIDIA
GPU_AUTOMATIC_SELECTION_VALIDATED
GPU_CPU_FALLBACK_PRESERVED
FROZEN_GPU_PACKAGING_PENDING
WINDOWS_GPU_VALIDATION_PENDING
EMBEDDED_HOST_INTEGRATION_READY
```

