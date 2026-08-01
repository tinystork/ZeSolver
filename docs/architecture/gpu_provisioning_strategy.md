# GPU Provisioning Strategy

ZeSolver treats CUDA/CuPy support as optional acceleration for ZeNear star
detection. CPU operation is always the safe baseline.

## Layers

1. Detection: `zesolver.gpu_support.probe` inspects platform, architecture,
   installed CuPy packages, NVIDIA visibility through `nvidia-smi`, and, when
   CuPy imports, a real CUDA allocation/calculation self-test.
2. Policy: `zesolver.gpu_support.policy` decides whether a provisioning plan is
   allowed for the current runtime context.
3. Provisioning: `zesolver.gpu_support.provisioning` executes or delegates a
   plan. GUI code only consumes these services.

## Runtime Contexts

- `SOURCE_MANAGED`: diagnostic plus guided package installation, only when the
  runtime is a provable virtual environment: not frozen, not embedded,
  `sys.prefix != sys.base_prefix`, and `sys.executable` is inside the active
  virtual environment.
- `FROZEN_STANDALONE`: diagnostic and guidance only. No `pip install` from a
  bundled executable.
- `EMBEDDED_HOST`: diagnostic and plan generation only unless the host supplies
  a provisioning callback.
- `UNKNOWN`: diagnostic only.

The wizard detects safe source-managed virtual environments automatically.
`ZESOLVER_ALLOW_GPU_PROVISIONING=1` remains an advanced/testing override, but it
does not make a system Python mutable. `ZESOLVER_DISABLE_GPU_PROVISIONING=1`
wins over every other signal and forces diagnostic-only behavior.

Provisioning commands always target exactly the active `sys.executable`. Pip
subprocesses receive a ZeSolver GPU temporary directory through `TMPDIR`, `TMP`
and `TEMP`; the parent process environment is not modified.

## Package Policy

The current candidate profile is `cupy-cuda12x[ctk]` for supported Linux/Windows
Python runtimes. It is an allowlisted package requirement, never derived from free text.
ZeSolver does not uninstall packages, install drivers, invoke system package
managers, or install multiple CuPy variants.

The profile remains a candidate until real Linux and Windows GPU installations
are validated on the target beta environments. A local throwaway-venv probe
showed `cupy-cuda12x` alone can fail the self-test when CUDA headers are absent,
so the candidate includes CuPy's CTK extra instead of relying on a system CUDA
Toolkit.

## Host Contract

Future ZeMosaic and ZeSeestarStacker integration should call:

```python
report = probe_gpu_capability(runtime_context)
plan = build_gpu_provisioning_plan(report, runtime_context)
```

The host remains responsible for UI, consent, restart behavior, settings, and
any provisioning callback. The diagnostic package imports no Qt and assumes no
ownership of a `QApplication`.

## Batch Runtime

When ZeNear `auto` detects a permanent missing GPU dependency such as
`CUPY_NOT_INSTALLED`, the batch disables CUDA once and routes subsequent images
directly to CPU. Transient CUDA failures keep the existing fallback path.
