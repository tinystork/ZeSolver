"""Runtime context detection for optional GPU provisioning."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Mapping

from .models import DistributionKind, GpuRuntimeContext


TRUE_VALUES = {"1", "true", "yes", "on"}


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES


def _path_inside(child: str | Path | None, parent: str | Path | None) -> bool:
    if child is None or parent is None:
        return False
    try:
        child_path = Path(child).expanduser().absolute()
        parent_path = Path(parent).expanduser().absolute()
        child_path.relative_to(parent_path)
        return True
    except Exception:
        return False


def default_gpu_temp_dir(
    *,
    platform_name: str | None = None,
    home: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    values = dict(os.environ if env is None else env)
    system = str(platform_name or platform.system() or sys.platform).lower()
    if system.startswith("win"):
        home_path = Path(home or values.get("USERPROFILE") or Path.home()).expanduser()
        base = values.get("LOCALAPPDATA")
        return (Path(base).expanduser() if base else home_path / "AppData" / "Local") / "ZeSolver" / "gpu-tmp"
    if system in {"darwin", "macos"}:
        home_path = Path(home or values.get("HOME") or Path.home()).expanduser()
        return home_path / "Library" / "Caches" / "ZeSolver" / "gpu-tmp"
    home_path = Path(home or values.get("HOME") or Path.home()).expanduser()
    xdg = values.get("XDG_CACHE_HOME")
    return (Path(xdg).expanduser() if xdg else home_path / ".cache") / "zesolver" / "gpu-tmp"


def detect_gpu_runtime_context(
    *,
    env: Mapping[str, str] | None = None,
    executable: str | None = None,
    prefix: str | None = None,
    base_prefix: str | None = None,
    frozen: bool | None = None,
    embedded_host: bool | None = None,
    platform_name: str | None = None,
    home: str | Path | None = None,
) -> GpuRuntimeContext:
    values = dict(os.environ if env is None else env)
    exe = str(executable or sys.executable)
    active_prefix = str(prefix if prefix is not None else sys.prefix)
    system_prefix = str(base_prefix if base_prefix is not None else getattr(sys, "base_prefix", sys.prefix))
    is_frozen = bool(getattr(sys, "frozen", False) if frozen is None else frozen)
    is_embedded = bool(_truthy(values.get("ZESOLVER_EMBEDDED_HOST")) if embedded_host is None else embedded_host)
    temp_dir = str(default_gpu_temp_dir(platform_name=platform_name, home=home, env=values))

    if _truthy(values.get("ZESOLVER_DISABLE_GPU_PROVISIONING")):
        return GpuRuntimeContext(
            DistributionKind.UNKNOWN,
            allow_environment_mutation=False,
            python_executable=exe,
            environment_reason="GPU provisioning disabled by ZESOLVER_DISABLE_GPU_PROVISIONING.",
            gpu_temp_dir=temp_dir,
        )
    if is_frozen:
        return GpuRuntimeContext(
            DistributionKind.FROZEN_STANDALONE,
            allow_environment_mutation=False,
            python_executable=exe,
            environment_reason="Frozen standalone runtime; ZeSolver will not run pip from the application.",
            gpu_temp_dir=temp_dir,
        )
    if is_embedded:
        return GpuRuntimeContext(
            DistributionKind.EMBEDDED_HOST,
            allow_environment_mutation=False,
            python_executable=exe,
            host_can_provision=False,
            environment_reason="Embedded host runtime; the host application owns GPU provisioning.",
            gpu_temp_dir=temp_dir,
        )

    in_virtualenv = bool(active_prefix and system_prefix and active_prefix != system_prefix)
    executable_in_env = _path_inside(exe, active_prefix)
    override_allow = _truthy(values.get("ZESOLVER_ALLOW_GPU_PROVISIONING"))
    if in_virtualenv and executable_in_env:
        reason = (
            "Source-managed virtual environment detected by ZESOLVER_ALLOW_GPU_PROVISIONING."
            if override_allow
            else "Source-managed virtual environment detected."
        )
        return GpuRuntimeContext(
            DistributionKind.SOURCE_MANAGED,
            allow_environment_mutation=True,
            python_executable=exe,
            environment_reason=reason,
            gpu_temp_dir=temp_dir,
        )

    reason = (
        "ZESOLVER_ALLOW_GPU_PROVISIONING was set, but this interpreter is not a provable virtual environment."
        if override_allow
        else "Python system or unproven environment; GPU provisioning is disabled."
    )
    return GpuRuntimeContext(
        DistributionKind.UNKNOWN,
        allow_environment_mutation=False,
        python_executable=exe,
        environment_reason=reason,
        gpu_temp_dir=temp_dir,
    )

