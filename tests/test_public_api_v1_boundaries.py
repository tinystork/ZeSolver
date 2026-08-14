"""Import-boundary and public-surface audit for ``zesolver.api.v1``.

These tests run in a fresh subprocess so that module state from earlier imports
cannot mask a heavy import that the public API must not trigger.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_EXPECTED_PUBLIC = {
    # version
    "API_VERSION",
    "API_MAJOR",
    "API_MINOR",
    # metadata / probe
    "ApiInfo",
    "get_api_info",
    "RuntimeProbe",
    "probe",
    # capabilities
    "CapabilityAvailability",
    "CapabilityUnavailableReason",
    "CapabilityState",
    # policies
    "BackendPolicy",
    "GpuPolicy",
    "NetworkPolicy",
    "WritePolicy",
    # solve models
    "SolveHints",
    "SolveOptions",
    "SolveRequest",
    "CanonicalWcsHeader",
    "SolveStatus",
    "FailureCode",
    "SolveResult",
    # progress / cancellation
    "ProgressPhase",
    "ProgressEvent",
    "CancellationToken",
    # errors
    "ZeSolverApiError",
    "SolverClosedError",
    "InvalidRequestError",
    # lifecycle
    "create_solver_runtime",
    "SolverRuntime",
    "SolverSession",
}

_FORBIDDEN_PUBLIC = {
    "SolverPipeline",
    "ProductSettings",
    "RuntimeOptions",
    "SolverCatalogResources",
    "SolverPipelinePort",
    "ProductionBlindSolverPort",
    "BlindPort",
    "Port",
    "profiles",
    "TerminalReasonCode",
    "batch",
    "BatchRequest",
    "SolveBatch",
    "NO_WRITE",
}


def _run_script(script: str) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=env,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"subprocess failed ({result.returncode}):\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return json.loads(result.stdout)


def test_public_import_is_lightweight() -> None:
    script = r"""
import builtins
import json
import sys

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "cupy" or name.startswith("cupy."):
        raise RuntimeError("cupy import blocked")
    if name == "PySide6" or name.startswith("PySide6.") or name.startswith("PyQt"):
        raise RuntimeError("Qt import blocked")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

import zesolver.api.v1 as v1

payload = {
    "api_version": v1.API_VERSION,
    "cupy_loaded": any(m == "cupy" or m.startswith("cupy.") for m in sys.modules),
    "qt_loaded": any(m.startswith("PySide6") or m.startswith("PyQt") for m in sys.modules),
    "gui_pipeline_loaded": any(m.startswith("zesolver.gui_pipeline") for m in sys.modules),
    "engine_loaded": "zesolver.zeblindsolver" in sys.modules,
    "catalog_loaded": "zesolver.catalog_resources" in sys.modules,
}
print(json.dumps(payload, sort_keys=True))
"""
    payload = _run_script(script)
    assert payload["api_version"] == "1.0"
    assert payload["cupy_loaded"] is False
    assert payload["qt_loaded"] is False
    assert payload["gui_pipeline_loaded"] is False
    assert payload["engine_loaded"] is False
    assert payload["catalog_loaded"] is False


def test_public_surface_is_exact() -> None:
    script = r"""
import json
import zesolver.api.v1 as v1
print(json.dumps({"all": sorted(v1.__all__)}))
"""
    payload = _run_script(script)
    assert set(payload["all"]) == _EXPECTED_PUBLIC


def test_forbidden_symbols_absent_from_public_surface() -> None:
    script = r"""
import json
import zesolver.api.v1 as v1
import zesolver.api.v1.models as models
forbidden = [
    "SolverPipeline", "ProductSettings", "RuntimeOptions", "SolverCatalogResources",
    "TerminalReasonCode", "batch", "NO_WRITE", "profiles", "Port",
]
result = {}
for name in forbidden:
    result[name] = {
        "in_all": name in v1.__all__,
        "on_v1": hasattr(v1, name),
        "on_models": hasattr(models, name),
    }
print(json.dumps(result, sort_keys=True))
"""
    payload = _run_script(script)
    for name, state in payload.items():
        assert state["in_all"] is False, name
        assert state["on_v1"] is False, name
        assert state["on_models"] is False, name


def test_importing_zesolver_root_does_not_load_engine() -> None:
    script = r"""
import json
import sys
import zesolver
payload = {
    "engine_loaded": "zesolver.zeblindsolver" in sys.modules,
    "cupy_loaded": any(m == "cupy" or m.startswith("cupy.") for m in sys.modules),
    "version": zesolver.__version__,
}
print(json.dumps(payload, sort_keys=True))
"""
    payload = _run_script(script)
    assert payload["engine_loaded"] is False
    assert payload["cupy_loaded"] is False
    assert payload["version"]


def test_historical_reexports_still_work_on_access() -> None:
    script = r"""
import json
import zesolver
# Triggering a historical re-export must import the engine lazily.
fn = zesolver.has_valid_wcs
print(json.dumps({"callable": callable(fn)}))
"""
    payload = _run_script(script)
    assert payload["callable"] is True


@pytest.mark.parametrize(
    "module",
    [
        "zesolver.api.v1.errors",
        "zesolver.api.v1.models",
        "zesolver.api.v1.cancellation",
        "zesolver.api.v1.probe",
    ],
)
def test_individual_public_modules_are_lightweight(module: str) -> None:
    script = f"""
import builtins
import json
import sys
real_import = builtins.__import__
def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "cupy" or name.startswith("cupy."):
        raise RuntimeError("cupy blocked")
    if name.startswith("PySide6") or name.startswith("PyQt"):
        raise RuntimeError("Qt blocked")
    return real_import(name, globals, locals, fromlist, level)
builtins.__import__ = guarded_import
import {module}
print(json.dumps({{"ok": True}}))
"""
    payload = _run_script(script)
    assert payload["ok"] is True
