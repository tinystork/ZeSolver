from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from zesolver.core.batch import BatchSolverPipeline


def test_legacy_batch_solver_still_available() -> None:
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("zesolver_entrypoint_batch_compat", root / "zesolver.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert hasattr(module, "BatchSolver")
    assert BatchSolverPipeline is not module.BatchSolver
