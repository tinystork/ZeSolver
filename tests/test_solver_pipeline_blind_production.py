from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from zeblindsolver.wcs_header import apply_wcs_solution_to_header
from zesolver.core import EngineSolveResult, SolveRequest, SolveStatus, SolverPipeline
from zesolver.core.blind_port import ProductionBlindSolverPort
from zesolver.settings import ProductSettings, RuntimeOptions

from solver_pipeline_fixtures import near_resources, sample_wcs


class FailingNear:
    calls = 0

    def solve(self, request, *, resources, configuration):
        self.calls += 1
        return EngineSolveResult(status=SolveStatus.UNSOLVED, backend="NEAR", error="near failed")


def _frame(path: Path) -> Path:
    fits.PrimaryHDU(data=np.arange(64, dtype=np.uint16).reshape(8, 8)).writeto(path)
    return path


def _pixel_bytes(path: Path) -> bytes:
    with fits.open(path, memmap=False) as hdul:
        return np.ascontiguousarray(hdul[0].data).tobytes()


def test_pipeline_uses_production_blind_port_and_writes_final_wcs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = _frame(tmp_path / "input.fit")
    before = _pixel_bytes(source)
    calls = {"blind": 0}

    def fake_solve_blind(self, request, *, resources, configuration):
        calls["blind"] += 1
        return EngineSolveResult(
            status=SolveStatus.SOLVED,
            backend="BLIND4D",
            wcs=sample_wcs(),
            wcs_written=False,
            inliers=47,
            rms_px=1.106,
            raw={"header_updates": {"SOLVED": 1, "SOLVER": "ZeSolver", "SOLVMODE": "BLIND4D"}},
        )

    monkeypatch.setattr(ProductionBlindSolverPort, "solve_blind", fake_solve_blind)

    pipeline = SolverPipeline(
        product_settings=ProductSettings(blind_enabled=True, web_fallback=False),
        runtime_options=RuntimeOptions(),
        catalog_resources=near_resources(tmp_path, blind_count=6),
        near_solver=FailingNear(),
    )
    result = pipeline.solve(SolveRequest(source, tmp_path / "out.fit", True, request_id="p2b2a"))

    assert result.status is SolveStatus.SOLVED
    assert result.backend == "BLIND4D"
    assert result.wcs_written is True
    assert result.inliers == 47
    assert calls["blind"] == 1
    assert _pixel_bytes(source) == before
    with fits.open(tmp_path / "out.fit", memmap=False) as hdul:
        assert bool(WCS(hdul[0].header, naxis=2, relax=True).has_celestial)
        assert hdul[0].header["SOLVMODE"] == "BLIND4D"


def test_legacy_image_solver_blind_path_still_available() -> None:
    import importlib.util
    import sys

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("zesolver_entrypoint_legacy_blind_available", root / "zesolver.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert hasattr(module.ImageSolver, "_run_blind_solver")
    assert hasattr(module.ImageSolver, "solve_path_blind_only")
