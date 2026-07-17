from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

from zesolver.core import EngineSolveResult, SolveRequest, SolveStatus, SolverPipeline
from zesolver.settings import ProductSettings, RuntimeOptions

from solver_pipeline_fixtures import near_resources, sample_wcs


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("zesolver_app_pipeline_compat", ROOT / "zesolver.py")
assert SPEC is not None and SPEC.loader is not None
zesolver = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = zesolver
SPEC.loader.exec_module(zesolver)


class NearSuccessPort:
    calls = 0

    def solve(self, request, *, resources, configuration):
        self.calls += 1
        return EngineSolveResult(status=SolveStatus.SOLVED, backend="NEAR", wcs=sample_wcs(), inliers=39, rms_px=0.2)


def test_solver_pipeline_uses_p2a_profile_ids_and_catalog_resources(tmp_path: Path) -> None:
    path = tmp_path / "frame.fit"
    fits.PrimaryHDU(data=np.ones((8, 8), dtype=np.uint16)).writeto(path)
    port = NearSuccessPort()
    pipeline = SolverPipeline(
        product_settings=ProductSettings(catalog_library_path=None),
        runtime_options=RuntimeOptions(worker_count_resolved=1),
        near_profile="zenear-v1",
        blind_profile="zeblind4d-v1",
        pipeline_profile="pipeline-v1",
        catalog_resources=near_resources(tmp_path, blind_count=6),
        near_solver=port,
    )

    result = pipeline.solve(SolveRequest(path, tmp_path / "out.fit", True))

    assert result.status is SolveStatus.SOLVED
    assert result.profile_ids == {"near": "zenear-v1", "blind": "zeblind4d-v1", "pipeline": "pipeline-v1"}
    assert result.catalog_status == "legacy"
    assert pipeline.last_telemetry["catalog_source"] == "legacy"
    assert pipeline.last_telemetry["catalog_coverage_fraction"] == 6 / 1476.0


def test_legacy_image_solver_near_entrypoint_remains_available() -> None:
    assert hasattr(zesolver.ImageSolver, "_run_index_near_solver")
    assert hasattr(zesolver.ImageSolver, "_run_blind_solver")
    assert hasattr(zesolver.ImageSolver, "_write_fits_solution")
