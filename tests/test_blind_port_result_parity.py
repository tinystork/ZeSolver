from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from zeblindsolver.wcs_header import apply_wcs_solution_to_header
from zesolver.core.blind_result_adapter import engine_result_from_blind_result
from zesolver.core.models import SolveStatus

from solver_pipeline_fixtures import sample_wcs


def _solved_frame(path: Path) -> Path:
    fits.PrimaryHDU(data=np.ones((8, 8), dtype=np.uint16)).writeto(path)
    with fits.open(path, mode="update", memmap=False) as hdul:
        apply_wcs_solution_to_header(
            hdul[0].header,
            sample_wcs(),
            header_updates={"SOLVED": 1, "SOLVER": "ZeSolver"},
        )
        hdul.flush()
    return path


def test_blind_result_adapter_preserves_engine_fields(tmp_path: Path) -> None:
    result = engine_result_from_blind_result(
        {
            "success": True,
            "message": "ZeBlind 4D - d50_2822",
            "elapsed_sec": 1.0,
            "tried_dbs": [str(tmp_path)],
            "used_db": "d50_2822",
            "wrote_wcs": True,
            "updated_keywords": {"SOLVED": 1, "SOLVER": "ZeSolver"},
            "output_path": str(tmp_path / "solved.fit"),
            "stats": {
                "astrometry_4d_runtime_accepted": True,
                "astrometry_4d_best_accepted_validation": {"inliers": 47, "rms_px": 1.106},
                "astrometry_4d_selected_origin_tile_key": "d50_2822",
            },
        },
        solved_path=_solved_frame(tmp_path / "solved.fit"),
    )

    assert result.status is SolveStatus.SOLVED
    assert result.backend == "BLIND4D"
    assert result.wcs is not None
    assert result.wcs_written is False
    assert result.inliers == 47
    assert result.rms_px == pytest.approx(1.106)
    assert result.raw["used_db"] == "d50_2822"
    assert result.raw["header_updates"]["SOLVER"] == "ZeSolver"


def test_blind_result_adapter_normalizes_failure(tmp_path: Path) -> None:
    path = tmp_path / "unsolved.fit"
    fits.PrimaryHDU(data=np.ones((8, 8), dtype=np.uint16)).writeto(path)

    result = engine_result_from_blind_result(
        {
            "success": False,
            "message": "no valid solution",
            "elapsed_sec": 0.2,
            "tried_dbs": [str(tmp_path)],
            "used_db": None,
            "wrote_wcs": False,
            "updated_keywords": {},
            "output_path": str(path),
            "stats": {"astrometry_4d_runtime_enabled": True},
        },
        solved_path=path,
    )

    assert result.status is SolveStatus.UNSOLVED
    assert result.backend == "BLIND4D"
    assert result.error == "no valid solution"
    assert result.wcs is None
