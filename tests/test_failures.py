from __future__ import annotations

import time
import numpy as np
from astropy.io import fits

import zeblindsolver.zeblindsolver as solver_module
from zeblindsolver.zeblindsolver import SolveConfig, solve_blind


def _build_star_array(positions: np.ndarray, mags: np.ndarray) -> np.ndarray:
    stars = np.zeros(positions.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("flux", "f4"), ("fwhm", "f4")])
    stars["x"] = positions[:, 0]
    stars["y"] = positions[:, 1]
    stars["flux"] = -mags
    stars["fwhm"] = 2.0
    return stars


def test_solver_returns_fail_quickly(tmp_path, synthetic_index, synthetic_star_catalog, monkeypatch):
    positions, mags = synthetic_star_catalog
    fits_path = tmp_path / "failure.fits"
    fits.PrimaryHDU(data=np.zeros((32, 32), dtype=np.float32)).writeto(fits_path)

    monkeypatch.setattr(solver_module, "read_fits_as_luma", lambda path: np.zeros((32, 32), dtype=np.float32))
    monkeypatch.setattr(solver_module, "remove_background", lambda img: img)
    monkeypatch.setattr(solver_module, "build_pyramid", lambda img: [img])
    monkeypatch.setattr(solver_module, "detect_stars", lambda img, **kwargs: _build_star_array(positions, mags))

    config = SolveConfig(
        max_candidates=4,
        max_stars=len(positions),
        max_quads=200,
        sip_order=2,
        quality_rms=1e-3,
        quality_inliers=100,
    )
    start = time.perf_counter()
    solution = solve_blind(str(fits_path), synthetic_index, config=config)
    elapsed = time.perf_counter() - start

    assert not solution.success
    assert elapsed < 3.0
