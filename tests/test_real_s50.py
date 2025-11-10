from __future__ import annotations

import os
from pathlib import Path

import pytest
from astropy.io import fits

from zeblindsolver.zeblindsolver import SolveConfig, solve_blind


@pytest.mark.skipif(
    "ZEBLIND_S50_INDEX" not in os.environ or "ZEBLIND_S50_FRAME" not in os.environ,
    reason="S50 index or frame not configured",
)
def test_real_s50_index_success(tmp_path):
    index_root = Path(os.environ["ZEBLIND_S50_INDEX"])
    frame_path = Path(os.environ["ZEBLIND_S50_FRAME"])
    solved = tmp_path / "s50_solved.fits"
    fits.PrimaryHDU(data=fits.getdata(frame_path, allow_pickle=False)).writeto(solved)

    config = SolveConfig(
        max_candidates=12,
        max_stars=800,
        max_quads=12000,
        sip_order=3,
        quality_rms=1.0,
        quality_inliers=60,
    )
    solution = solve_blind(str(solved), index_root, config=config)
    assert solution.success
    assert solution.stats["inliers"] >= 60
    header = fits.getheader(solved)
    assert header["QUALITY"] == "GOOD"
