import math
import pytest
from zeblindsolver import presets as P


def test_compute_scale_and_fov_basic():
    geo = P.compute_scale_and_fov(
        focal_mm=2800.0, pixel_um=3.76, res_w=3008, res_h=3008, reducer=0.63, binning=1
    )
    assert geo["eff_focal_mm"] == pytest.approx(1764.0, rel=1e-6)
    # Typical scale for C11 0.63x + 3.76um should be about 0.44"/px
    assert geo["scale_arcsec_per_px"] == pytest.approx(0.44, rel=0.1)
    assert geo["fov_w_deg"] > 0
    assert geo["fov_h_deg"] > 0
    assert geo["fov_diag_deg"] > 0


def test_recommend_params_transitions():
    # Wide/undersampled
    r1 = P.recommend_params(scale_arcsec_per_px=4.0, fov_diag_deg=5.0)
    assert r1["mag_cap"] <= 14.5
    # Moderate
    r2 = P.recommend_params(scale_arcsec_per_px=2.0, fov_diag_deg=1.5)
    assert 14.0 <= r2["mag_cap"] <= 16.0
    # Narrow/oversampled
    r3 = P.recommend_params(scale_arcsec_per_px=0.6, fov_diag_deg=0.5)
    assert r3["mag_cap"] >= 16.0
