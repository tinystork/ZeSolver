from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from zeblindsolver.metadata_solver import validate_strict_astap_iso_candidate
from zeblindsolver.profiles import HISTORICAL_PROFILE, ZEBLIND_4D_EXPERIMENTAL_PROFILE


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_zesolver_entrypoint():
    path = REPO_ROOT / "zesolver.py"
    spec = importlib.util.spec_from_file_location("zesolver_entrypoint_zn35_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _wcs(*, ra: float = 10.0, dec: float = 40.0, scale_arcsec: float = 2.4, singular: bool = False) -> WCS:
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [50.0, 50.0]
    s = scale_arcsec / 3600.0
    if singular:
        w.wcs.cd = np.asarray([[s, 0.0], [s, 0.0]], dtype=float)
    else:
        w.wcs.cd = np.asarray([[-s, 0.0], [0.0, s]], dtype=float)
    return w


def _matches(w: WCS, *, degenerate: bool = False) -> np.ndarray:
    if degenerate:
        pts = np.asarray([[20.0 + i, 50.0] for i in range(18)], dtype=float)
    else:
        xs = np.linspace(15.0, 85.0, 5)
        ys = np.linspace(15.0, 85.0, 4)
        pts = np.asarray([[x, y] for y in ys for x in xs], dtype=float)
    world = np.asarray(w.wcs_pix2world(pts, 0), dtype=float)
    return np.column_stack([pts, world[:, :2]])


def test_zn35_acceptance_gate_accepts_runtime_consistent_wcs() -> None:
    w = _wcs()
    diag = validate_strict_astap_iso_candidate(
        w,
        _matches(w),
        width=100,
        height=100,
        ra_hint_deg=10.0,
        dec_hint_deg=40.0,
        search_radius_deg=1.0,
        approx_fov_deg=0.12,
        approx_scale_arcsec=2.4,
        pixel_tolerance=2.5,
        iso_refs=6,
    )

    assert diag["decision"] == "ACCEPT"
    assert diag["reason"] == "STRICT_ACCEPTANCE_ACCEPTED"
    assert diag["unique_matches"] >= 12
    assert diag["holdout_matches"] >= 8


def test_zn35_acceptance_gate_rejects_wrong_center_bad_cd_and_degenerate_support() -> None:
    ok = _wcs()
    wrong_center = validate_strict_astap_iso_candidate(
        _wcs(ra=120.0),
        _matches(_wcs(ra=120.0)),
        width=100,
        height=100,
        ra_hint_deg=10.0,
        dec_hint_deg=40.0,
        search_radius_deg=0.5,
        approx_fov_deg=0.12,
        approx_scale_arcsec=2.4,
        pixel_tolerance=2.5,
        iso_refs=0,
    )
    bad_cd = validate_strict_astap_iso_candidate(
        _wcs(singular=True),
        _matches(_wcs()),
        width=100,
        height=100,
        ra_hint_deg=10.0,
        dec_hint_deg=40.0,
        search_radius_deg=1.0,
        approx_fov_deg=0.12,
        approx_scale_arcsec=2.4,
        pixel_tolerance=2.5,
        iso_refs=0,
    )
    poor_support = validate_strict_astap_iso_candidate(
        ok,
        _matches(ok, degenerate=True),
        width=100,
        height=100,
        ra_hint_deg=10.0,
        dec_hint_deg=40.0,
        search_radius_deg=1.0,
        approx_fov_deg=0.12,
        approx_scale_arcsec=2.4,
        pixel_tolerance=2.5,
        iso_refs=0,
    )

    assert wrong_center["reason"] == "STRICT_ACCEPTANCE_CENTER"
    assert bad_cd["reason"] == "STRICT_ACCEPTANCE_BAD_CD"
    assert poor_support["reason"] == "STRICT_ACCEPTANCE_POOR_SPATIAL_COVERAGE"


def test_zn35_near_wrapper_never_falls_back_to_historical(monkeypatch, tmp_path: Path) -> None:
    from zesolver import zeblindsolver as near_wrapper

    fits_path = tmp_path / "frame.fit"
    fits.PrimaryHDU(data=np.zeros((16, 16), dtype=np.float32)).writeto(fits_path)

    def fake_near(*args, **kwargs):
        return SimpleNamespace(success=False, message="forced near failure", tile_key=None, header_updates={}, stats={})

    def forbidden_blind(*args, **kwargs):  # pragma: no cover - should never execute
        raise AssertionError("historical blind backend was called")

    monkeypatch.setattr(near_wrapper, "_internal_solve_near", fake_near)
    monkeypatch.setattr(near_wrapper, "_internal_solve_blind", forbidden_blind)

    result = near_wrapper.near_solve(str(fits_path), str(tmp_path), fallback_to_blind=True, skip_if_valid=False, log=lambda _: None)

    assert result["success"] is False
    assert result["wrote_wcs"] is False
    assert result["stats"]["historical_blind_called"] is False
    assert result["stats"]["final_status"] == "BLIND4D_CONFIGURATION_REQUIRED"


def test_zn35_app_blind_runner_blocks_historical_profile_in_source() -> None:
    source = (REPO_ROOT / "zesolver.py").read_text(encoding="utf-8")
    start = source.index("    def _run_blind_solver(")
    end = source.index("    def _run_blind_on_raster", start)
    block = source[start:end]

    assert "legacy blind backend profile" in block
    assert "BLIND4D_CONFIGURATION_REQUIRED" in block
    assert "\"historical_blind_called\": False" in block
    assert "index_root = self.config.blind_index_path" not in block


def test_zn35_reports_keep_historical_out_of_chain() -> None:
    path = REPO_ROOT / "reports" / "zenear_zn35_final_chain_matrix.json"
    if not path.exists():
        pytest.skip("ZN3.5 reports not generated")
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["appels historiques observés"] == 0
    assert data["faux WCS Near acceptés"] == 0
    assert data["faux WCS 4D acceptés"] == 0
