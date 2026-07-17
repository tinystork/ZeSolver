from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from zeblindsolver.wcs_header import apply_wcs_solution_to_header
from zesolver.core.models import SolveRequest, SolveStatus
from zesolver.core.preflight import run_preflight

from solver_pipeline_fixtures import empty_resources, near_resources, sample_wcs


def test_preflight_rejects_invalid_fits(tmp_path: Path) -> None:
    path = tmp_path / "bad.fit"
    path.write_text("not fits", encoding="utf-8")

    result = run_preflight(SolveRequest(path, None, True), catalog_resources=near_resources(tmp_path))

    assert result.ok is False
    assert result.status is SolveStatus.INVALID_INPUT
    assert "fits_unreadable" in str(result.error)


def test_preflight_rejects_existing_wcs_when_overwrite_forbidden(tmp_path: Path) -> None:
    path = tmp_path / "with-wcs.fit"
    hdu = fits.PrimaryHDU(data=np.ones((16, 16), dtype=np.uint16))
    apply_wcs_solution_to_header(hdu.header, sample_wcs())
    hdu.writeto(path)

    result = run_preflight(SolveRequest(path, None, False), catalog_resources=near_resources(tmp_path))

    assert result.ok is False
    assert result.status is SolveStatus.INVALID_INPUT
    assert result.error == "existing_wcs_overwrite_forbidden"
    assert result.has_existing_wcs is True


def test_preflight_reports_catalog_unavailable(tmp_path: Path) -> None:
    path = tmp_path / "frame.fit"
    fits.PrimaryHDU(data=np.ones((8, 8), dtype=np.uint16)).writeto(path)

    result = run_preflight(SolveRequest(path, None, True), catalog_resources=empty_resources())

    assert result.ok is False
    assert result.status is SolveStatus.CATALOG_UNAVAILABLE
    assert result.error == "catalog_resources_unavailable"


def test_preflight_accepts_partial_blind_resources(tmp_path: Path) -> None:
    path = tmp_path / "frame.fit"
    fits.PrimaryHDU(data=np.ones((8, 8), dtype=np.uint16)).writeto(path)

    result = run_preflight(SolveRequest(path, None, True), catalog_resources=near_resources(tmp_path, blind_count=6))

    assert result.ok is True
    assert result.image_shape == (8, 8)
    assert "blind4d_coverage_not_all_sky" in result.warnings
