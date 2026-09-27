from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from zeblindsolver.wcs_header import apply_wcs_solution_to_header
from zesolver.core import EngineSolveResult, SolveRequest, SolveStatus, SolverPipeline
from zesolver.core.preflight import run_preflight
from zesolver.core.wcs_io import pixel_fingerprint, write_wcs_safely
from zesolver.settings import ProductSettings, RuntimeOptions

from solver_pipeline_fixtures import near_resources, sample_wcs


class FakePort:
    def __init__(self, *results: EngineSolveResult) -> None:
        self.results = list(results)
        self.calls = 0

    def solve(self, request, *, resources, configuration):
        self.calls += 1
        if not self.results:
            return EngineSolveResult(status=SolveStatus.UNSOLVED, backend="FAKE", error="no fake result")
        return self.results.pop(0)


_CALPROV_JSON = b'{"calibration":"ZeCalibrator","version":1,"sources":3}'


def _calprov_data() -> np.ndarray:
    return np.frombuffer(_CALPROV_JSON, dtype=np.uint8)


def _ze_calibrator_hdul(*, with_wcs: bool = False) -> fits.HDUList:
    """ZeCalibrator-shaped FITS: PRIMARY 2D science, DQ 2D, CALPROV 1D.

    Built WITHOUT importing ZeCalibrator.
    """
    primary = fits.PrimaryHDU(data=np.zeros((16, 16), dtype=np.float32))
    if with_wcs:
        apply_wcs_solution_to_header(primary.header, sample_wcs())
    dq = fits.ImageHDU(data=np.zeros((16, 16), dtype=np.uint16), name="DQ")
    calprov = fits.ImageHDU(data=_calprov_data(), name="CALPROV")
    return fits.HDUList([primary, dq, calprov])


def _write(path: Path, hdul: fits.HDUList) -> Path:
    hdul.writeto(path, overwrite=True)
    return path


# --- TEST A: preflight accepts ZeCalibrator-shaped FITS ---


def test_preflight_accepts_ze_calibrator_shape(tmp_path: Path) -> None:
    path = _write(tmp_path / "zc.fit", _ze_calibrator_hdul())

    result = run_preflight(SolveRequest(path, None, True), catalog_resources=near_resources(tmp_path))

    assert result.ok is True
    assert result.status is None
    assert result.error is None
    assert result.has_existing_wcs is False
    assert result.image_shape == (16, 16)
    assert np.dtype(result.image_dtype).type is np.float32


def test_preflight_accepts_unrelated_1d_extension(tmp_path: Path) -> None:
    """A 1D aux extension under a non-CALPROV name still passes (no name hack)."""
    primary = fits.PrimaryHDU(data=np.zeros((16, 16), dtype=np.float32))
    meta = fits.ImageHDU(data=np.frombuffer(_CALPROV_JSON, dtype=np.uint8), name="META1")
    path = _write(tmp_path / "meta.fit", fits.HDUList([primary, meta]))

    result = run_preflight(SolveRequest(path, None, True), catalog_resources=near_resources(tmp_path))

    assert result.ok is True
    assert result.status is None
    assert result.error is None


def test_preflight_missing_file(tmp_path: Path) -> None:
    result = run_preflight(SolveRequest(tmp_path / "nope.fit", None, True))

    assert result.ok is False
    assert result.status is SolveStatus.INVALID_INPUT
    assert "input_missing" in str(result.error)


def test_preflight_invalid_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.fit"
    path.write_text("not fits", encoding="utf-8")

    result = run_preflight(SolveRequest(path, None, True))

    assert result.ok is False
    assert result.status is SolveStatus.INVALID_INPUT
    assert "fits_unreadable" in str(result.error)


def test_preflight_fits_without_image(tmp_path: Path) -> None:
    hdu = fits.PrimaryHDU()
    path = _write(tmp_path / "empty.fit", fits.HDUList([hdu]))

    result = run_preflight(SolveRequest(path, None, True))

    assert result.ok is False
    assert result.status is SolveStatus.INVALID_INPUT
    assert result.error == "fits_image_missing"


def test_preflight_invalid_science_dimensions(tmp_path: Path) -> None:
    hdu = fits.PrimaryHDU(data=np.zeros((16,), dtype=np.float32))
    path = _write(tmp_path / "oned.fit", fits.HDUList([hdu]))

    result = run_preflight(SolveRequest(path, None, True))

    assert result.ok is False
    assert result.status is SolveStatus.INVALID_INPUT
    assert "fits_invalid_dimensions" in str(result.error)


def test_preflight_single_hdu(tmp_path: Path) -> None:
    hdu = fits.PrimaryHDU(data=np.zeros((8, 8), dtype=np.uint16))
    path = _write(tmp_path / "single.fit", fits.HDUList([hdu]))

    result = run_preflight(SolveRequest(path, None, True), catalog_resources=near_resources(tmp_path))

    assert result.ok is True
    assert result.status is None
    assert result.image_shape == (8, 8)


# --- TEST B: existing celestial WCS on eligible HDU is still detected ---


def test_preflight_detects_existing_wcs_on_ze_calibrator_shape(tmp_path: Path) -> None:
    path = _write(tmp_path / "zc-wcs.fit", _ze_calibrator_hdul(with_wcs=True))

    result = run_preflight(SolveRequest(path, None, False), catalog_resources=near_resources(tmp_path))

    assert result.ok is False
    assert result.status is SolveStatus.INVALID_INPUT
    assert result.error == "existing_wcs_overwrite_forbidden"
    assert result.has_existing_wcs is True


# --- TEST C: safe WCS write preserves every HDU ---


def test_wcs_write_preserves_ze_calibrator_shape(tmp_path: Path) -> None:
    src = _write(tmp_path / "zc-src.fit", _ze_calibrator_hdul())
    dst = tmp_path / "zc-dst.fit"

    with fits.open(src, memmap=False) as hdul:
        before_fp = pixel_fingerprint(src)
        before_nhdus = len(hdul)
        before_extnames = [hdu.header.get("EXTNAME") for hdu in hdul]
        before_dtypes = [hdu.data.dtype.str if hdu.data is not None else None for hdu in hdul]
        before_shapes = [hdu.data.shape if hdu.data is not None else None for hdu in hdul]
        before_bytes = [hdu.data.tobytes() if hdu.data is not None else None for hdu in hdul]

    result = write_wcs_safely(input_path=src, output_path=dst, wcs=sample_wcs(), overwrite_wcs=True)

    assert result.ok is True
    assert result.wcs_written is True
    assert result.pixels_unchanged is True

    with fits.open(dst, memmap=False) as hdul:
        assert len(hdul) == before_nhdus == 3
        assert [hdu.header.get("EXTNAME") for hdu in hdul] == before_extnames == [None, "DQ", "CALPROV"]
        assert [hdu.data.dtype.str for hdu in hdul] == before_dtypes
        assert [hdu.data.shape for hdu in hdul] == before_shapes
        for hdu, before in zip(hdul, before_bytes):
            assert hdu.data.tobytes() == before
        # Celestial WCS landed on PRIMARY only.
        assert bool(WCS(hdul[0].header, naxis=2, relax=True).has_celestial)
        assert not bool(WCS(hdul[1].header, naxis=2, relax=True).has_celestial)

    assert pixel_fingerprint(dst) == before_fp
    # Source is untouched.
    assert not bool(WCS(fits.getheader(src), naxis=2, relax=True).has_celestial)


# --- TEST D: end-to-end pipeline over ZeCalibrator-shaped FITS ---


def test_pipeline_solves_ze_calibrator_shape_end_to_end(tmp_path: Path) -> None:
    src = _write(tmp_path / "zc-in.fit", _ze_calibrator_hdul())
    out = tmp_path / "zc-out.fit"

    with fits.open(src, memmap=False) as hdul:
        before_dq = hdul["DQ"].data.tobytes()
        before_calprov = hdul["CALPROV"].data.tobytes()

    pipeline = SolverPipeline(
        product_settings=ProductSettings(),
        runtime_options=RuntimeOptions(),
        catalog_resources=near_resources(tmp_path, blind_count=6),
        near_solver=FakePort(
            EngineSolveResult(status=SolveStatus.SOLVED, backend="NEAR", wcs=sample_wcs())
        ),
        blind_solver=FakePort(EngineSolveResult(status=SolveStatus.UNSOLVED, backend="BLIND4D")),
    )

    result = pipeline.solve(SolveRequest(src, out, True))

    assert result.status is SolveStatus.SOLVED
    assert result.wcs_written is True
    assert result.backend == "NEAR"

    with fits.open(out, memmap=False) as hdul:
        assert len(hdul) == 3
        assert bool(WCS(hdul[0].header, naxis=2, relax=True).has_celestial)
        assert hdul["DQ"].data.tobytes() == before_dq
        assert hdul["CALPROV"].data.tobytes() == before_calprov
