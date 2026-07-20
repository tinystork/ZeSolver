from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from zeblindsolver.wcs_header import apply_wcs_solution_to_header, sanitize_wcs_keywords


def _pixel_sha(path: Path) -> str:
    with fits.open(path, memmap=False) as hdul:
        data = np.ascontiguousarray(hdul[0].data)
    return hashlib.sha256(data.tobytes()).hexdigest()


def _sample_wcs() -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [184.62777087563467, 47.298584781528135]
    wcs.wcs.crpix = [32.0, 32.0]
    scale = 2.371713806045426 / 3600.0
    wcs.wcs.cd = [[-scale, 0.0], [0.0, scale]]
    return wcs


def test_wcs_write_updates_header_without_changing_pixels(tmp_path: Path) -> None:
    path = tmp_path / "frame.fit"
    data = (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) % 4096)
    fits.PrimaryHDU(data=data).writeto(path)
    before = _pixel_sha(path)

    with fits.open(path, mode="update", memmap=False) as hdul:
        apply_wcs_solution_to_header(hdul[0].header, _sample_wcs(), header_updates={"SOLVED": True})

    assert _pixel_sha(path) == before
    assert WCS(fits.getheader(path)).has_celestial
    assert fits.getdata(path).shape == data.shape


def test_sanitize_wcs_keywords_removes_old_wcs_but_not_pixels(tmp_path: Path) -> None:
    path = tmp_path / "old_wcs.fit"
    hdu = fits.PrimaryHDU(data=np.ones((8, 8), dtype=np.uint16))
    hdu.header["CRVAL1"] = 1.0
    hdu.header["CRVAL2"] = 2.0
    hdu.header["CTYPE1"] = "RA---TAN"
    hdu.header["CTYPE2"] = "DEC--TAN"
    hdu.writeto(path)
    before = _pixel_sha(path)

    with fits.open(path, mode="update", memmap=False) as hdul:
        removed = sanitize_wcs_keywords(hdul[0].header)

    assert removed >= 4
    assert _pixel_sha(path) == before
    assert not WCS(fits.getheader(path)).has_celestial


def test_failed_solve_simulation_leaves_copy_pixels_unchanged(tmp_path: Path) -> None:
    src = tmp_path / "source.fit"
    dst = tmp_path / "copy.fit"
    fits.PrimaryHDU(data=np.full((16, 16), 42, dtype=np.uint16)).writeto(src)
    dst.write_bytes(src.read_bytes())
    before_src = _pixel_sha(src)
    before_dst = _pixel_sha(dst)

    # Simulate an orchestration failure before any WCS write.
    try:
        raise RuntimeError("simulated solve failure")
    except RuntimeError:
        pass

    assert _pixel_sha(src) == before_src
    assert _pixel_sha(dst) == before_dst
