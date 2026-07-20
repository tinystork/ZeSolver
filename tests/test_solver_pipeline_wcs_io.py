from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from zeblindsolver.wcs_header import apply_wcs_solution_to_header
from zesolver.core.wcs_io import pixel_fingerprint, write_wcs_safely

from solver_pipeline_fixtures import sample_wcs


def test_wcs_io_writes_copy_without_changing_pixels(tmp_path: Path) -> None:
    src = tmp_path / "src.fit"
    dst = tmp_path / "dst.fit"
    fits.PrimaryHDU(data=np.arange(64, dtype=np.uint16).reshape(8, 8)).writeto(src)
    before_src = pixel_fingerprint(src)

    result = write_wcs_safely(input_path=src, output_path=dst, wcs=sample_wcs(), overwrite_wcs=True)

    assert result.ok is True
    assert result.wcs_written is True
    assert result.pixels_unchanged is True
    assert pixel_fingerprint(src) == before_src
    assert pixel_fingerprint(dst) == before_src
    assert WCS(fits.getheader(dst), naxis=2, relax=True).has_celestial
    assert not WCS(fits.getheader(src), naxis=2, relax=True).has_celestial


def test_wcs_io_rejects_existing_wcs_when_overwrite_forbidden(tmp_path: Path) -> None:
    src = tmp_path / "src.fit"
    hdu = fits.PrimaryHDU(data=np.ones((8, 8), dtype=np.uint16))
    apply_wcs_solution_to_header(hdu.header, sample_wcs())
    hdu.writeto(src)
    before = pixel_fingerprint(src)

    result = write_wcs_safely(input_path=src, output_path=None, wcs=sample_wcs(), overwrite_wcs=False)

    assert result.ok is False
    assert result.error == "existing_wcs_overwrite_forbidden"
    assert pixel_fingerprint(src) == before


def test_wcs_io_reports_write_failure(tmp_path: Path) -> None:
    src = tmp_path / "src.fit"
    fits.PrimaryHDU(data=np.ones((8, 8), dtype=np.uint16)).writeto(src)
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")

    result = write_wcs_safely(input_path=src, output_path=blocked / "out.fit", wcs=sample_wcs(), overwrite_wcs=True)

    assert result.ok is False
    assert result.wcs_written is False
    assert result.error
