from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from astropy.io import fits


def load_zesolver_app():
    path = Path(__file__).resolve().parents[1] / "zesolver.py"
    spec = importlib.util.spec_from_file_location("zesolver_app_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_fits(path: Path, *, primary_wcs: bool, extension_wcs: bool = False) -> Path:
    header = fits.Header()
    if primary_wcs:
        _add_wcs(header)
    primary = fits.PrimaryHDU(data=[[1.0, 2.0], [3.0, 4.0]], header=header)
    hdus = [primary]
    if extension_wcs:
        ext_header = fits.Header()
        _add_wcs(ext_header)
        hdus.append(fits.ImageHDU(data=[[5.0, 6.0], [7.0, 8.0]], header=ext_header))
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def _add_wcs(header: fits.Header) -> None:
    header["CRPIX1"] = 1.0
    header["CRPIX2"] = 1.0
    header["CRVAL1"] = 10.0
    header["CRVAL2"] = 20.0
    header["CD1_1"] = 0.001
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 0.001
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
