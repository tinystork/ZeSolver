from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("corpus_fixture_derivation", ROOT / "tools" / "corpus_fixture_derivation.py")
assert SPEC is not None and SPEC.loader is not None
fixtures = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fixtures
SPEC.loader.exec_module(fixtures)


def _add_wcs_and_solver_cards(header: fits.Header) -> None:
    header["WCSAXES"] = 2
    header["CTYPE1"] = "RA---TAN-SIP"
    header["CTYPE2"] = "DEC--TAN-SIP"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CRPIX1"] = 5.0
    header["CRPIX2"] = 6.0
    header["CRVAL1"] = 184.6
    header["CRVAL2"] = 47.2
    header["CD1_1"] = -0.0006
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 0.0006
    header["PV1_0"] = 0.0
    header["A_ORDER"] = 2
    header["A_0_2"] = 1.0e-8
    header["B_ORDER"] = 2
    header["B_2_0"] = -1.0e-8
    header["SOLVED"] = True
    header["SOLVER"] = "ZeSolver"
    header["SOLVMODE"] = "near"


def _write_parent(path: Path) -> None:
    primary_data = np.arange(64, dtype=np.uint16).reshape(8, 8)
    ext_data = (np.arange(36, dtype=np.uint16).reshape(6, 6) + 100)
    primary = fits.PrimaryHDU(data=primary_data)
    primary.header["RA"] = 184.91667
    primary.header["DEC"] = 47.162222
    primary.header["OBJCTRA"] = "12 19 40"
    primary.header["OBJCTDEC"] = "+47 09 44"
    primary.header["OBJECT"] = "M 106"
    primary.header["FOCALLEN"] = 250
    primary.header["XPIXSZ"] = 2.9
    primary.header["YPIXSZ"] = 2.9
    primary.header["DATE-OBS"] = "2025-05-19T03:20:36"
    primary.header["EXPTIME"] = 20.0
    _add_wcs_and_solver_cards(primary.header)

    image = fits.ImageHDU(data=ext_data, name="SCI")
    _add_wcs_and_solver_cards(image.header)
    fits.HDUList([primary, image]).writeto(path, checksum=True)


def test_strip_solver_wcs_cards_is_deterministic_and_preserves_pixels_and_input_metadata(tmp_path: Path) -> None:
    parent = tmp_path / "parent.fit"
    out_a = tmp_path / "canonical-a.fit"
    out_b = tmp_path / "canonical-b.fit"
    _write_parent(parent)

    parent_audit = fixtures.audit_fits(parent)
    parent_pixels = [h.pixel_hash for h in parent_audit]
    assert any(h.has_celestial_wcs for h in parent_audit)
    assert any("SOLVED" in h.solver_cards for h in parent_audit)
    assert any(h.checksum_cards for h in parent_audit)

    report_a = fixtures.strip_solver_wcs_cards(parent, out_a)
    report_b = fixtures.strip_solver_wcs_cards(parent, out_b)

    assert report_a.output_sha256 == report_b.output_sha256
    assert report_a.pixels_unchanged is True
    assert [h.pixel_hash for h in report_a.output_hdus] == parent_pixels
    assert not any(h.has_celestial_wcs for h in report_a.output_hdus)
    assert not any(h.solver_cards for h in report_a.output_hdus)
    assert not any(h.checksum_cards for h in report_a.output_hdus)

    with fits.open(out_a, memmap=False) as hdul:
        primary = hdul[0].header
        assert not WCS(primary, naxis=2, relax=True).has_celestial
        assert primary["RA"] == 184.91667
        assert primary["DEC"] == 47.162222
        assert primary["OBJCTRA"] == "12 19 40"
        assert primary["OBJCTDEC"] == "+47 09 44"
        assert primary["OBJECT"] == "M 106"
        assert primary["FOCALLEN"] == 250
        assert primary["XPIXSZ"] == 2.9
        assert primary["YPIXSZ"] == 2.9
        assert primary["DATE-OBS"] == "2025-05-19T03:20:36"
        assert primary["EXPTIME"] == 20.0
        for header in (hdul[0].header, hdul[1].header):
            for forbidden in ("CTYPE1", "CRVAL1", "CRPIX1", "CD1_1", "PV1_0", "A_ORDER", "A_0_2"):
                assert forbidden not in header
            for forbidden in ("SOLVED", "SOLVER", "SOLVMODE", "CHECKSUM", "DATASUM"):
                assert forbidden not in header
