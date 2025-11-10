from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from zesolver import zeblindsolver


def _populate_valid_wcs(header: fits.Header) -> None:
    header["CRVAL1"] = 120.5
    header["CRVAL2"] = -15.2
    header["CRPIX1"] = 512.0
    header["CRPIX2"] = 512.0
    header["CD1_1"] = -2.3e-4
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 2.3e-4
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["RADESYS"] = "ICRS"


def test_has_valid_wcs_rejects_cdelt1_1deg() -> None:
    header = fits.Header()
    _populate_valid_wcs(header)
    header["CDELT1"] = 1.0
    assert not zeblindsolver.has_valid_wcs(header)
    del header["CDELT1"]
    assert zeblindsolver.has_valid_wcs(header)


def test_sanitize_removes_wcs_keys() -> None:
    header = fits.Header()
    for key in ("CTYPE1", "CRVAL1", "CD1_1", "RADESYS"):
        header[key] = 1.0
    removed = zeblindsolver.sanitize_wcs(header)
    assert removed == 4
    for key in ("CTYPE1", "CRVAL1", "CD1_1", "RADESYS"):
        assert key not in header


def test_cli_skip_if_valid(tmp_path, capsys) -> None:
    path = tmp_path / "valid.fits"
    hdu = fits.PrimaryHDU(data=np.zeros((4, 4), dtype=np.float32))
    _populate_valid_wcs(hdu.header)
    hdu.header["ZESOLVER_HINT"] = 1
    hdu.writeto(path)
    code = zeblindsolver.main(
        [
            "--input",
            str(path),
            "--db",
            "D50",
            "--skip-if-valid",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "skipped" in captured.out


def test_cli_fail_when_no_db(monkeypatch, tmp_path) -> None:
    path = tmp_path / "missing_db.fits"
    fits.PrimaryHDU(data=np.zeros((2, 2), dtype=np.float32)).writeto(path)

    def fake_resolver(_: str | None) -> str:
        raise zeblindsolver.AstapNotFoundError("no astap")

    monkeypatch.setattr(zeblindsolver, "_resolve_astap_executable", fake_resolver)
    code = zeblindsolver.main(["--input", str(path), "--db", "D50"])
    assert code == 3


def test_blind_solve_writes_tags(monkeypatch, tmp_path) -> None:
    path = tmp_path / "solve_me.fits"
    fits.PrimaryHDU(data=np.ones((8, 8), dtype=np.float32)).writeto(path)

    monkeypatch.setattr(zeblindsolver, "_resolve_astap_executable", lambda _: "/usr/bin/astap")

    captured_cmd: list[str] = []

    def fake_run(cmd, capture_output, text, timeout, check):  # noqa: D401 - pytest helper
        captured_cmd[:] = cmd
        with fits.open(path, mode="update", memmap=False) as hdul:
            hdr = hdul[0].header
            _populate_valid_wcs(hdr)
        class Proc:
            returncode = 0
            stdout = "ok"
            stderr = ""
        return Proc()

    monkeypatch.setattr(zeblindsolver.subprocess, "run", fake_run)
    result = zeblindsolver.blind_solve(
        fits_path=str(path),
        db_roots=["D50"],
        skip_if_valid=False,
        timeout_sec=10,
        ra_hint=30.0,
        dec_hint=-10.0,
    )
    assert result["success"]
    header = fits.getheader(path)
    assert header["SOLVED"] == 1
    assert header["USED_DB"] == "D50"
    assert header["ZESOLVER_HINT"] == 1
    assert result["wrote_wcs"]
    assert result["updated_keywords"]["USED_DB"] == "D50"
    assert "-ra" in captured_cmd
    assert "-spd" in captured_cmd
    ra_index = captured_cmd.index("-ra")
    spd_index = captured_cmd.index("-spd")
    assert pytest.approx(float(captured_cmd[ra_index + 1]), rel=1e-6) == pytest.approx(30.0 / 15.0, rel=1e-6)
    assert pytest.approx(float(captured_cmd[spd_index + 1]), rel=1e-6) == pytest.approx(80.0, rel=1e-6)
