from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits
from pathlib import Path

from zesolver import zeblindsolver
from zeblindsolver.image_io import load_raster_image
from zeblindsolver.image_prep import downsample_image
from zeblindsolver.quad_index_builder import select_tiles_in_cone
from types import SimpleNamespace
from typing import Any, Optional


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


def test_blind_solve_skips_valid_header(tmp_path) -> None:
    path = tmp_path / "valid.fits"
    hdu = fits.PrimaryHDU(data=np.zeros((4, 4), dtype=np.float32))
    _populate_valid_wcs(hdu.header)
    hdu.writeto(path)
    result = zeblindsolver.blind_solve(
        fits_path=str(path),
        index_root=str(tmp_path),
        skip_if_valid=True,
    )
    assert result["success"]
    assert "skipped" in result["message"]
    assert not result["wrote_wcs"]


def test_blind_solve_delegates_to_internal(monkeypatch, tmp_path) -> None:
    path = tmp_path / "input.fits"
    fits.PrimaryHDU(data=np.zeros((2, 2), dtype=np.float32)).writeto(path)
    index_root = tmp_path / "index"
    index_root.mkdir()

    captured: list[tuple[str, str]] = []

    def fake_internal(
        input_fits: str,
        index_root_arg: str,
        *,
        config: Optional[Any],
        cancel_check: Optional[Any] = None,
    ) -> SimpleNamespace:
        captured.append((input_fits, index_root_arg))
        return SimpleNamespace(
            success=True,
            message="ok",
            tile_key="tile42",
            header_updates={"SOLVED": 1},
        )

    monkeypatch.setattr(zeblindsolver, "_internal_solve_blind", fake_internal)
    result = zeblindsolver.blind_solve(
        fits_path=str(path),
        index_root=str(index_root),
        skip_if_valid=False,
    )
    assert result["success"]
    assert result["used_db"] == "tile42"
    assert result["tried_dbs"] == [str(Path(index_root).expanduser())]
    assert result["updated_keywords"]["SOLVED"] == 1
    assert captured and captured[0][0] == str(path)


def test_manifest_cone_filter():
    manifest = {
        "tiles": [
            {
                "tile_key": "near",
                "bounds": {"dec_min": -5.0, "dec_max": 5.0, "ra_segments": [[10.0, 15.0]]},
            },
            {
                "tile_key": "far",
                "bounds": {"dec_min": 50.0, "dec_max": 60.0, "ra_segments": [[200.0, 205.0]]},
            },
        ]
    }
    selected = select_tiles_in_cone(manifest, ra_deg=12.0, dec_deg=0.0, radius_deg=6.0)
    assert selected == [0]
    selected_all = select_tiles_in_cone(manifest, ra_deg=0.0, dec_deg=80.0, radius_deg=50.0)
    assert set(selected_all) == {1}


def test_downsample_image_reduces_shape():
    img = np.arange(64, dtype=np.float32).reshape(8, 8)
    reduced = downsample_image(img, 2)
    assert reduced.shape == (4, 4)
    reduced_four = downsample_image(img, 4)
    assert reduced_four.shape == (2, 2)


def test_load_raster_image(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    data = (np.linspace(0, 255, num=25, dtype=np.uint8).reshape(5, 5))
    path = tmp_path / "sample.png"
    Image.fromarray(data).save(path)
    array, meta = load_raster_image(path)
    assert array.shape == (5, 5)
    assert array.dtype == np.float32
    assert 0.0 <= float(array.max()) <= 1.0
    assert meta.get("backend")
