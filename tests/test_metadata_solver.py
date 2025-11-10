from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from zeblindsolver.metadata_solver import NearSolveConfig, solve_near
from zeblindsolver.projections import project_tan


def _build_test_index(index_root: Path, ra_vals: np.ndarray, dec_vals: np.ndarray, center_ra: float, center_dec: float) -> None:
    index_root.mkdir(parents=True, exist_ok=True)
    tiles_dir = index_root / "tiles"
    tiles_dir.mkdir(exist_ok=True)
    x_deg, y_deg = project_tan(ra_vals, dec_vals, center_ra, center_dec)
    mag = np.linspace(8.5, 11.0, ra_vals.size, dtype=np.float32)
    tile_path = tiles_dir / "test01.npz"
    np.savez(
        tile_path,
        ra_deg=ra_vals.astype(np.float64),
        dec_deg=dec_vals.astype(np.float64),
        mag=mag,
        x_deg=x_deg.astype(np.float32),
        y_deg=y_deg.astype(np.float32),
    )
    manifest = {
        "tiles": [
            {
                "tile_key": "TEST01",
                "tile_file": f"tiles/{tile_path.name}",
                "center_ra_deg": center_ra,
                "center_dec_deg": center_dec,
                "bounds": {
                    "dec_min": center_dec - 2.0,
                    "dec_max": center_dec + 2.0,
                    "ra_segments": [[center_ra - 2.0, center_ra + 2.0]],
                },
            }
        ]
    }
    (index_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _make_synthetic_fits(path: Path, star_px: np.ndarray, center_ra: float, center_dec: float) -> None:
    image = np.zeros((200, 200), dtype=np.float32)
    for x, y in star_px:
        xi = int(round(x))
        yi = int(round(y))
        image[max(0, yi - 1) : yi + 2, max(0, xi - 1) : xi + 2] = 5000.0
    header = fits.Header()
    header["RA"] = center_ra
    header["DEC"] = center_dec
    header["FOCALLEN"] = 150.0
    header["XPIXSZ"] = 3.76
    header["YPIXSZ"] = 3.76
    fits.PrimaryHDU(data=image, header=header).writeto(path)


def test_metadata_solver_solves_synthetic_frame(tmp_path):
    center_ra = 33.0
    center_dec = 12.0
    scale_arcsec = 5.0
    scale_deg = scale_arcsec / 3600.0
    base_pixels = np.array([[100, 100], [120, 105], [82, 123], [140, 140]], dtype=np.float64)
    offsets = (base_pixels - np.array([100.0, 100.0])) * scale_deg
    ra_offsets = offsets[:, 0] / np.cos(np.radians(center_dec))
    ra_vals = center_ra + ra_offsets
    dec_vals = center_dec + offsets[:, 1]
    index_root = tmp_path / "index"
    _build_test_index(index_root, ra_vals, dec_vals, center_ra, center_dec)
    fits_path = tmp_path / "frame.fits"
    _make_synthetic_fits(fits_path, base_pixels, center_ra, center_dec)
    config = NearSolveConfig(
        max_img_stars=50,
        max_cat_stars=50,
        pixel_tolerance=4.0,
        quality_inliers=3,
    )
    result = solve_near(fits_path, index_root, config=config)
    assert result.success, result.message
    assert result.stats.get("quality") == "GOOD"
    with fits.open(fits_path) as hdul:
        header = hdul[0].header
        assert header["SOLVED"] == 1
        assert "NEAR_VER" in header
        assert abs(header["CRVAL1"] - center_ra) < 0.1
