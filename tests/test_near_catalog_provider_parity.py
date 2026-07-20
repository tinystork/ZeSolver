from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

import zeblindsolver.metadata_solver as metadata_solver
from near_catalog_provider_helpers import make_synthetic_near_fits, write_astap_1476_tile, write_legacy_index_from_tile
from zeblindsolver.metadata_solver import NearSolveConfig, solve_near
from zeblindsolver.near_catalog_provider import AstapNearCatalogProvider, LegacyIndexNearCatalogProvider


def _patch_strict_detector(monkeypatch, star_px: np.ndarray) -> None:
    stars = np.zeros(star_px.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("flux", "f4")])
    stars["x"] = star_px[:, 0].astype(np.float32)
    stars["y"] = star_px[:, 1].astype(np.float32)
    stars["flux"] = np.linspace(4000, 1000, star_px.shape[0], dtype=np.float32)

    def _strict_detect_stub(*args, **kwargs):
        return stars.copy(), {
            "raw_candidates": int(stars.size),
            "selected_count": int(stars.size),
            "source": "unit_test_strict_detector_stub",
        }

    monkeypatch.setattr(metadata_solver, "astap_adaptive_image_detection", _strict_detect_stub)


def _patch_iso_success(monkeypatch, scale_arcsec: float) -> None:
    def _iso_success(*args, **kwargs):
        center_offset = -scale_arcsec * np.array([100.0, 100.0], dtype=np.float64)
        return (
            metadata_solver.SimilarityTransform(
                scale=scale_arcsec,
                rotation=0.0,
                translation=(float(center_offset[0]), float(center_offset[1])),
                parity=1,
            ),
            np.eye(2, dtype=np.float64) * scale_arcsec,
            center_offset,
            8,
        )

    monkeypatch.setattr(metadata_solver, "_astap_iso_hypothesis", _iso_success)


def test_astap_native_matches_strict_astap_stars_and_solver_result(tmp_path: Path, monkeypatch) -> None:
    center_ra = 2.0
    center_dec = -18.0
    scale_arcsec = 5.0
    scale_deg = scale_arcsec / 3600.0
    star_px = np.array(
        [[100, 100], [120, 105], [82, 123], [140, 140], [60, 80], [155, 82], [70, 150], [132, 166]],
        dtype=np.float64,
    )
    offsets = (star_px - np.array([100.0, 100.0])) * scale_deg
    ra_vals = center_ra + offsets[:, 0] / np.cos(np.radians(center_dec))
    dec_vals = center_dec + offsets[:, 1]
    mags = np.linspace(8.0, 11.0, star_px.shape[0], dtype=np.float32)

    astap_root = tmp_path / "astap"
    write_astap_1476_tile(astap_root, family="d50", tile_code="1501", ra_deg=ra_vals, dec_deg=dec_vals, mag=mags)
    astap_provider = AstapNearCatalogProvider(astap_root, families=("d50",))
    astap_tile = astap_provider.select_tiles(center_ra, center_dec, 2.0, 1)[0]
    bounds = astap_tile.to_manifest_entry()["bounds"]
    index_root = write_legacy_index_from_tile(
        tmp_path / "legacy-index",
        tile_key=astap_tile.tile_key,
        family=astap_tile.family,
        tile_code=astap_tile.tile_code,
        center_ra_deg=astap_tile.center_ra_deg,
        center_dec_deg=astap_tile.center_dec_deg,
        bounds=bounds,
        ra_deg=ra_vals,
        dec_deg=dec_vals,
        mag=mags,
        db_root=astap_root,
    )
    legacy_provider = LegacyIndexNearCatalogProvider(index_root)

    legacy_tiles = legacy_provider.select_tiles(center_ra, center_dec, 2.0, 4, families=("d50",))
    astap_tiles = astap_provider.select_tiles(center_ra, center_dec, 2.0, 4, families=("d50",))
    assert [tile.tile_key for tile in astap_tiles] == [tile.tile_key for tile in legacy_tiles]

    legacy_stars = legacy_provider.load_stars(legacy_tiles[0])
    astap_stars = astap_provider.load_stars(astap_tiles[0])
    assert legacy_stars.size == astap_stars.size
    assert np.allclose(legacy_stars.ra_deg, astap_stars.ra_deg, atol=2e-5)
    assert np.allclose(legacy_stars.dec_deg, astap_stars.dec_deg, atol=2e-5)
    assert np.allclose(legacy_stars.mag, astap_stars.mag, atol=0.06)

    fits_path = tmp_path / "frame.fits"
    make_synthetic_near_fits(fits_path, center_ra=center_ra, center_dec=center_dec, star_px=star_px)
    astap_fits = tmp_path / "frame_astap.fits"
    shutil.copy2(fits_path, astap_fits)
    legacy_fits = tmp_path / "frame_legacy.fits"
    shutil.copy2(fits_path, legacy_fits)

    _patch_strict_detector(monkeypatch, star_px)
    _patch_iso_success(monkeypatch, scale_arcsec)
    cfg = NearSolveConfig(
        astap_iso_strict=True,
        family="d50",
        quality_inliers=3,
        quality_rms=10.0,
        max_tile_candidates=4,
    )
    legacy_result = solve_near(legacy_fits, index_root, config=cfg)
    astap_result = solve_near(astap_fits, None, config=cfg, catalog_provider=astap_provider)

    assert legacy_result.success, legacy_result.message
    assert astap_result.success, astap_result.message
    assert not list(astap_root.rglob("*.npz"))
    assert legacy_result.tile_key == astap_result.tile_key
    assert legacy_result.stats["inliers"] == astap_result.stats["inliers"]
    assert astap_result.stats["near_catalog_provider"] == "astap_native"
    assert legacy_result.stats["near_catalog_provider"] == "legacy_index"
