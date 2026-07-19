from __future__ import annotations

import numpy as np
import pytest

from near_catalog_provider_helpers import write_astap_1476_tile
from zeblindsolver import astap_4d_builder as builder
from zeblindsolver.astap_4d_builder import AstapTileMaterializationConfig, materialize_astap_tile_for_4d
from zeblindsolver.astap_db_reader import TileMeta, iter_tiles
from zewcs290.catalog290 import STAR_DTYPE, SkyBox


def _tile_meta(root, tile_key="d50_1501"):
    return next(meta for meta in iter_tiles(root) if meta.key == tile_key)


def test_materialize_astap_tile_matches_historical_filters(tmp_path):
    root = tmp_path / "astap"
    ra = np.asarray([12.0, 12.01, 12.02, 12.03, 12.04], dtype=np.float64)
    dec = np.asarray([1.0, 1.01, 1.02, 1.03, 1.04], dtype=np.float64)
    mag = np.asarray([9.0, 12.0, 8.5, 16.0, 10.0], dtype=np.float32)
    write_astap_1476_tile(root, family="d50", tile_code="1501", ra_deg=ra, dec_deg=dec, mag=mag)

    tile = materialize_astap_tile_for_4d(
        root,
        _tile_meta(root),
        config=AstapTileMaterializationConfig(mag_cap=12.0, source_max_stars=3, max_stars_per_tile=2),
    )

    assert tile.tile_key == "d50_1501"
    assert tile.family == "d50"
    assert tile.post_mag_count == 4
    assert tile.post_source_limit_count == 3
    assert tile.post_4d_limit_count == 2
    assert np.all(tile.mag <= 12.0)
    assert np.array_equal(tile.source_star_indices, np.asarray([0, 1, 2], dtype=np.int32))
    assert np.isfinite(tile.x_deg).all()
    assert np.isfinite(tile.y_deg).all()


def test_materialize_astap_tile_supports_brightest_mag_source_limit(tmp_path):
    root = tmp_path / "astap"
    ra = np.linspace(22.0, 22.05, 6)
    dec = np.linspace(-3.0, -2.95, 6)
    mag = np.asarray([12.0, 8.0, 11.0, 7.0, 10.0, 9.0], dtype=np.float32)
    write_astap_1476_tile(root, family="d50", tile_code="1501", ra_deg=ra, dec_deg=dec, mag=mag)

    tile = materialize_astap_tile_for_4d(
        root,
        _tile_meta(root),
        config=AstapTileMaterializationConfig(
            mag_cap=None,
            source_max_stars=3,
            source_star_truncation_mode="brightest_mag",
        ),
    )

    assert np.allclose(np.sort(tile.mag), np.asarray([7.0, 8.0, 9.0], dtype=np.float32))
    assert np.array_equal(tile.source_star_indices, np.asarray([3, 1, 5], dtype=np.int32))


def test_materialize_astap_tile_handles_empty_tile(tmp_path):
    root = tmp_path / "astap"
    write_astap_1476_tile(root, family="d50", tile_code="1501", ra_deg=np.asarray([]), dec_deg=np.asarray([]), mag=np.asarray([]))

    tile = materialize_astap_tile_for_4d(root, _tile_meta(root))

    assert tile.post_source_limit_count == 0
    assert tile.ra_deg.shape == (0,)
    assert tile.center_ra_deg == pytest.approx(_tile_meta(root).center_ra_deg)


def test_materialize_astap_tile_filters_non_finite_projection(monkeypatch, tmp_path):
    meta = TileMeta(
        key="d50_9999",
        family="d50",
        tile_code="9999",
        path=tmp_path / "d50_9999.1476",
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        bounds=SkyBox(ra_segments=((0.0, 1.0),), dec_min=0.0, dec_max=1.0),
        ring_index=99,
        tile_index=99,
    )
    stars = np.zeros(5, dtype=STAR_DTYPE)
    stars["ra_deg"] = np.linspace(10.0, 10.4, 5)
    stars["dec_deg"] = np.linspace(20.0, 20.4, 5)
    stars["mag"] = np.linspace(8.0, 12.0, 5)
    monkeypatch.setattr(builder, "load_tile_stars", lambda _root, _meta: stars)

    def fake_project(ra, dec, _center_ra, _center_dec):
        x = np.asarray([0.0, np.nan, 2.0, np.inf, 4.0], dtype=np.float64)
        y = np.asarray([0.0, 1.0, np.nan, 3.0, 4.0], dtype=np.float64)
        return x, y

    monkeypatch.setattr(builder, "project_tan", fake_project)

    tile = materialize_astap_tile_for_4d(tmp_path, meta, config=AstapTileMaterializationConfig(mag_cap=None))

    assert tile.post_source_limit_count == 2
    assert np.allclose(tile.x_deg, [0.0, 4.0])
    assert np.allclose(tile.mag, [8.0, 12.0])


def test_materialize_astap_tile_falls_back_to_layout_center(monkeypatch, tmp_path):
    root = tmp_path / "astap"
    write_astap_1476_tile(
        root,
        family="d50",
        tile_code="1501",
        ra_deg=np.asarray([12.0, 12.1, 12.2]),
        dec_deg=np.asarray([1.0, 1.1, 1.2]),
        mag=np.asarray([8.0, 9.0, 10.0]),
    )
    calls = {"n": 0}
    real_project = builder.project_tan

    def flaky_project(ra, dec, center_ra, center_dec):
        calls["n"] += 1
        if calls["n"] == 1:
            return np.full_like(ra, np.nan, dtype=np.float64), np.full_like(dec, np.nan, dtype=np.float64)
        return real_project(ra, dec, center_ra, center_dec)

    monkeypatch.setattr(builder, "project_tan", flaky_project)

    tile = materialize_astap_tile_for_4d(root, _tile_meta(root), config=AstapTileMaterializationConfig(mag_cap=None))

    assert tile.used_layout_fallback is True
    assert calls["n"] == 2
    assert np.isfinite(tile.x_deg).all()
