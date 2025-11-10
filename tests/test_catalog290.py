from __future__ import annotations

from pathlib import Path

import numpy as np

from zewcs290 import CatalogDB


DB_ROOT = Path(__file__).resolve().parents[1] / "database"


def _get_tile(db: CatalogDB, tile_code: str) -> int:
    for idx, tile in enumerate(db.tiles):
        if tile.tile_code == tile_code:
            return idx
    raise AssertionError(f"tile {tile_code} not found")


def test_decode_g05_polar_tile():
    db = CatalogDB(DB_ROOT, families=["g05"])
    tile = db.tiles[_get_tile(db, "0101")]
    block = db._load_tile(tile)  # pylint: disable=protected-access
    assert block.star_count > 0
    assert block.header_records > 0
    assert np.isfinite(block.stars["ra_deg"]).all()
    assert np.all(block.stars["dec_deg"] <= tile.bounds.dec_max + 1e-6)
    assert np.all(block.stars["dec_deg"] >= tile.bounds.dec_min - 1e-6)

    stars = db.query_box(0, 360, tile.bounds.dec_min, tile.bounds.dec_max, families=["g05"])
    assert stars.size == block.star_count


def test_decode_d50_tile_and_cone_query():
    db = CatalogDB(DB_ROOT, families=["d50"])
    tile = db.tiles[_get_tile(db, "0501")]
    block = db._load_tile(tile)  # pylint: disable=protected-access
    assert block.star_count > 0
    assert np.isnan(block.stars["bp_rp"]).all()

    ra_center = float(np.mean(block.stars["ra_deg"]))
    dec_center = float(np.mean(block.stars["dec_deg"]))
    subset = db.query_cone(ra_center, dec_center, radius_deg=0.5, families=["d50"], max_stars=100)
    assert subset.size > 0
    assert subset.size <= 100
