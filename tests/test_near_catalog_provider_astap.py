from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from near_catalog_provider_helpers import write_astap_1476_tile
from zeblindsolver.near_catalog_provider import AstapNearCatalogProvider, NearCatalogProviderError


def _write_probe_db(root: Path) -> None:
    write_astap_1476_tile(
        root,
        family="d50",
        tile_code="0501",
        ra_deg=np.array([1.0, 2.0, 3.0]),
        dec_deg=np.array([-70.0, -69.5, -69.0]),
        mag=np.array([8.0, 9.0, 10.0], dtype=np.float32),
    )
    write_astap_1476_tile(
        root,
        family="d50",
        tile_code="1501",
        ra_deg=np.array([1.0, 1.2, 1.4]),
        dec_deg=np.array([-18.2, -18.0, -17.8]),
        mag=np.array([7.5, 8.5, 9.5], dtype=np.float32),
    )
    write_astap_1476_tile(
        root,
        family="d80",
        tile_code="1501",
        ra_deg=np.array([1.0]),
        dec_deg=np.array([-18.0]),
        mag=np.array([8.0], dtype=np.float32),
    )


def test_astap_provider_discovers_allowed_families_and_loads_stars(tmp_path: Path) -> None:
    _write_probe_db(tmp_path)

    provider = AstapNearCatalogProvider(tmp_path, families=("d50",))
    tiles = provider.select_tiles(1.0, -18.0, 2.0, 4)
    stars = provider.load_stars(tiles[0])

    assert provider.kind == "astap_native"
    assert provider.families == ("d50",)
    assert tiles[0].family == "d50"
    assert stars.size == 3
    assert stars.ra_deg.dtype == np.float64
    assert stars.dec_deg.dtype == np.float64
    assert stars.mag.dtype == np.float32


def test_astap_provider_selects_ra_wrap_and_polar_regions(tmp_path: Path) -> None:
    write_astap_1476_tile(
        tmp_path,
        family="d50",
        tile_code="0101",
        ra_deg=np.array([0.0]),
        dec_deg=np.array([-88.8]),
    )
    write_astap_1476_tile(
        tmp_path,
        family="d50",
        tile_code="0501",
        ra_deg=np.array([1.0]),
        dec_deg=np.array([-69.5]),
    )
    provider = AstapNearCatalogProvider(tmp_path, families=("d50",))

    wrap = provider.select_tiles(359.5, -69.5, 2.0, 8)
    polar = provider.select_tiles(180.0, -89.0, 1.0, 8)

    assert any(tile.tile_code == "0501" for tile in wrap)
    assert any(tile.tile_code == "0101" for tile in polar)


def test_astap_provider_limits_deterministically_and_rejects_missing_family(tmp_path: Path) -> None:
    _write_probe_db(tmp_path)
    provider = AstapNearCatalogProvider(tmp_path, families=("d50",))

    selected_once = provider.select_tiles(1.0, -20.0, 60.0, 1)
    selected_twice = provider.select_tiles(1.0, -20.0, 60.0, 1)

    assert len(selected_once) == 1
    assert [tile.tile_key for tile in selected_once] == [tile.tile_key for tile in selected_twice]
    with pytest.raises(NearCatalogProviderError, match="no usable Near tiles|missing requested"):
        AstapNearCatalogProvider(tmp_path, families=("g05",))


def test_astap_provider_is_read_only_and_cache_returns_copies(tmp_path: Path) -> None:
    _write_probe_db(tmp_path)
    before = {path: path.stat().st_mtime_ns for path in tmp_path.iterdir()}
    provider = AstapNearCatalogProvider(tmp_path, families=("d50",))
    tile = provider.select_tiles(1.0, -18.0, 2.0, 1)[0]

    first = provider.load_stars(tile)
    first.ra_deg[0] = 123.0
    second = provider.load_stars(tile)
    after = {path: path.stat().st_mtime_ns for path in tmp_path.iterdir()}

    assert second.ra_deg[0] != pytest.approx(123.0)
    assert before == after
