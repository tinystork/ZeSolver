from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from near_catalog_provider_helpers import write_astap_1476_tile, write_legacy_index_from_tile
from zeblindsolver.near_catalog_provider import LegacyIndexNearCatalogProvider, NearCatalogProviderError


def _write_two_tile_legacy_index(root: Path) -> Path:
    write_legacy_index_from_tile(
        root,
        tile_key="d50_near",
        family="d50",
        tile_code="1501",
        center_ra_deg=1.0,
        center_dec_deg=-18.0,
        bounds={"dec_min": -20.0, "dec_max": -16.0, "ra_segments": [[0.0, 4.0]]},
        ra_deg=np.array([1.0, 1.1]),
        dec_deg=np.array([-18.0, -17.9]),
        mag=np.array([8.0, 9.0], dtype=np.float32),
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    far_tile = root / "tiles" / "d50_far.npz"
    np.savez(
        far_tile,
        ra_deg=np.array([50.0], dtype=np.float64),
        dec_deg=np.array([-18.0], dtype=np.float64),
        mag=np.array([10.0], dtype=np.float32),
    )
    manifest["tiles"].append(
        {
            "tile_key": "d50_far",
            "tile_file": "tiles/d50_far.npz",
            "family": "d50",
            "tile_code": "1510",
            "center_ra_deg": 52.0,
            "center_dec_deg": -18.0,
            "bounds": {"dec_min": -20.0, "dec_max": -16.0, "ra_segments": [[49.0, 55.0]]},
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_legacy_provider_loads_manifest_and_preserves_selection_order(tmp_path: Path) -> None:
    provider = LegacyIndexNearCatalogProvider(_write_two_tile_legacy_index(tmp_path))

    selected = provider.select_tiles(1.0, -18.0, 80.0, 2)

    assert provider.kind == "legacy_index"
    assert provider.families == ("d50",)
    assert [tile.tile_key for tile in selected] == ["d50_near", "d50_far"]


def test_legacy_provider_filters_family_and_loads_npz(tmp_path: Path) -> None:
    provider = LegacyIndexNearCatalogProvider(_write_two_tile_legacy_index(tmp_path))

    selected = provider.select_tiles(1.0, -18.0, 80.0, 4, families=("d80",))
    stars = provider.load_stars(provider.select_tiles(1.0, -18.0, 2.0, 1)[0])

    assert selected == ()
    assert stars.size == 2
    assert np.allclose(stars.ra_deg, [1.0, 1.1])


def test_legacy_provider_reports_missing_tile(tmp_path: Path) -> None:
    provider = LegacyIndexNearCatalogProvider(_write_two_tile_legacy_index(tmp_path))
    tile = provider.select_tiles(1.0, -18.0, 2.0, 1)[0]
    (tmp_path / "tiles" / "d50_near.npz").unlink()

    with pytest.raises(NearCatalogProviderError, match="failed to read legacy tile"):
        provider.load_stars(tile)


def test_legacy_provider_can_use_read_only_astap_fallback(tmp_path: Path) -> None:
    astap_root = tmp_path / "astap"
    write_astap_1476_tile(
        astap_root,
        family="d50",
        tile_code="1501",
        ra_deg=np.array([1.0, 1.1]),
        dec_deg=np.array([-18.0, -17.9]),
        mag=np.array([8.0, 9.0], dtype=np.float32),
    )
    index_root = write_legacy_index_from_tile(
        tmp_path / "index",
        tile_key="d50_1501",
        family="d50",
        tile_code="1501",
        center_ra_deg=1.0,
        center_dec_deg=-18.0,
        bounds={"dec_min": -20.0, "dec_max": -16.0, "ra_segments": [[0.0, 4.0]]},
        ra_deg=np.array([1.0]),
        dec_deg=np.array([-18.0]),
        mag=np.array([8.0], dtype=np.float32),
        db_root=astap_root,
    )
    provider = LegacyIndexNearCatalogProvider(index_root)
    tile = provider.select_tiles(1.0, -18.0, 2.0, 1)[0]
    (index_root / "tiles" / "d50_1501.npz").unlink()

    stars = provider.load_stars(tile)

    assert stars.size == 2
    assert not (index_root / "tiles" / "d50_1501.npz").exists()
