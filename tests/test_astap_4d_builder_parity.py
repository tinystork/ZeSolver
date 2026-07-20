from __future__ import annotations

import json

import numpy as np
import pytest

from near_catalog_provider_helpers import write_astap_1476_tile
from zeblindsolver.astap_4d_builder import (
    Astap4DBuildConfig,
    AstapTileMaterializationConfig,
    build_4d_index_from_astap,
    materialize_astap_tile_for_4d,
)
from zeblindsolver.astap_db_reader import iter_tiles
from zeblindsolver.db_convert import build_index_from_astap
from zeblindsolver.quad_index_4d import Quad4DIndex, build_experimental_4d_index, compare_4d_indexes


def _write_catalog(root):
    ra = np.asarray([42.00, 42.03, 42.08, 42.13, 42.18, 42.23, 42.29, 42.35, 42.42], dtype=np.float64)
    dec = np.asarray([3.00, 3.07, 2.96, 3.11, 3.02, 3.16, 3.08, 3.20, 3.14], dtype=np.float64)
    mag = np.asarray([9.0, 8.0, 9.5, 10.0, 8.0, 11.0, 8.7, 10.5, 9.2], dtype=np.float32)
    write_astap_1476_tile(root, family="d50", tile_code="1501", ra_deg=ra, dec_deg=dec, mag=mag)


def _build_config() -> Astap4DBuildConfig:
    return Astap4DBuildConfig(
        family="d50",
        tile_keys=("d50_1501",),
        mag_cap=11.0,
        source_max_stars=9,
        source_star_truncation_mode="native_prefix",
        max_stars_per_tile=9,
        max_quads_per_tile=14,
        sampler_tag="legacy_brightness",
    )


def test_direct_materialization_matches_historical_tile_npz(tmp_path):
    astap = tmp_path / "astap"
    _write_catalog(astap)
    legacy_root = tmp_path / "legacy"
    cfg = _build_config()
    build_index_from_astap(
        astap,
        legacy_root,
        mag_cap=cfg.mag_cap,
        max_stars=cfg.source_max_stars,
        skip_quads=True,
        star_truncation_mode=cfg.source_star_truncation_mode,
    )
    meta = next(meta for meta in iter_tiles(astap) if meta.key == "d50_1501")
    direct = materialize_astap_tile_for_4d(astap, meta, config=cfg.materialization_config())

    with np.load(legacy_root / "tiles" / "d50_1501.npz", allow_pickle=False) as hist:
        assert np.array_equal(direct.ra_deg, hist["ra_deg"])
        assert np.array_equal(direct.dec_deg, hist["dec_deg"])
        assert np.array_equal(direct.mag, hist["mag"])
        assert np.array_equal(direct.x_deg, hist["x_deg"])
        assert np.array_equal(direct.y_deg, hist["y_deg"])
        assert np.array_equal(direct.source_star_indices, hist["sweep_rank"])

    manifest = json.loads((legacy_root / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["tiles"][0]
    assert direct.center_ra_deg == entry["center_ra_deg"]
    assert direct.center_dec_deg == entry["center_dec_deg"]


def test_direct_payload_matches_historical_payload_arrays(tmp_path):
    astap = tmp_path / "astap"
    _write_catalog(astap)
    legacy_root = tmp_path / "legacy"
    cfg = _build_config()
    build_index_from_astap(
        astap,
        legacy_root,
        mag_cap=cfg.mag_cap,
        max_stars=cfg.source_max_stars,
        skip_quads=True,
        star_truncation_mode=cfg.source_star_truncation_mode,
    )
    historical = tmp_path / "historical_4d.npz"
    direct = tmp_path / "direct_4d.npz"

    build_experimental_4d_index(
        legacy_root,
        historical,
        tile_keys=cfg.tile_keys,
        level=cfg.level,
        max_stars_per_tile=cfg.max_stars_per_tile,
        max_quads_per_tile=cfg.max_quads_per_tile,
        sampler_tag=cfg.sampler_tag,
        code_tol_recommended=cfg.code_tol_recommended,
        dtype=cfg.dtype,
    )
    build_4d_index_from_astap(astap, direct, config=cfg)

    report = compare_4d_indexes(historical, direct)
    assert report["exact"] is True
    assert Quad4DIndex.load(direct).metadata["source_catalog"] == "astap_raw"


def test_compare_4d_indexes_reports_first_divergence(tmp_path):
    astap = tmp_path / "astap"
    _write_catalog(astap)
    cfg = _build_config()
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    build_4d_index_from_astap(astap, left, config=cfg)
    build_4d_index_from_astap(astap, right, config=cfg)
    loaded = Quad4DIndex.load(right)
    changed = loaded.codes_4d.copy()
    changed[0, 0] += 0.25
    np.savez_compressed(
        right,
        codes_4d=changed,
        quad_star_indices=loaded.quad_star_indices,
        source_quad_indices=loaded.source_quad_indices,
        tile_key_indices=loaded.tile_key_indices,
        ratio_hashes=loaded.ratio_hashes,
        tile_keys=np.asarray(loaded.tile_keys, dtype="<U16"),
        catalog_ra_dec=loaded.catalog_ra_dec,
        catalog_xy=loaded.catalog_xy,
        metadata=np.asarray([json.dumps(loaded.metadata, sort_keys=True)], dtype="<U4096"),
    )

    report = compare_4d_indexes(left, right)

    assert report["exact"] is False
    assert report["arrays"]["codes_4d"]["first_difference_index"] == [0, 0]
    assert report["arrays"]["codes_4d"]["max_abs_diff"] > 0.0


def test_materialization_config_rejects_unknown_policy():
    cfg = AstapTileMaterializationConfig(source_star_truncation_mode="mystery")
    with pytest.raises(ValueError, match="source_star_truncation_mode"):
        materialize_astap_tile_for_4d("/unused", object(), config=cfg)  # type: ignore[arg-type]
