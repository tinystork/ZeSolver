from __future__ import annotations

import time

import numpy as np

from near_catalog_provider_helpers import write_astap_1476_tile
from zeblindsolver.astap_4d_builder import Astap4DBuildConfig, build_4d_index_from_astap
from zeblindsolver.quad_index_4d import Quad4DIndex, compare_4d_indexes, scientific_payload_fingerprint


def _write_catalog(root):
    ra = np.asarray([32.00, 32.03, 32.08, 32.11, 32.15, 32.21, 32.27, 32.34], dtype=np.float64)
    dec = np.asarray([-2.00, -1.92, -2.06, -1.88, -2.01, -1.84, -1.96, -1.80], dtype=np.float64)
    mag = np.asarray([8.1, 8.8, 9.4, 8.6, 10.0, 9.2, 10.4, 9.8], dtype=np.float32)
    write_astap_1476_tile(root, family="d50", tile_code="1501", ra_deg=ra, dec_deg=dec, mag=mag)


def _config() -> Astap4DBuildConfig:
    return Astap4DBuildConfig(
        family="d50",
        tile_keys=("d50_1501",),
        mag_cap=11.0,
        source_max_stars=8,
        max_stars_per_tile=8,
        max_quads_per_tile=10,
        sampler_tag="legacy_brightness",
    )


def test_direct_builder_scientific_fingerprint_is_deterministic(tmp_path):
    astap = tmp_path / "astap"
    _write_catalog(astap)
    first = tmp_path / "a" / "direct.npz"
    second = tmp_path / "b" / "direct.npz"

    build_4d_index_from_astap(astap, first, config=_config())
    time.sleep(1.0)
    build_4d_index_from_astap(astap, second, config=_config())

    assert scientific_payload_fingerprint(first) == scientific_payload_fingerprint(second)
    assert compare_4d_indexes(first, second)["exact"] is True


def test_direct_builder_detects_scientific_change(tmp_path):
    astap_a = tmp_path / "astap_a"
    astap_b = tmp_path / "astap_b"
    _write_catalog(astap_a)
    ra = np.asarray([32.00, 32.03, 32.08, 32.11, 32.15, 32.21, 32.27, 32.40], dtype=np.float64)
    dec = np.asarray([-2.00, -1.92, -2.06, -1.88, -2.01, -1.84, -1.96, -1.80], dtype=np.float64)
    mag = np.asarray([8.1, 8.8, 9.4, 8.6, 10.0, 9.2, 10.4, 9.8], dtype=np.float32)
    write_astap_1476_tile(astap_b, family="d50", tile_code="1501", ra_deg=ra, dec_deg=dec, mag=mag)
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"

    build_4d_index_from_astap(astap_a, first, config=_config())
    build_4d_index_from_astap(astap_b, second, config=_config())

    assert scientific_payload_fingerprint(first) != scientific_payload_fingerprint(second)
    assert compare_4d_indexes(Quad4DIndex.load(first), Quad4DIndex.load(second))["exact"] is False
