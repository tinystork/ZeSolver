from __future__ import annotations

import json

import numpy as np
import pytest

from near_catalog_provider_helpers import write_astap_1476_tile
from zeblindsolver.astap_4d_builder import Astap4DBuildConfig, build_4d_index_from_astap
from zeblindsolver.index_manifest_4d import MANIFEST_SCHEMA, MANIFEST_VERSION, load_4d_index_manifest, sha256_file
from zeblindsolver.quad_index_4d import Quad4DIndex, Quad4DPayloadTile, build_4d_index_from_payload_tiles


def _stars(offset: float = 0.0):
    ra = np.asarray([12.00, 12.03, 12.06, 12.10, 12.14, 12.19, 12.25, 12.31], dtype=np.float64) + offset
    dec = np.asarray([1.00, 1.06, 0.98, 1.12, 1.02, 1.16, 1.08, 1.20], dtype=np.float64)
    mag = np.asarray([8.0, 9.5, 8.7, 10.2, 9.0, 11.0, 9.2, 10.8], dtype=np.float32)
    return ra, dec, mag


def _write_catalog(root, *, tiles=("1501",)):
    for idx, tile_code in enumerate(tiles):
        ra, dec, mag = _stars(offset=idx * 0.5)
        write_astap_1476_tile(root, family="d50", tile_code=tile_code, ra_deg=ra, dec_deg=dec, mag=mag)


def _config(*tile_keys: str) -> Astap4DBuildConfig:
    return Astap4DBuildConfig(
        family="d50",
        tile_keys=tuple(tile_keys),
        mag_cap=12.0,
        source_max_stars=8,
        max_stars_per_tile=8,
        max_quads_per_tile=12,
        sampler_tag="legacy_brightness",
    )


def test_build_4d_index_from_astap_one_tile_loadable(tmp_path):
    astap = tmp_path / "astap"
    _write_catalog(astap)
    out = tmp_path / "direct" / "d50_1501_4d.npz"

    result = build_4d_index_from_astap(astap, out, config=_config("d50_1501"))
    loaded = Quad4DIndex.load(result)

    assert result == out.resolve()
    assert loaded.codes_4d.shape[1] == 4
    assert loaded.quad_star_indices.shape[1] == 4
    assert loaded.metadata["source_catalog"] == "astap_raw"
    assert loaded.metadata["source_family"] == "d50"
    assert loaded.metadata["build_parameters"]["source_star_truncation_mode"] == "native_prefix"
    assert loaded.tile_keys == ("d50_1501",)


def test_build_4d_index_from_astap_multiple_tiles(tmp_path):
    astap = tmp_path / "astap"
    _write_catalog(astap, tiles=("1501", "1502"))
    out = tmp_path / "direct" / "two_tiles.npz"

    build_4d_index_from_astap(astap, out, config=_config("d50_1501", "d50_1502"))
    loaded = Quad4DIndex.load(out)

    assert loaded.tile_keys == ("d50_1501", "d50_1502")
    assert set(loaded.tile_key_indices.tolist()) == {0, 1}


def test_build_4d_index_from_astap_requires_existing_family_and_tile(tmp_path):
    astap = tmp_path / "astap"
    _write_catalog(astap)

    with pytest.raises(KeyError):
        build_4d_index_from_astap(astap, tmp_path / "missing_family.npz", config=Astap4DBuildConfig(family="g05", tile_keys=("g05_0101",)))
    with pytest.raises(KeyError):
        build_4d_index_from_astap(astap, tmp_path / "missing_tile.npz", config=_config("d50_9999"))


def test_build_4d_index_from_astap_refuses_existing_output(tmp_path):
    astap = tmp_path / "astap"
    _write_catalog(astap)
    out = tmp_path / "direct.npz"
    out.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        build_4d_index_from_astap(astap, out, config=_config("d50_1501"))


def test_build_4d_index_from_astap_reports_no_quads(tmp_path):
    astap = tmp_path / "astap"
    write_astap_1476_tile(
        astap,
        family="d50",
        tile_code="1501",
        ra_deg=np.asarray([12.0, 12.1, 12.2]),
        dec_deg=np.asarray([1.0, 1.1, 1.2]),
        mag=np.asarray([8.0, 9.0, 10.0], dtype=np.float32),
    )

    with pytest.raises(RuntimeError, match="no 4D codes generated"):
        build_4d_index_from_astap(astap, tmp_path / "empty.npz", config=_config("d50_1501"))


def test_build_4d_index_from_astap_progress_and_metadata_are_json(tmp_path):
    astap = tmp_path / "astap"
    _write_catalog(astap)
    events = []

    build_4d_index_from_astap(astap, tmp_path / "direct.npz", config=_config("d50_1501"), progress_callback=events.append)
    loaded = Quad4DIndex.load(tmp_path / "direct.npz")

    assert [event["stage"] for event in events] == ["materialized_tile", "written"]
    json.dumps(loaded.metadata, sort_keys=True)
    assert loaded.metadata["provenance_fingerprint"] == loaded.metadata["source_fingerprint"]


def test_direct_payload_loads_through_strict_4d_manifest(tmp_path):
    astap = tmp_path / "astap"
    _write_catalog(astap)
    out = tmp_path / "direct.npz"
    build_4d_index_from_astap(astap, out, config=_config("d50_1501"))
    loaded = Quad4DIndex.load(out)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "manifest_version": MANIFEST_VERSION,
        "indexes": [
            {
                "id": "direct-d50-1501",
                "enabled": True,
                "path": str(out),
                "filename": out.name,
                "quad_schema": loaded.metadata["schema"],
                "index_version": loaded.metadata["version"],
                "level": loaded.metadata["level"],
                "tile_keys": list(loaded.tile_keys),
                "star_count": int(loaded.catalog_ra_dec.shape[0]),
                "quad_count": int(loaded.codes_4d.shape[0]),
                "sampler_tag": loaded.metadata["sampler_tag"],
                "sha256": sha256_file(out),
                "code_tol_recommended": loaded.metadata["code_tol_recommended"],
                "catalog_source": "astap_raw",
            }
        ],
    }
    manifest_path = tmp_path / "strict_4d_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    strict = load_4d_index_manifest(manifest_path)

    assert strict.index_ids == ("direct-d50-1501",)
    assert strict.tile_keys == ("d50_1501",)
    assert strict.entries[0].catalog_source == "astap_raw"


def _payload_tile(tile_key: str, *, empty: bool = False) -> Quad4DPayloadTile:
    if empty:
        arr64 = np.asarray([], dtype=np.float64)
        arr32 = np.asarray([], dtype=np.float32)
        return Quad4DPayloadTile(tile_key=tile_key, ra_deg=arr64, dec_deg=arr64, mag=arr32, x_deg=arr32, y_deg=arr32)
    if tile_key.endswith("_b"):
        x = np.asarray([0.0, 1.0, 0.58, 0.83, 0.15, 0.95], dtype=np.float64)
        y = np.asarray([0.0, 1.0, 0.31, 0.36, 0.85, 0.18], dtype=np.float64)
    else:
        x = np.asarray([0.0, 1.0, 0.68, 0.73, 0.2, 0.9], dtype=np.float64)
        y = np.asarray([0.0, 1.0, 0.21, 0.42, 0.9, 0.2], dtype=np.float64)
    mag = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.float32)
    return Quad4DPayloadTile(
        tile_key=tile_key,
        ra_deg=180.0 + x,
        dec_deg=45.0 + y,
        mag=mag,
        x_deg=x.astype(np.float32),
        y_deg=y.astype(np.float32),
    )


@pytest.mark.parametrize(
    ("tiles", "expected_keys"),
    [
        ([_payload_tile("empty", empty=True), _payload_tile("valid")], ("valid",)),
        ([_payload_tile("valid"), _payload_tile("empty", empty=True)], ("valid",)),
        ([_payload_tile("valid_a"), _payload_tile("empty", empty=True), _payload_tile("valid_b")], ("valid_a", "valid_b")),
    ],
)
def test_payload_tile_indices_skip_empty_tiles_without_out_of_bounds(tmp_path, tiles, expected_keys):
    out = tmp_path / "payload.npz"
    build_4d_index_from_payload_tiles(
        out,
        tiles=tiles,
        max_stars_per_tile=6,
        max_quads_per_tile=1,
        sampler_tag="legacy_brightness",
        source_catalog="test",
    )
    loaded = Quad4DIndex.load(out)

    assert loaded.tile_keys == expected_keys
    assert loaded.tile_key_indices.min() >= 0
    assert loaded.tile_key_indices.max() < len(loaded.tile_keys)
    for row_idx, tile_idx in enumerate(loaded.tile_key_indices):
        query = type(
            "Q",
            (),
            {
                "source_quad_index": 0,
                "ordered_indices": tuple(int(v) for v in loaded.quad_star_indices[row_idx]),
                "code": loaded.codes_4d[row_idx],
            },
        )()
        hits = loaded.search_records([query], code_tol=0.001, max_hits=1)
        assert any(hit.tile_key == loaded.tile_keys[int(tile_idx)] for hit in hits)
