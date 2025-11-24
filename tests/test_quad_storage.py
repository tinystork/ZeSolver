from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import zeblindsolver.db_convert as db_convert_mod
from zeblindsolver.db_convert import build_index_from_astap
from zeblindsolver.levels import LEVEL_SPECS
from zeblindsolver.quad_index_builder import QuadIndex, build_quad_index, lookup_hashes, _INDEX_CACHE


def _create_base_index(tmp_path: Path, positions: np.ndarray, mags: np.ndarray) -> Path:
    index_root = tmp_path / "base_index"
    tiles_dir = index_root / "tiles"
    tiles_dir.mkdir(parents=True)
    tile_key = "SYNTH"
    tile_path = tiles_dir / f"{tile_key}.npz"
    np.savez_compressed(
        tile_path,
        ra_deg=positions[:, 0].astype(np.float64),
        dec_deg=positions[:, 1].astype(np.float64),
        mag=mags.astype(np.float32),
        x_deg=positions[:, 0].astype(np.float32),
        y_deg=positions[:, 1].astype(np.float32),
    )
    manifest = {
        "version": 1,
        "mag_cap": float(np.max(mags) + 1.0),
        "max_stars": int(len(positions)),
        "levels": [level.to_manifest() for level in LEVEL_SPECS],
        "generated_at": "",
        "db_root": "",
        "tile_count": 1,
        "tiles": [
            {
                "tile_index": 0,
                "tile_key": tile_key,
                "family": "TEST",
                "tile_code": "TEST001",
                "center_ra_deg": 0.0,
                "center_dec_deg": 0.0,
                "bounds": {
                    "dec_min": -1.0,
                    "dec_max": 1.0,
                    "ra_segments": [[0.0, 1.0]],
                },
                "stars": int(len(positions)),
                "tile_file": tile_path.relative_to(index_root).as_posix(),
                "usable_ratio": 1.0,
            }
        ],
    }
    manifest_path = index_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return index_root


def test_quad_index_storage_variants(tmp_path, synthetic_star_catalog):
    positions, mags = synthetic_star_catalog
    base_root = _create_base_index(tmp_path, positions, mags)
    results: dict[str, dict[str, object]] = {}
    for fmt in ("npz", "npz_uncompressed", "npy"):
        dst = tmp_path / f"index_{fmt}"
        shutil.copytree(base_root, dst)
        build_quad_index(dst, "S", max_quads_per_tile=200, storage_format=fmt)
        _INDEX_CACHE.clear()
        index = QuadIndex.load(dst, "S")
        sample = index.hashes[: min(5, index.hashes.shape[0])]
        slices = [(slc.start, slc.stop) for slc in lookup_hashes(dst, "S", sample)]
        results[fmt] = {
            "hashes": index.hashes.copy(),
            "tiles": index.tile_indices.copy(),
            "quads": index.quad_indices.copy(),
            "slices": slices,
        }
    baseline = results["npz"]
    for fmt, payload in results.items():
        assert np.array_equal(payload["hashes"], baseline["hashes"])
        assert np.array_equal(payload["tiles"], baseline["tiles"])
        assert np.array_equal(payload["quads"], baseline["quads"])
        assert payload["slices"] == baseline["slices"]


def test_quads_only_rebuild(tmp_path, synthetic_star_catalog, monkeypatch):
    s_spec = next(spec for spec in db_convert_mod.LEVEL_SPECS if spec.name == "S")
    monkeypatch.setattr(db_convert_mod, "LEVEL_SPECS", (s_spec,))
    positions, mags = synthetic_star_catalog
    index_root = _create_base_index(tmp_path, positions, mags)
    manifest = index_root / "manifest.json"
    assert manifest.exists()
    hash_dir = index_root / "hash_tables"
    if hash_dir.exists():
        shutil.rmtree(hash_dir)
    rebuilt = build_index_from_astap(
        db_root=tmp_path / "fake_db",
        index_root=index_root,
        max_quads_per_tile=128,
        quads_only=True,
        quad_storage="npz",
        tile_compression="compressed",
        workers=1,
    )
    assert rebuilt == manifest
    for level in db_convert_mod.LEVEL_SPECS:
        table = hash_dir / f"quads_{level.name}.npz"
        assert table.exists()
        with np.load(table, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"][0]))
        assert metadata["sampler"] == "pairwise_multiscale_v1"


def test_quads_only_requires_manifest(tmp_path):
    index_root = tmp_path / "no_manifest"
    index_root.mkdir()
    with pytest.raises(FileNotFoundError):
        build_index_from_astap(
            db_root=tmp_path / "any",
            index_root=index_root,
            quads_only=True,
        )
