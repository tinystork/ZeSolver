from __future__ import annotations

import numpy as np
from pathlib import Path

from zeblindsolver import zeblindsolver as solver


def _make_tile(tmp_path: Path, mags: np.ndarray) -> tuple[Path, dict[str, str]]:
    index_root = tmp_path / "index"
    tiles_dir = index_root / "tiles"
    tiles_dir.mkdir(parents=True)
    tile_path = tiles_dir / "cache_tile.npz"
    count = mags.size
    ra = np.linspace(10.0, 20.0, count, dtype=np.float64)
    dec = np.linspace(-5.0, 5.0, count, dtype=np.float64)
    x = np.linspace(0.0, 1.0, count, dtype=np.float32)
    y = np.linspace(0.0, 1.0, count, dtype=np.float32)
    np.savez_compressed(
        tile_path,
        ra_deg=ra,
        dec_deg=dec,
        mag=mags.astype(np.float32),
        x_deg=x,
        y_deg=y,
    )
    entry = {"tile_file": tile_path.relative_to(index_root).as_posix()}
    return index_root, entry


def test_tile_cache_reuses_tile_reads(tmp_path, monkeypatch):
    index_root, entry = _make_tile(tmp_path, np.array([9.0, 10.0, 11.0], dtype=np.float32))
    calls = {"count": 0}
    original_load = solver.np.load

    def counting_load(*args, **kwargs):
        calls["count"] += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(solver.np, "load", counting_load)
    solver._tile_cache_clear()
    solver._configure_tile_cache(8)
    xy1, world1 = solver._load_tile_positions(index_root, entry)
    xy2, world2 = solver._load_tile_positions(index_root, entry)
    assert calls["count"] == 1
    assert np.array_equal(xy1, xy2)
    assert np.array_equal(world1, world2)


def test_tile_cache_invalidates_on_file_change(tmp_path, monkeypatch):
    index_root, entry = _make_tile(tmp_path, np.array([8.0, 9.0, 10.0, 11.0], dtype=np.float32))
    tile_path = index_root / Path(entry["tile_file"])
    calls = {"count": 0}
    original_load = solver.np.load

    def counting_load(*args, **kwargs):
        calls["count"] += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(solver.np, "load", counting_load)
    solver._tile_cache_clear()
    solver._configure_tile_cache(8)
    solver._load_tile_positions(index_root, entry)
    # Rewrite tile with modified content
    mags = np.array([7.0, 8.0, 9.0, 10.0, 11.0], dtype=np.float32)
    np.savez_compressed(
        tile_path,
        ra_deg=np.linspace(30.0, 40.0, mags.size, dtype=np.float64),
        dec_deg=np.linspace(-15.0, -5.0, mags.size, dtype=np.float64),
        mag=mags,
        x_deg=np.linspace(0.2, 1.2, mags.size, dtype=np.float32),
        y_deg=np.linspace(0.3, 1.3, mags.size, dtype=np.float32),
    )
    xy2, world2 = solver._load_tile_positions(index_root, entry)
    assert calls["count"] == 2
    assert xy2.shape[0] == mags.size
    assert world2.shape[0] == mags.size
