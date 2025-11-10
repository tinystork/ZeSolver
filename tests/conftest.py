from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from zeblindsolver.levels import LEVEL_SPECS
from zeblindsolver.quad_index_builder import build_quad_index

SYNTHETIC_STAR_POS = np.array(
    [
        (0.0, 0.0),
        (4.0, 0.0),
        (4.0, 4.0),
        (0.0, 4.0),
        (1.5, 1.5),
        (2.5, 1.5),
        (2.5, 2.5),
        (1.5, 2.5),
    ],
    dtype=np.float64,
)
SYNTHETIC_STAR_MAGS = np.array([6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0], dtype=np.float64)


@pytest.fixture(scope="module")
def synthetic_star_catalog() -> tuple[np.ndarray, np.ndarray]:
    return SYNTHETIC_STAR_POS.copy(), SYNTHETIC_STAR_MAGS.copy()


@pytest.fixture(scope="module")
def synthetic_index(tmp_path_factory, synthetic_star_catalog: tuple[np.ndarray, np.ndarray]) -> Path:
    positions, mags = synthetic_star_catalog
    index_root = tmp_path_factory.mktemp("synthetic_index")
    tiles_dir = index_root / "tiles"
    tiles_dir.mkdir(parents=True)
    tile_key = "SYNTH"
    ra_center = 150.0
    dec_center = -20.0
    scale = 0.05
    mean_x = float(np.mean(positions[:, 0]))
    mean_y = float(np.mean(positions[:, 1]))
    ra_deg = (ra_center + (positions[:, 0] - mean_x) * scale).astype(np.float32)
    dec_deg = (dec_center + (positions[:, 1] - mean_y) * scale).astype(np.float32)
    tile_path = tiles_dir / f"{tile_key}.npz"
    np.savez_compressed(
        tile_path,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        mag=mags.astype(np.float32),
        x_deg=positions[:, 0].astype(np.float32),
        y_deg=positions[:, 1].astype(np.float32),
    )
    manifest = {
        "version": 1,
        "mag_cap": float(np.max(mags) + 0.5),
        "max_stars": len(positions),
        "levels": [level.to_manifest() for level in LEVEL_SPECS],
        "generated_at": "",
        "db_root": str(ROOT / "database"),
        "tile_count": 1,
        "tiles": [
            {
                "tile_index": 0,
                "tile_key": tile_key,
                "family": "SYNTH",
                "tile_code": "SYNTH001",
                "center_ra_deg": ra_center,
                "center_dec_deg": dec_center,
                "bounds": {
                    "dec_min": float(np.min(dec_deg)),
                    "dec_max": float(np.max(dec_deg)),
                    "ra_segments": [[float(np.min(ra_deg)), float(np.max(ra_deg))]],
                },
                "stars": int(len(positions)),
                "tile_file": str(tile_path.relative_to(index_root)),
                "usable_ratio": 1.0,
            }
        ],
    }
    manifest_path = index_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for level in LEVEL_SPECS:
        build_quad_index(index_root, level.name, max_quads_per_tile=1000)
    return index_root
