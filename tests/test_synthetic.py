from __future__ import annotations

import numpy as np
from pathlib import Path

from zeblindsolver.asterisms import hash_quads, sample_quads
from zeblindsolver.candidate_search import tally_candidates
from zeblindsolver.verify import validate_solution
from zeblindsolver.wcs_fit import fit_wcs_tan


def _load_tile_arrays(index_root: Path) -> tuple[np.ndarray, np.ndarray]:
    tiles_dir = index_root / "tiles"
    tile_file = next(tiles_dir.iterdir())
    with np.load(tile_file) as data:
        ra = data["ra_deg"].astype(np.float32)
        dec = data["dec_deg"].astype(np.float32)
    return ra, dec


def test_synthetic_index_produces_candidate(synthetic_index, synthetic_star_catalog):
    positions, mags = synthetic_star_catalog
    ra_deg, dec_deg = _load_tile_arrays(synthetic_index)
    image_points = positions.astype(np.float32)
    obs_stars = np.zeros(image_points.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("mag", "f4")])
    obs_stars["x"] = image_points[:, 0]
    obs_stars["y"] = image_points[:, 1]
    obs_stars["mag"] = mags.astype(np.float32)

    quads = sample_quads(obs_stars, max_quads=200)
    assert quads.size > 0
    obs_hash = hash_quads(quads, image_points)
    assert obs_hash.hashes.size > 0

    candidates = tally_candidates(obs_hash.hashes, synthetic_index, levels=["L", "M", "S"])
    assert candidates and candidates[0][0] == "SYNTH"

    matches = np.column_stack((image_points, np.column_stack((ra_deg, dec_deg))))
    wcs, _ = fit_wcs_tan(matches)
    stats = validate_solution(wcs, matches, thresholds={"rms_px": 1.0, "inliers": 4})
    assert stats["quality"] == "GOOD"
    assert stats["rms_px"] < 1.0


def test_tally_candidates_respects_allowed_tiles(synthetic_index, synthetic_star_catalog):
    positions, mags = synthetic_star_catalog
    image_points = positions.astype(np.float32)
    obs_stars = np.zeros(image_points.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("mag", "f4")])
    obs_stars["x"] = image_points[:, 0]
    obs_stars["y"] = image_points[:, 1]
    obs_stars["mag"] = mags.astype(np.float32)

    quads = sample_quads(obs_stars, max_quads=100)
    assert quads.size > 0
    obs_hash = hash_quads(quads, image_points)
    assert obs_hash.hashes.size > 0

    unrestricted = tally_candidates(obs_hash.hashes, synthetic_index, levels=["L", "M", "S"])
    assert unrestricted and unrestricted[0][0] == "SYNTH"

    # Restrict to a non-existent tile index; expect no candidates
    restricted_none = tally_candidates(
        obs_hash.hashes,
        synthetic_index,
        levels=["L", "M", "S"],
        allowed_tiles={42},
    )
    assert restricted_none == []

    # Restrict to the actual tile index (0) and ensure it still returns the candidate
    restricted_hit = tally_candidates(
        obs_hash.hashes,
        synthetic_index,
        levels=["L", "M", "S"],
        allowed_tiles={0},
    )
    assert restricted_hit and restricted_hit[0][0] == "SYNTH"
