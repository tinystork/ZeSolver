# """
# STANDARDIZED_PROJECT_HEADER_V1
# ╔═══════════════════════════════════════════════════════════════════════════════════╗
# ║ ZeSolver Project (ZeMosaic / ZeSeestarStacker ecosystem)                         ║
# ║                                                                                   ║
# ║ Auteur principal : Tinystork (Tristan Nauleau)                                   ║
# ║ Partenaire IA   : J.A.R.V.I.S. (OpenAI ChatGPT)                                  ║
# ║                                                                                   ║
# ║ Licence du dépôt : MIT (voir pyproject.toml / repository metadata)               ║
# ║                                                                                   ║
# ║ Remerciements amont :                                                             ║
# ║ - ASTAP, par Han Kleijn                                                           ║
# ║ - Astrometry.net, par Dustin Lang, David W. Hogg, Keir Mierle, et al.            ║
# ║                                                                                   ║
# ║ Description FR :                                                                  ║
# ║ Ce code sert à transformer des nuages de photons en solutions WCS et en images   ║
# ║ astronomiques exploitables. Merci de créditer les auteurs et projets amont lors   ║
# ║ de toute réutilisation.                                                           ║
# ║                                                                                   ║
# ║ EN Description:                                                                    ║
# ║ This code helps turn clouds of photons into usable WCS solutions and astronomical ║
# ║ imagery outputs. Please credit both project authors and upstream references when  ║
# ║ reusing this work.                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════════════╝
# """

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
    stats = validate_solution(
        wcs,
        matches,
        thresholds={"rms_px": 1.0, "inliers": 4, "scale_max_arcsec": 300.0},
    )
    assert stats["quality"] == "GOOD"
    assert stats["rms_px"] < 1.0


def test_validate_solution_parity_mode_keeps_fail_quality_but_marks_progress(synthetic_index, synthetic_star_catalog):
    positions, _mags = synthetic_star_catalog
    ra_deg, dec_deg = _load_tile_arrays(synthetic_index)
    image_points = positions.astype(np.float32)
    matches = np.column_stack((image_points, np.column_stack((ra_deg, dec_deg))))
    wcs, _ = fit_wcs_tan(matches)
    stats = validate_solution(
        wcs,
        matches,
        thresholds={"rms_px": 1.0, "inliers": 10, "astrometry_parity_mode": True},
    )
    assert stats["quality"] == "FAIL"
    assert stats["success"] is False
    assert stats["validation_metrics_only"] is True
    assert stats["validation_progress_eligible"] is True
    assert str(stats["reason"]).startswith("validation_metrics_only[")


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


def test_tally_candidates_accepts_weighted_hashes(synthetic_index, synthetic_star_catalog):
    positions, mags = synthetic_star_catalog
    image_points = positions.astype(np.float32)
    obs_stars = np.zeros(image_points.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("mag", "f4")])
    obs_stars["x"] = image_points[:, 0]
    obs_stars["y"] = image_points[:, 1]
    obs_stars["mag"] = mags.astype(np.float32)

    quads = sample_quads(obs_stars, max_quads=120)
    obs_hash = hash_quads(quads, image_points)
    assert obs_hash.hashes.size > 0
    duplicated = np.repeat(obs_hash.hashes, 3)
    baseline = tally_candidates(duplicated, synthetic_index, levels=["L", "M", "S"])
    unique_hashes, _, counts = np.unique(duplicated, return_index=True, return_counts=True)
    weighted = tally_candidates((unique_hashes, counts), synthetic_index, levels=["L", "M", "S"])
    assert baseline == weighted
    restricted = tally_candidates(
        (unique_hashes, counts),
        synthetic_index,
        levels=["L", "M", "S"],
        allowed_tiles={0},
    )
    assert restricted and restricted[0][0] == "SYNTH"
