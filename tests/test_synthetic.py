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

import itertools

import numpy as np
from pathlib import Path

from zeblindsolver.asterisms import hash_quads, opposite_edge_ratio_code, sample_quads
from zeblindsolver.zeblindsolver import _hash_from_ordered_quad_points, _quad_ratio_code
from zeblindsolver.candidate_search import tally_candidates
from zeblindsolver.verify import validate_solution
from zeblindsolver.wcs_fit import fit_wcs_tan


def test_quad_hash_is_invariant_to_vertex_ids_and_similarity():
    points = np.array(
        [
            [0.2, -0.4],
            [3.8, 0.7],
            [2.9, 4.6],
            [-1.1, 2.3],
        ],
        dtype=np.float64,
    )
    baseline = int(hash_quads(np.array([[0, 1, 2, 3]]), points).hashes[0])

    for permutation in itertools.permutations(range(4)):
        renumbered = points[list(permutation)]
        hashed = hash_quads(np.array([[0, 1, 2, 3]]), renumbered)
        assert int(hashed.hashes[0]) == baseline

    angle = np.deg2rad(37.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    transformed = (points @ rotation.T) * 7.25 + np.array([123.0, -48.0])
    reflected = transformed * np.array([-1.0, 1.0])
    assert int(hash_quads(np.array([[0, 1, 2, 3]]), transformed).hashes[0]) == baseline
    assert int(hash_quads(np.array([[0, 1, 2, 3]]), reflected).hashes[0]) == baseline

    components = (
        (baseline >> 48) & 0xFFFF,
        (baseline >> 32) & 0xFFFF,
        (baseline >> 16) & 0xFFFF,
    )
    assert all(0 <= value <= 255 for value in components)
    assert baseline & 0xFFFF == 0

    expected_code = opposite_edge_ratio_code(points)
    assert expected_code is not None
    assert np.allclose(_quad_ratio_code(points), expected_code)
    assert _hash_from_ordered_quad_points(points) == baseline


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


def test_catalog_coverage_first_spreads_fixed_budget_across_seeds():
    rng = np.random.default_rng(42)
    count = 160
    xs, ys = np.meshgrid(np.linspace(0.0, 1.0, 20), np.linspace(0.0, 1.0, 8))
    points = np.column_stack((xs.ravel(), ys.ravel()))[:count]
    points += rng.normal(0.0, 0.003, points.shape)
    stars = np.zeros(count, dtype=[("x", "f4"), ("y", "f4"), ("mag", "f4")])
    stars["x"] = points[:, 0].astype(np.float32)
    stars["y"] = points[:, 1].astype(np.float32)
    stars["mag"] = np.arange(count, dtype=np.float32)

    legacy = sample_quads(stars, max_quads=240, strategy="log_spaced")
    coverage = sample_quads(stars, max_quads=240, strategy="catalog_coverage_first")

    assert legacy.shape[0] == coverage.shape[0] == 240
    assert np.unique(coverage[:, 0]).shape[0] > np.unique(legacy[:, 0]).shape[0] * 10
    assert np.unique(coverage).shape[0] > np.unique(legacy).shape[0] * 2


def test_catalog_ring_coverage_reaches_mid_rank_local_geometry():
    count = 180
    xs, ys = np.meshgrid(np.linspace(0.0, 1.0, 20), np.linspace(0.0, 1.0, 9))
    points = np.column_stack((xs.ravel(), ys.ravel()))[:count]
    stars = np.zeros(count, dtype=[("x", "f4"), ("y", "f4"), ("mag", "f4")])
    stars["x"] = points[:, 0].astype(np.float32)
    stars["y"] = points[:, 1].astype(np.float32)
    stars["mag"] = np.arange(count, dtype=np.float32)

    target = tuple(sorted((41, 42, 61, 84)))
    legacy = sample_quads(stars, max_quads=1200, strategy="catalog_coverage_first")
    ring = sample_quads(stars, max_quads=1200, strategy="catalog_ring_coverage")

    legacy_sets = {tuple(sorted(map(int, row))) for row in legacy}
    ring_sets = {tuple(sorted(map(int, row))) for row in ring}

    assert target not in legacy_sets
    assert target in ring_sets
    assert ring.shape[0] == 1200


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
