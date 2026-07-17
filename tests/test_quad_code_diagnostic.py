from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import fit_wcs_from_points
from scipy.spatial import cKDTree

from zeblindsolver.asterisms import hash_quads
from zeblindsolver.zeblindsolver import (
    SolveConfig,
    _astrometry_4d_accept_policy,
    _astrometry_4d_accept_sort_key,
    _astrometry_4d_dedup_catalog_world,
    _astrometry_4d_index_paths,
    _astrometry_4d_runtime_requested,
    _astrometry_4d_source_policy,
    _astrometry_4d_validate_pixel_matches,
    _astrometry_4d_validation_catalog_policy,
    _solve_astrometry_4d_runtime_route,
)
from zeblindsolver.quad_index_4d import (
    ASTROMETRY_AB_CODE_4D_SCHEMA,
    Quad4DIndex,
    build_experimental_4d_index,
)
from zeblindsolver.quad_code_diagnostic import (
    astrometry_ab_code_4d,
    astrometry_code_flip_parity,
    build_memory_quad_code_index,
    canonicalize_astrometry_abcd,
)


def _points_from_code(code: np.ndarray) -> np.ndarray:
    cx, cy, dx, dy = np.asarray(code, dtype=np.float64)
    return np.asarray(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [cy, cx],
            [dy, dx],
        ],
        dtype=np.float64,
    )


def test_astrometry_ab_code_is_similarity_invariant() -> None:
    points = _points_from_code(np.asarray([0.21, 0.68, 0.42, 0.73], dtype=np.float64))
    baseline = astrometry_ab_code_4d(points)
    assert baseline is not None

    angle = np.deg2rad(31.0)
    rot = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    moved = points @ rot.T * 8.5 + np.asarray([123.0, -77.0])
    transformed = astrometry_ab_code_4d(moved)

    assert transformed is not None
    assert np.allclose(transformed, baseline, atol=1e-12)


def test_ab_inversion_returns_one_minus_code() -> None:
    code = np.asarray([0.22, 0.61, 0.41, 0.74], dtype=np.float64)
    points = _points_from_code(code)

    inverted = astrometry_ab_code_4d(points[[1, 0, 2, 3]])

    assert inverted is not None
    assert np.allclose(inverted, 1.0 - code, atol=1e-12)


def test_canonicalization_orders_cd_and_enforces_mean_x() -> None:
    points = _points_from_code(np.asarray([0.72, 0.40, 0.62, 0.20], dtype=np.float64))

    canonical = canonicalize_astrometry_abcd(points)

    assert canonical is not None
    cx, _cy, dx, _dy = canonical.code
    assert cx <= dx
    assert (cx + dx) <= 1.0


def test_astrometry_parity_flip_swaps_code_axes() -> None:
    code = np.asarray([0.23, 0.62, 0.41, 0.77], dtype=np.float64)

    flipped = astrometry_code_flip_parity(code)

    assert np.allclose(flipped, np.asarray([0.62, 0.23, 0.77, 0.41]))


def test_exact_ratio_hash_can_miss_when_4d_range_search_succeeds() -> None:
    base = np.asarray([0.21, 0.68, 0.42, 0.73], dtype=np.float64)
    base_points = _points_from_code(base)
    base_hash = int(hash_quads(np.asarray([[0, 1, 2, 3]], dtype=np.uint16), base_points).hashes[0])

    found = None
    for delta in np.linspace(0.001, 0.014, 14):
        candidate = base + np.asarray([delta, 0.0, 0.0, -delta], dtype=np.float64)
        candidate_points = _points_from_code(candidate)
        candidate_hash = int(hash_quads(np.asarray([[0, 1, 2, 3]], dtype=np.uint16), candidate_points).hashes[0])
        if candidate_hash != base_hash and np.linalg.norm(candidate - base) < 0.02:
            found = (candidate, candidate_hash)
            break

    assert found is not None
    candidate, candidate_hash = found
    assert candidate_hash != base_hash

    tree = cKDTree(np.asarray([base], dtype=np.float64))
    neighbors = tree.query_ball_point(np.asarray([candidate], dtype=np.float64), r=0.02)
    assert [list(row) for row in neighbors] == [[0]]


def test_memory_4d_index_finds_noisy_neighbor() -> None:
    base = _points_from_code(np.asarray([0.21, 0.68, 0.42, 0.73], dtype=np.float64))
    quads = np.asarray([[0, 1, 2, 3]], dtype=np.uint16)
    index = build_memory_quad_code_index(quads, base, tile_key="synthetic")

    query = astrometry_ab_code_4d(
        _points_from_code(np.asarray([0.214, 0.681, 0.419, 0.728], dtype=np.float64))
    )

    assert query is not None
    hits = index.query(query, code_tol=0.01)
    assert len(hits) == 1
    assert hits[0][0] == 0


def test_synthetic_4d_hit_can_fit_coherent_wcs() -> None:
    image_points = np.asarray(
        [
            [100.0, 120.0],
            [300.0, 320.0],
            [168.0, 222.0],
            [226.0, 258.0],
        ],
        dtype=np.float64,
    )
    ra0, dec0 = 184.0, 47.0
    scale = 0.00065
    world = np.column_stack(
        (
            ra0 + (image_points[:, 0] - 200.0) * scale,
            dec0 + (image_points[:, 1] - 220.0) * scale,
        )
    )
    coords = SkyCoord(ra=world[:, 0] * u.deg, dec=world[:, 1] * u.deg, frame="icrs")
    wcs = fit_wcs_from_points((image_points[:, 0], image_points[:, 1]), coords, projection="TAN")
    projected = np.asarray(wcs.all_world2pix(world[:, 0], world[:, 1], 0), dtype=np.float64).T

    assert np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1))) < 0.1


def test_building_memory_4d_index_does_not_change_ratio_hash_backend() -> None:
    points = _points_from_code(np.asarray([0.21, 0.68, 0.42, 0.73], dtype=np.float64))
    quads = np.asarray([[0, 1, 2, 3]], dtype=np.uint16)
    before = int(hash_quads(quads, points).hashes[0])

    _index = build_memory_quad_code_index(quads, points, tile_key="synthetic")

    after = int(hash_quads(quads, points).hashes[0])
    assert after == before


def test_disk_4d_index_roundtrip_and_search(tmp_path) -> None:
    index_root = tmp_path / "index"
    tiles = index_root / "tiles"
    tiles.mkdir(parents=True)
    points = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.68, 0.21],
            [0.73, 0.42],
            [0.2, 0.9],
            [0.9, 0.2],
        ],
        dtype=np.float64,
    )
    tile_path = tiles / "SYNTH.npz"
    np.savez_compressed(
        tile_path,
        ra_deg=(180.0 + points[:, 0]).astype(np.float64),
        dec_deg=(45.0 + points[:, 1]).astype(np.float64),
        x_deg=points[:, 0].astype(np.float32),
        y_deg=points[:, 1].astype(np.float32),
        mag=np.arange(points.shape[0], dtype=np.float32),
    )
    manifest = {
        "tiles": [
            {
                "tile_index": 0,
                "tile_key": "SYNTH",
                "tile_file": "tiles/SYNTH.npz",
            }
        ]
    }
    (index_root / "manifest.json").write_text(__import__("json").dumps(manifest), encoding="utf-8")

    out = build_experimental_4d_index(
        index_root,
        tmp_path / "quad4d.npz",
        tile_keys=["SYNTH"],
        max_stars_per_tile=6,
        max_quads_per_tile=8,
    )
    loaded = Quad4DIndex.load(out)
    query = SimpleNamespace(
        source_quad_index=0,
        ordered_indices=tuple(int(v) for v in loaded.quad_star_indices[0]),
        code=loaded.codes_4d[0],
    )
    hits = loaded.search_records([query], code_tol=0.001)

    assert loaded.metadata["schema"] == ASTROMETRY_AB_CODE_4D_SCHEMA
    assert loaded.codes_4d.shape[1] == 4
    assert loaded.quad_star_indices.shape[1] == 4
    assert hits
    assert hits[0].tile_key == "SYNTH"


def test_disk_4d_index_build_does_not_change_ratio_hash_backend(tmp_path) -> None:
    index_root = tmp_path / "index"
    tiles = index_root / "tiles"
    tiles.mkdir(parents=True)
    points = _points_from_code(np.asarray([0.21, 0.68, 0.42, 0.73], dtype=np.float64))
    quads = np.asarray([[0, 1, 2, 3]], dtype=np.uint16)
    baseline = int(hash_quads(quads, points).hashes[0])
    tile_path = tiles / "SYNTH.npz"
    np.savez_compressed(
        tile_path,
        ra_deg=(180.0 + points[:, 0]).astype(np.float64),
        dec_deg=(45.0 + points[:, 1]).astype(np.float64),
        x_deg=points[:, 0].astype(np.float32),
        y_deg=points[:, 1].astype(np.float32),
        mag=np.arange(points.shape[0], dtype=np.float32),
    )
    (index_root / "manifest.json").write_text(
        __import__("json").dumps({"tiles": [{"tile_index": 0, "tile_key": "SYNTH", "tile_file": "tiles/SYNTH.npz"}]}),
        encoding="utf-8",
    )

    build_experimental_4d_index(index_root, tmp_path / "quad4d.npz", tile_keys=["SYNTH"], max_stars_per_tile=4, max_quads_per_tile=1)

    assert int(hash_quads(quads, points).hashes[0]) == baseline


def test_runtime_4d_route_is_off_by_default() -> None:
    cfg = SolveConfig()

    assert cfg.quad_hash_schema == "opposite_edge_ratio_8bit_v1"
    assert cfg.blind_astrometry_4d_index_enabled is False
    assert cfg.blind_astrometry_4d_index_paths == ()
    assert cfg.blind_astrometry_4d_validation_catalog_policy == "source_tile"
    assert cfg.blind_astrometry_4d_accept_policy == "first_accept"
    assert cfg.blind_astrometry_4d_source_policy == "standard_runtime"
    assert cfg.blind_star_quality_filter is True
    assert _astrometry_4d_runtime_requested(cfg) is False
    assert _astrometry_4d_source_policy(cfg) == "standard_runtime"
    assert _astrometry_4d_validation_catalog_policy(cfg) == "source_tile"
    assert _astrometry_4d_accept_policy(cfg) == "first_accept"


def test_runtime_4d_route_requires_explicit_schema() -> None:
    cfg_flag_only = SolveConfig(
        blind_astrometry_4d_index_enabled=True,
        blind_astrometry_4d_index_path="/tmp/quad4d.npz",
        blind_astrometry_4d_source_policy="diagnostic_unfiltered",
    )
    cfg_paths_only = SolveConfig(
        blind_astrometry_4d_index_paths=("/tmp/quad4d-a.npz", "/tmp/quad4d-b.npz"),
        blind_astrometry_4d_validation_catalog_policy="union_candidate_tiles",
        blind_astrometry_4d_accept_policy="best_within_budget",
    )
    cfg_schema = SolveConfig(quad_hash_schema=ASTROMETRY_AB_CODE_4D_SCHEMA)

    assert _astrometry_4d_runtime_requested(cfg_flag_only) is False
    assert _astrometry_4d_runtime_requested(cfg_paths_only) is False
    assert _astrometry_4d_runtime_requested(cfg_schema) is True


def test_runtime_4d_source_policy_accepts_diagnostic_opt_in() -> None:
    cfg = SolveConfig(
        quad_hash_schema=ASTROMETRY_AB_CODE_4D_SCHEMA,
        blind_astrometry_4d_index_enabled=True,
        blind_astrometry_4d_index_path="/tmp/quad4d.npz",
        blind_astrometry_4d_source_policy="diagnostic_unfiltered",
    )

    assert _astrometry_4d_runtime_requested(cfg) is True
    assert _astrometry_4d_source_policy(cfg) == "diagnostic_unfiltered"
    assert cfg.blind_star_quality_filter is True


def test_runtime_4d_multi_index_policy_helpers_are_explicit() -> None:
    cfg = SolveConfig(
        quad_hash_schema=ASTROMETRY_AB_CODE_4D_SCHEMA,
        blind_astrometry_4d_index_paths=("a.npz", "b.npz", "a.npz"),
        blind_astrometry_4d_validation_catalog_policy="union_candidate_tiles",
        blind_astrometry_4d_accept_policy="best_within_budget",
    )

    assert _astrometry_4d_runtime_requested(cfg) is True
    assert _astrometry_4d_index_paths(cfg) == ("a.npz", "b.npz")
    assert _astrometry_4d_validation_catalog_policy(cfg) == "union_candidate_tiles"
    assert _astrometry_4d_accept_policy(cfg) == "best_within_budget"


def test_runtime_4d_index_paths_keeps_legacy_single_path_guarded_by_schema() -> None:
    cfg = SolveConfig(
        quad_hash_schema=ASTROMETRY_AB_CODE_4D_SCHEMA,
        blind_astrometry_4d_index_enabled=True,
        blind_astrometry_4d_index_path="/tmp/quad4d.npz",
    )

    assert _astrometry_4d_index_paths(cfg) == ("/tmp/quad4d.npz",)


def test_runtime_4d_schema_without_index_paths_fails_explicitly() -> None:
    cfg = SolveConfig(quad_hash_schema=ASTROMETRY_AB_CODE_4D_SCHEMA)
    obs = np.zeros(0, dtype=[("x", "f4"), ("y", "f4"), ("mag", "f4")])

    result = _solve_astrometry_4d_runtime_route(
        config=cfg,
        obs_stars=obs,
        image_positions_solver=np.empty((0, 2), dtype=np.float64),
        image_shape=(100, 100),
        scale_bounds_arcsec=None,
        cancel_check=None,
    )

    assert result.success is False
    assert "missing explicit 4D index paths" in result.message
    assert result.stats["astrometry_4d_fallback_forbidden"] is True
    assert result.stats["astrometry_4d_stop_reason"] == "missing_explicit_index_paths"


def test_runtime_4d_absent_index_fails_without_fallback(tmp_path) -> None:
    missing = tmp_path / "missing_quad4d.npz"
    cfg = SolveConfig(
        quad_hash_schema=ASTROMETRY_AB_CODE_4D_SCHEMA,
        blind_astrometry_4d_index_paths=(str(missing),),
        blind_astrometry_4d_validation_catalog_policy="union_candidate_tiles",
        blind_astrometry_4d_accept_policy="best_within_budget",
    )
    obs = np.zeros(4, dtype=[("x", "f4"), ("y", "f4"), ("mag", "f4")])
    obs["x"] = [10.0, 20.0, 30.0, 40.0]
    obs["y"] = [10.0, 20.0, 30.0, 40.0]

    result = _solve_astrometry_4d_runtime_route(
        config=cfg,
        obs_stars=obs,
        image_positions_solver=np.column_stack((obs["x"], obs["y"])),
        image_shape=(100, 100),
        scale_bounds_arcsec=None,
        cancel_check=None,
    )

    assert result.success is False
    assert "missing explicit index path" in result.message
    assert result.stats["astrometry_4d_stop_reason"] == "index_absent"
    assert result.stats["astrometry_4d_fallback_forbidden"] is True
    assert result.stats["reject_reason_counts"] == {"index_absent": 1}


def test_runtime_4d_incompatible_index_fails_without_fallback(tmp_path) -> None:
    bad = tmp_path / "tile_catalog_only.npz"
    np.savez_compressed(
        bad,
        ra_deg=np.asarray([10.0, 11.0], dtype=np.float64),
        dec_deg=np.asarray([20.0, 21.0], dtype=np.float64),
    )
    cfg = SolveConfig(
        quad_hash_schema=ASTROMETRY_AB_CODE_4D_SCHEMA,
        blind_astrometry_4d_index_paths=(str(bad),),
    )
    obs = np.zeros(0, dtype=[("x", "f4"), ("y", "f4"), ("mag", "f4")])

    result = _solve_astrometry_4d_runtime_route(
        config=cfg,
        obs_stars=obs,
        image_positions_solver=np.empty((0, 2), dtype=np.float64),
        image_shape=(100, 100),
        scale_bounds_arcsec=None,
        cancel_check=None,
    )

    assert result.success is False
    assert result.stats["astrometry_4d_stop_reason"] == "index_load_failed"
    assert result.stats["astrometry_4d_fallback_forbidden"] is True


def test_runtime_4d_explicit_multi_index_order_is_deterministic() -> None:
    cfg = SolveConfig(
        quad_hash_schema=ASTROMETRY_AB_CODE_4D_SCHEMA,
        blind_astrometry_4d_index_paths=("field-a.npz", "field-b.npz"),
        blind_astrometry_4d_validation_catalog_policy="union_candidate_tiles",
        blind_astrometry_4d_accept_policy="best_within_budget",
    )
    reversed_cfg = SolveConfig(
        quad_hash_schema=ASTROMETRY_AB_CODE_4D_SCHEMA,
        blind_astrometry_4d_index_paths=("field-b.npz", "field-a.npz"),
        blind_astrometry_4d_validation_catalog_policy="union_candidate_tiles",
        blind_astrometry_4d_accept_policy="best_within_budget",
    )

    assert _astrometry_4d_index_paths(cfg) == ("field-a.npz", "field-b.npz")
    assert _astrometry_4d_index_paths(reversed_cfg) == ("field-b.npz", "field-a.npz")
    assert _astrometry_4d_validation_catalog_policy(cfg) == "union_candidate_tiles"
    assert _astrometry_4d_accept_policy(cfg) == "best_within_budget"


def test_runtime_4d_has_no_all_sky_autodiscovery_knob() -> None:
    cfg = SolveConfig(quad_hash_schema=ASTROMETRY_AB_CODE_4D_SCHEMA)

    assert not hasattr(cfg, "blind_astrometry_4d_all_sky")
    assert not hasattr(cfg, "blind_astrometry_4d_auto_discover_indexes")
    assert _astrometry_4d_index_paths(cfg) == ()


def test_runtime_4d_union_catalog_deduplicates_ra_dec() -> None:
    first = np.asarray([[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]], dtype=np.float64)
    second = np.asarray([[11.0, 21.0], [12.0 + 1e-10, 22.0 - 1e-10], [13.0, 23.0]], dtype=np.float64)

    union = _astrometry_4d_dedup_catalog_world([first, second])

    assert union.shape == (4, 2)
    assert np.allclose(union[0], [10.0, 20.0])
    assert np.allclose(union[-1], [13.0, 23.0])


def test_runtime_4d_best_within_budget_prefers_rms_before_inliers() -> None:
    early = {
        "rank": 1,
        "validation": {"quality": "GOOD", "success": True, "rms_px": 0.9, "inliers": 70, "geo_cov_area": 0.20},
    }
    better_rms = {
        "rank": 9,
        "validation": {"quality": "GOOD", "success": True, "rms_px": 0.3, "inliers": 57, "geo_cov_area": 0.18},
    }

    assert max([early, better_rms], key=_astrometry_4d_accept_sort_key) is better_rms


def _simple_tan_wcs():
    image = np.asarray(
        [
            [100.0, 100.0],
            [200.0, 100.0],
            [100.0, 200.0],
            [200.0, 200.0],
        ],
        dtype=np.float64,
    )
    scale_deg = 2.373471335866922 / 3600.0
    world = np.column_stack(
        (
            184.0 + (image[:, 0] - 150.0) * scale_deg,
            47.0 + (image[:, 1] - 150.0) * scale_deg,
        )
    )
    coords = SkyCoord(ra=world[:, 0] * u.deg, dec=world[:, 1] * u.deg, frame="icrs")
    return fit_wcs_from_points((image[:, 0], image[:, 1]), coords, projection="TAN")


def test_runtime_4d_final_metric_uses_supplied_world2pix_distances() -> None:
    wcs = _simple_tan_wcs()
    matches = np.zeros((40, 4), dtype=np.float64)
    # Deliberately nonsense sky columns: the direct 4D contract must use the
    # already-collected pixel distances, not recompute these pairs in reverse.
    matches[:, 2:] = 999.0
    distances = np.full(40, 1.1, dtype=np.float64)

    validation = _astrometry_4d_validate_pixel_matches(
        wcs,
        matches,
        distances,
        {"rms_px": 1.2, "inliers": 40, "scale_min_arcsec": 1.79, "scale_max_arcsec": 2.99},
    )

    assert validation["quality"] == "GOOD"
    assert validation["success"] is True
    assert validation["inliers"] == 40
    assert validation["rms_px"] == 1.1
    assert validation["residual_metric"] == "catalog_world2pix_to_image_px"


def test_runtime_4d_final_metric_is_pair_order_invariant() -> None:
    wcs = _simple_tan_wcs()
    matches = np.zeros((42, 4), dtype=np.float64)
    distances = np.linspace(0.2, 1.5, 42, dtype=np.float64)
    order = np.arange(41, -1, -1)

    thresholds = {"rms_px": 1.2, "inliers": 40, "scale_min_arcsec": 1.79, "scale_max_arcsec": 2.99}
    base = _astrometry_4d_validate_pixel_matches(wcs, matches, distances, thresholds)
    shuffled = _astrometry_4d_validate_pixel_matches(wcs, matches[order], distances[order], thresholds)

    assert base["success"] is True
    assert shuffled["success"] is True
    assert base["inliers"] == shuffled["inliers"]
    assert base["rms_px"] == shuffled["rms_px"]


def test_runtime_4d_legacy_inverse_metrics_do_not_affect_accept_sorting() -> None:
    accepted_direct_with_bad_legacy = {
        "rank": 95,
        "validation": {
            "quality": "GOOD",
            "success": True,
            "rms_px": 1.1021042545198643,
            "inliers": 42,
            "geo_cov_area": 0.82,
            "legacy_inverse_quality": "FAIL",
            "legacy_inverse_inliers": 41,
            "legacy_inverse_rms_px": 1.2033882037876618,
        },
    }
    rejected_direct_with_good_legacy = {
        "rank": 92,
        "validation": {
            "quality": "FAIL",
            "success": False,
            "rms_px": 0.5,
            "inliers": 10,
            "geo_cov_area": 0.9,
            "legacy_inverse_quality": "GOOD",
            "legacy_inverse_inliers": 80,
            "legacy_inverse_rms_px": 0.2,
        },
    }

    chosen = max(
        [accepted_direct_with_bad_legacy, rejected_direct_with_good_legacy],
        key=_astrometry_4d_accept_sort_key,
    )

    assert chosen is accepted_direct_with_bad_legacy


def test_runtime_4d_234013_boundary_direct_accepts_legacy_rejects() -> None:
    wcs = _simple_tan_wcs()
    matches = np.zeros((42, 4), dtype=np.float64)
    direct_distances = np.full(42, 1.1021042545198643, dtype=np.float64)
    thresholds = {"rms_px": 1.2, "inliers": 40, "scale_min_arcsec": 1.79, "scale_max_arcsec": 2.99}

    direct = _astrometry_4d_validate_pixel_matches(wcs, matches, direct_distances, thresholds)
    legacy_inverse = {
        "quality": "FAIL",
        "success": False,
        "inliers": 41,
        "rms_px": 1.2033882037876618,
    }
    accepted_row = {
        "rank": 95,
        "validation": {
            **direct,
            "legacy_inverse_quality": legacy_inverse["quality"],
            "legacy_inverse_inliers": legacy_inverse["inliers"],
            "legacy_inverse_rms_px": legacy_inverse["rms_px"],
        },
    }

    assert direct["quality"] == "GOOD"
    assert direct["success"] is True
    assert direct["inliers"] == 42
    assert direct["rms_px"] == 1.1021042545198643
    assert legacy_inverse["quality"] == "FAIL"
    assert legacy_inverse["success"] is False
    assert _astrometry_4d_accept_sort_key(accepted_row)[0] == 1
