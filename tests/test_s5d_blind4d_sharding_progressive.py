from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from zeblindsolver.quad_index_4d import ASTROMETRY_AB_CODE_4D_SCHEMA, Quad4DIndex
from zeblindsolver.zeblindsolver import SolveConfig, WcsSolution


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("shard_blind4d_index", ROOT / "tools" / "shard_blind4d_index.py")
assert SPEC is not None and SPEC.loader is not None
sharder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sharder
SPEC.loader.exec_module(sharder)


def _metadata_array(metadata: dict[str, object]) -> np.ndarray:
    text = json.dumps(metadata, sort_keys=True)
    return np.asarray([text], dtype=f"<U{len(text)}")


def _write_fake_monolith(path: Path) -> Path:
    tile_keys = np.asarray(["d50_0101", "d50_0102", "d50_0201", "d50_0202"], dtype="<U8")
    tile_count = len(tile_keys)
    stars_per_tile = 3
    quads_per_tile = 2
    catalog = np.arange(tile_count * stars_per_tile * 2, dtype=np.float64).reshape(tile_count * stars_per_tile, 2)
    xy = catalog + 1000.0
    codes = (np.arange(tile_count * quads_per_tile * 4, dtype=np.float32).reshape(tile_count * quads_per_tile, 4) / 1000.0)
    qsi = []
    tile_idx = []
    source_idx = []
    ratio_hashes = []
    for tile in range(tile_count):
        for quad in range(quads_per_tile):
            base_star = tile * stars_per_tile
            qsi.append([base_star, base_star + 1, base_star + 2, base_star])
            tile_idx.append(tile)
            source_idx.append(quad)
            ratio_hashes.append(tile * 100 + quad)
    metadata = {
        "schema": ASTROMETRY_AB_CODE_4D_SCHEMA,
        "version": 1,
        "level": "S",
        "sampler_tag": "test",
        "source_catalog": "unit",
        "code_tol_recommended": 0.015,
    }
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata=_metadata_array(metadata),
            codes_4d=codes,
            quad_star_indices=np.asarray(qsi, dtype=np.int32),
            source_quad_indices=np.asarray(source_idx, dtype=np.int32),
            tile_key_indices=np.asarray(tile_idx, dtype=np.int32),
            ratio_hashes=np.asarray(ratio_hashes, dtype=np.int64),
            tile_keys=tile_keys,
            catalog_ra_dec=catalog,
            catalog_xy=xy,
        )
    return path


def test_shard_tool_extracts_payload_blocks_and_remaps_local_quad_stars(tmp_path: Path) -> None:
    source = _write_fake_monolith(tmp_path / "d50_4d.npz")
    out_dir = tmp_path / "shards"
    manifest = out_dir / "manifest.json"

    rc = sharder.main(
        [
            "--source",
            str(source),
            "--output-dir",
            str(out_dir),
            "--manifest-path",
            str(manifest),
            "--topology",
            "oracle",
            "--tile-keys",
            "d50_0102,d50_0201",
        ]
    )

    assert rc == 0
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert [entry["tile_keys"] for entry in payload["indexes"]] == [["d50_0102"], ["d50_0201"]]
    with np.load(source, allow_pickle=False) as mono, np.load(out_dir / "direct-d50-oracle-000-d50_0102.npz", allow_pickle=False) as shard:
        assert np.array_equal(shard["codes_4d"], mono["codes_4d"][2:4])
        assert np.array_equal(shard["catalog_ra_dec"], mono["catalog_ra_dec"][3:6])
        assert np.array_equal(shard["catalog_xy"], mono["catalog_xy"][3:6])
        assert np.array_equal(shard["source_quad_indices"], mono["source_quad_indices"][2:4])
        assert np.array_equal(shard["ratio_hashes"], mono["ratio_hashes"][2:4])
        assert np.array_equal(shard["tile_key_indices"], np.zeros(2, dtype=np.int32))
        assert np.array_equal(shard["quad_star_indices"], mono["quad_star_indices"][2:4] - 3)
        metadata = json.loads(str(shard["metadata"][0]))
        assert metadata["source_monolith"] == str(source.resolve())
        assert metadata["source_monolith_sha256"]


def test_shard_plans_are_deterministic_for_ring_and_fixed_topologies() -> None:
    keys = ("d50_0101", "d50_0102", "d50_0201", "d50_0202", "d50_0203")

    rings = sharder.build_plan(keys, topology="ring", tiles_per_shard=16, oracle_tile_keys=())
    fixed = sharder.build_plan(keys, topology="fixed", tiles_per_shard=2, oracle_tile_keys=())

    assert [(p.shard_id, p.tile_keys) for p in rings] == [
        ("direct-d50-ring-01", ("d50_0101", "d50_0102")),
        ("direct-d50-ring-02", ("d50_0201", "d50_0202", "d50_0203")),
    ]
    assert [p.tile_keys for p in fixed] == [
        ("d50_0101", "d50_0102"),
        ("d50_0201", "d50_0202"),
        ("d50_0203",),
    ]


def _write_diversity_index(path: Path) -> Path:
    metadata = {
        "schema": ASTROMETRY_AB_CODE_4D_SCHEMA,
        "version": 1,
        "level": "S",
        "sampler_tag": "test",
        "source_catalog": "unit",
        "code_tol_recommended": 0.5,
    }
    codes = np.asarray(
        [
            [0.000, 0.0, 0.0, 0.0],
            [0.001, 0.0, 0.0, 0.0],
            [0.002, 0.0, 0.0, 0.0],
            [0.200, 0.0, 0.0, 0.0],
            [0.210, 0.0, 0.0, 0.0],
            [0.220, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata=_metadata_array(metadata),
            codes_4d=codes,
            quad_star_indices=np.asarray([[0, 1, 2, 3]] * 3 + [[4, 5, 6, 7]] * 3, dtype=np.int32),
            source_quad_indices=np.arange(6, dtype=np.int32),
            tile_key_indices=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32),
            ratio_hashes=np.arange(6, dtype=np.int64),
            tile_keys=np.asarray(["d50_0101", "d50_0102"], dtype="<U8"),
            catalog_ra_dec=np.arange(16, dtype=np.float64).reshape(8, 2),
            catalog_xy=np.arange(16, dtype=np.float64).reshape(8, 2),
        )
    return path


def test_diversified_search_keeps_tile_quota_instead_of_global_nearest_only(tmp_path: Path) -> None:
    index = Quad4DIndex.load(_write_diversity_index(tmp_path / "idx.npz"))
    records = [SimpleNamespace(code=np.zeros(4), source_quad_index=0, ordered_indices=(0, 1, 2, 3), ratio_hash=None)]

    plain = index.search_records(records, code_tol=0.5, max_hits=4, max_hits_per_image_quad=4)
    diversified = index.search_records_diversified(
        records,
        code_tol=0.5,
        max_hits=4,
        max_hits_per_image_quad=4,
        max_hits_per_tile=2,
        max_hits_per_image_quad_tile=2,
    )

    assert {hit.tile_key for hit in plain} == {"d50_0101", "d50_0102"}
    assert [hit.tile_key for hit in diversified].count("d50_0101") == 2
    assert [hit.tile_key for hit in diversified].count("d50_0102") == 2


def test_progressive_shard_runner_stops_after_first_accepted_shard(monkeypatch, tmp_path: Path) -> None:
    import zeblindsolver.zeblindsolver as solver

    calls: list[tuple[str, tuple[str, ...], int]] = []
    prepared_payload = solver._Astrometry4DImageQuadPayload(
        image_positions=np.zeros((4, 2), dtype=np.float64),
        verification_image_positions=np.zeros((4, 2), dtype=np.float64),
        image_quads=np.zeros((1, 4), dtype=np.uint16),
        image_records=(object(),),
        stats={
            "astrometry_4d_quad_build_s": 0.25,
            "astrometry_4d_image_quad_build_count": 1,
            "astrometry_4d_image_quad_count": 1,
            "image_quad_build_count": 1,
            "image_quad_build_time": 0.25,
            "image_quad_count": 1,
        },
    )
    prepare_calls = 0

    def fake_prepare(**kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        return prepared_payload

    def fake_route(**kwargs):
        cfg = kwargs["config"]
        paths = tuple(cfg.blind_astrometry_4d_index_paths)
        calls.append((str(paths[0]), paths, id(kwargs["image_quad_payload"])))
        stats = {
            "astrometry_4d_hits": 3,
            "astrometry_4d_hits_tested": 1,
            "astrometry_4d_accepted_candidates": 1 if "001" in paths[0] else 0,
            "astrometry_4d_stop_reason": "confident_accept" if "001" in paths[0] else "candidate_exhausted",
            "astrometry_4d_time_to_first_test_s": 0.01,
            "astrometry_4d_time_to_first_accept_s": 0.02 if "001" in paths[0] else None,
            "astrometry_4d_quad_build_s": 0.0,
            "inliers": 41 if "001" in paths[0] else 0,
            "rms_px": 1.0,
        }
        return WcsSolution("001" in paths[0], "ok", None, stats, "d50_0102" if "001" in paths[0] else None, {})

    monkeypatch.setattr(solver, "_astrometry_4d_prepare_image_quads", fake_prepare)
    monkeypatch.setattr(solver, "_solve_astrometry_4d_runtime_route", fake_route)
    result = solver._solve_astrometry_4d_progressive_shards(
        config=SolveConfig(
            blind_astrometry_4d_index_paths=(),
            blind_astrometry_4d_accept_policy="first_accept",
            blind_astrometry_4d_search_budget_s=45.0,
        ),
        obs_stars=np.empty((0,)),
        image_positions_solver=np.empty((0, 2)),
        verification_image_positions_solver=None,
        image_shape=(10, 10),
        scale_bounds_arcsec=None,
        cancel_check=None,
        cancel_reason_provider=None,
        index_paths=(tmp_path / "shard-000.npz", tmp_path / "shard-001.npz", tmp_path / "shard-002.npz"),
        base_stats={"astrometry_4d_index_count": 3},
        started_at=solver.time.perf_counter(),
    )

    assert result.success is True
    assert calls == [
        (str(tmp_path / "shard-000.npz"), (str(tmp_path / "shard-000.npz"),), id(prepared_payload)),
        (str(tmp_path / "shard-001.npz"), (str(tmp_path / "shard-001.npz"),), id(prepared_payload)),
    ]
    assert prepare_calls == 1
    assert result.stats["astrometry_4d_shards_opened"] == 2
    assert result.stats["astrometry_4d_hits_tested"] == 2
    assert result.stats["image_quad_build_count"] == 1
    assert result.stats["astrometry_4d_stop_reason"] == "confident_accept"


def test_progressive_budget_allows_minimum_validation_floor(monkeypatch) -> None:
    import zeblindsolver.zeblindsolver as solver

    monkeypatch.setattr(solver.time, "perf_counter", lambda: 100.0)

    allowed, overrun = solver._astrometry_4d_budget_allows_validation(
        route_budget_s=1.0,
        t_total0=0.0,
        tested=0,
        candidate_count=12,
        max_hypotheses=16,
        min_hypotheses_per_nonempty_shard=8,
    )

    assert allowed is True
    assert overrun is True

    allowed, overrun = solver._astrometry_4d_budget_allows_validation(
        route_budget_s=1.0,
        t_total0=0.0,
        tested=8,
        candidate_count=12,
        max_hypotheses=16,
        min_hypotheses_per_nonempty_shard=8,
    )

    assert allowed is False
    assert overrun is False
