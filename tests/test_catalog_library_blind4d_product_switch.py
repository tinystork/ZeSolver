from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from catalog_resource_helpers import sha256_file, strict_entry, write_catalog_library, write_fake_4d_index, write_strict_manifest
from zesolver.catalog_resources import resolve_catalog_resources
from zesolver.core.blind_port import ProductionBlindSolverPort
from zesolver.core.models import EngineSolveResult, SolveRequest, SolveStatus
from zesolver.core.pipeline import SolverPipeline
from zesolver.settings import ProductSettings, RuntimeOptions, build_solver_configuration


def _with_runtime_order(root: Path, order: list[str]) -> Path:
    path = root / "catalog.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["runtime_order"] = {"blind4d": order}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return root


def _library_with_blind(tmp_path: Path) -> tuple[Path, Path]:
    index = write_fake_4d_index(tmp_path / "d50_2823_S_q.npz", "d50_2823")
    root = write_catalog_library(tmp_path / "library", index_paths=[index])
    _with_runtime_order(root, ["blind4d-0"])
    return root, index


def _fits(path: Path) -> Path:
    fits.PrimaryHDU(np.ones((8, 8), dtype=np.float32)).writeto(path, overwrite=True)
    return path


def _full_sharded_library_with_compat_monolith(tmp_path: Path, *, shard_count: int = 47) -> tuple[Path, list[Path], Path]:
    root = tmp_path / "library"
    root.mkdir(parents=True, exist_ok=True)
    source_dir = root / "sources" / "astap" / "d50"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "d50_0001.1476").write_bytes(b"fixture")
    shards: list[Path] = []
    indexes: list[dict[str, object]] = []
    tile_keys: list[str] = []
    for i in range(shard_count):
        tile_key = f"d50_{i + 1:04d}"
        index_path = write_fake_4d_index(root / "indexes" / f"direct-d50-fixed32-{i:03d}.npz", tile_key)
        shards.append(index_path)
        tile_keys.append(tile_key)
        indexes.append(
            {
                "id": f"direct-d50-fixed32-{i:03d}",
                "engine": "blind4d",
                "schema": "zeblind.astrometry_4d_index_manifest.v1",
                "algorithm_version": "astrometry_ab_code_4d_v1",
                "path": {"kind": "external_reference", "value": str(index_path.resolve())},
                "manifest_path": None,
                "source_ids": ["astap-d50"],
                "source_tiles": [tile_key],
                "coverage": {
                    "status": "PARTIAL",
                    "all_sky": False,
                    "families": ["d50"],
                    "tile_keys": [tile_key],
                    "covered_tiles": 1,
                    "total_tiles": shard_count,
                    "fraction": 1 / shard_count,
                },
                "integrity": {
                    "files": [
                        {
                            "path": str(index_path.resolve()),
                            "sha256": sha256_file(index_path),
                            "size_bytes": index_path.stat().st_size,
                        }
                    ]
                },
                "status": "FULL_VERIFIED",
                "category": "product",
                "build_parameters": {
                    "quad_schema": "astrometry_ab_code_4d_v1",
                    "quad_version": 1,
                    "level": "S",
                    "sampler_tag": "catalog_ring_coverage",
                    "code_tol_recommended": 0.015,
                    "catalog_source": "unit-test",
                },
            }
        )
    monolith = write_fake_4d_index(root / "indexes" / "direct-d50.npz", "d50_9999")
    indexes.append(
        {
            "id": "direct-d50",
            "engine": "blind4d",
            "schema": "zeblind.astrometry_4d_index_manifest.v1",
            "algorithm_version": "astrometry_ab_code_4d_v1",
            "path": {"kind": "external_reference", "value": str(monolith.resolve())},
            "manifest_path": None,
            "source_ids": ["astap-d50"],
            "source_tiles": tile_keys,
            "coverage": {
                "status": "FULL",
                "all_sky": True,
                "families": ["d50"],
                "tile_keys": tile_keys,
                "covered_tiles": shard_count,
                "total_tiles": shard_count,
                "fraction": 1.0,
            },
            "integrity": {
                "files": [
                    {
                        "path": str(monolith.resolve()),
                        "sha256": sha256_file(monolith),
                        "size_bytes": monolith.stat().st_size,
                    }
                ]
            },
            "status": "FULL_VERIFIED",
            "category": "compatibility",
        }
    )
    payload = {
        "schema_version": 1,
        "library_id": "s5d2-divergence",
        "created_at": "2026-07-25T00:00:00Z",
        "created_by": "tests",
        "minimum_zesolver_version": None,
        "status": "READY_FULL",
        "sources": [
            {
                "id": "astap-d50",
                "kind": "astap_hnsky",
                "family": "d50",
                "format": "1476-5",
                "path": {"kind": "relative", "value": "sources/astap/d50"},
                "tile_count": shard_count,
                "layout": "hnsky_1476",
                "coverage": {
                    "status": "FULL",
                    "all_sky": True,
                    "families": ["d50"],
                    "tile_keys": tile_keys,
                    "covered_tiles": shard_count,
                    "total_tiles": shard_count,
                    "fraction": 1.0,
                },
                "integrity": {"files": []},
                "status": "FAST_VERIFIED",
            }
        ],
        "derived_indexes": indexes,
        "coverage": {
            "status": "FULL",
            "all_sky": True,
            "families": ["d50"],
            "tile_keys": tile_keys,
            "covered_tiles": shard_count,
            "total_tiles": shard_count,
            "fraction": 1.0,
        },
        "integrity": {"checksum_algorithm": "sha256"},
        "provenance": {"notes": "s5d2 divergence fixture"},
        "runtime_order": {"blind4d": [f"direct-d50-fixed32-{i:03d}" for i in range(shard_count)]},
    }
    (root / "catalog.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return root, shards, monolith


def test_blind_port_uses_library_view_and_ignores_invalid_external_manifest(tmp_path: Path, monkeypatch) -> None:
    library_root, index = _library_with_blind(tmp_path)
    resources = resolve_catalog_resources(catalog_library=library_root)
    configuration = build_solver_configuration(
        product_settings=ProductSettings(catalog_library_path=library_root, blind4d_catalog_mode="auto"),
        runtime_options=RuntimeOptions(),
    )
    calls: list[tuple[str, ...]] = []

    def fake_blind_solve(*, fits_path, index_root, config, **kwargs):
        calls.append(tuple(str(path) for path in config.blind_astrometry_4d_index_paths))
        return {
            "success": False,
            "message": "synthetic miss",
            "elapsed_sec": 0.0,
            "tried_dbs": [],
            "used_db": None,
            "wrote_wcs": False,
            "updated_keywords": {},
            "output_path": fits_path,
            "stats": {},
        }

    monkeypatch.setattr("zesolver.core.blind_port.blind_solve", fake_blind_solve)
    result = ProductionBlindSolverPort().solve(
        SolveRequest(_fits(tmp_path / "input.fit"), None, True),
        resources=resources,
        configuration=configuration,
    )

    assert result.status is SolveStatus.UNSOLVED
    assert calls == [(str(index.resolve()),)]
    assert result.raw["blind4d_catalog_source"] == "catalog_library_view"
    assert result.raw["blind4d_catalog_mode_effective"] == "library-view"
    assert result.raw["blind4d_index_ids"] == ["blind4d-0"]
    assert result.raw["blind4d_external_fallback_used"] is False


def test_s5d2_library_view_prefers_47_shards_over_compatibility_monolith(tmp_path: Path, monkeypatch) -> None:
    library_root, shards, monolith = _full_sharded_library_with_compat_monolith(tmp_path)
    resources = resolve_catalog_resources(catalog_library=library_root)
    configuration = build_solver_configuration(
        product_settings=ProductSettings(catalog_library_path=library_root, blind4d_catalog_mode="auto"),
        runtime_options=RuntimeOptions(),
    )
    calls: list[tuple[str, ...]] = []

    def fake_blind_solve(*, fits_path, index_root, config, **kwargs):
        calls.append(tuple(str(path) for path in config.blind_astrometry_4d_index_paths))
        return {
            "success": False,
            "message": "synthetic miss",
            "elapsed_sec": 0.0,
            "tried_dbs": [],
            "used_db": None,
            "wrote_wcs": False,
            "updated_keywords": {},
            "output_path": fits_path,
            "stats": {"astrometry_4d_index_count": len(config.blind_astrometry_4d_index_paths)},
        }

    monkeypatch.setattr("zesolver.core.blind_port.blind_solve", fake_blind_solve)
    result = ProductionBlindSolverPort().solve(
        SolveRequest(_fits(tmp_path / "input.fit"), None, True),
        resources=resources,
        configuration=configuration,
    )

    assert result.status is SolveStatus.UNSOLVED
    assert calls == [tuple(str(path.resolve()) for path in shards)]
    assert str(monolith.resolve()) not in calls[0]
    assert result.raw["blind4d_catalog_source"] == "catalog_library_view"
    assert result.raw["blind4d_catalog_mode_effective"] == "library-view"
    assert result.raw["blind4d_index_count"] == 47
    assert result.raw["blind4d_covered_tiles"] == 47
    assert result.raw["blind4d_total_tiles"] == 47
    assert result.raw["blind4d_all_sky"] is True
    assert result.raw["blind4d_external_fallback_used"] is False
    assert result.raw["blind4d_runtime_order"] == [f"direct-d50-fixed32-{i:03d}" for i in range(47)]
    assert "direct-d50" not in result.raw["blind4d_runtime_order"]


def test_s5f_catalog_resources_publish_final_library_view_coverage_without_compat_warning(tmp_path: Path) -> None:
    library_root, shards, monolith = _full_sharded_library_with_compat_monolith(tmp_path)
    resources = resolve_catalog_resources(catalog_library=library_root)

    assert resources.source == "library"
    assert resources.blind4d_index_count == 47
    assert tuple(path.resolve() for path in resources.blind4d_runtime_paths) == tuple(path.resolve() for path in shards)
    assert monolith.resolve() not in {path.resolve() for path in resources.blind4d_runtime_paths}
    assert resources.coverage is not None
    assert resources.coverage.covered_tiles == 47
    assert resources.coverage.total_tiles == 47
    assert resources.all_sky_blind4d is True
    assert "blind4d_coverage_not_all_sky" not in resources.warnings


def test_blind_port_forced_external_rollback_uses_external_manifest(tmp_path: Path, monkeypatch) -> None:
    library_root, _index = _library_with_blind(tmp_path)
    external_index = write_fake_4d_index(tmp_path / "external" / "d50_2822_S_q.npz", "d50_2822")
    external = write_strict_manifest(tmp_path / "external" / "manifest.json", [strict_entry("external-idx", external_index, "d50_2822")])
    resources = resolve_catalog_resources(legacy_blind4d_manifest=external)
    configuration = build_solver_configuration(
        product_settings=ProductSettings(catalog_library_path=library_root, blind4d_catalog_mode="external-manifest"),
        runtime_options=RuntimeOptions(),
    )
    calls: list[tuple[str, ...]] = []

    def fake_blind_solve(*, fits_path, index_root, config, **kwargs):
        calls.append(tuple(str(path) for path in config.blind_astrometry_4d_index_paths))
        return {
            "success": False,
            "message": "synthetic miss",
            "elapsed_sec": 0.0,
            "tried_dbs": [],
            "used_db": None,
            "wrote_wcs": False,
            "updated_keywords": {},
            "output_path": fits_path,
            "stats": {},
        }

    monkeypatch.setattr("zesolver.core.blind_port.blind_solve", fake_blind_solve)
    result = ProductionBlindSolverPort().solve(
        SolveRequest(_fits(tmp_path / "input.fit"), None, True),
        resources=resources,
        configuration=configuration,
    )

    assert result.status is SolveStatus.UNSOLVED
    assert calls == [(str(external_index.resolve()),)]
    assert result.raw["blind4d_catalog_source"] == "external_manifest"
    assert result.raw["blind4d_catalog_mode_effective"] == "external-manifest"


class _NoopNear:
    def solve(self, request, *, resources, configuration):
        return EngineSolveResult(status=SolveStatus.UNSOLVED, backend="NEAR", error="synthetic near miss")


class _NoopBlind:
    def solve(self, request, *, resources, configuration):
        return EngineSolveResult(status=SolveStatus.UNSOLVED, backend="BLIND4D", error="synthetic blind miss")


def test_pipeline_telemetry_reports_library_view_source(tmp_path: Path, monkeypatch) -> None:
    library_root, _index = _library_with_blind(tmp_path)
    resources = resolve_catalog_resources(catalog_library=library_root)

    def fake_blind_solve(*, fits_path, index_root, config, **kwargs):
        return {
            "success": False,
            "message": "synthetic miss",
            "elapsed_sec": 0.0,
            "tried_dbs": [],
            "used_db": None,
            "wrote_wcs": False,
            "updated_keywords": {},
            "output_path": fits_path,
            "stats": {},
        }

    monkeypatch.setattr("zesolver.core.blind_port.blind_solve", fake_blind_solve)
    pipeline = SolverPipeline(
        product_settings=ProductSettings(catalog_library_path=library_root, blind4d_catalog_mode="auto"),
        runtime_options=RuntimeOptions(),
        catalog_resources=resources,
        near_solver=_NoopNear(),
    )
    result = pipeline.solve(SolveRequest(_fits(tmp_path / "input.fit"), None, True))

    assert result.status is SolveStatus.UNSOLVED
    assert pipeline.last_telemetry is not None
    assert pipeline.last_telemetry["blind4d_catalog_source"] == "catalog_library_view"
    assert pipeline.last_telemetry["blind4d_catalog_mode_effective"] == "library-view"
    assert pipeline.last_telemetry["blind4d_external_fallback_used"] is False
    assert not any(str(value).startswith(str(tmp_path)) for value in pipeline.last_telemetry.values())


def test_blind_port_reuses_runtime_selection_for_same_context(tmp_path: Path, monkeypatch) -> None:
    library_root, _index = _library_with_blind(tmp_path)
    resources = resolve_catalog_resources(catalog_library=library_root)
    configuration = build_solver_configuration(
        product_settings=ProductSettings(catalog_library_path=library_root, blind4d_catalog_mode="auto"),
        runtime_options=RuntimeOptions(),
    )
    calls = {"resolve": 0}
    original_resolve = __import__("zesolver.core.blind_port", fromlist=["resolve_blind4d_runtime"]).resolve_blind4d_runtime

    def counted_resolve(*args, **kwargs):
        calls["resolve"] += 1
        return original_resolve(*args, **kwargs)

    def fake_blind_solve(*, fits_path, index_root, config, **kwargs):
        return {
            "success": False,
            "message": "synthetic miss",
            "elapsed_sec": 0.0,
            "tried_dbs": [],
            "used_db": None,
            "wrote_wcs": False,
            "updated_keywords": {},
            "output_path": fits_path,
            "stats": {},
        }

    monkeypatch.setattr("zesolver.core.blind_port.resolve_blind4d_runtime", counted_resolve)
    monkeypatch.setattr("zesolver.core.blind_port.blind_solve", fake_blind_solve)
    port = ProductionBlindSolverPort()
    for idx in range(2):
        result = port.solve(
            SolveRequest(_fits(tmp_path / f"input-{idx}.fit"), None, True),
            resources=resources,
            configuration=configuration,
        )
        assert result.status is SolveStatus.UNSOLVED

    assert calls["resolve"] == 1


def test_pipeline_leaves_runtime_selection_to_blind_port(tmp_path: Path, monkeypatch) -> None:
    library_root, _index = _library_with_blind(tmp_path)
    resources = resolve_catalog_resources(catalog_library=library_root)
    calls = {"resolve": 0}
    original_resolve = __import__("zesolver.core.pipeline", fromlist=["resolve_blind4d_runtime"]).resolve_blind4d_runtime

    def counted_resolve(*args, **kwargs):
        calls["resolve"] += 1
        return original_resolve(*args, **kwargs)

    monkeypatch.setattr("zesolver.core.pipeline.resolve_blind4d_runtime", counted_resolve)
    pipeline = SolverPipeline(
        product_settings=ProductSettings(catalog_library_path=library_root, blind4d_catalog_mode="auto"),
        runtime_options=RuntimeOptions(),
        catalog_resources=resources,
        near_solver=_NoopNear(),
        blind_solver=_NoopBlind(),
    )
    for idx in range(2):
        result = pipeline.solve(SolveRequest(_fits(tmp_path / f"pipeline-{idx}.fit"), None, True))
        assert result.status is SolveStatus.UNSOLVED

    assert calls["resolve"] == 0
