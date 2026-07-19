from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from catalog_resource_helpers import strict_entry, write_catalog_library, write_fake_4d_index, write_strict_manifest
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


def test_pipeline_reuses_runtime_selection_for_same_context(tmp_path: Path, monkeypatch) -> None:
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

    assert calls["resolve"] == 1
