from __future__ import annotations

from pathlib import Path

import numpy as np

from catalog_resource_helpers import write_catalog_library
from near_catalog_provider_helpers import write_astap_1476_tile, write_legacy_index_from_tile
from zesolver.catalog_resources import resolve_catalog_resources
from zesolver.core.models import SolveRequest, SolveStatus
from zesolver.core.pipeline import ExistingNearSolverPort
from zesolver.settings import ProductSettings, RuntimeOptions, build_solver_configuration


def _library_with_astap_tile(tmp_path: Path) -> Path:
    root = write_catalog_library(tmp_path / "library", include_source=True, index_paths=[])
    write_astap_1476_tile(
        root / "sources" / "astap" / "d50",
        family="d50",
        tile_code="2823",
        ra_deg=np.asarray([184.6, 184.7], dtype=np.float64),
        dec_deg=np.asarray([47.2, 47.3], dtype=np.float64),
    )
    return root


def _legacy_index(tmp_path: Path) -> Path:
    return write_legacy_index_from_tile(
        tmp_path / "legacy-index",
        tile_key="d50_2823",
        family="d50",
        tile_code="2823",
        center_ra_deg=184.6,
        center_dec_deg=47.3,
        bounds={"dec_min": 46.0, "dec_max": 48.0, "ra_min": 183.0, "ra_max": 186.0},
        ra_deg=np.asarray([184.6, 184.7], dtype=np.float64),
        dec_deg=np.asarray([47.2, 47.3], dtype=np.float64),
        mag=np.asarray([10.0, 10.2], dtype=np.float32),
    )


def test_pipeline_near_port_uses_astap_provider_without_index_root(tmp_path: Path, monkeypatch) -> None:
    library_root = _library_with_astap_tile(tmp_path)
    resources = resolve_catalog_resources(catalog_library=library_root, legacy_index_root=tmp_path / "bad-index")
    configuration = build_solver_configuration(
        product_settings=ProductSettings(catalog_library_path=library_root),
        runtime_options=RuntimeOptions(),
    )
    calls: list[dict[str, object]] = []

    def fake_near_solve(fits_path, index_root, *, catalog_provider=None, **kwargs):
        calls.append({"index_root": index_root, "provider": getattr(catalog_provider, "kind", None)})
        return {"success": False, "message": "synthetic miss", "stats": {}, "wrote_wcs": False}

    monkeypatch.setattr("zesolver.core.pipeline.near_solve", fake_near_solve)
    result = ExistingNearSolverPort().solve(
        SolveRequest(tmp_path / "input.fit", None, True),
        resources=resources,
        configuration=configuration,
    )

    assert result.status is SolveStatus.UNSOLVED
    assert calls == [{"index_root": None, "provider": "astap_native"}]
    assert result.raw["stats"]["near_catalog_mode_requested"] == "auto"
    assert result.raw["stats"]["near_catalog_mode_effective"] == "astap-native"
    assert result.raw["stats"]["near_catalog_provider"] == "astap_native"
    assert result.raw["stats"]["near_catalog_source"] == "library"
    assert "sentinel-missing-legacy" not in str(result.raw["stats"])


def test_pipeline_near_port_uses_legacy_provider_only_when_requested(tmp_path: Path, monkeypatch) -> None:
    library_root = _library_with_astap_tile(tmp_path)
    legacy_root = _legacy_index(tmp_path)
    resources = resolve_catalog_resources(catalog_library=library_root)
    configuration = build_solver_configuration(
        product_settings=ProductSettings(catalog_library_path=library_root, near_catalog_mode="legacy-index"),
        runtime_options=RuntimeOptions(),
    )
    calls: list[dict[str, object]] = []

    def fake_near_solve(fits_path, index_root, *, catalog_provider=None, **kwargs):
        calls.append({"index_root": index_root, "provider": getattr(catalog_provider, "kind", None)})
        return {"success": False, "message": "synthetic miss", "stats": {}, "wrote_wcs": False}

    monkeypatch.setattr("zesolver.core.pipeline.near_solve", fake_near_solve)
    result = ExistingNearSolverPort().solve(
        SolveRequest(tmp_path / "input.fit", None, True),
        resources=resources,
        configuration=configuration,
    )
    assert result.status is SolveStatus.CATALOG_UNAVAILABLE
    assert not calls

    resources_with_legacy = resolve_catalog_resources(
        legacy_db_root=tmp_path / "astap",
        legacy_families=("d50",),
        legacy_index_root=legacy_root,
        enable_environment_discovery=False,
    )
    result = ExistingNearSolverPort().solve(
        SolveRequest(tmp_path / "input.fit", None, True),
        resources=resources_with_legacy,
        configuration=configuration,
    )

    assert result.status is SolveStatus.UNSOLVED
    assert calls == [{"index_root": str(legacy_root), "provider": "legacy_index"}]
    assert result.raw["stats"]["near_catalog_mode_effective"] == "legacy-index"
    assert result.raw["stats"]["near_catalog_provider"] == "legacy_index"
