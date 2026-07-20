from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from catalog_resource_helpers import write_catalog_library
from near_catalog_provider_helpers import write_astap_1476_tile, write_legacy_index_from_tile


def _load_entrypoint():
    path = Path(__file__).resolve().parents[1] / "zesolver.py"
    spec = importlib.util.spec_from_file_location("zesolver_entrypoint_p1d1b_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeCatalogDB:
    tiles = ()
    families = ("d50",)

    def __init__(self, *args, **kwargs) -> None:
        pass


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


def test_image_solver_builds_astap_native_provider_from_catalog_library_without_legacy_index(tmp_path: Path, monkeypatch) -> None:
    zs = _load_entrypoint()
    monkeypatch.setattr(zs, "CatalogDB", _FakeCatalogDB)
    solver = zs.ImageSolver(
        zs.SolveConfig(
            db_root=tmp_path / "legacy-db",
            input_dir=tmp_path,
            families=("g05",),
            catalog_library_path=_library_with_astap_tile(tmp_path),
            blind_index_path=None,
        )
    )

    assert solver.near_catalog_runtime.available is True
    assert solver.near_catalog_runtime.provider_kind == "astap_native"
    assert solver.near_catalog_runtime.legacy_index_root is None


def test_image_solver_passes_astap_provider_to_near_without_index_root(tmp_path: Path, monkeypatch) -> None:
    zs = _load_entrypoint()
    monkeypatch.setattr(zs, "CatalogDB", _FakeCatalogDB)
    calls: list[dict[str, object]] = []

    def fake_near_solve(*, fits_path, index_root, catalog_provider=None, **kwargs):
        calls.append({"index_root": index_root, "provider": getattr(catalog_provider, "kind", None)})
        return {"success": False, "message": "synthetic miss", "stats": {}, "wrote_wcs": False}

    monkeypatch.setattr(zs, "near_solve", fake_near_solve)
    solver = zs.ImageSolver(
        zs.SolveConfig(
            db_root=tmp_path / "legacy-db",
            input_dir=tmp_path,
            families=("g05",),
            catalog_library_path=_library_with_astap_tile(tmp_path),
            blind_index_path=None,
        )
    )

    result = solver._run_index_near_solver(
        tmp_path / "frame.fit",
        zs.ImageMetadata(
            path=tmp_path / "frame.fit",
            kind="fits",
            width=100,
            height=100,
            ra_deg=184.6,
            dec_deg=47.3,
            source="header",
            has_wcs=False,
        ),
        allow_blind_fallback=False,
    )

    assert result is None
    assert calls == [
        {"index_root": None, "provider": "astap_native"},
        {"index_root": None, "provider": "astap_native"},
    ]


def test_image_solver_explicit_legacy_rollback_uses_legacy_provider(tmp_path: Path, monkeypatch) -> None:
    zs = _load_entrypoint()
    monkeypatch.setattr(zs, "CatalogDB", _FakeCatalogDB)
    legacy_root = _legacy_index(tmp_path)
    solver = zs.ImageSolver(
        zs.SolveConfig(
            db_root=tmp_path / "legacy-db",
            input_dir=tmp_path,
            families=("d50",),
            catalog_library_path=_library_with_astap_tile(tmp_path),
            blind_index_path=legacy_root,
            near_catalog_mode="legacy-index",
        )
    )

    assert solver.near_catalog_runtime.available is True
    assert solver.near_catalog_runtime.provider_kind == "legacy_index"
    assert solver.near_catalog_runtime.legacy_index_root == legacy_root
