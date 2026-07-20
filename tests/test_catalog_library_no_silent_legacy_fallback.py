from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from catalog_resource_helpers import strict_entry, write_catalog_library, write_fake_4d_index, write_strict_manifest
from near_catalog_provider_helpers import write_legacy_index_from_tile


def _load_entrypoint():
    path = Path(__file__).resolve().parents[1] / "zesolver.py"
    spec = importlib.util.spec_from_file_location("zesolver_entrypoint_p1d1b_no_fallback", path)
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


def _library_without_near(tmp_path: Path) -> Path:
    idx = write_fake_4d_index(tmp_path / "d50_A_S_q.npz", "d50_A")
    manifest = write_strict_manifest(tmp_path / "manifest.json", [strict_entry("a", idx, "d50_A")])
    return write_catalog_library(
        tmp_path / "library",
        include_source=False,
        index_paths=[idx],
        strict_manifest_path=manifest,
    )


def test_library_without_near_records_astap_error_and_does_not_build_legacy_provider(tmp_path: Path, monkeypatch) -> None:
    zs = _load_entrypoint()
    monkeypatch.setattr(zs, "CatalogDB", _FakeCatalogDB)
    legacy_root = _legacy_index(tmp_path)
    solver = zs.ImageSolver(
        zs.SolveConfig(
            db_root=tmp_path / "legacy-db",
            input_dir=tmp_path,
            families=("d50",),
            catalog_library_path=_library_without_near(tmp_path),
            blind_index_path=legacy_root,
            near_catalog_mode="auto",
        )
    )

    runtime = solver.near_catalog_runtime
    assert runtime.provider is None
    assert runtime.provider_kind is None
    assert runtime.error_code == "ASTAP_NEAR_RESOURCE_REQUIRED"
    assert runtime.legacy_index_root is None


def test_forced_astap_native_with_missing_resource_does_not_use_residual_legacy(tmp_path: Path, monkeypatch) -> None:
    zs = _load_entrypoint()
    monkeypatch.setattr(zs, "CatalogDB", _FakeCatalogDB)
    legacy_root = _legacy_index(tmp_path)
    solver = zs.ImageSolver(
        zs.SolveConfig(
            db_root=tmp_path / "legacy-db",
            input_dir=tmp_path,
            families=("d50",),
            catalog_library_path=_library_without_near(tmp_path),
            blind_index_path=legacy_root,
            near_catalog_mode="astap-native",
        )
    )

    assert solver.near_catalog_runtime.provider is None
    assert solver.near_catalog_runtime.error_code == "ASTAP_NEAR_RESOURCE_REQUIRED"
    assert solver.near_catalog_runtime.effective_mode.value == "astap-native"
