from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from zeblindsolver.profiles import ZEBLIND_4D_EXPERIMENTAL_PROFILE

from catalog_resource_helpers import (
    strict_entry,
    write_catalog_library,
    write_fake_4d_index,
    write_strict_manifest,
)


def _load_entrypoint():
    path = Path(__file__).resolve().parents[1] / "zesolver.py"
    spec = importlib.util.spec_from_file_location("zesolver_entrypoint_p1c_pipeline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pipeline_config_uses_library_near_and_blind4d_before_legacy(tmp_path: Path) -> None:
    zs = _load_entrypoint()
    idx_lib = write_fake_4d_index(tmp_path / "d50_LIB_S_q.npz", "d50_LIB")
    idx_legacy = write_fake_4d_index(tmp_path / "d50_OLD_S_q.npz", "d50_OLD")
    library_manifest = write_strict_manifest(
        tmp_path / "library-manifest.json",
        [strict_entry("lib", idx_lib, "d50_LIB")],
    )
    legacy_manifest = write_strict_manifest(
        tmp_path / "legacy-manifest.json",
        [strict_entry("old", idx_legacy, "d50_OLD")],
    )
    library_root = write_catalog_library(
        tmp_path / "library",
        index_paths=[idx_lib],
        strict_manifest_path=library_manifest,
    )
    config = zs.SolveConfig(
        db_root=tmp_path / "legacy-db",
        input_dir=tmp_path,
        families=("g05",),
        blind_backend_profile=ZEBLIND_4D_EXPERIMENTAL_PROFILE,
        blind_4d_manifest_path=legacy_manifest,
        catalog_library_path=library_root,
    )

    resolved, resources = zs.apply_catalog_resources_to_config(config)
    blind_cfg = zs.build_blind_solve_config(resolved)

    assert resources.source == "library"
    assert resolved.db_root == (library_root / "sources" / "astap" / "d50").resolve()
    assert resolved.families == ("d50",)
    assert resolved.blind_4d_manifest_path == library_manifest.resolve()
    assert tuple(Path(p) for p in blind_cfg.blind_astrometry_4d_index_paths) == (idx_lib.resolve(),)


def test_partial_library_telemetry_is_preserved(tmp_path: Path) -> None:
    zs = _load_entrypoint()
    idx = write_fake_4d_index(tmp_path / "d50_A_S_q.npz", "d50_A")
    manifest = write_strict_manifest(tmp_path / "manifest.json", [strict_entry("a", idx, "d50_A")])
    library_root = write_catalog_library(
        tmp_path / "library",
        index_paths=[idx],
        strict_manifest_path=manifest,
    )
    config = zs.SolveConfig(
        db_root=tmp_path / "legacy-db",
        input_dir=tmp_path,
        families=("g05",),
        catalog_library_path=library_root,
    )

    _resolved, resources = zs.apply_catalog_resources_to_config(config)
    telemetry = resources.telemetry()

    assert telemetry["catalog_source"] == "library"
    assert telemetry["catalog_library_status"] == "READY_PARTIAL"
    assert telemetry["blind4d_index_count"] == 1
    assert telemetry["blind4d_all_sky"] is False
    assert "blind4d_coverage_not_all_sky" in telemetry["warnings"]
