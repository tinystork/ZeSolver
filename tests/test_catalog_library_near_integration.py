from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from zesolver.catalog_resources import CatalogResourceResolutionError, resolve_catalog_resources

from catalog_resource_helpers import strict_entry, write_catalog_library, write_fake_4d_index, write_strict_manifest


def _load_entrypoint():
    path = Path(__file__).resolve().parents[1] / "zesolver.py"
    spec = importlib.util.spec_from_file_location("zesolver_entrypoint_p1c_near", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_apply_catalog_resources_overrides_near_root_and_families(tmp_path: Path) -> None:
    zs = _load_entrypoint()
    library_root = write_catalog_library(tmp_path / "library", include_source=True, index_paths=[])
    config = zs.SolveConfig(
        db_root=tmp_path / "legacy-db",
        input_dir=tmp_path,
        families=("g05",),
        catalog_library_path=library_root,
    )

    resolved, resources = zs.apply_catalog_resources_to_config(config)

    assert resources.source == "library"
    assert resolved.db_root == (library_root / "sources" / "astap" / "d50").resolve()
    assert resolved.families == ("d50",)


def test_without_library_legacy_near_values_are_preserved(tmp_path: Path) -> None:
    zs = _load_entrypoint()
    config = zs.SolveConfig(
        db_root=tmp_path / "legacy-db",
        input_dir=tmp_path,
        families=("g05",),
    )

    resolved, resources = zs.apply_catalog_resources_to_config(config)

    assert resources.source == "legacy"
    assert resolved.db_root == tmp_path / "legacy-db"
    assert resolved.families == ("g05",)


def test_library_without_source_does_not_announce_near(tmp_path: Path) -> None:
    idx = write_fake_4d_index(tmp_path / "d50_A_S_q.npz", "d50_A")
    manifest = write_strict_manifest(tmp_path / "manifest.json", [strict_entry("a", idx, "d50_A")])
    library_root = write_catalog_library(
        tmp_path / "library",
        include_source=False,
        index_paths=[idx],
        strict_manifest_path=manifest,
    )

    resources = resolve_catalog_resources(catalog_library=library_root)

    assert resources.source == "library"
    assert resources.near is None
    assert resources.blind4d_runtime_paths == (idx.resolve(),)


def test_corrupt_source_is_explicit_error(tmp_path: Path) -> None:
    library_root = write_catalog_library(
        tmp_path / "library",
        include_source=True,
        source_status="CORRUPT",
        index_paths=[],
    )

    with pytest.raises(CatalogResourceResolutionError, match="SOURCE_STATUS_CORRUPT"):
        resolve_catalog_resources(catalog_library=library_root)
