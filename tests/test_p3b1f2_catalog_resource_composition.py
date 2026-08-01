from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from catalog_resource_helpers import strict_entry, write_catalog_library, write_fake_4d_index, write_strict_manifest
from near_catalog_provider_helpers import write_astap_1476_tile
from zesolver.catalog_resources import (
    CatalogResourceResolutionError,
    resolve_blind4d_runtime,
    resolve_catalog_resources,
    resolve_near_catalog_runtime,
)


def _astap_root(root: Path) -> Path:
    write_astap_1476_tile(
        root,
        family="d50",
        tile_code="2823",
        ra_deg=np.asarray([184.6, 184.7, 184.8], dtype=np.float64),
        dec_deg=np.asarray([47.2, 47.3, 47.4], dtype=np.float64),
        mag=np.asarray([10.0, 10.2, 10.4], dtype=np.float32),
    )
    return root


def _library_with_blind(root: Path) -> tuple[Path, Path]:
    index = write_fake_4d_index(root.parent / "indexes" / "d50_2823_S_q.npz", "d50_2823")
    manifest = write_strict_manifest(root.parent / "manifest.json", [strict_entry("blind4d-0", index, "d50_2823")])
    library = write_catalog_library(root, include_source=True, index_paths=[index], strict_manifest_path=manifest, all_sky_index=True)
    return library, index


def _corrupt_manifest(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not-json", encoding="utf-8")
    return path


def test_library_blind_is_preserved_when_astap_near_is_selected(tmp_path: Path) -> None:
    library, index = _library_with_blind(tmp_path / "library")
    astap = _astap_root(tmp_path / "external-astap")

    resources = resolve_catalog_resources(
        catalog_library=library,
        legacy_db_root=astap,
        legacy_families=("d50",),
        prefer_legacy_near=True,
    )

    assert resources.source == "library"
    assert resources.library_path == library.resolve()
    assert resources.near is not None
    assert resources.near.root == astap
    assert resources.near.external_reference is True
    assert resources.blind4d_runtime_paths == (index.resolve(),)
    assert resources.all_sky_blind4d is True
    assert "astap_near_overrides_library_near" in resources.warnings


def test_hybrid_resources_keep_astap_near_and_library_blind_after_re_resolution(tmp_path: Path) -> None:
    library, index = _library_with_blind(tmp_path / "library")
    astap = _astap_root(tmp_path / "external-astap")

    resources = resolve_catalog_resources(
        catalog_library=library,
        legacy_db_root=astap,
        legacy_families=("d50",),
        prefer_legacy_near=True,
        strict_legacy_blind4d_manifest=False,
    )

    assert resources.near is not None
    assert resources.near.root == astap
    assert resources.blind4d_runtime_paths == (index.resolve(),)
    assert resources.blind4d_manifest_path is not None


def test_astap_only_with_missing_legacy_manifest_degrades_to_near_only(tmp_path: Path) -> None:
    astap = _astap_root(tmp_path / "astap")

    resources = resolve_catalog_resources(
        legacy_db_root=astap,
        legacy_families=("d50",),
        legacy_blind4d_manifest=tmp_path / "missing" / "manifest.json",
        enable_environment_discovery=False,
        strict_legacy_blind4d_manifest=False,
    )

    assert resources.near is not None
    assert resources.near.root == astap
    assert not resources.blind4d_available
    assert resources.blind4d_manifest_path is None
    assert any(warning.startswith("legacy_blind4d_manifest_invalid_ignored:") for warning in resources.warnings)


def test_astap_only_with_corrupt_legacy_manifest_degrades_to_near_only(tmp_path: Path) -> None:
    astap = _astap_root(tmp_path / "astap")
    corrupt = _corrupt_manifest(tmp_path / "legacy" / "manifest.json")

    resources = resolve_catalog_resources(
        legacy_db_root=astap,
        legacy_families=("d50",),
        legacy_blind4d_manifest=corrupt,
        enable_environment_discovery=False,
        strict_legacy_blind4d_manifest=False,
    )
    runtime = resolve_near_catalog_runtime(resources, mode="astap-native")

    assert runtime.available
    assert runtime.provider_kind == "astap_native"
    assert not resources.blind4d_available


def test_strict_invalid_legacy_manifest_is_blocking(tmp_path: Path) -> None:
    astap = _astap_root(tmp_path / "astap")
    corrupt = _corrupt_manifest(tmp_path / "legacy" / "manifest.json")

    with pytest.raises(CatalogResourceResolutionError, match="legacy_blind4d_manifest_invalid"):
        resolve_catalog_resources(
            legacy_db_root=astap,
            legacy_blind4d_manifest=corrupt,
            enable_environment_discovery=False,
            strict_legacy_blind4d_manifest=True,
        )


def test_external_manifest_mode_invalid_legacy_manifest_is_blocking(tmp_path: Path) -> None:
    astap = _astap_root(tmp_path / "astap")
    corrupt = _corrupt_manifest(tmp_path / "legacy" / "manifest.json")

    with pytest.raises(CatalogResourceResolutionError, match="legacy_blind4d_manifest_invalid"):
        resolve_catalog_resources(
            legacy_db_root=astap,
            legacy_blind4d_manifest=corrupt,
            enable_environment_discovery=False,
            strict_legacy_blind4d_manifest=True,
        )


def test_near_runtime_remains_available_when_auto_blind_is_absent(tmp_path: Path) -> None:
    astap = _astap_root(tmp_path / "astap")
    resources = resolve_catalog_resources(
        legacy_db_root=astap,
        legacy_families=("d50",),
        legacy_blind4d_manifest=tmp_path / "legacy" / "manifest.json",
        enable_environment_discovery=False,
        strict_legacy_blind4d_manifest=False,
    )
    near_runtime = resolve_near_catalog_runtime(resources, mode="astap-native")
    blind_runtime = resolve_blind4d_runtime(resources, mode="auto")

    assert near_runtime.available
    assert near_runtime.provider_kind == "astap_native"
    assert not blind_runtime.available
    assert blind_runtime.error_code == "BLIND4D_RUNTIME_RESOURCE_UNAVAILABLE"
