from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from catalog_resource_helpers import strict_entry, write_catalog_library, write_fake_4d_index, write_strict_manifest
from near_catalog_provider_helpers import write_astap_1476_tile, write_legacy_index_from_tile
from zesolver.catalog_resources import (
    ASTAP_NEAR_RESOURCE_REQUIRED,
    LEGACY_NEAR_INDEX_REQUIRED,
    NEAR_CATALOG_MODE_INVALID,
    NearCatalogMode,
    NearCatalogRuntimeError,
    resolve_catalog_resources,
    resolve_near_catalog_runtime,
)


def _library_with_astap_tile(tmp_path: Path) -> Path:
    root = write_catalog_library(tmp_path / "library", include_source=True, index_paths=[])
    source = root / "sources" / "astap" / "d50"
    write_astap_1476_tile(
        source,
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


def _library_without_near(tmp_path: Path) -> Path:
    idx = write_fake_4d_index(tmp_path / "d50_A_S_q.npz", "d50_A")
    manifest = write_strict_manifest(tmp_path / "manifest.json", [strict_entry("a", idx, "d50_A")])
    return write_catalog_library(
        tmp_path / "library",
        include_source=False,
        index_paths=[idx],
        strict_manifest_path=manifest,
    )


def test_auto_prefers_catalog_library_astap_native_over_residual_legacy_path(tmp_path: Path) -> None:
    legacy_root = tmp_path / "sentinel-missing-legacy"
    resources = resolve_catalog_resources(catalog_library=_library_with_astap_tile(tmp_path))

    runtime = resolve_near_catalog_runtime(resources, mode="auto", legacy_index_root=legacy_root)

    assert runtime.effective_mode is NearCatalogMode.ASTAP_NATIVE
    assert runtime.provider is not None
    assert runtime.provider.kind == "astap_native"
    assert runtime.legacy_index_root is None


def test_auto_uses_legacy_when_no_catalog_library_is_selected(tmp_path: Path) -> None:
    legacy_root = _legacy_index(tmp_path)
    resources = resolve_catalog_resources(
        legacy_db_root=tmp_path / "astap",
        legacy_families=("d50",),
        legacy_index_root=legacy_root,
        enable_environment_discovery=False,
    )

    runtime = resolve_near_catalog_runtime(resources, mode="auto")

    assert runtime.effective_mode is NearCatalogMode.LEGACY_INDEX
    assert runtime.provider is not None
    assert runtime.provider.kind == "legacy_index"
    assert runtime.legacy_index_root == legacy_root


def test_auto_library_without_near_does_not_fall_back_to_legacy(tmp_path: Path) -> None:
    legacy_root = _legacy_index(tmp_path)
    resources = resolve_catalog_resources(
        catalog_library=_library_without_near(tmp_path),
        legacy_index_root=legacy_root,
        enable_environment_discovery=False,
    )

    runtime = resolve_near_catalog_runtime(resources, mode="auto", legacy_index_root=legacy_root)

    assert runtime.provider is None
    assert runtime.effective_mode is NearCatalogMode.ASTAP_NATIVE
    assert runtime.error_code == ASTAP_NEAR_RESOURCE_REQUIRED
    assert runtime.legacy_index_root is None


def test_forced_astap_native_requires_near_resource(tmp_path: Path) -> None:
    resources = resolve_catalog_resources(
        catalog_library=_library_without_near(tmp_path),
        enable_environment_discovery=False,
    )

    with pytest.raises(NearCatalogRuntimeError) as excinfo:
        resolve_near_catalog_runtime(resources, mode="astap-native")

    assert excinfo.value.code == ASTAP_NEAR_RESOURCE_REQUIRED


def test_forced_legacy_requires_explicit_legacy_index(tmp_path: Path) -> None:
    resources = resolve_catalog_resources(catalog_library=_library_with_astap_tile(tmp_path))

    with pytest.raises(NearCatalogRuntimeError) as excinfo:
        resolve_near_catalog_runtime(resources, mode="legacy-index")

    assert excinfo.value.code == LEGACY_NEAR_INDEX_REQUIRED


def test_invalid_near_catalog_mode_is_stable_error(tmp_path: Path) -> None:
    resources = resolve_catalog_resources(catalog_library=_library_with_astap_tile(tmp_path))

    with pytest.raises(NearCatalogRuntimeError) as excinfo:
        resolve_near_catalog_runtime(resources, mode="banana")

    assert excinfo.value.code == NEAR_CATALOG_MODE_INVALID
