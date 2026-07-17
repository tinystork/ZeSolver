from __future__ import annotations

from pathlib import Path

import pytest

from zesolver.catalog_library import CatalogLibrary, CatalogStatus
from zesolver.catalog_resources import CatalogResourceResolutionError, resolve_catalog_resources

from catalog_resource_helpers import (
    strict_entry,
    write_catalog_library,
    write_fake_4d_index,
    write_strict_manifest,
)


def test_no_library_uses_legacy_resources(tmp_path: Path) -> None:
    idx = write_fake_4d_index(tmp_path / "d50_A_S_q.npz", "d50_A")
    manifest = write_strict_manifest(tmp_path / "manifest.json", [strict_entry("a", idx, "d50_A")])

    resources = resolve_catalog_resources(
        legacy_db_root=tmp_path / "database",
        legacy_families=("d50",),
        legacy_blind4d_manifest=manifest,
        legacy_index_root=tmp_path / "legacy-index",
    )

    assert resources.source == "legacy"
    assert resources.near is not None
    assert resources.near.root == tmp_path / "database"
    assert resources.near.families == ("d50",)
    assert resources.blind4d_runtime_paths == (idx.resolve(),)
    assert resources.blind4d_manifest_path == manifest


def test_valid_library_takes_priority_over_contradictory_legacy(tmp_path: Path) -> None:
    idx = write_fake_4d_index(tmp_path / "d50_A_S_q.npz", "d50_A")
    manifest = write_strict_manifest(tmp_path / "manifest.json", [strict_entry("a", idx, "d50_A")])
    library_root = write_catalog_library(
        tmp_path / "library",
        index_paths=[idx],
        strict_manifest_path=manifest,
    )

    resources = resolve_catalog_resources(
        catalog_library=CatalogLibrary.open(library_root),
        legacy_db_root=tmp_path / "legacy-db",
        legacy_families=("g05",),
        legacy_blind4d_manifest=tmp_path / "legacy-manifest.json",
    )

    assert resources.source == "library"
    assert resources.library_status is CatalogStatus.READY_PARTIAL
    assert resources.near is not None
    assert resources.near.root == (library_root / "sources" / "astap" / "d50").resolve()
    assert resources.near.families == ("d50",)
    assert resources.blind4d_manifest_path == manifest.resolve()
    assert resources.blind4d_runtime_paths == (idx.resolve(),)


def test_explicit_corrupt_library_does_not_fall_back_silently(tmp_path: Path) -> None:
    idx = write_fake_4d_index(tmp_path / "d50_A_S_q.npz", "d50_A")
    manifest = write_strict_manifest(tmp_path / "manifest.json", [strict_entry("a", idx, "d50_A")])
    library_root = write_catalog_library(
        tmp_path / "library",
        index_paths=[idx],
        strict_manifest_path=manifest,
        bad_index_sha=True,
    )

    with pytest.raises(CatalogResourceResolutionError, match="SHA256_MISMATCH"):
        resolve_catalog_resources(
            catalog_library=library_root,
            legacy_db_root=tmp_path / "legacy-db",
            legacy_blind4d_manifest=manifest,
        )


def test_absent_catalogues_are_explicit_none() -> None:
    resources = resolve_catalog_resources(enable_environment_discovery=False)

    assert resources.source == "none"
    assert resources.near is None
    assert resources.blind4d_runtime_paths == ()
    assert resources.warnings == ("catalog_resources_absent",)
