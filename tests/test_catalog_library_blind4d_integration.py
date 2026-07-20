from __future__ import annotations

from pathlib import Path

import pytest

from zesolver.catalog_resources import CatalogResourceResolutionError, resolve_catalog_resources

from catalog_resource_helpers import (
    strict_entry,
    write_catalog_library,
    write_fake_4d_index,
    write_strict_manifest,
)


def test_library_blind4d_preserves_strict_manifest_order(tmp_path: Path) -> None:
    idx_a = write_fake_4d_index(tmp_path / "d50_A_S_q.npz", "d50_A")
    idx_b = write_fake_4d_index(tmp_path / "d50_B_S_q.npz", "d50_B")
    manifest = write_strict_manifest(
        tmp_path / "manifest.json",
        [strict_entry("a", idx_a, "d50_A"), strict_entry("b", idx_b, "d50_B")],
    )
    library_root = write_catalog_library(
        tmp_path / "library",
        index_paths=[idx_a, idx_b],
        strict_manifest_path=manifest,
    )

    resources = resolve_catalog_resources(catalog_library=library_root)

    assert resources.blind4d_manifest_path == manifest.resolve()
    assert resources.blind4d_runtime_paths == (idx_a.resolve(), idx_b.resolve())
    assert [index.id for index in resources.blind4d_indexes] == ["blind4d-0", "blind4d-1"]


def test_library_bad_fingerprint_blocks_blind4d_use(tmp_path: Path) -> None:
    idx = write_fake_4d_index(tmp_path / "d50_A_S_q.npz", "d50_A")
    manifest = write_strict_manifest(tmp_path / "manifest.json", [strict_entry("a", idx, "d50_A")])
    library_root = write_catalog_library(
        tmp_path / "library",
        index_paths=[idx],
        strict_manifest_path=manifest,
        bad_index_sha=True,
    )

    with pytest.raises(CatalogResourceResolutionError, match="SHA256_MISMATCH"):
        resolve_catalog_resources(catalog_library=library_root)


def test_six_indexes_remain_partial_not_all_sky(tmp_path: Path) -> None:
    indexes = [
        write_fake_4d_index(tmp_path / f"d50_{2600 + i}_S_q.npz", f"d50_{2600 + i}")
        for i in range(6)
    ]
    manifest = write_strict_manifest(
        tmp_path / "manifest.json",
        [strict_entry(f"idx-{i}", index, f"d50_{2600 + i}") for i, index in enumerate(indexes)],
    )
    library_root = write_catalog_library(
        tmp_path / "library",
        index_paths=indexes,
        strict_manifest_path=manifest,
    )

    resources = resolve_catalog_resources(catalog_library=library_root)

    assert resources.blind4d_index_count == 6
    assert resources.all_sky_blind4d is False
    assert "blind4d_coverage_not_all_sky" in resources.warnings
    assert resources.telemetry()["blind4d_all_sky"] is False


def test_legacy_manifest_still_works_without_library(tmp_path: Path) -> None:
    idx = write_fake_4d_index(tmp_path / "d50_A_S_q.npz", "d50_A")
    manifest = write_strict_manifest(tmp_path / "manifest.json", [strict_entry("a", idx, "d50_A")])

    resources = resolve_catalog_resources(legacy_blind4d_manifest=manifest)

    assert resources.source == "legacy"
    assert resources.blind4d_manifest_path == manifest
    assert resources.blind4d_runtime_paths == (idx.resolve(),)
