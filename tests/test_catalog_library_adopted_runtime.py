from __future__ import annotations

from pathlib import Path

from catalog_resource_helpers import strict_entry, write_fake_4d_index, write_strict_manifest
from zesolver.catalog_library import CatalogLibrary, CatalogLibraryAdoptionPlan, CatalogLibraryAdoptionWriter
from zesolver.catalog_resources import NearCatalogMode, resolve_catalog_resources, resolve_near_catalog_runtime


def test_adopted_manifest_resolves_near_astap_native_and_blind4d(tmp_path: Path) -> None:
    astap = tmp_path / "astap"
    astap.mkdir()
    (astap / "d50_2823.1476").write_bytes(b"tile")
    index = write_fake_4d_index(tmp_path / "d50_2823_S_q40000.npz", "d50_2823")
    strict_manifest = write_strict_manifest(tmp_path / "manifest.json", [strict_entry("d50_2823", index, "d50_2823")])
    library = tmp_path / "library"
    library.mkdir()
    plan = CatalogLibraryAdoptionPlan.reference_existing(
        library_root=library,
        astap_roots=astap,
        blind4d_manifest=strict_manifest,
        legacy_index_root=tmp_path / "legacy-does-not-matter",
    )

    CatalogLibraryAdoptionWriter.commit(plan, mode="create")
    opened = CatalogLibrary.open(library)
    resources = resolve_catalog_resources(catalog_library=opened)
    near = resolve_near_catalog_runtime(resources, mode=NearCatalogMode.AUTO, legacy_index_root=tmp_path / "stale-legacy")

    assert resources.source == "library"
    assert resources.near is not None
    assert resources.near.root == astap.resolve()
    assert resources.blind4d_manifest_path == strict_manifest.resolve()
    assert resources.blind4d_runtime_paths == (index.resolve(),)
    assert resources.all_sky_blind4d is False
    assert resources.coverage is not None
    assert resources.coverage.covered_tiles == 1
    assert resources.coverage.total_tiles == 1476
    assert near.provider_kind == "astap_native"
    assert near.legacy_index_root is None
