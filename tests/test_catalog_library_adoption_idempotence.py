from __future__ import annotations

from pathlib import Path

from zesolver.catalog_library import CatalogAdoptionCommitStatus, CatalogLibraryAdoptionPlan, CatalogLibraryAdoptionWriter


def test_two_successive_commits_of_same_plan_have_no_second_effect(tmp_path: Path) -> None:
    astap = tmp_path / "astap"
    astap.mkdir()
    (astap / "d50_2823.1476").write_bytes(b"tile")
    library = tmp_path / "library"
    library.mkdir()
    plan = CatalogLibraryAdoptionPlan.reference_existing(library_root=library, astap_roots=astap)

    first = CatalogLibraryAdoptionWriter.commit(plan, mode="create")
    before = first.catalog_path.read_bytes()
    before_mtime = first.catalog_path.stat().st_mtime_ns
    second = CatalogLibraryAdoptionWriter.commit(plan, mode="replace", expected_existing_sha256=first.manifest_sha256)

    assert second.status == CatalogAdoptionCommitStatus.NO_CHANGE
    assert second.files_written == 0
    assert second.catalog_path.read_bytes() == before
    assert second.catalog_path.stat().st_mtime_ns == before_mtime
