from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from zesolver.catalog_library import (
    CatalogAdoptionCommitStatus,
    CatalogLibrary,
    CatalogLibraryAdoptionError,
    CatalogLibraryAdoptionPlan,
    CatalogLibraryAdoptionWriter,
)


def _astap_root(root: Path, *, payload: bytes = b"tile") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "d50_2823.1476").write_bytes(payload)
    return root


def _plan(tmp_path: Path, *, library_name: str = "library"):
    library_root = tmp_path / library_name
    library_root.mkdir()
    return CatalogLibraryAdoptionPlan.reference_existing(
        library_root=library_root,
        astap_roots=_astap_root(tmp_path / f"astap-{library_name}"),
        fingerprint_policy="fast",
        generated_at="2026-07-19T00:00:00Z",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_create_new_writes_catalog_atomically_and_reloads(tmp_path: Path) -> None:
    plan = _plan(tmp_path)

    result = CatalogLibraryAdoptionWriter.commit(plan, mode="create")

    assert result.status == CatalogAdoptionCommitStatus.CREATED
    assert result.created is True
    assert result.catalog_path.exists()
    assert result.lock_used is True
    assert result.atomic_replace_used is True
    assert result.post_write_validation is True
    assert CatalogLibrary.open(result.library_root).validate().capabilities.near is True
    assert not list(result.library_root.glob(".catalog.json.*.tmp"))
    assert not (result.library_root / ".catalog-adoption.lock").exists()


def test_create_new_refuses_existing_manifest(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    CatalogLibraryAdoptionWriter.commit(plan, mode="create")
    different_plan = CatalogLibraryAdoptionPlan.reference_existing(
        library_root=plan.library_root,
        astap_roots=tmp_path / "astap-library",
        fingerprint_policy="fast",
        generated_at="different",
    )

    with pytest.raises(CatalogLibraryAdoptionError) as exc:
        CatalogLibraryAdoptionWriter.commit(different_plan, mode="create")

    assert exc.value.code == "CATALOG_ADOPTION_TARGET_EXISTS"


def test_second_identical_commit_is_no_change_without_mtime_change(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    first = CatalogLibraryAdoptionWriter.commit(plan, mode="create")
    before_mtime = first.catalog_path.stat().st_mtime_ns

    second = CatalogLibraryAdoptionWriter.commit(plan, mode="create")

    assert second.status == CatalogAdoptionCommitStatus.NO_CHANGE
    assert second.unchanged is True
    assert second.files_written == 0
    assert first.catalog_path.stat().st_mtime_ns == before_mtime
    assert second.backup_path is None


def test_create_new_refuses_symlink_target(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    target = plan.library_root / "catalog.json"
    target.symlink_to(tmp_path / "elsewhere.json")

    with pytest.raises(CatalogLibraryAdoptionError) as exc:
        CatalogLibraryAdoptionWriter.commit(plan, mode="create")

    assert exc.value.code == "CATALOG_ADOPTION_TARGET_SYMLINK"


def test_create_new_requires_existing_library_root(tmp_path: Path) -> None:
    astap = _astap_root(tmp_path / "astap")
    plan = CatalogLibraryAdoptionPlan.reference_existing(
        library_root=tmp_path / "missing-library",
        astap_roots=astap,
    )

    with pytest.raises(CatalogLibraryAdoptionError) as exc:
        CatalogLibraryAdoptionWriter.commit(plan, mode="create")

    assert exc.value.code == "CATALOG_ADOPTION_READ_ONLY"
