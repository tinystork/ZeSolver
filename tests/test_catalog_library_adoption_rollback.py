from __future__ import annotations

import os
from pathlib import Path

import pytest

from zesolver.catalog_library import CatalogAdoptionCommitStatus, CatalogLibrary, CatalogLibraryAdoptionError, CatalogLibraryAdoptionPlan, CatalogLibraryAdoptionWriter
import zesolver.catalog_library.atomic_adoption as atomic_adoption


def _astap(root: Path, data: bytes) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "d50_2823.1476").write_bytes(data)
    return root


def _initial_and_replacement(tmp_path: Path):
    library = tmp_path / "library"
    library.mkdir()
    first = CatalogLibraryAdoptionPlan.reference_existing(library_root=library, astap_roots=_astap(tmp_path / "astap-a", b"a"))
    first_result = CatalogLibraryAdoptionWriter.commit(first, mode="create")
    replacement = CatalogLibraryAdoptionPlan.reference_existing(library_root=library, astap_roots=_astap(tmp_path / "astap-b", b"b"), generated_at="replacement")
    return first_result, replacement


def test_replace_existing_creates_byte_identical_backup(tmp_path: Path) -> None:
    first_result, replacement = _initial_and_replacement(tmp_path)
    original_bytes = first_result.catalog_path.read_bytes()

    result = CatalogLibraryAdoptionWriter.commit(
        replacement,
        mode="replace",
        expected_existing_sha256=first_result.manifest_sha256,
    )

    assert result.status == CatalogAdoptionCommitStatus.REPLACED
    assert result.replaced is True
    assert result.backup_path is not None
    assert result.backup_path.read_bytes() == original_bytes
    assert CatalogLibrary.open(result.library_root).validate().capabilities.near is True


def test_replace_failure_after_backup_rolls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first_result, replacement = _initial_and_replacement(tmp_path)
    original_bytes = first_result.catalog_path.read_bytes()
    real_open = atomic_adoption.CatalogLibrary.open
    calls = {"count": 0}

    def flaky_open(path):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("post validation boom")
        return real_open(path)

    monkeypatch.setattr(atomic_adoption.CatalogLibrary, "open", flaky_open)

    with pytest.raises(CatalogLibraryAdoptionError) as exc:
        CatalogLibraryAdoptionWriter.commit(
            replacement,
            mode="replace",
            expected_existing_sha256=first_result.manifest_sha256,
        )

    assert exc.value.code == "CATALOG_ADOPTION_VALIDATION_FAILED"
    assert exc.value.result is not None
    assert exc.value.result.status == CatalogAdoptionCommitStatus.ROLLED_BACK
    assert first_result.catalog_path.read_bytes() == original_bytes
    assert exc.value.result.backup_path is not None
    assert exc.value.result.backup_path.exists()


def test_create_validation_failure_removes_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    library = tmp_path / "library"
    library.mkdir()
    plan = CatalogLibraryAdoptionPlan.reference_existing(library_root=library, astap_roots=_astap(tmp_path / "astap", b"a"))

    def fail_open(path):
        raise RuntimeError("post validation boom")

    monkeypatch.setattr(atomic_adoption.CatalogLibrary, "open", fail_open)

    with pytest.raises(CatalogLibraryAdoptionError) as exc:
        CatalogLibraryAdoptionWriter.commit(plan, mode="create")

    assert exc.value.code == "CATALOG_ADOPTION_VALIDATION_FAILED"
    assert not (library / "catalog.json").exists()


def test_temp_write_failure_leaves_no_catalog_or_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    library = tmp_path / "library"
    library.mkdir()
    plan = CatalogLibraryAdoptionPlan.reference_existing(library_root=library, astap_roots=_astap(tmp_path / "astap", b"a"))

    def fail_temp(root, data):
        raise atomic_adoption.CatalogLibraryAdoptionError("CATALOG_ADOPTION_TEMP_WRITE_FAILED")

    monkeypatch.setattr(atomic_adoption, "_write_temp_manifest", fail_temp)

    with pytest.raises(CatalogLibraryAdoptionError) as exc:
        CatalogLibraryAdoptionWriter.commit(plan, mode="create")

    assert exc.value.code == "CATALOG_ADOPTION_TEMP_WRITE_FAILED"
    assert not (library / "catalog.json").exists()
    assert not (library / ".catalog-adoption.lock").exists()


def test_temp_fsync_failure_cleans_temp_and_releases_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    library = tmp_path / "library"
    library.mkdir()
    plan = CatalogLibraryAdoptionPlan.reference_existing(library_root=library, astap_roots=_astap(tmp_path / "astap", b"a"))
    real_fsync = atomic_adoption.os.fsync
    calls = {"count": 0}

    def flaky_fsync(fd):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("fsync boom")
        return real_fsync(fd)

    monkeypatch.setattr(atomic_adoption.os, "fsync", flaky_fsync)

    with pytest.raises(CatalogLibraryAdoptionError) as exc:
        CatalogLibraryAdoptionWriter.commit(plan, mode="create")

    assert exc.value.code == "CATALOG_ADOPTION_TEMP_WRITE_FAILED"
    assert not (library / "catalog.json").exists()
    assert not list(library.glob(".catalog.json.*.tmp"))
    assert not (library / ".catalog-adoption.lock").exists()


def test_replace_os_replace_failure_preserves_original_and_cleans_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first_result, replacement = _initial_and_replacement(tmp_path)
    original_bytes = first_result.catalog_path.read_bytes()
    calls = {"count": 0}
    real_replace = atomic_adoption._atomic_replace

    def flaky_replace(src, dst):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("replace boom")
        return real_replace(src, dst)

    monkeypatch.setattr(atomic_adoption, "_atomic_replace", flaky_replace)

    with pytest.raises(CatalogLibraryAdoptionError) as exc:
        CatalogLibraryAdoptionWriter.commit(replacement, mode="replace", expected_existing_sha256=first_result.manifest_sha256)

    assert exc.value.code == "CATALOG_ADOPTION_REPLACE_FAILED"
    assert first_result.catalog_path.read_bytes() == original_bytes
    assert not list(first_result.library_root.glob(".catalog.json.*.tmp"))


def test_backup_creation_failure_preserves_original_and_releases_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first_result, replacement = _initial_and_replacement(tmp_path)
    original_bytes = first_result.catalog_path.read_bytes()

    def fail_backup(path, sha):
        raise atomic_adoption.CatalogLibraryAdoptionError("CATALOG_ADOPTION_REPLACE_FAILED")

    monkeypatch.setattr(atomic_adoption, "_create_backup", fail_backup)

    with pytest.raises(CatalogLibraryAdoptionError) as exc:
        CatalogLibraryAdoptionWriter.commit(replacement, mode="replace", expected_existing_sha256=first_result.manifest_sha256)

    assert exc.value.code == "CATALOG_ADOPTION_REPLACE_FAILED"
    assert first_result.catalog_path.read_bytes() == original_bytes
    assert not (first_result.library_root / ".catalog-adoption.lock").exists()


def test_rollback_failure_reports_stable_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first_result, replacement = _initial_and_replacement(tmp_path)
    real_open = atomic_adoption.CatalogLibrary.open

    def fail_post_open(path):
        raise RuntimeError("post validation boom")

    def fail_rollback(backup, catalog, mode, root):
        raise atomic_adoption.CatalogLibraryAdoptionError("CATALOG_ADOPTION_ROLLBACK_FAILED")

    monkeypatch.setattr(atomic_adoption.CatalogLibrary, "open", fail_post_open)
    monkeypatch.setattr(atomic_adoption, "_attempt_rollback", fail_rollback)

    with pytest.raises(CatalogLibraryAdoptionError) as exc:
        CatalogLibraryAdoptionWriter.commit(replacement, mode="replace", expected_existing_sha256=first_result.manifest_sha256)

    monkeypatch.setattr(atomic_adoption.CatalogLibrary, "open", real_open)
    assert exc.value.code == "CATALOG_ADOPTION_ROLLBACK_FAILED"
    assert exc.value.result is not None
    assert exc.value.result.rollback_performed is True
