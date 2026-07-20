from __future__ import annotations

from pathlib import Path

import pytest

from zesolver.catalog_library import CatalogLibraryAdoptionError, CatalogLibraryAdoptionPlan, CatalogLibraryAdoptionWriter


def _write_astap(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "d50_2823.1476").write_bytes(b"tile")
    return root


def _plan(tmp_path: Path):
    library = tmp_path / "library"
    library.mkdir()
    return CatalogLibraryAdoptionPlan.reference_existing(library_root=library, astap_roots=_write_astap(tmp_path / "astap"))


def test_replace_requires_existing_catalog(tmp_path: Path) -> None:
    plan = _plan(tmp_path)

    with pytest.raises(CatalogLibraryAdoptionError) as exc:
        CatalogLibraryAdoptionWriter.commit(plan, mode="replace", expected_existing_sha256="0" * 64)

    assert exc.value.code == "CATALOG_ADOPTION_TARGET_MISSING"


def test_replace_requires_expected_sha(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    CatalogLibraryAdoptionWriter.commit(plan, mode="create")
    new_plan = CatalogLibraryAdoptionPlan.reference_existing(library_root=plan.library_root, astap_roots=tmp_path / "astap", generated_at="changed")

    with pytest.raises(CatalogLibraryAdoptionError) as exc:
        CatalogLibraryAdoptionWriter.commit(new_plan, mode="replace")

    assert exc.value.code == "CATALOG_ADOPTION_CONFLICT"


def test_replace_refuses_wrong_expected_sha(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    CatalogLibraryAdoptionWriter.commit(plan, mode="create")
    new_plan = CatalogLibraryAdoptionPlan.reference_existing(library_root=plan.library_root, astap_roots=tmp_path / "astap", generated_at="changed")

    with pytest.raises(CatalogLibraryAdoptionError) as exc:
        CatalogLibraryAdoptionWriter.commit(new_plan, mode="replace", expected_existing_sha256="0" * 64)

    assert exc.value.code == "CATALOG_ADOPTION_CONFLICT"


def test_existing_lock_blocks_commit_and_is_not_removed(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    lock = plan.library_root / ".catalog-adoption.lock"
    lock.write_text("busy", encoding="utf-8")

    with pytest.raises(CatalogLibraryAdoptionError) as exc:
        CatalogLibraryAdoptionWriter.commit(plan, mode="create")

    assert exc.value.code == "CATALOG_ADOPTION_LOCKED"
    assert lock.exists()


def test_missing_source_between_plan_and_commit_is_conflict(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    for path in (tmp_path / "astap").iterdir():
        path.unlink()

    with pytest.raises(CatalogLibraryAdoptionError) as exc:
        CatalogLibraryAdoptionWriter.commit(plan, mode="create")

    assert exc.value.code == "CATALOG_ADOPTION_CONFLICT"
