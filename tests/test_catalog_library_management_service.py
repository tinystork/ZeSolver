from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest

from near_catalog_provider_helpers import write_astap_1476_tile
from zesolver.catalog_library import CatalogLibrary, CatalogStatus
from zesolver.catalog_library.management import (
    CatalogLibraryManagementCancelled,
    CatalogLibraryManagementError,
    CatalogLibraryManagementService,
    LibraryCreateOptions,
    LibraryInstallOptions,
)
from zeblindsolver.index_manifest_4d import sha256_file


def _write_astap(root: Path, *, subdir: str | None = None, family: str = "d50", tile: str = "1501") -> Path:
    target = root / subdir if subdir else root
    ra = np.asarray([12.00, 12.03, 12.06, 12.10, 12.14, 12.19, 12.25, 12.31], dtype=np.float64)
    dec = np.asarray([1.00, 1.06, 0.98, 1.12, 1.02, 1.16, 1.08, 1.20], dtype=np.float64)
    mag = np.asarray([8.0, 9.5, 8.7, 10.2, 9.0, 11.0, 9.2, 10.8], dtype=np.float32)
    write_astap_1476_tile(target, family=family, tile_code=tile, ra_deg=ra, dec_deg=dec, mag=mag)
    return target


def _package_metadata(library: Path) -> dict:
    hashes = {
        path.relative_to(library).as_posix(): sha256_file(path)
        for path in sorted(library.rglob("*"))
        if path.is_file()
    }
    return {
        "library_id": "fixture-library",
        "version": "1.0",
        "format_version": 1,
        "astap_families": ["d50"],
        "near_coverage": {"status": "partial"},
        "blind4d_coverage": {"status": "partial"},
        "all_sky_blind4d": False,
        "installed_size_bytes": sum(path.stat().st_size for path in library.rglob("*") if path.is_file()),
        "sha256": hashes,
        "provenance": {"source": "pytest"},
        "astap_credit": "ASTAP by Han Kleijn",
        "astrometry_credit": "Astrometry.net",
        "license": "fixture-only",
        "generated_at": "2026-07-22T00:00:00Z",
    }


def _make_created_library(tmp_path: Path) -> Path:
    astap = tmp_path / "astap"
    _write_astap(astap)
    dest = tmp_path / "library"
    service = CatalogLibraryManagementService()
    service.create_from_astap(LibraryCreateOptions(astap_root=astap, destination=dest, families=("d50",)))
    return dest


def test_detect_astap_families_direct_and_subdirectories(tmp_path: Path) -> None:
    direct = tmp_path / "direct"
    nested = tmp_path / "program-files-astap"
    _write_astap(direct)
    _write_astap(nested, subdir="d50")

    service = CatalogLibraryManagementService()

    assert service.detect_astap_families(direct)[0].family == "d50"
    detected = service.detect_astap_families(nested)
    assert [(item.family, item.shard_count, item.root.name) for item in detected] == [("d50", 1, "d50")]


def test_detect_astap_families_empty_missing_case_and_stable_order(tmp_path: Path) -> None:
    service = CatalogLibraryManagementService()
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "notes.txt").write_text("not a catalogue", encoding="utf-8")

    assert service.detect_astap_families(empty) == ()
    with pytest.raises(CatalogLibraryManagementError, match="ASTAP_SOURCE_MISSING"):
        service.detect_astap_families(tmp_path / "missing")

    mixed = tmp_path / "mixed"
    _write_astap(mixed, family="d80", tile="1501")
    _write_astap(mixed, family="d50", tile="1501")
    upper = mixed / "d50_1501.1476"
    upper.rename(mixed / "D50_1501.1476")
    (mixed / "D50_README.1476.txt").write_text("ignore", encoding="utf-8")

    detected = service.detect_astap_families(mixed)
    assert [item.family for item in detected] == ["d50", "d80"]
    assert [item.shard_count for item in detected] == [1, 1]


def test_standard_selection_with_d50_only_builds_only_d50_without_absent_family_noise(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    astap = tmp_path / "astap"
    _write_astap(astap, family="d50", tile="1501")
    dest = tmp_path / "library"
    events = []

    result = CatalogLibraryManagementService(progress_callback=events.append).create_from_astap(
        LibraryCreateOptions(astap_root=astap, destination=dest, families=(), storage_policy="reference")
    )

    manifest = json.loads(result.blind4d_manifest.read_text(encoding="utf-8"))
    assert result.library_root == dest.resolve()
    assert [entry["id"] for entry in manifest["indexes"]] == ["direct-d50"]
    assert {event.family for event in events if event.family} == {"d50"}
    assert all(event.overall_total in {0, 1, 7} for event in events)
    assert "no tiles matched" not in caplog.text


def test_d50_uppercase_filename_builds_with_canonical_family(tmp_path: Path) -> None:
    astap = tmp_path / "astap"
    source = _write_astap(astap, family="d50", tile="1501") / "d50_1501.1476"
    source.rename(astap / "D50_1501.1476")
    dest = tmp_path / "library"

    result = CatalogLibraryManagementService().create_from_astap(
        LibraryCreateOptions(astap_root=astap, destination=dest, families=(), storage_policy="reference")
    )

    catalog = json.loads((dest / "catalog.json").read_text(encoding="utf-8"))
    manifest = json.loads(result.blind4d_manifest.read_text(encoding="utf-8"))
    assert [source["family"] for source in catalog["sources"]] == ["d50"]
    assert [entry["id"] for entry in manifest["indexes"]] == ["direct-d50"]


def test_custom_selection_builds_selected_subset_only(tmp_path: Path) -> None:
    astap = tmp_path / "astap"
    _write_astap(astap, family="d20", tile="1501")
    _write_astap(astap, family="d50", tile="1501")
    dest = tmp_path / "library"

    result = CatalogLibraryManagementService().create_from_astap(
        LibraryCreateOptions(astap_root=astap, destination=dest, families=("D50", "d50"), storage_policy="reference")
    )

    manifest = json.loads(result.blind4d_manifest.read_text(encoding="utf-8"))
    assert [entry["id"] for entry in manifest["indexes"]] == ["direct-d50"]
    catalog = json.loads((dest / "catalog.json").read_text(encoding="utf-8"))
    assert [source["family"] for source in catalog["sources"]] == ["d50"]


def test_selected_absent_family_errors_before_publication(tmp_path: Path) -> None:
    astap = tmp_path / "astap"
    _write_astap(astap, family="d50", tile="1501")
    dest = tmp_path / "library"

    with pytest.raises(CatalogLibraryManagementError, match="ASTAP_FAMILY_MISSING.*d20.*deselect"):
        CatalogLibraryManagementService().create_from_astap(
            LibraryCreateOptions(astap_root=astap, destination=dest, families=("d20",), storage_policy="reference")
        )

    assert not dest.exists()
    assert not list(tmp_path.glob("library.partial-*"))


def test_empty_astap_root_rejected_before_publication(tmp_path: Path) -> None:
    astap = tmp_path / "empty"
    astap.mkdir()
    dest = tmp_path / "library"

    with pytest.raises(CatalogLibraryManagementError, match="ASTAP_NO_FAMILIES_DETECTED"):
        CatalogLibraryManagementService().create_from_astap(LibraryCreateOptions(astap_root=astap, destination=dest))

    assert not dest.exists()


def test_create_from_astap_reference_builds_atomic_library(tmp_path: Path) -> None:
    astap = tmp_path / "astap"
    _write_astap(astap)
    dest = tmp_path / "library"
    events = []

    result = CatalogLibraryManagementService(progress_callback=events.append).create_from_astap(
        LibraryCreateOptions(astap_root=astap, destination=dest, families=("d50",), storage_policy="reference")
    )

    assert result.library_root == dest.resolve()
    assert (dest / "catalog.json").is_file()
    assert (dest / "indexes" / "blind4d" / "strict_4d_manifest.json").is_file()
    assert CatalogLibrary.open(dest).validate().capabilities.near is True
    assert CatalogLibrary.open(dest).validate().capabilities.blind4d is True
    assert not list(tmp_path.glob("library.partial-*"))
    assert [event.stage for event in events]


def test_create_from_astap_copy_policy_never_moves_source(tmp_path: Path) -> None:
    astap = tmp_path / "astap"
    source_file = _write_astap(astap) / "d50_1501.1476"
    dest = tmp_path / "library-copy"

    CatalogLibraryManagementService().create_from_astap(
        LibraryCreateOptions(astap_root=astap, destination=dest, families=("d50",), storage_policy="copy")
    )

    assert source_file.exists()
    assert (dest / "sources" / "astap" / source_file.name).is_file()
    manifest = json.loads((dest / "catalog.json").read_text(encoding="utf-8"))
    assert "ASTAP" in manifest["provenance"]["astap_credit"]


@pytest.mark.parametrize("destination", ["same", "inside-source", "source-inside-dest"])
def test_create_rejects_source_destination_overlaps(tmp_path: Path, destination: str) -> None:
    astap = tmp_path / "astap"
    _write_astap(astap)
    if destination == "same":
        dest = astap
    elif destination == "inside-source":
        dest = astap / "library"
    else:
        dest = tmp_path / "parent"
        astap = dest / "astap"
        _write_astap(astap)

    with pytest.raises(CatalogLibraryManagementError, match="SOURCE|DESTINATION"):
        CatalogLibraryManagementService().create_from_astap(LibraryCreateOptions(astap_root=astap, destination=dest))


def test_create_cleans_staging_after_error_and_cancel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    astap = tmp_path / "astap"
    _write_astap(astap)

    service = CatalogLibraryManagementService(cancel_callback=lambda: True)
    with pytest.raises(CatalogLibraryManagementCancelled):
        service.create_from_astap(LibraryCreateOptions(astap_root=astap, destination=tmp_path / "cancelled"))
    assert not list(tmp_path.glob("cancelled.partial-*"))
    assert not (tmp_path / "cancelled" / "catalog.json").exists()

    import zesolver.catalog_library.management as management

    def fail_build(*_args, **_kwargs):
        raise RuntimeError("forced")

    monkeypatch.setattr(management, "build_4d_index_from_astap", fail_build)
    with pytest.raises(CatalogLibraryManagementError, match="forced"):
        CatalogLibraryManagementService().create_from_astap(LibraryCreateOptions(astap_root=astap, destination=tmp_path / "failed"))
    assert not list(tmp_path.glob("failed.partial-*"))
    assert not (tmp_path / "failed" / "catalog.json").exists()


def test_install_package_valid_and_auto_selectable_result(tmp_path: Path) -> None:
    library = _make_created_library(tmp_path / "build")
    package = tmp_path / "package"
    shutil.copytree(library, package / "library")
    metadata = _package_metadata(package / "library")
    (package / "zesolver-library-package.json").write_text(json.dumps(metadata), encoding="utf-8")

    dest = tmp_path / "installed"
    result = CatalogLibraryManagementService().install_package(LibraryInstallOptions(package_path=package, destination=dest))

    assert result.library_root == dest.resolve()
    assert (dest / "catalog.json").is_file()
    assert result.metadata["library_id"] == "fixture-library"


def test_install_package_from_valid_zip_archive(tmp_path: Path) -> None:
    library = _make_created_library(tmp_path / "build")
    package_dir = tmp_path / "package"
    shutil.copytree(library, package_dir / "library")
    metadata = _package_metadata(package_dir / "library")
    (package_dir / "zesolver-library-package.json").write_text(json.dumps(metadata), encoding="utf-8")
    archive = tmp_path / "package.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in sorted(package_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(package_dir).as_posix())

    result = CatalogLibraryManagementService().install_package(
        LibraryInstallOptions(package_path=archive, destination=tmp_path / "installed-zip")
    )

    assert result.library_root == (tmp_path / "installed-zip").resolve()
    assert result.status in {CatalogStatus.READY_PARTIAL, CatalogStatus.READY_FULL}


def test_install_package_rejects_bad_sha_and_unsafe_archive(tmp_path: Path) -> None:
    library = _make_created_library(tmp_path / "build")
    package = tmp_path / "package"
    shutil.copytree(library, package / "library")
    metadata = _package_metadata(package / "library")
    metadata["sha256"]["catalog.json"] = "0" * 64
    (package / "zesolver-library-package.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(CatalogLibraryManagementError, match="SHA256"):
        CatalogLibraryManagementService().install_package(LibraryInstallOptions(package_path=package, destination=tmp_path / "bad"))

    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../evil.txt", "nope")
    with pytest.raises(CatalogLibraryManagementError, match="UNSAFE_PATH"):
        CatalogLibraryManagementService().install_package(LibraryInstallOptions(package_path=archive, destination=tmp_path / "unsafe"))


def test_install_package_rejects_metadata_and_disk_space(tmp_path: Path) -> None:
    library = _make_created_library(tmp_path / "build")
    package = tmp_path / "package"
    shutil.copytree(library, package / "library")
    metadata = _package_metadata(package / "library")
    metadata.pop("license")
    (package / "zesolver-library-package.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(CatalogLibraryManagementError, match="INCOMPLETE"):
        CatalogLibraryManagementService().install_package(LibraryInstallOptions(package_path=package, destination=tmp_path / "missing-meta"))

    metadata = _package_metadata(package / "library")
    (package / "zesolver-library-package.json").write_text(json.dumps(metadata), encoding="utf-8")
    disk_usage = lambda _path: shutil._ntuple_diskusage(total=1, used=1, free=0)
    with pytest.raises(CatalogLibraryManagementError, match="DISK_SPACE"):
        CatalogLibraryManagementService(disk_usage=disk_usage).install_package(
            LibraryInstallOptions(package_path=package, destination=tmp_path / "no-space")
        )


def test_analyze_complete_near_only_and_repair_plan(tmp_path: Path) -> None:
    complete = _make_created_library(tmp_path / "complete")
    analysis = CatalogLibraryManagementService().analyze_library(complete)
    assert analysis.status in {"READY_PARTIAL", "READY_FULL"}
    assert not analysis.repair_plan.actions

    astap = tmp_path / "near-only-astap"
    _write_astap(astap)
    near_only = tmp_path / "near-only"
    near_only.mkdir()
    from zesolver.catalog_library import CatalogLibraryAdoptionPlan, CatalogLibraryAdoptionWriter

    plan = CatalogLibraryAdoptionPlan.reference_existing(library_root=near_only, astap_roots=astap)
    CatalogLibraryAdoptionWriter.commit(plan, mode="create")
    analysis = CatalogLibraryManagementService().analyze_library(near_only)

    assert "build_missing_blind4d" in analysis.repair_plan.actions
    assert any(item.element == "Blind 4D" and item.state == "missing" for item in analysis.items)


def test_repair_near_only_builds_missing_blind4d_without_rebuilding_complete_library(tmp_path: Path) -> None:
    astap = tmp_path / "astap"
    _write_astap(astap)
    near_only = tmp_path / "near-only"
    near_only.mkdir()
    from zesolver.catalog_library import CatalogLibraryAdoptionPlan, CatalogLibraryAdoptionWriter

    plan = CatalogLibraryAdoptionPlan.reference_existing(library_root=near_only, astap_roots=astap)
    CatalogLibraryAdoptionWriter.commit(plan, mode="create")
    service = CatalogLibraryManagementService()
    analysis = service.analyze_library(near_only)

    result = service.repair_library(analysis.repair_plan)

    assert result.library_root == near_only.resolve()
    assert (near_only / "indexes" / "blind4d" / "strict_4d_manifest.json").is_file()
    assert CatalogLibrary.open(near_only).validate().capabilities.blind4d is True
