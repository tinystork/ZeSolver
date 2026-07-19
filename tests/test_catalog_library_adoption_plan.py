from __future__ import annotations

import json
from pathlib import Path

from catalog_resource_helpers import strict_entry, write_fake_4d_index, write_strict_manifest
from zesolver.catalog_library import CatalogLibrary, CatalogLibraryAdoptionPlan, CatalogStatus


def _astap_root(root: Path, *, family: str = "d50", tiles: tuple[str, ...] = ("2823",)) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for tile in tiles:
        (root / f"{family}_{tile}.1476").write_bytes(f"{family}-{tile}".encode("ascii"))
    return root


def test_adoption_plan_astap_only_is_source_only_and_writes_nothing(tmp_path: Path) -> None:
    astap = _astap_root(tmp_path / "ASTAP With Spaces")
    before = {path: path.stat().st_mtime_ns for path in astap.iterdir()}

    plan = CatalogLibraryAdoptionPlan.reference_existing(
        library_root=tmp_path / "library",
        astap_roots=astap,
        fingerprint_policy="fast",
    )

    assert plan.status == CatalogStatus.SOURCE_ONLY
    assert plan.sources[0].id == "astap-d50"
    assert plan.sources[0].path.external_reference is True
    assert plan.telemetry.files_written == 0
    assert plan.telemetry.builder_called is False
    assert {path: path.stat().st_mtime_ns for path in astap.iterdir()} == before


def test_adoption_plan_fast_inventory_does_not_hash_source_shards(tmp_path: Path) -> None:
    astap = _astap_root(tmp_path / "astap")

    plan = CatalogLibraryAdoptionPlan.reference_existing(
        library_root=tmp_path / "library",
        astap_roots=astap,
        fingerprint_policy="fast",
    )

    assert plan.telemetry.source_file_count == 1
    assert plan.telemetry.source_hashed_count == 0
    assert plan.sources[0].shards[0].sha256 is None
    assert any(action.code == "VERIFY_SOURCE_SHA256" for action in plan.repair_actions)


def test_adoption_plan_full_hashes_source_shards(tmp_path: Path) -> None:
    astap = _astap_root(tmp_path / "astap")
    seen: list[tuple[int, int, Path]] = []

    plan = CatalogLibraryAdoptionPlan.reference_existing(
        library_root=tmp_path / "library",
        astap_roots=astap,
        fingerprint_policy="full",
        progress_callback=lambda current, total, path: seen.append((current, total, path)),
    )

    assert plan.telemetry.source_hashed_count == 1
    assert plan.sources[0].shards[0].sha256 is not None
    assert seen and seen[0][1] == 1


def test_adoption_plan_astap_blind4d_preview_is_valid_catalog_manifest(tmp_path: Path) -> None:
    astap = _astap_root(tmp_path / "astap", tiles=("2823",))
    index = write_fake_4d_index(tmp_path / "d50_2823_S_q40000.npz", "d50_2823")
    strict_manifest = write_strict_manifest(tmp_path / "manifest.json", [strict_entry("d50_2823", index, "d50_2823")])

    plan = CatalogLibraryAdoptionPlan.reference_existing(
        library_root=tmp_path / "library",
        astap_roots=astap,
        blind4d_manifest=strict_manifest,
        fingerprint_policy="fast",
        generated_at="2026-07-19T00:00:00Z",
    )
    preview_root = tmp_path / "preview"
    preview_root.mkdir()
    (preview_root / "catalog.json").write_text(json.dumps(plan.manifest_preview, indent=2), encoding="utf-8")

    library = CatalogLibrary.open(preview_root)
    report = library.validate()

    assert plan.status == CatalogStatus.READY_PARTIAL
    assert report.status == CatalogStatus.READY_PARTIAL
    assert report.capabilities.near is True
    assert report.capabilities.blind4d is True
    assert report.capabilities.all_sky_blind4d is False
    assert plan.indexes[0].source_ids == ("astap-d50",)
    assert plan.indexes[0].source_tiles == ("d50_2823",)


def test_adoption_plan_keeps_legacy_index_as_compatibility_only(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy-index"
    (legacy / "tiles").mkdir(parents=True)
    (legacy / "hash_tables").mkdir()
    (legacy / "manifest.json").write_text("{}", encoding="utf-8")

    plan = CatalogLibraryAdoptionPlan.reference_existing(
        library_root=tmp_path / "library",
        legacy_index_root=legacy,
        fingerprint_policy="fast",
    )

    assert plan.indexes == ()
    assert plan.compatibility_resources[0].category == "compatibility"
    assert plan.manifest_preview["compatibility_resources"][0]["category"] == "compatibility"


def test_adoption_plan_reports_missing_4d_manifest_as_prescription(tmp_path: Path) -> None:
    plan = CatalogLibraryAdoptionPlan.reference_existing(
        library_root=tmp_path / "library",
        blind4d_manifest=tmp_path / "missing-manifest.json",
    )

    assert plan.status == CatalogStatus.MISSING
    assert any(error.code == "BLIND4D_MANIFEST_MISSING" for error in plan.errors)
    assert any(action.code == "LOCATE_MISSING_INDEX" for action in plan.repair_actions)


def test_adoption_preview_is_deterministic_except_informative_generated_at(tmp_path: Path) -> None:
    astap = _astap_root(tmp_path / "astap")

    first = CatalogLibraryAdoptionPlan.reference_existing(
        library_root=tmp_path / "library",
        astap_roots=astap,
        generated_at="2026-07-19T00:00:00Z",
    ).manifest_preview
    second = CatalogLibraryAdoptionPlan.reference_existing(
        library_root=tmp_path / "library",
        astap_roots=astap,
        generated_at="2026-07-20T00:00:00Z",
    ).manifest_preview
    first_no_time = {**first, "created_at": None}
    second_no_time = {**second, "created_at": None}

    assert first_no_time == second_no_time
