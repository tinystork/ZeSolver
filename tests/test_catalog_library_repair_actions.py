from __future__ import annotations

from pathlib import Path

from catalog_resource_helpers import strict_entry, write_fake_4d_index, write_strict_manifest
from zesolver.catalog_library import CatalogLibraryAdoptionPlan


def test_rebuild_blind4d_index_action_is_prescription_for_p1d3_or_later(tmp_path: Path) -> None:
    index = write_fake_4d_index(tmp_path / "d50_2823_S_q40000.npz", "d50_2823")
    entry = strict_entry("d50_2823", index, "d50_2823")
    entry.pop("code_tol_recommended")
    strict_manifest = write_strict_manifest(tmp_path / "manifest.json", [entry])

    plan = CatalogLibraryAdoptionPlan.reference_existing(
        library_root=tmp_path / "library",
        blind4d_manifest=strict_manifest,
    )

    actions = [action for action in plan.repair_actions if action.code == "REBUILD_BLIND4D_INDEX"]
    assert actions
    assert actions[0].execution_phase == "P1D-3_OR_LATER"
    assert actions[0].automatic is False
    assert plan.telemetry.builder_called is False


def test_stale_legacy_reference_repair_is_compatibility_cleanup_only(tmp_path: Path) -> None:
    plan = CatalogLibraryAdoptionPlan.reference_existing(
        library_root=tmp_path / "library",
        legacy_index_root=tmp_path / "missing-legacy",
    )

    actions = [action for action in plan.repair_actions if action.code == "REMOVE_STALE_COMPATIBILITY_REFERENCE"]
    assert actions
    assert actions[0].resource_id == "legacy-index"
    assert actions[0].automatic is False
