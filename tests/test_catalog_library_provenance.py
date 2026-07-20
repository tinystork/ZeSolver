from __future__ import annotations

from pathlib import Path

from catalog_resource_helpers import strict_entry, write_fake_4d_index, write_strict_manifest
from zesolver.catalog_library import CatalogLibraryAdoptionPlan


def test_blind4d_provenance_links_index_to_astap_source_and_tiles(tmp_path: Path) -> None:
    astap = tmp_path / "astap"
    astap.mkdir()
    (astap / "d50_2823.1476").write_bytes(b"tile")
    index = write_fake_4d_index(tmp_path / "d50_2823_S_q40000.npz", "d50_2823")
    strict_manifest = write_strict_manifest(tmp_path / "manifest.json", [strict_entry("d50_2823", index, "d50_2823")])

    plan = CatalogLibraryAdoptionPlan.reference_existing(
        library_root=tmp_path / "library",
        astap_roots=astap,
        blind4d_manifest=strict_manifest,
    )

    derived = plan.indexes[0]
    assert derived.source_ids == ("astap-d50",)
    assert derived.source_tiles == ("d50_2823",)
    assert derived.families == ("d50",)
    assert derived.provenance_fingerprint
    assert derived.parameter_status.value in {"PARTIAL", "KNOWN"}
    assert derived.reconstruction_status == "PARTIAL"
