from __future__ import annotations

from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[1] / "zesolver.py").read_text(encoding="utf-8")


def test_gui_has_targeted_messages_for_common_catalog_path_confusions() -> None:
    required = [
        "CATALOG_LIBRARY_USED_AS_LEGACY_NEAR_INDEX",
        "LEGACY_NEAR_INDEX_USED_AS_CATALOG_LIBRARY",
        "ASTAP_SOURCE_USED_AS_CATALOG_LIBRARY",
        "BLIND4D_MANIFEST_FILE_REQUIRED",
        "Ce dossier est une Bibliothèque ZeSolver, pas un index Near historique.",
        "This folder is a ZeSolver library, not a historical Near index.",
        "Utilisez ce dossier dans le champ « Bibliothèque ZeSolver ».",
        "Use this folder in the ZeSolver library field.",
    ]
    missing = [needle for needle in required if needle not in SOURCE]
    assert not missing


def test_gui_uses_typed_validators_before_save_and_run() -> None:
    required = [
        "validate_catalog_library_root(path)",
        "validate_astap_root(db_root)",
        "validate_astap_root(db_root_text)",
        "validate_legacy_near_index_root(index_root)",
        "validate_legacy_near_index_root(index_root_text)",
        "validate_blind4d_manifest_file(manifest_text_for_save)",
        "validate_blind4d_manifest_file(manifest_text_for_run)",
        "catalog_resources_for_config is None or near_catalog_mode == \"legacy-index\"",
    ]
    missing = [needle for needle in required if needle not in SOURCE]
    assert not missing

