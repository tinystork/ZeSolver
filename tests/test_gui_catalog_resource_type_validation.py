from __future__ import annotations

from pathlib import Path

import numpy as np

from near_catalog_provider_helpers import write_astap_1476_tile, write_legacy_index_from_tile
from zesolver.gui_catalog_validation import (
    validate_astap_root,
    validate_blind4d_manifest_file,
    validate_catalog_library_root,
    validate_legacy_near_index_root,
)


def test_catalog_library_root_rejects_legacy_index_and_astap_source(tmp_path: Path) -> None:
    legacy = write_legacy_index_from_tile(
        tmp_path / "legacy-index",
        tile_key="d50_0001",
        family="d50",
        tile_code="0001",
        center_ra_deg=0.0,
        center_dec_deg=0.0,
        bounds={},
        ra_deg=np.array([0.0, 0.01, 0.02, 0.03]),
        dec_deg=np.array([0.0, 0.01, 0.02, 0.03]),
        mag=np.array([10.0, 10.1, 10.2, 10.3], dtype=np.float32),
    )
    astap = tmp_path / "astap"
    write_astap_1476_tile(
        astap,
        ra_deg=np.array([0.0, 0.01]),
        dec_deg=np.array([0.0, 0.01]),
    )

    assert validate_catalog_library_root(legacy).code == "LEGACY_NEAR_INDEX_USED_AS_CATALOG_LIBRARY"
    assert validate_catalog_library_root(astap).code == "ASTAP_SOURCE_USED_AS_CATALOG_LIBRARY"


def test_legacy_index_root_rejects_catalog_library(tmp_path: Path) -> None:
    library = tmp_path / "ZeSolverCatalog"
    library.mkdir()
    (library / "catalog.json").write_text("{}", encoding="utf-8")

    result = validate_legacy_near_index_root(library)

    assert not result.ok
    assert result.code == "CATALOG_LIBRARY_USED_AS_LEGACY_NEAR_INDEX"


def test_typed_validators_distinguish_expected_resources(tmp_path: Path) -> None:
    astap = tmp_path / "astap source"
    write_astap_1476_tile(
        astap,
        ra_deg=np.array([0.0, 0.01]),
        dec_deg=np.array([0.0, 0.01]),
    )

    assert validate_astap_root(astap).ok
    assert validate_blind4d_manifest_file(astap).code == "BLIND4D_MANIFEST_FILE_REQUIRED"

