from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from near_catalog_provider_helpers import write_astap_1476_tile
from zesolver.catalog_library.models import CatalogCoverage, CoverageStatus, NearCatalogDescriptor
from zesolver.catalog_resources import SolverCatalogResources, build_near_catalog_provider


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_near_catalog_provider_has_no_product_or_gui_imports() -> None:
    path = REPO_ROOT / "zeblindsolver" / "near_catalog_provider.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden_prefixes = (
        "zesolver",
        "zesolver.catalog_library",
        "zesolver.catalog_resources",
        "PySide6",
    )
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")

    assert not [
        name
        for name in imports
        for forbidden in forbidden_prefixes
        if name == forbidden or name.startswith(forbidden + ".")
    ]


def test_astap_provider_source_does_not_reference_index_roots_or_blind4d() -> None:
    source = (REPO_ROOT / "zeblindsolver" / "near_catalog_provider.py").read_text(encoding="utf-8")
    astap_class = source.split("class AstapNearCatalogProvider:", 1)[1]

    assert "index_root" not in astap_class
    assert "manifest.json" not in astap_class
    assert "quad_index_4d" not in astap_class
    assert "np.save" not in astap_class


def test_product_adapter_builds_astap_provider_from_near_descriptor(tmp_path: Path) -> None:
    write_astap_1476_tile(
        tmp_path,
        family="d50",
        tile_code="1501",
        ra_deg=np.array([1.0]),
        dec_deg=np.array([-18.0]),
    )
    resources = SolverCatalogResources(
        library_path=None,
        library_status=None,
        near=NearCatalogDescriptor(
            root=tmp_path,
            families=("d50",),
            formats=("1476-5",),
            coverage=CatalogCoverage(status=CoverageStatus.UNKNOWN),
            external_reference=True,
        ),
        blind4d_indexes=(),
        blind4d_runtime_paths=(),
        blind4d_manifest_path=None,
        legacy_index_root=None,
        source="test",
        warnings=(),
    )

    provider = build_near_catalog_provider(resources)

    assert provider.kind == "astap_native"
    assert provider.families == ("d50",)
