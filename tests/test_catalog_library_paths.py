from __future__ import annotations

import json
from pathlib import Path

import pytest

from catalog_library_fixtures import make_catalog_library
from zesolver.catalog_library import CatalogLibrary, CatalogManifestError


def test_relative_managed_path_accepts_windows_separators(tmp_path: Path) -> None:
    make_catalog_library(tmp_path)
    payload = json.loads((tmp_path / "catalog.json").read_text(encoding="utf-8"))
    payload["sources"][0]["path"] = {"kind": "relative", "value": "sources\\astap\\d50"}
    (tmp_path / "catalog.json").write_text(json.dumps(payload), encoding="utf-8")

    library = CatalogLibrary.open(tmp_path)

    assert library.manifest.sources[0].path.resolved == (tmp_path / "sources" / "astap" / "d50").resolve()


def test_managed_path_rejects_windows_separator_traversal(tmp_path: Path) -> None:
    make_catalog_library(tmp_path)
    payload = json.loads((tmp_path / "catalog.json").read_text(encoding="utf-8"))
    payload["sources"][0]["path"] = {"kind": "relative", "value": "sources\\..\\outside"}
    (tmp_path / "catalog.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CatalogManifestError, match="PATH_ESCAPES_LIBRARY"):
        CatalogLibrary.open(tmp_path)


def test_external_absolute_path_outside_library_is_allowed(tmp_path: Path) -> None:
    external = tmp_path / "outside library"
    external.mkdir()
    (external / "d50_2823.1476").write_bytes(b"tile")
    make_catalog_library(tmp_path)
    payload = json.loads((tmp_path / "catalog.json").read_text(encoding="utf-8"))
    payload["sources"][0]["path"] = {"kind": "external_reference", "value": str(external)}
    (tmp_path / "catalog.json").write_text(json.dumps(payload), encoding="utf-8")

    report = CatalogLibrary.open(tmp_path).validate()

    assert report.capabilities.near is True


def test_external_relative_path_is_rejected(tmp_path: Path) -> None:
    make_catalog_library(tmp_path)
    payload = json.loads((tmp_path / "catalog.json").read_text(encoding="utf-8"))
    payload["sources"][0]["path"] = {"kind": "external_reference", "value": "relative/astap"}
    (tmp_path / "catalog.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CatalogManifestError, match="EXTERNAL_PATH_NOT_ABSOLUTE"):
        CatalogLibrary.open(tmp_path)
