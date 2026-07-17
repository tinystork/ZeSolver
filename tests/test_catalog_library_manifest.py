from __future__ import annotations

import json
from pathlib import Path

import pytest

from catalog_library_fixtures import make_catalog_library
from zesolver.catalog_library import CatalogLibrary, CatalogManifestError, CatalogMissingError, CatalogVersionError


def test_catalog_library_opens_valid_manifest_from_directory(tmp_path: Path) -> None:
    make_catalog_library(tmp_path)

    library = CatalogLibrary.open(tmp_path)

    assert library.manifest.library_id == "test-library"
    assert library.manifest.sources[0].path.resolved == (tmp_path / "sources/astap/d50").resolve()


def test_catalog_library_opens_valid_manifest_from_file(tmp_path: Path) -> None:
    make_catalog_library(tmp_path)

    library = CatalogLibrary.open(tmp_path / "catalog.json")

    assert library.root == tmp_path.resolve()


def test_catalog_library_rejects_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(CatalogMissingError, match="MANIFEST_MISSING"):
        CatalogLibrary.open(tmp_path)


def test_catalog_library_rejects_invalid_json(tmp_path: Path) -> None:
    (tmp_path / "catalog.json").write_text("{", encoding="utf-8")

    with pytest.raises(CatalogManifestError, match="MANIFEST_INVALID_JSON"):
        CatalogLibrary.open(tmp_path)


def test_catalog_library_rejects_missing_required_field(tmp_path: Path) -> None:
    make_catalog_library(tmp_path)
    payload = json.loads((tmp_path / "catalog.json").read_text(encoding="utf-8"))
    payload.pop("library_id")
    (tmp_path / "catalog.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CatalogManifestError, match="REQUIRED_FIELD_MISSING: library_id"):
        CatalogLibrary.open(tmp_path)


def test_catalog_library_rejects_newer_schema_version(tmp_path: Path) -> None:
    make_catalog_library(tmp_path)
    payload = json.loads((tmp_path / "catalog.json").read_text(encoding="utf-8"))
    payload["schema_version"] = 999
    (tmp_path / "catalog.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CatalogVersionError, match="SCHEMA_UNSUPPORTED"):
        CatalogLibrary.open(tmp_path)


def test_catalog_library_allows_declared_external_absolute_path(tmp_path: Path) -> None:
    external = tmp_path / "external-astap"
    external.mkdir()
    (external / "d50_2823.1476").write_bytes(b"x")
    make_catalog_library(tmp_path)
    payload = json.loads((tmp_path / "catalog.json").read_text(encoding="utf-8"))
    payload["sources"][0]["path"] = {"kind": "external_reference", "value": str(external)}
    (tmp_path / "catalog.json").write_text(json.dumps(payload), encoding="utf-8")

    library = CatalogLibrary.open(tmp_path)

    assert library.manifest.sources[0].path.external_reference is True
    assert library.validate().capabilities.near is True


def test_catalog_library_rejects_relative_path_escape(tmp_path: Path) -> None:
    make_catalog_library(tmp_path)
    payload = json.loads((tmp_path / "catalog.json").read_text(encoding="utf-8"))
    payload["sources"][0]["path"] = {"kind": "relative", "value": "../outside"}
    (tmp_path / "catalog.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CatalogManifestError, match="PATH_ESCAPES_LIBRARY"):
        CatalogLibrary.open(tmp_path)


def test_catalog_library_rejects_env_expansion_in_manifest_path(tmp_path: Path) -> None:
    make_catalog_library(tmp_path)
    payload = json.loads((tmp_path / "catalog.json").read_text(encoding="utf-8"))
    payload["sources"][0]["path"] = {"kind": "external_reference", "value": "${ZESOLVER_ASTAP_ROOT}"}
    (tmp_path / "catalog.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CatalogManifestError, match="PATH_ENV_EXPANSION_NOT_ALLOWED"):
        CatalogLibrary.open(tmp_path)
