from __future__ import annotations

from pathlib import Path

import pytest

from catalog_library_fixtures import make_catalog_library
from zesolver.catalog_library import CatalogIncompleteError, CatalogLibrary


def test_near_descriptor_exposes_catalogdb_inputs(tmp_path: Path) -> None:
    make_catalog_library(tmp_path)

    near = CatalogLibrary.open(tmp_path).near_source()

    assert near.root == (tmp_path / "sources/astap/d50").resolve()
    assert near.families == ("d50",)
    assert near.formats == ("1476-5",)


def test_near_descriptor_fails_without_valid_source(tmp_path: Path) -> None:
    make_catalog_library(tmp_path, include_source=False, include_index=True)

    with pytest.raises(CatalogIncompleteError, match="NEAR_SOURCE_UNAVAILABLE"):
        CatalogLibrary.open(tmp_path).near_source()


def test_blind4d_descriptors_preserve_manifest_order(tmp_path: Path) -> None:
    make_catalog_library(tmp_path)

    indexes = CatalogLibrary.open(tmp_path).blind4d_indexes()

    assert tuple(index.id for index in indexes) == ("blind4d-d50-2823",)
    assert indexes[0].path == (tmp_path / "indexes/blind4d/d50_2823.npz").resolve()
    assert indexes[0].family == "d50"
    assert indexes[0].tile_keys == ("d50_2823",)


def test_blind4d_runtime_paths_are_ordered(tmp_path: Path) -> None:
    make_catalog_library(tmp_path)

    paths = CatalogLibrary.open(tmp_path).blind4d_runtime_paths()

    assert paths == ((tmp_path / "indexes/blind4d/d50_2823.npz").resolve(),)


def test_blind4d_descriptors_exclude_invalid_indexes(tmp_path: Path) -> None:
    make_catalog_library(tmp_path, index_sha="0" * 64)

    with pytest.raises(CatalogIncompleteError, match="BLIND4D_INDEXES_UNAVAILABLE"):
        CatalogLibrary.open(tmp_path).blind4d_indexes()
