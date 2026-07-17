from __future__ import annotations

import json
from pathlib import Path

from catalog_library_fixtures import make_catalog_library
from zesolver.catalog_library import CatalogLibrary, CatalogStatus


def _status(root: Path) -> CatalogStatus:
    return CatalogLibrary.open(root).validate().status


def test_status_missing_for_empty_manifest(tmp_path: Path) -> None:
    make_catalog_library(tmp_path, include_source=False, include_index=False)

    assert _status(tmp_path) == CatalogStatus.MISSING


def test_status_corrupt_for_sha_mismatch(tmp_path: Path) -> None:
    make_catalog_library(tmp_path, index_sha="0" * 64)

    assert _status(tmp_path) == CatalogStatus.CORRUPT


def test_status_incompatible_for_unknown_index_engine(tmp_path: Path) -> None:
    make_catalog_library(tmp_path, index_engine="future-engine")

    assert _status(tmp_path) == CatalogStatus.INCOMPATIBLE


def test_status_source_only_without_declared_indexes(tmp_path: Path) -> None:
    make_catalog_library(tmp_path, include_source=True, include_index=False)

    assert _status(tmp_path) == CatalogStatus.SOURCE_ONLY


def test_status_near_only_when_index_declared_but_unavailable(tmp_path: Path) -> None:
    make_catalog_library(tmp_path, include_source=True, include_index=True, index_exists=False)

    assert _status(tmp_path) == CatalogStatus.NEAR_ONLY


def test_status_blind4d_only_without_astap_source(tmp_path: Path) -> None:
    make_catalog_library(tmp_path, include_source=False, include_index=True)

    assert _status(tmp_path) == CatalogStatus.BLIND4D_ONLY


def test_status_ready_partial_for_source_and_partial_4d(tmp_path: Path) -> None:
    make_catalog_library(tmp_path, include_source=True, include_index=True)

    report = CatalogLibrary.open(tmp_path).validate()

    assert report.status == CatalogStatus.READY_PARTIAL
    assert report.capabilities.all_sky_blind4d is False


def test_status_ready_full_requires_all_sky_blind4d(tmp_path: Path) -> None:
    make_catalog_library(tmp_path, include_source=True, include_index=True, all_sky_index=True)

    assert _status(tmp_path) == CatalogStatus.READY_FULL


def test_current_six_tile_style_coverage_never_becomes_ready_full(tmp_path: Path) -> None:
    make_catalog_library(tmp_path, include_source=True, include_index=True)
    payload = json.loads((tmp_path / "catalog.json").read_text(encoding="utf-8"))
    payload["derived_indexes"][0]["coverage"]["tile_keys"] = [
        "d50_2602",
        "d50_2644",
        "d50_2645",
        "d50_2702",
        "d50_2822",
        "d50_2823",
    ]
    payload["derived_indexes"][0]["source_tiles"] = payload["derived_indexes"][0]["coverage"]["tile_keys"]
    payload["derived_indexes"][0]["coverage"]["all_sky"] = False
    (tmp_path / "catalog.json").write_text(json.dumps(payload), encoding="utf-8")

    report = CatalogLibrary.open(tmp_path).validate()

    assert report.status == CatalogStatus.READY_PARTIAL
    assert report.capabilities.blind4d is True
    assert report.capabilities.all_sky_blind4d is False
