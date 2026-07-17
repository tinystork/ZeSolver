from __future__ import annotations

import json
from pathlib import Path

from catalog_library_fixtures import make_catalog_library
from zesolver.catalog_library import CatalogLibrary, CatalogStatus


def _issue_codes(root: Path) -> set[str]:
    return {issue.code for issue in CatalogLibrary.open(root).validate().issues}


def test_validation_accepts_correct_sha(tmp_path: Path) -> None:
    make_catalog_library(tmp_path)

    report = CatalogLibrary.open(tmp_path).validate()

    assert report.status == CatalogStatus.READY_PARTIAL
    assert "INDEX_SHA256_MISMATCH" not in {issue.code for issue in report.issues}


def test_validation_reports_incorrect_sha_as_corrupt(tmp_path: Path) -> None:
    make_catalog_library(tmp_path, index_sha="0" * 64)

    report = CatalogLibrary.open(tmp_path).validate()

    assert report.status == CatalogStatus.CORRUPT
    assert "INDEX_SHA256_MISMATCH" in {issue.code for issue in report.issues}


def test_validation_reports_missing_source_path(tmp_path: Path) -> None:
    make_catalog_library(tmp_path, source_exists=False, include_index=False)

    assert "SOURCE_PATH_MISSING" in _issue_codes(tmp_path)


def test_validation_reports_missing_index_path(tmp_path: Path) -> None:
    make_catalog_library(tmp_path, index_exists=False)

    assert "INDEX_PATH_MISSING" in _issue_codes(tmp_path)


def test_validation_reports_unknown_index_source_without_disabling_blind4d(tmp_path: Path) -> None:
    make_catalog_library(tmp_path, include_source=False, include_index=True, source_ids=["missing-source"])

    report = CatalogLibrary.open(tmp_path).validate()

    assert report.status == CatalogStatus.BLIND4D_ONLY
    assert report.capabilities.blind4d is True
    assert "INDEX_SOURCE_UNKNOWN" in {issue.code for issue in report.issues}


def test_validation_reports_unsupported_source_family(tmp_path: Path) -> None:
    make_catalog_library(tmp_path)
    payload = json.loads((tmp_path / "catalog.json").read_text(encoding="utf-8"))
    payload["sources"][0]["family"] = "x99"
    (tmp_path / "catalog.json").write_text(json.dumps(payload), encoding="utf-8")

    assert "SOURCE_FAMILY_UNSUPPORTED" in _issue_codes(tmp_path)


def test_validation_reports_coverage_fraction_out_of_range(tmp_path: Path) -> None:
    make_catalog_library(tmp_path)
    payload = json.loads((tmp_path / "catalog.json").read_text(encoding="utf-8"))
    payload["derived_indexes"][0]["coverage"]["status"] = "FULL"
    payload["derived_indexes"][0]["coverage"]["all_sky"] = False
    (tmp_path / "catalog.json").write_text(json.dumps(payload), encoding="utf-8")

    assert "COVERAGE_INCONSISTENT" in _issue_codes(tmp_path)


def test_validation_report_lists_checked_components(tmp_path: Path) -> None:
    make_catalog_library(tmp_path)

    report = CatalogLibrary.open(tmp_path).validate()

    assert report.checked_sources == ("astap-d50",)
    assert report.checked_indexes == ("blind4d-d50-2823",)
