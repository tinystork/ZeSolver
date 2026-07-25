from __future__ import annotations

import json
from pathlib import Path

import pytest

from catalog_resource_helpers import write_catalog_library, write_fake_4d_index
from zeblindsolver.index_manifest_4d import load_4d_index_manifest
from zesolver.catalog_library import (
    CatalogBlind4DManifestViewError,
    CatalogLibrary,
    CatalogStatus,
    build_blind4d_manifest_view,
)
from zesolver.catalog_library.blind4d_view import (
    BLIND4D_VIEW_CHECKSUM_MISMATCH,
    BLIND4D_VIEW_INDEX_MISSING,
    BLIND4D_VIEW_MATERIALIZATION_FAILED,
    BLIND4D_VIEW_NO_INDEXES,
    BLIND4D_VIEW_COVERAGE_INCONSISTENT,
    BLIND4D_VIEW_RUNTIME_ORDER_DUPLICATE,
    BLIND4D_VIEW_RUNTIME_ORDER_MISSING,
    BLIND4D_VIEW_TILE_DUPLICATE,
)
from zesolver.catalog_resources import Blind4DCatalogMode, resolve_blind4d_runtime, resolve_catalog_resources


def _payload(root: Path) -> dict[str, object]:
    return json.loads((root / "catalog.json").read_text(encoding="utf-8"))


def _write_payload(root: Path, payload: dict[str, object]) -> None:
    (root / "catalog.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _with_runtime_order(root: Path, order: list[str]) -> Path:
    payload = _payload(root)
    payload["runtime_order"] = {"blind4d": order}
    _write_payload(root, payload)
    return root


def _library_with_two_indexes(tmp_path: Path) -> tuple[Path, Path, Path]:
    idx_a = write_fake_4d_index(tmp_path / "d50_2823_S_q.npz", "d50_2823")
    idx_b = write_fake_4d_index(tmp_path / "d50_2822_S_q.npz", "d50_2822")
    root = write_catalog_library(tmp_path / "library", index_paths=[idx_a, idx_b])
    _with_runtime_order(root, ["blind4d-0", "blind4d-1"])
    return root, idx_a, idx_b


def _library_with_one_index(tmp_path: Path, *, all_sky_index: bool = False) -> tuple[Path, Path]:
    idx = write_fake_4d_index(tmp_path / "d50_2823_S_q.npz", "d50_2823")
    root = write_catalog_library(tmp_path / "library", index_paths=[idx], all_sky_index=all_sky_index)
    _with_runtime_order(root, ["blind4d-0"])
    return root, idx


def _set_blind4d_coverage(
    root: Path,
    *,
    status: str,
    all_sky: bool,
    tile_keys: list[str] | None = None,
    covered_tiles: int | None = None,
    total_tiles: int | None = None,
    fraction: float | None = None,
) -> None:
    payload = _payload(root)
    coverage = {
        "status": status,
        "all_sky": all_sky,
        "families": ["d50"],
        "tile_keys": ["d50_2823"] if tile_keys is None else tile_keys,
    }
    if covered_tiles is not None:
        coverage["covered_tiles"] = covered_tiles
    if total_tiles is not None:
        coverage["total_tiles"] = total_tiles
    if fraction is not None:
        coverage["fraction"] = fraction
    payload["derived_indexes"][0]["coverage"] = coverage
    payload["coverage"] = dict(coverage)
    payload["status"] = "READY_FULL" if status == "FULL" and all_sky else "READY_PARTIAL"
    _write_payload(root, payload)


def test_view_builds_strict_manifest_from_catalog_library(tmp_path: Path) -> None:
    root, idx_a, idx_b = _library_with_two_indexes(tmp_path)

    view = build_blind4d_manifest_view(CatalogLibrary.open(root))

    assert view.errors == ()
    assert view.payload["schema"] == "zeblind.astrometry_4d_index_manifest.v1"
    assert [entry["id"] for entry in view.entries] == ["blind4d-0", "blind4d-1"]
    assert [entry["path"] for entry in view.entries] == [str(idx_a.resolve()), str(idx_b.resolve())]
    assert [entry["tile_keys"] for entry in view.entries] == [["d50_2823"], ["d50_2822"]]
    assert view.coverage.status.value == "PARTIAL"
    assert view.coverage.covered_tiles == 2
    assert view.coverage.all_sky is False


def test_full_all_sky_view_is_valid_and_ready_full(tmp_path: Path) -> None:
    root, idx = _library_with_one_index(tmp_path, all_sky_index=True)
    _set_blind4d_coverage(root, status="FULL", all_sky=True, covered_tiles=1, total_tiles=1, fraction=1.0)

    library = CatalogLibrary.open(root)
    view = build_blind4d_manifest_view(library)

    assert view.errors == ()
    assert view.coverage.status.value == "FULL"
    assert view.coverage.all_sky is True
    assert view.coverage.covered_tiles == 1
    assert view.coverage.total_tiles == 1
    assert view.telemetry["validation"] == "valid"
    assert view.telemetry["entry_count"] == 1
    assert [entry["id"] for entry in view.entries] == ["blind4d-0"]
    assert [entry["path"] for entry in view.entries] == [str(idx.resolve())]
    assert library.validate().status is CatalogStatus.READY_FULL


def test_full_all_sky_library_view_runtime_is_available(tmp_path: Path) -> None:
    root, idx = _library_with_one_index(tmp_path, all_sky_index=True)
    _set_blind4d_coverage(root, status="FULL", all_sky=True, covered_tiles=1, total_tiles=1, fraction=1.0)
    resources = resolve_catalog_resources(catalog_library=root)

    runtime = resolve_blind4d_runtime(resources, mode="auto")

    assert runtime.available
    assert runtime.mode_effective is Blind4DCatalogMode.LIBRARY_VIEW
    assert runtime.error_code is None
    assert runtime.index_ids == ("blind4d-0",)
    assert runtime.index_paths == (idx.resolve(),)
    assert runtime.coverage is not None
    assert runtime.coverage.status.value == "FULL"
    assert runtime.coverage.all_sky is True


def test_materialized_view_loads_with_strict_loader(tmp_path: Path) -> None:
    root, idx_a, idx_b = _library_with_two_indexes(tmp_path)
    view = build_blind4d_manifest_view(root)
    out = tmp_path / "strict-view.json"

    written = view.materialize(out)
    loaded = load_4d_index_manifest(written)

    assert loaded.enabled_index_paths == (idx_a.resolve(), idx_b.resolve())
    assert loaded.index_ids == ("blind4d-0", "blind4d-1")
    assert loaded.tile_keys == ("d50_2823", "d50_2822")


def test_materialization_refuses_overwrite_without_explicit_option(tmp_path: Path) -> None:
    root, _, _ = _library_with_two_indexes(tmp_path)
    view = build_blind4d_manifest_view(root)
    out = tmp_path / "strict-view.json"
    out.write_text("{}", encoding="utf-8")

    with pytest.raises(CatalogBlind4DManifestViewError) as exc:
        view.materialize(out)

    assert exc.value.code == BLIND4D_VIEW_MATERIALIZATION_FAILED
    assert out.read_text(encoding="utf-8") == "{}"


def test_view_without_blind_indexes_reports_stable_error(tmp_path: Path) -> None:
    root = write_catalog_library(tmp_path / "library", index_paths=[])

    view = build_blind4d_manifest_view(root)

    assert [issue.code for issue in view.errors] == [BLIND4D_VIEW_NO_INDEXES]
    assert view.payload["indexes"] == []


def test_view_rejects_full_status_without_all_sky(tmp_path: Path) -> None:
    root, _ = _library_with_one_index(tmp_path)
    _set_blind4d_coverage(root, status="FULL", all_sky=False, covered_tiles=1, total_tiles=1, fraction=1.0)

    view = build_blind4d_manifest_view(root)

    assert BLIND4D_VIEW_COVERAGE_INCONSISTENT in {issue.code for issue in view.errors}
    assert "full_status_requires_all_sky" in ";".join(issue.message for issue in view.errors)
    assert view.telemetry["validation"] == "invalid"


def test_view_rejects_partial_status_with_all_sky(tmp_path: Path) -> None:
    root, _ = _library_with_one_index(tmp_path)
    _set_blind4d_coverage(root, status="PARTIAL", all_sky=True, covered_tiles=1, total_tiles=2, fraction=0.5)

    view = build_blind4d_manifest_view(root)

    assert BLIND4D_VIEW_COVERAGE_INCONSISTENT in {issue.code for issue in view.errors}
    assert "all_sky_requires_full_status" in ";".join(issue.message for issue in view.errors)


def test_view_rejects_declared_full_with_partial_tile_count(tmp_path: Path) -> None:
    root, _ = _library_with_one_index(tmp_path, all_sky_index=True)
    _set_blind4d_coverage(root, status="FULL", all_sky=True, covered_tiles=1, total_tiles=1476, fraction=1 / 1476)

    view = build_blind4d_manifest_view(root)

    assert BLIND4D_VIEW_COVERAGE_INCONSISTENT in {issue.code for issue in view.errors}
    assert "full_status_requires_complete_tile_count" in ";".join(issue.message for issue in view.errors)


def test_view_rejects_partial_coverage_without_tiles(tmp_path: Path) -> None:
    root, _ = _library_with_one_index(tmp_path)
    _set_blind4d_coverage(root, status="PARTIAL", all_sky=False, tile_keys=[], covered_tiles=0, total_tiles=1476, fraction=0.0)

    view = build_blind4d_manifest_view(root)

    assert BLIND4D_VIEW_COVERAGE_INCONSISTENT in {issue.code for issue in view.errors}
    assert "non_empty_status_requires_covered_tiles" in ";".join(issue.message for issue in view.errors)


def test_view_requires_explicit_runtime_order_for_blind4d(tmp_path: Path) -> None:
    idx = write_fake_4d_index(tmp_path / "d50_2823_S_q.npz", "d50_2823")
    root = write_catalog_library(tmp_path / "library", index_paths=[idx])

    view = build_blind4d_manifest_view(root)

    assert [issue.code for issue in view.errors] == [BLIND4D_VIEW_RUNTIME_ORDER_MISSING]


def test_view_rejects_duplicate_runtime_order(tmp_path: Path) -> None:
    root, _, _ = _library_with_two_indexes(tmp_path)
    _with_runtime_order(root, ["blind4d-0", "blind4d-0"])

    view = build_blind4d_manifest_view(root)

    assert BLIND4D_VIEW_RUNTIME_ORDER_DUPLICATE in {issue.code for issue in view.errors}


def test_view_rejects_order_referencing_missing_index(tmp_path: Path) -> None:
    root, _, _ = _library_with_two_indexes(tmp_path)
    _with_runtime_order(root, ["blind4d-0", "blind4d-1", "blind4d-404"])

    view = build_blind4d_manifest_view(root)

    assert BLIND4D_VIEW_INDEX_MISSING in {issue.code for issue in view.errors}


def test_view_rejects_order_missing_declared_index(tmp_path: Path) -> None:
    root, _, _ = _library_with_two_indexes(tmp_path)
    _with_runtime_order(root, ["blind4d-0"])

    view = build_blind4d_manifest_view(root)

    assert BLIND4D_VIEW_RUNTIME_ORDER_MISSING in {issue.code for issue in view.errors}


def test_view_rejects_checksum_mismatch(tmp_path: Path) -> None:
    idx = write_fake_4d_index(tmp_path / "d50_2823_S_q.npz", "d50_2823")
    root = write_catalog_library(tmp_path / "library", index_paths=[idx], bad_index_sha=True)
    _with_runtime_order(root, ["blind4d-0"])

    view = build_blind4d_manifest_view(root)

    assert [issue.code for issue in view.errors] == [BLIND4D_VIEW_CHECKSUM_MISMATCH]


def test_view_rejects_duplicate_tiles(tmp_path: Path) -> None:
    idx_a = write_fake_4d_index(tmp_path / "d50_A_S_q1.npz", "d50_A")
    idx_b = write_fake_4d_index(tmp_path / "d50_A_S_q2.npz", "d50_A")
    root = write_catalog_library(tmp_path / "library", index_paths=[idx_a, idx_b])
    _with_runtime_order(root, ["blind4d-0", "blind4d-1"])

    view = build_blind4d_manifest_view(root)

    assert BLIND4D_VIEW_TILE_DUPLICATE in {issue.code for issue in view.errors}


def test_view_fingerprint_is_deterministic_and_sensitive_to_runtime_fields(tmp_path: Path) -> None:
    root, _, _ = _library_with_two_indexes(tmp_path)
    first = build_blind4d_manifest_view(root).fingerprint
    second = build_blind4d_manifest_view(root).fingerprint
    assert first == second

    payload = _payload(root)
    payload["runtime_order"] = {"blind4d": ["blind4d-1", "blind4d-0"]}
    _write_payload(root, payload)
    reordered = build_blind4d_manifest_view(root).fingerprint
    assert reordered != first

    payload = _payload(root)
    payload["runtime_order"] = {"blind4d": ["blind4d-0", "blind4d-1"]}
    payload["derived_indexes"][0]["parameters"] = {"code_tol_recommended": 0.02}
    _write_payload(root, payload)
    changed_tol = build_blind4d_manifest_view(root).fingerprint
    assert changed_tol != first
    assert changed_tol != reordered
