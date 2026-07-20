from __future__ import annotations

import json
from pathlib import Path

import pytest

from catalog_resource_helpers import write_catalog_library, write_fake_4d_index
from zeblindsolver.index_manifest_4d import load_4d_index_manifest
from zesolver.catalog_library import (
    CatalogBlind4DManifestViewError,
    CatalogLibrary,
    build_blind4d_manifest_view,
)
from zesolver.catalog_library.blind4d_view import (
    BLIND4D_VIEW_CHECKSUM_MISMATCH,
    BLIND4D_VIEW_INDEX_MISSING,
    BLIND4D_VIEW_MATERIALIZATION_FAILED,
    BLIND4D_VIEW_NO_INDEXES,
    BLIND4D_VIEW_RUNTIME_ORDER_DUPLICATE,
    BLIND4D_VIEW_RUNTIME_ORDER_MISSING,
    BLIND4D_VIEW_TILE_DUPLICATE,
)


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
