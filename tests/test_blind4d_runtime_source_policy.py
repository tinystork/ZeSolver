from __future__ import annotations

import json
from pathlib import Path

import pytest

from catalog_resource_helpers import strict_entry, write_catalog_library, write_fake_4d_index, write_strict_manifest
from zesolver.catalog_resources import (
    BLIND4D_EXTERNAL_MANIFEST_REQUIRED,
    BLIND4D_LIBRARY_NO_INDEXES,
    BLIND4D_LIBRARY_REQUIRED,
    BLIND4D_RUNTIME_ORDER_INVALID,
    Blind4DCatalogMode,
    Blind4DRuntimeError,
    resolve_blind4d_runtime,
    resolve_catalog_resources,
)


def _with_runtime_order(root: Path, order: list[str]) -> Path:
    path = root / "catalog.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["runtime_order"] = {"blind4d": order}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return root


def _library_with_blind(tmp_path: Path, *, manifest_path: Path | None = None) -> tuple[Path, Path]:
    index = write_fake_4d_index(tmp_path / "d50_2823_S_q.npz", "d50_2823")
    root = write_catalog_library(tmp_path / "library", index_paths=[index], strict_manifest_path=manifest_path)
    _with_runtime_order(root, ["blind4d-0"])
    return root, index


def test_auto_uses_library_view_and_ignores_stale_external_manifest(tmp_path: Path) -> None:
    root, index = _library_with_blind(tmp_path)
    resources = resolve_catalog_resources(catalog_library=root)
    stale = tmp_path / "missing-manifest.json"

    runtime = resolve_blind4d_runtime(resources, mode="auto", external_manifest_path=stale)

    assert runtime.available
    assert runtime.mode_effective is Blind4DCatalogMode.LIBRARY_VIEW
    assert runtime.source == "catalog_library_view"
    assert runtime.index_paths == (index.resolve(),)
    assert runtime.runtime_order == ("blind4d-0",)
    assert runtime.telemetry()["blind4d_external_fallback_used"] is False


def test_auto_uses_external_manifest_without_library(tmp_path: Path) -> None:
    index = write_fake_4d_index(tmp_path / "d50_2823_S_q.npz", "d50_2823")
    manifest = write_strict_manifest(tmp_path / "manifest.json", [strict_entry("idx", index, "d50_2823")])
    resources = resolve_catalog_resources(legacy_blind4d_manifest=manifest)

    runtime = resolve_blind4d_runtime(resources, mode="auto")

    assert runtime.available
    assert runtime.mode_effective is Blind4DCatalogMode.EXTERNAL_MANIFEST
    assert runtime.source == "external_manifest"
    assert runtime.index_ids == ("idx",)


def test_auto_library_near_only_does_not_fall_back_to_external(tmp_path: Path) -> None:
    root = write_catalog_library(tmp_path / "library", include_source=True, index_paths=[])
    index = write_fake_4d_index(tmp_path / "d50_2823_S_q.npz", "d50_2823")
    external = write_strict_manifest(tmp_path / "manifest.json", [strict_entry("idx", index, "d50_2823")])
    resources = resolve_catalog_resources(catalog_library=root, legacy_blind4d_manifest=external)

    runtime = resolve_blind4d_runtime(resources, mode="auto", external_manifest_path=external)

    assert not runtime.available
    assert runtime.source == "unavailable"
    assert runtime.error_code == BLIND4D_LIBRARY_NO_INDEXES
    assert runtime.external_manifest_path is None


def test_auto_library_bad_runtime_order_is_explicit_error_without_external_fallback(tmp_path: Path) -> None:
    index = write_fake_4d_index(tmp_path / "d50_2823_S_q.npz", "d50_2823")
    root = write_catalog_library(tmp_path / "library", index_paths=[index])
    external = write_strict_manifest(tmp_path / "manifest.json", [strict_entry("idx", index, "d50_2823")])
    resources = resolve_catalog_resources(catalog_library=root, legacy_blind4d_manifest=external)

    runtime = resolve_blind4d_runtime(resources, mode="auto", external_manifest_path=external)

    assert not runtime.available
    assert runtime.error_code == BLIND4D_RUNTIME_ORDER_INVALID
    assert runtime.source == "unavailable"


def test_forced_library_requires_library(tmp_path: Path) -> None:
    index = write_fake_4d_index(tmp_path / "d50_2823_S_q.npz", "d50_2823")
    manifest = write_strict_manifest(tmp_path / "manifest.json", [strict_entry("idx", index, "d50_2823")])
    resources = resolve_catalog_resources(legacy_blind4d_manifest=manifest)

    with pytest.raises(Blind4DRuntimeError) as exc:
        resolve_blind4d_runtime(resources, mode="library-view")

    assert exc.value.code == BLIND4D_LIBRARY_REQUIRED


def test_forced_external_requires_explicit_manifest(tmp_path: Path) -> None:
    resources = resolve_catalog_resources(enable_environment_discovery=False)

    with pytest.raises(Blind4DRuntimeError) as exc:
        resolve_blind4d_runtime(resources, mode="external-manifest")

    assert exc.value.code == BLIND4D_EXTERNAL_MANIFEST_REQUIRED
