from __future__ import annotations

import json
from pathlib import Path

import pytest

from catalog_resource_helpers import strict_entry, write_fake_4d_index, write_strict_manifest
from zeblindsolver.index_manifest_4d import (
    IndexManifestIntegrityError,
    IndexManifestSchemaError,
    load_4d_index_manifest,
    load_4d_index_manifest_payload,
)


def test_payload_loader_matches_path_loader(tmp_path: Path) -> None:
    index = write_fake_4d_index(tmp_path / "d50_2823_S_q.npz", "d50_2823")
    manifest = write_strict_manifest(tmp_path / "manifest.json", [strict_entry("idx", index, "d50_2823")])
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    from_path = load_4d_index_manifest(manifest)
    from_payload = load_4d_index_manifest_payload(payload, manifest_path=manifest)

    assert from_payload.index_ids == from_path.index_ids
    assert from_payload.enabled_index_paths == from_path.enabled_index_paths
    assert from_payload.checksums == from_path.checksums
    assert from_payload.tile_keys == from_path.tile_keys


def test_payload_loader_uses_explicit_root_for_relative_paths(tmp_path: Path) -> None:
    root = tmp_path / "indexes"
    index = write_fake_4d_index(root / "d50_2823_S_q.npz", "d50_2823")
    payload = {
        "schema": "zeblind.astrometry_4d_index_manifest.v1",
        "manifest_version": 1,
        "indexes": [strict_entry("idx", index, "d50_2823")],
    }

    loaded = load_4d_index_manifest_payload(payload, index_root=root)

    assert loaded.enabled_index_paths == (index.resolve(),)


def test_payload_loader_rejects_relative_paths_without_root(tmp_path: Path) -> None:
    index = write_fake_4d_index(tmp_path / "d50_2823_S_q.npz", "d50_2823")
    payload = {
        "schema": "zeblind.astrometry_4d_index_manifest.v1",
        "manifest_version": 1,
        "indexes": [strict_entry("idx", index, "d50_2823")],
    }

    with pytest.raises(IndexManifestSchemaError, match="manifest_relative_path_without_root"):
        load_4d_index_manifest_payload(payload)


def test_payload_loader_preserves_duplicate_validation(tmp_path: Path) -> None:
    index = write_fake_4d_index(tmp_path / "d50_2823_S_q.npz", "d50_2823")
    entry = strict_entry("idx", index, "d50_2823")
    payload = {
        "schema": "zeblind.astrometry_4d_index_manifest.v1",
        "manifest_version": 1,
        "indexes": [entry, dict(entry)],
    }

    with pytest.raises(IndexManifestIntegrityError, match="manifest_duplicate_id"):
        load_4d_index_manifest_payload(payload, manifest_path=tmp_path / "manifest.json")
