from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from zeblindsolver.index_manifest_4d import sha256_file
from zeblindsolver.quad_index_4d import ASTROMETRY_AB_CODE_4D_SCHEMA


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def metadata_array(metadata: dict[str, object]) -> np.ndarray:
    text = json.dumps(metadata, sort_keys=True)
    return np.asarray([text], dtype=f"<U{len(text)}")


def write_fake_4d_index(path: Path, tile_key: str) -> Path:
    metadata = {
        "schema": ASTROMETRY_AB_CODE_4D_SCHEMA,
        "version": 1,
        "level": "S",
        "sampler_tag": "catalog_ring_coverage",
        "code_tol_recommended": 0.015,
        "source_catalog": "unit-test",
        "generated_at": "2026-07-17T00:00:00Z",
        "max_stars_per_tile": 4,
        "max_quads_per_tile": 1,
        "entry_count": 1,
        "star_count": 4,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        codes_4d=np.asarray([[0.2, 0.3, 0.4, 0.5]], dtype=np.float32),
        quad_star_indices=np.asarray([[0, 1, 2, 3]], dtype=np.int32),
        source_quad_indices=np.asarray([0], dtype=np.int32),
        tile_key_indices=np.asarray([0], dtype=np.int32),
        ratio_hashes=np.asarray([-1], dtype=np.int64),
        tile_keys=np.asarray([tile_key], dtype=f"<U{len(tile_key)}"),
        catalog_ra_dec=np.asarray([[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3]], dtype=np.float64),
        catalog_xy=np.asarray([[0.0, 0.0], [1.0, 1.0], [0.2, 0.3], [0.4, 0.5]], dtype=np.float64),
        metadata=metadata_array(metadata),
    )
    return path


def strict_entry(index_id: str, path: Path, tile_key: str, *, enabled: bool = True) -> dict[str, object]:
    return {
        "id": index_id,
        "enabled": enabled,
        "path": path.name,
        "filename": path.name,
        "quad_schema": ASTROMETRY_AB_CODE_4D_SCHEMA,
        "index_version": 1,
        "level": "S",
        "tile_keys": [tile_key],
        "star_count": 4,
        "quad_count": 1,
        "sampler_tag": "catalog_ring_coverage",
        "code_tol_recommended": 0.015,
        "catalog_source": "unit-test",
        "sha256": sha256_file(path) if path.exists() else "0" * 64,
    }


def write_strict_manifest(path: Path, entries: list[dict[str, object]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "zeblind.astrometry_4d_index_manifest.v1",
                "manifest_version": 1,
                "indexes": entries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def write_catalog_library(
    root: Path,
    *,
    include_source: bool = True,
    source_status: str = "PRESENT",
    index_paths: list[Path] | None = None,
    strict_manifest_path: Path | None = None,
    bad_index_sha: bool = False,
    all_sky_index: bool = False,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    sources: list[dict[str, object]] = []
    if include_source:
        source_dir = root / "sources" / "astap" / "d50"
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / "d50_2823.1476").write_bytes(b"fixture-tile")
        sources.append(
            {
                "id": "astap-d50",
                "kind": "astap_hnsky",
                "family": "d50",
                "format": "1476-5",
                "path": {"kind": "relative", "value": "sources/astap/d50"},
                "tile_count": 1,
                "layout": "hnsky_1476",
                "coverage": {
                    "status": "FULL",
                    "all_sky": True,
                    "families": ["d50"],
                    "tile_keys": [],
                    "dec_min_deg": -90.0,
                    "dec_max_deg": 90.0,
                    "ra_segments_deg": [[0.0, 360.0]],
                },
                "integrity": {"files": []},
                "status": source_status,
            }
        )
    indexes: list[dict[str, object]] = []
    for i, index_path in enumerate(index_paths or []):
        tile_key = f"d50_TEST{i}"
        if index_path.name.startswith("d50_"):
            tile_key = index_path.stem.split("_S_", 1)[0]
        indexes.append(
            {
                "id": f"blind4d-{i}",
                "engine": "blind4d",
                "schema": "zeblind.astrometry_4d_index_manifest.v1",
                "algorithm_version": "astrometry_ab_code_4d_v1",
                "path": {"kind": "external_reference", "value": str(index_path.resolve())},
                "manifest_path": (
                    {"kind": "external_reference", "value": str(strict_manifest_path.resolve())}
                    if strict_manifest_path is not None
                    else None
                ),
                "source_ids": ["astap-d50"] if include_source else [],
                "source_tiles": [tile_key],
                "coverage": {
                    "status": "FULL" if all_sky_index else "PARTIAL",
                    "all_sky": bool(all_sky_index),
                    "families": ["d50"],
                    "tile_keys": [tile_key],
                    "dec_min_deg": 46.28571429,
                    "dec_max_deg": 51.42857143,
                    "ra_segments_deg": [],
                },
                "integrity": {
                    "files": [
                        {
                            "path": str(index_path.resolve()),
                            "sha256": ("0" * 64 if bad_index_sha else sha256_file(index_path)),
                            "size_bytes": index_path.stat().st_size,
                        }
                    ]
                },
                "status": "PRESENT",
            }
        )
    manifest = {
        "schema_version": 1,
        "library_id": "test-library",
        "created_at": "2026-07-17T00:00:00Z",
        "created_by": "tests",
        "minimum_zesolver_version": None,
        "status": "READY_PARTIAL",
        "sources": sources,
        "derived_indexes": indexes,
        "coverage": {
            "status": "PARTIAL",
            "all_sky": False,
            "families": ["d50"],
            "tile_keys": [item["source_tiles"][0] for item in indexes],
            "dec_min_deg": 46.28571429 if indexes else None,
            "dec_max_deg": 51.42857143 if indexes else None,
            "ra_segments_deg": [],
        },
        "integrity": {"checksum_algorithm": "sha256"},
        "provenance": {"notes": "test fixture"},
    }
    (root / "catalog.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root
