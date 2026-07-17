from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_catalog_library(
    root: Path,
    *,
    include_source: bool = True,
    include_index: bool = True,
    source_exists: bool = True,
    index_exists: bool = True,
    index_sha: str | None = None,
    index_engine: str = "blind4d",
    source_status: str = "PRESENT",
    index_status: str = "PRESENT",
    all_sky_index: bool = False,
    source_ids: list[str] | None = None,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    sources: list[dict[str, Any]] = []
    indexes: list[dict[str, Any]] = []
    if include_source:
        source_dir = root / "sources" / "astap" / "d50"
        if source_exists:
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
    if include_index:
        index_dir = root / "indexes" / "blind4d"
        index_path = index_dir / "d50_2823.npz"
        data = b"fixture-index"
        if index_exists:
            index_dir.mkdir(parents=True, exist_ok=True)
            index_path.write_bytes(data)
        expected_sha = index_sha if index_sha is not None else sha256_bytes(data)
        index_all_sky = bool(all_sky_index)
        indexes.append(
            {
                "id": "blind4d-d50-2823",
                "engine": index_engine,
                "schema": "zeblind.astrometry_4d_index_manifest.v1",
                "algorithm_version": "astrometry_ab_code_4d_v1",
                "path": {"kind": "relative", "value": "indexes/blind4d/d50_2823.npz"},
                "manifest_path": None,
                "source_ids": source_ids if source_ids is not None else (["astap-d50"] if include_source else []),
                "source_tiles": ["d50_2823"],
                "coverage": {
                    "status": "FULL" if index_all_sky else "PARTIAL",
                    "all_sky": index_all_sky,
                    "families": ["d50"],
                    "tile_keys": ["d50_2823"],
                    "dec_min_deg": 46.28571429,
                    "dec_max_deg": 51.42857143,
                    "ra_segments_deg": [],
                },
                "integrity": {
                    "files": [
                        {
                            "path": "indexes/blind4d/d50_2823.npz",
                            "sha256": expected_sha,
                            "size_bytes": len(data) if index_exists else None,
                        }
                    ]
                },
                "status": index_status,
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
            "tile_keys": ["d50_2823"] if include_index else [],
            "dec_min_deg": 46.28571429 if include_index else None,
            "dec_max_deg": 51.42857143 if include_index else None,
            "ra_segments_deg": [],
        },
        "integrity": {"checksum_algorithm": "sha256"},
        "provenance": {"notes": "test fixture"},
    }
    (root / "catalog.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
