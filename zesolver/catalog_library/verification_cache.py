"""Lightweight CatalogLibrary verification cache.

The FAST path deliberately avoids hashing large payloads. It fingerprints the
library manifest, referenced manifests, runtime order, declared coverage, and
file identity metadata (path, size, mtime) only.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .manifest import CatalogLibrary
from .models import CatalogStatus

VERIFICATION_SCHEMA_VERSION = 1
APPLICATION_COMPATIBILITY_VERSION = "s5e-v1"
FULL_LEVEL = "FULL"
FAST_LEVEL = "FAST"
STATUS_VALID = "valid"
STATUS_INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class CatalogVerificationFingerprint:
    canonical_library_path: str
    library_id: str
    catalog_manifest_fingerprint: str
    blind4d_view_fingerprint: str
    runtime_order: tuple[str, ...]
    blind4d_index_count: int
    covered_tiles: int
    total_tiles: int | None
    all_sky: bool
    fingerprint: str
    inspected_file_count: int
    payload_hash_count: int = 0
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CachedVerificationState:
    status: str
    message: str
    fingerprint: CatalogVerificationFingerprint | None
    record: dict[str, Any] | None = None
    cache_reused: bool = False
    invalidation_reason: str | None = None
    duration_s: float = 0.0


def catalog_verification_record(
    *,
    fingerprint: CatalogVerificationFingerprint,
    verification_level: str,
    verification_status: str = STATUS_VALID,
    verified_at: float | None = None,
) -> dict[str, Any]:
    return {
        "canonical_library_path": fingerprint.canonical_library_path,
        "library_id": fingerprint.library_id,
        "catalog_manifest_fingerprint": fingerprint.catalog_manifest_fingerprint,
        "blind4d_view_fingerprint": fingerprint.blind4d_view_fingerprint,
        "verification_level": str(verification_level or FAST_LEVEL).upper(),
        "verification_status": verification_status,
        "verified_at": float(time.time() if verified_at is None else verified_at),
        "verification_schema_version": VERIFICATION_SCHEMA_VERSION,
        "application_compatibility_version": APPLICATION_COMPATIBILITY_VERSION,
        "lightweight_fingerprint": fingerprint.fingerprint,
        "runtime_order": list(fingerprint.runtime_order),
        "blind4d_index_count": fingerprint.blind4d_index_count,
        "blind4d_covered_tiles": fingerprint.covered_tiles,
        "blind4d_total_tiles": fingerprint.total_tiles,
        "blind4d_all_sky": fingerprint.all_sky,
        "payload_hash_count": fingerprint.payload_hash_count,
        "inspected_file_count": fingerprint.inspected_file_count,
    }


def restore_cached_catalog_verification(path: str | Path, record: object) -> CachedVerificationState:
    start = time.perf_counter()
    if not isinstance(record, dict):
        return CachedVerificationState("missing", "no_cached_verification", None, duration_s=time.perf_counter() - start)
    try:
        fingerprint = build_lightweight_catalog_fingerprint(path)
    except Exception as exc:
        return CachedVerificationState(
            STATUS_INVALID,
            "Bibliothèque invalide",
            None,
            record=record,
            invalidation_reason=str(exc),
            duration_s=time.perf_counter() - start,
        )
    reason = _cache_mismatch_reason(record, fingerprint)
    if reason:
        return CachedVerificationState(
            STATUS_INVALID,
            "Modification détectée — nouvelle vérification requise",
            fingerprint,
            record=record,
            invalidation_reason=reason,
            duration_s=time.perf_counter() - start,
        )
    level = str(record.get("verification_level") or FAST_LEVEL).upper()
    if level == FULL_LEVEL:
        message = "Vérifiée — cache valide"
    else:
        message = "Vérification rapide réussie"
    return CachedVerificationState(
        STATUS_VALID,
        message,
        fingerprint,
        record=record,
        cache_reused=True,
        duration_s=time.perf_counter() - start,
    )


def build_lightweight_catalog_fingerprint(path: str | Path) -> CatalogVerificationFingerprint:
    library = CatalogLibrary.open(path)
    root = library.root.resolve()
    manifest_path = library.manifest.manifest_path.resolve()
    catalog_manifest_fingerprint = _sha256_file(manifest_path)
    runtime_order = tuple(str(item) for item in library.manifest.runtime_order.get("blind4d", ()))
    by_id = {index.id: index for index in library.manifest.derived_indexes if index.engine == "blind4d" and index.category != "compatibility"}
    ordered_indexes = tuple(by_id[index_id] for index_id in runtime_order if index_id in by_id)
    coverage = _merged_blind4d_coverage(ordered_indexes)
    file_records: list[dict[str, Any]] = [_file_identity(manifest_path, label="catalog.json", required=True)]
    index_records: list[dict[str, Any]] = []
    for index_id in runtime_order:
        index = by_id.get(index_id)
        if index is None:
            index_records.append({"id": index_id, "missing_from_manifest": True})
            continue
        manifest_identity = None
        if index.manifest_path is not None:
            manifest_identity = _file_identity(index.manifest_path.resolved, label=f"{index.id}:manifest", required=True)
            file_records.append(manifest_identity)
        path_identity = _file_identity(index.path.resolved, label=index.id, required=True)
        file_records.append(path_identity)
        integrity = [
            {
                "path": item.path,
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
                "mtime_ns": item.mtime_ns,
            }
            for item in (*index.integrity_files, *index.derived_files)
        ]
        index_records.append(
            {
                "id": index.id,
                "engine": index.engine,
                "schema": index.schema,
                "status": index.status.value,
                "category": index.category,
                "path_ref": {"kind": index.path.kind.value, "value": index.path.value},
                "manifest_ref": (
                    {"kind": index.manifest_path.kind.value, "value": index.manifest_path.value}
                    if index.manifest_path is not None
                    else None
                ),
                "path_identity": path_identity,
                "manifest_identity": manifest_identity,
                "source_tiles": list(index.source_tiles),
                "coverage": _coverage_payload(index.coverage),
                "integrity": integrity,
                "provenance_fingerprint": index.provenance_fingerprint,
                "reconstruction_status": index.reconstruction_status,
            }
        )
    payload = {
        "verification_schema_version": VERIFICATION_SCHEMA_VERSION,
        "application_compatibility_version": APPLICATION_COMPATIBILITY_VERSION,
        "canonical_library_path": str(root),
        "library_id": library.manifest.library_id,
        "catalog_manifest_fingerprint": catalog_manifest_fingerprint,
        "runtime_order": list(runtime_order),
        "indexes": index_records,
        "coverage": _coverage_payload(coverage),
    }
    blind4d_view_fingerprint = hashlib.sha256(_canonical_json({"runtime_order": list(runtime_order), "indexes": index_records, "coverage": _coverage_payload(coverage)})).hexdigest()
    payload["blind4d_view_fingerprint"] = blind4d_view_fingerprint
    fingerprint = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return CatalogVerificationFingerprint(
        canonical_library_path=str(root),
        library_id=library.manifest.library_id,
        catalog_manifest_fingerprint=catalog_manifest_fingerprint,
        blind4d_view_fingerprint=blind4d_view_fingerprint,
        runtime_order=runtime_order,
        blind4d_index_count=len(ordered_indexes),
        covered_tiles=int(coverage.get("covered_tiles") or 0),
        total_tiles=coverage.get("total_tiles"),
        all_sky=bool(coverage.get("all_sky")),
        fingerprint=fingerprint,
        inspected_file_count=len(file_records),
        payload_hash_count=0,
        payload=payload,
    )


def _cache_mismatch_reason(record: dict[str, Any], fingerprint: CatalogVerificationFingerprint) -> str | None:
    checks = (
        ("verification_schema_version", VERIFICATION_SCHEMA_VERSION),
        ("application_compatibility_version", APPLICATION_COMPATIBILITY_VERSION),
        ("canonical_library_path", fingerprint.canonical_library_path),
        ("library_id", fingerprint.library_id),
        ("catalog_manifest_fingerprint", fingerprint.catalog_manifest_fingerprint),
        ("blind4d_view_fingerprint", fingerprint.blind4d_view_fingerprint),
        ("lightweight_fingerprint", fingerprint.fingerprint),
    )
    for key, expected in checks:
        if record.get(key) != expected:
            return key
    if str(record.get("verification_status") or "") != STATUS_VALID:
        return "verification_status"
    return None


def _merged_blind4d_coverage(indexes: tuple[object, ...]) -> dict[str, Any]:
    tile_keys: list[str] = []
    families: set[str] = set()
    total_tiles = None
    all_sky = bool(indexes)
    for index in indexes:
        cov = index.coverage
        tile_keys.extend(str(tile) for tile in cov.tile_keys)
        families.update(str(family) for family in cov.families)
        if cov.total_tiles is not None:
            total_tiles = cov.total_tiles
        all_sky = all_sky and bool(cov.status.value == "FULL" and cov.all_sky)
    unique_tiles = tuple(sorted(set(tile_keys)))
    if total_tiles is None:
        total_tiles = len(unique_tiles) or None
    all_sky = all_sky or (bool(total_tiles) and len(unique_tiles) == total_tiles)
    status = "FULL" if all_sky else ("PARTIAL" if unique_tiles else "MISSING")
    return {
        "status": status,
        "all_sky": all_sky,
        "families": sorted(families),
        "tile_keys": list(unique_tiles),
        "covered_tiles": len(unique_tiles),
        "total_tiles": total_tiles,
        "fraction": (len(unique_tiles) / total_tiles) if total_tiles else None,
    }


def _coverage_payload(coverage: object) -> dict[str, Any]:
    if isinstance(coverage, dict):
        return dict(coverage)
    return {
        "status": coverage.status.value,
        "all_sky": coverage.all_sky,
        "families": list(coverage.families),
        "tile_keys": list(coverage.tile_keys),
        "covered_tiles": coverage.covered_tiles,
        "total_tiles": coverage.total_tiles,
        "fraction": coverage.fraction,
    }


def _file_identity(path: Path, *, label: str, required: bool) -> dict[str, Any]:
    exists = path.exists()
    payload: dict[str, Any] = {
        "label": label,
        "path": str(path),
        "exists": exists,
        "required": required,
    }
    if exists:
        stat = path.stat()
        payload.update({"size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    return payload


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


__all__ = [
    "APPLICATION_COMPATIBILITY_VERSION",
    "FAST_LEVEL",
    "FULL_LEVEL",
    "STATUS_INVALID",
    "STATUS_VALID",
    "VERIFICATION_SCHEMA_VERSION",
    "CachedVerificationState",
    "CatalogVerificationFingerprint",
    "build_lightweight_catalog_fingerprint",
    "catalog_verification_record",
    "restore_cached_catalog_verification",
]
