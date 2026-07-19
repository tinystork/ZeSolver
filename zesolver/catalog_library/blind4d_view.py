"""CatalogLibrary-owned strict Blind 4D manifest view."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from zeblindsolver.index_manifest_4d import (
    MANIFEST_SCHEMA,
    MANIFEST_VERSION,
    load_4d_index_manifest,
    load_4d_index_manifest_payload,
    sha256_file,
)
from zeblindsolver.quad_index_4d import ASTROMETRY_AB_CODE_4D_SCHEMA, Quad4DIndex

from .coverage import merge_coverages
from .manifest import CatalogLibrary
from .models import CatalogCoverage, CatalogDataStatus, CatalogIndex, CatalogIssue, CoverageStatus, IssueSeverity

BLIND4D_VIEW_NO_INDEXES = "BLIND4D_VIEW_NO_INDEXES"
BLIND4D_VIEW_INDEX_MISSING = "BLIND4D_VIEW_INDEX_MISSING"
BLIND4D_VIEW_INDEX_CORRUPT = "BLIND4D_VIEW_INDEX_CORRUPT"
BLIND4D_VIEW_CHECKSUM_MISMATCH = "BLIND4D_VIEW_CHECKSUM_MISMATCH"
BLIND4D_VIEW_SCHEMA_UNSUPPORTED = "BLIND4D_VIEW_SCHEMA_UNSUPPORTED"
BLIND4D_VIEW_RUNTIME_ORDER_MISSING = "BLIND4D_VIEW_RUNTIME_ORDER_MISSING"
BLIND4D_VIEW_RUNTIME_ORDER_DUPLICATE = "BLIND4D_VIEW_RUNTIME_ORDER_DUPLICATE"
BLIND4D_VIEW_TILE_DUPLICATE = "BLIND4D_VIEW_TILE_DUPLICATE"
BLIND4D_VIEW_COVERAGE_INCONSISTENT = "BLIND4D_VIEW_COVERAGE_INCONSISTENT"
BLIND4D_VIEW_PATH_INVALID = "BLIND4D_VIEW_PATH_INVALID"
BLIND4D_VIEW_MATERIALIZATION_FAILED = "BLIND4D_VIEW_MATERIALIZATION_FAILED"

_VIEW_FINGERPRINT_EXCLUDE_KEYS = frozenset({"description", "generated_at", "resolved_path", "actual_sha256"})


class CatalogBlind4DManifestViewError(RuntimeError):
    """Stable failure raised by explicit Blind 4D view materialization."""

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        super().__init__(message or code)


@dataclass(frozen=True, slots=True)
class CatalogBlind4DManifestView:
    payload: dict[str, Any]
    schema: str
    version: int
    entries: tuple[dict[str, Any], ...]
    coverage: CatalogCoverage
    fingerprint: str
    warnings: tuple[CatalogIssue, ...]
    errors: tuple[CatalogIssue, ...]
    source_library_id: str
    source_manifest_fingerprint: str | None
    telemetry: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_library(cls, library: CatalogLibrary | str | Path) -> "CatalogBlind4DManifestView":
        catalog = library if isinstance(library, CatalogLibrary) else CatalogLibrary.open(library)
        return _build_view(catalog)

    @property
    def valid(self) -> bool:
        return not self.errors

    def materialize(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Atomically write the strict view to a caller-provided path."""

        if self.errors:
            raise CatalogBlind4DManifestViewError(
                BLIND4D_VIEW_MATERIALIZATION_FAILED,
                ", ".join(issue.code for issue in self.errors),
            )
        target = Path(path).expanduser()
        parent = target.parent
        if not parent.exists() or not parent.is_dir():
            raise CatalogBlind4DManifestViewError(BLIND4D_VIEW_PATH_INVALID, f"parent directory missing: {parent}")
        if target.exists() and not overwrite:
            raise CatalogBlind4DManifestViewError(BLIND4D_VIEW_MATERIALIZATION_FAILED, f"target exists: {target}")
        if target.is_symlink():
            raise CatalogBlind4DManifestViewError(BLIND4D_VIEW_PATH_INVALID, f"target is symlink: {target}")
        data = _serialize_payload(self.payload)
        tmp = parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
        try:
            with tmp.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            load_4d_index_manifest_payload(self.payload, manifest_path=tmp)
            os.replace(tmp, target)
            _fsync_dir(parent)
            load_4d_index_manifest(target)
        except CatalogBlind4DManifestViewError:
            raise
        except Exception as exc:
            try:
                if tmp.exists() and not tmp.is_symlink():
                    tmp.unlink()
            finally:
                pass
            raise CatalogBlind4DManifestViewError(BLIND4D_VIEW_MATERIALIZATION_FAILED, str(exc)) from exc
        return target.resolve()


def build_blind4d_manifest_view(library: CatalogLibrary | str | Path) -> CatalogBlind4DManifestView:
    return CatalogBlind4DManifestView.from_library(library)


def _build_view(library: CatalogLibrary) -> CatalogBlind4DManifestView:
    warnings: list[CatalogIssue] = []
    errors: list[CatalogIssue] = []
    indexes = tuple(
        index
        for index in library.manifest.derived_indexes
        if index.engine == "blind4d" and index.category != "compatibility"
    )
    ordered_indexes = _ordered_indexes(library, indexes, errors)
    entries: list[dict[str, Any]] = []
    seen_tiles: set[str] = set()
    for priority, index in enumerate(ordered_indexes):
        entry = _entry_from_index(index, priority=priority, warnings=warnings, errors=errors)
        if entry is None:
            continue
        duplicate_tiles = tuple(tile for tile in entry["tile_keys"] if tile in seen_tiles)
        if duplicate_tiles:
            errors.append(_issue(BLIND4D_VIEW_TILE_DUPLICATE, IssueSeverity.ERROR, index.id, index.path.resolved))
            continue
        seen_tiles.update(str(tile) for tile in entry["tile_keys"])
        entries.append(entry)
    coverage = _view_coverage(tuple(ordered_indexes))
    if coverage.all_sky or coverage.status is CoverageStatus.FULL:
        errors.append(_issue(BLIND4D_VIEW_COVERAGE_INCONSISTENT, IssueSeverity.ERROR, "blind4d", None))
    payload = {
        "schema": MANIFEST_SCHEMA,
        "manifest_version": MANIFEST_VERSION,
        "description": "CatalogLibrary-owned strict Blind 4D manifest view.",
        "indexes": entries,
    }
    fingerprint = _view_fingerprint(payload, library=library, coverage=coverage)
    return CatalogBlind4DManifestView(
        payload=payload,
        schema=MANIFEST_SCHEMA,
        version=MANIFEST_VERSION,
        entries=tuple(entries),
        coverage=coverage,
        fingerprint=fingerprint,
        warnings=tuple(warnings),
        errors=tuple(errors),
        source_library_id=library.manifest.library_id,
        source_manifest_fingerprint=_source_manifest_fingerprint(library),
        telemetry={
            "entry_count": len(entries),
            "coverage_status": coverage.status.value,
            "covered_tiles": coverage.covered_tiles,
            "total_tiles": coverage.total_tiles,
            "all_sky": coverage.all_sky,
        },
    )


def _ordered_indexes(
    library: CatalogLibrary,
    indexes: tuple[CatalogIndex, ...],
    errors: list[CatalogIssue],
) -> tuple[CatalogIndex, ...]:
    if not indexes:
        errors.append(_issue(BLIND4D_VIEW_NO_INDEXES, IssueSeverity.ERROR, "blind4d", None))
        return ()
    order = library.manifest.runtime_order.get("blind4d")
    if not order:
        errors.append(_issue(BLIND4D_VIEW_RUNTIME_ORDER_MISSING, IssueSeverity.ERROR, "blind4d", None))
        return ()
    seen: set[str] = set()
    duplicates: set[str] = set()
    for index_id in order:
        if index_id in seen:
            duplicates.add(index_id)
        seen.add(index_id)
    if duplicates:
        errors.append(_issue(BLIND4D_VIEW_RUNTIME_ORDER_DUPLICATE, IssueSeverity.ERROR, ",".join(sorted(duplicates)), None))
    by_id = {index.id: index for index in indexes}
    missing_from_order = sorted(set(by_id) - set(order))
    extra_in_order = sorted(set(order) - set(by_id))
    if missing_from_order:
        errors.append(_issue(BLIND4D_VIEW_RUNTIME_ORDER_MISSING, IssueSeverity.ERROR, ",".join(missing_from_order), None))
    for index_id in extra_in_order:
        errors.append(_issue(BLIND4D_VIEW_INDEX_MISSING, IssueSeverity.ERROR, index_id, None))
    if duplicates or missing_from_order or extra_in_order:
        return ()
    return tuple(by_id[index_id] for index_id in order)


def _entry_from_index(
    index: CatalogIndex,
    *,
    priority: int,
    warnings: list[CatalogIssue],
    errors: list[CatalogIssue],
) -> dict[str, Any] | None:
    if index.status in {CatalogDataStatus.MISSING, CatalogDataStatus.CORRUPT, CatalogDataStatus.INCOMPATIBLE, CatalogDataStatus.MISMATCH}:
        errors.append(_issue(BLIND4D_VIEW_INDEX_CORRUPT, IssueSeverity.ERROR, index.id, index.path.resolved))
        return None
    path = index.path.resolved
    if not path.exists():
        errors.append(_issue(BLIND4D_VIEW_INDEX_MISSING, IssueSeverity.ERROR, index.id, path))
        return None
    expected_sha = _expected_sha(index)
    if not expected_sha:
        errors.append(_issue(BLIND4D_VIEW_INDEX_CORRUPT, IssueSeverity.ERROR, index.id, path))
        return None
    actual_sha = sha256_file(path)
    if actual_sha.lower() != expected_sha.lower():
        errors.append(_issue(BLIND4D_VIEW_CHECKSUM_MISMATCH, IssueSeverity.ERROR, index.id, path))
        return None
    try:
        loaded = Quad4DIndex.load(path)
    except Exception as exc:
        errors.append(_issue(BLIND4D_VIEW_INDEX_CORRUPT, IssueSeverity.ERROR, index.id, path, message=str(exc)))
        return None
    quad_schema = str(loaded.metadata.get("schema") or index.schema)
    if quad_schema != ASTROMETRY_AB_CODE_4D_SCHEMA:
        errors.append(_issue(BLIND4D_VIEW_SCHEMA_UNSUPPORTED, IssueSeverity.ERROR, index.id, path))
        return None
    tile_keys = tuple(str(tile) for tile in loaded.tile_keys)
    if tuple(index.source_tiles) != tile_keys:
        errors.append(_issue(BLIND4D_VIEW_COVERAGE_INCONSISTENT, IssueSeverity.ERROR, index.id, path))
        return None
    params = _merged_parameters(index, loaded.metadata)
    code_tol = params.get("code_tol_recommended")
    if code_tol is None:
        warnings.append(_issue("BLIND4D_VIEW_CODE_TOL_DEFAULTED_FROM_RUNTIME", IssueSeverity.WARNING, index.id, path))
    entry = {
        "id": index.id,
        "enabled": True,
        "path": str(path),
        "filename": path.name,
        "quad_schema": quad_schema,
        "index_version": int(loaded.metadata.get("version", 1)),
        "level": str(params.get("level") or loaded.metadata.get("level") or ""),
        "tile_keys": list(tile_keys),
        "star_count": int(loaded.catalog_ra_dec.shape[0]),
        "quad_count": int(loaded.codes_4d.shape[0]),
        "sampler_tag": str(params.get("sampler_tag") or loaded.metadata.get("sampler_tag") or ""),
        "code_tol_recommended": float(code_tol) if code_tol is not None else None,
        "catalog_source": str(params.get("catalog_source") or loaded.metadata.get("source_catalog") or ""),
        "sha256": actual_sha,
        "priority": int(priority),
        "file_size_bytes": int(path.stat().st_size),
    }
    return entry


def _merged_parameters(index: CatalogIndex, metadata: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    result.update(metadata)
    result.update(index.build_parameters)
    result.update(index.parameters)
    if "catalog_source" not in result and "source_catalog" in metadata:
        result["catalog_source"] = metadata.get("source_catalog")
    return result


def _expected_sha(index: CatalogIndex) -> str | None:
    for item in (*index.derived_files, *index.integrity_files):
        if item.sha256 and (item.resolved_path is None or item.resolved_path.resolve() == index.path.resolved):
            return item.sha256.lower()
    return None


def _view_coverage(indexes: tuple[CatalogIndex, ...]) -> CatalogCoverage:
    if not indexes:
        return CatalogCoverage(status=CoverageStatus.MISSING, provenance="blind4d-view")
    return merge_coverages(index.coverage for index in indexes)


def _view_fingerprint(payload: dict[str, Any], *, library: CatalogLibrary, coverage: CatalogCoverage) -> str:
    entries: list[dict[str, Any]] = []
    indexes_by_id = {index.id: index for index in library.manifest.derived_indexes}
    for entry in payload.get("indexes", []):
        index = indexes_by_id.get(str(entry.get("id") or ""))
        path_payload = None
        if index is not None:
            path_payload = {"kind": index.path.kind.value, "value": index.path.value}
        entries.append(
            {
                key: _strip_for_fingerprint(value)
                for key, value in entry.items()
                if key not in _VIEW_FINGERPRINT_EXCLUDE_KEYS and key != "path"
            }
            | {"path": path_payload or entry.get("path")}
        )
    canonical = {
        "schema": payload.get("schema"),
        "manifest_version": payload.get("manifest_version"),
        "indexes": entries,
        "coverage": {
            "status": coverage.status.value,
            "all_sky": coverage.all_sky,
            "families": list(coverage.families),
            "tile_keys": list(coverage.tile_keys),
            "covered_tiles": coverage.covered_tiles,
            "total_tiles": coverage.total_tiles,
            "fraction": coverage.fraction,
        },
    }
    return hashlib.sha256(_canonical_json(canonical)).hexdigest()


def _strip_for_fingerprint(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _strip_for_fingerprint(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _VIEW_FINGERPRINT_EXCLUDE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_strip_for_fingerprint(item) for item in value]
    return value


def _source_manifest_fingerprint(library: CatalogLibrary) -> str | None:
    path = library.manifest.manifest_path
    if path.exists() and path.is_file():
        return sha256_file(path)
    return None


def _serialize_payload(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _issue(
    code: str,
    severity: IssueSeverity,
    component_id: str,
    path: Path | None,
    *,
    message: str | None = None,
) -> CatalogIssue:
    return CatalogIssue(
        code=code,
        severity=severity,
        message=message or f"{code}: {component_id}",
        path=path,
        component_id=component_id,
    )


__all__ = [
    "BLIND4D_VIEW_CHECKSUM_MISMATCH",
    "BLIND4D_VIEW_COVERAGE_INCONSISTENT",
    "BLIND4D_VIEW_INDEX_CORRUPT",
    "BLIND4D_VIEW_INDEX_MISSING",
    "BLIND4D_VIEW_MATERIALIZATION_FAILED",
    "BLIND4D_VIEW_NO_INDEXES",
    "BLIND4D_VIEW_PATH_INVALID",
    "BLIND4D_VIEW_RUNTIME_ORDER_DUPLICATE",
    "BLIND4D_VIEW_RUNTIME_ORDER_MISSING",
    "BLIND4D_VIEW_SCHEMA_UNSUPPORTED",
    "BLIND4D_VIEW_TILE_DUPLICATE",
    "CatalogBlind4DManifestView",
    "CatalogBlind4DManifestViewError",
    "build_blind4d_manifest_view",
]
