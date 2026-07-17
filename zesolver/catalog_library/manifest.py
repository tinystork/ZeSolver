"""Manifest loading and public CatalogLibrary entry point."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .coverage import coverage_from_payload, merge_coverages
from .models import (
    CatalogCapabilities,
    CatalogCoverage,
    CatalogDataStatus,
    CatalogIndex,
    CatalogIssue,
    CatalogManifest,
    CatalogSource,
    CatalogStatus,
    CatalogValidationReport,
    IntegrityFile,
    IssueSeverity,
    PathKind,
    PathReference,
    SUPPORTED_INDEX_ENGINES,
    SUPPORTED_MANIFEST_SCHEMA_VERSION,
    SUPPORTED_SOURCE_FAMILIES,
    SUPPORTED_SOURCE_FORMATS,
)


class CatalogLibraryError(RuntimeError):
    """Base class for CatalogLibrary failures."""


class CatalogManifestError(CatalogLibraryError):
    """Raised when catalog.json is malformed."""


class CatalogMissingError(CatalogLibraryError):
    """Raised when the requested library or manifest is absent."""


class CatalogIncompleteError(CatalogLibraryError):
    """Raised when a requested capability is unavailable."""


class CatalogCorruptionError(CatalogLibraryError):
    """Raised when integrity validation finds corrupted data."""


class CatalogCompatibilityError(CatalogLibraryError):
    """Raised when data are incompatible with this ZeSolver version."""


class CatalogVersionError(CatalogCompatibilityError):
    """Raised when the manifest schema version is unsupported."""


class CatalogLibrary:
    """Read-only facade over a ZeSolver catalogue library manifest."""

    def __init__(self, manifest: CatalogManifest):
        self.manifest = manifest
        self.root = manifest.root

    @classmethod
    def open(cls, path: str | Path) -> "CatalogLibrary":
        manifest = load_manifest(path)
        return cls(manifest)

    @classmethod
    def discover_existing(cls, **kwargs: Any):
        from .discovery import discover_existing

        return discover_existing(**kwargs)

    @property
    def status(self) -> CatalogStatus:
        return self.validate().status

    @property
    def capabilities(self) -> CatalogCapabilities:
        return self.validate().capabilities

    @property
    def coverage(self) -> CatalogCoverage:
        return self.validate().coverage

    def validate(self) -> CatalogValidationReport:
        from .validation import validate_library

        return validate_library(self)

    def near_source(self):
        from .adapters import near_source_descriptor

        return near_source_descriptor(self)

    def blind4d_indexes(self):
        from .adapters import blind4d_index_descriptors

        return blind4d_index_descriptors(self)

    def blind4d_runtime_paths(self) -> tuple[Path, ...]:
        return tuple(index.path for index in self.blind4d_indexes())


def load_manifest(path: str | Path) -> CatalogManifest:
    raw_path = Path(path).expanduser()
    manifest_path = raw_path if raw_path.name == "catalog.json" else raw_path / "catalog.json"
    if not manifest_path.exists():
        raise CatalogMissingError(f"MANIFEST_MISSING: {manifest_path}")
    if not manifest_path.is_file():
        raise CatalogManifestError(f"MANIFEST_NOT_FILE: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CatalogManifestError(f"MANIFEST_INVALID_JSON: {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CatalogManifestError("MANIFEST_INVALID_JSON: expected object")
    root = manifest_path.parent.resolve()
    return manifest_from_payload(payload, root=root, manifest_path=manifest_path.resolve())


def manifest_from_payload(payload: dict[str, Any], *, root: Path, manifest_path: Path) -> CatalogManifest:
    required = ("schema_version", "library_id", "sources", "derived_indexes", "coverage", "integrity", "provenance", "status")
    for field in required:
        if field not in payload:
            raise CatalogManifestError(f"REQUIRED_FIELD_MISSING: {field}")
    version = _as_int(payload.get("schema_version"), "schema_version")
    if version != SUPPORTED_MANIFEST_SCHEMA_VERSION:
        raise CatalogVersionError(f"SCHEMA_UNSUPPORTED: {version}")
    library_id = _as_nonempty_str(payload.get("library_id"), "library_id")
    declared_status = _catalog_status(str(payload.get("status")))
    sources = tuple(_source_from_payload(item, root=root) for item in _as_list(payload.get("sources"), "sources"))
    indexes = tuple(_index_from_payload(item, root=root) for item in _as_list(payload.get("derived_indexes"), "derived_indexes"))
    coverage = coverage_from_payload(payload.get("coverage"), provenance="manifest")
    return CatalogManifest(
        schema_version=version,
        library_id=library_id,
        root=root,
        manifest_path=manifest_path,
        created_at=_optional_str(payload.get("created_at")),
        created_by=_optional_str(payload.get("created_by")),
        minimum_zesolver_version=_optional_str(payload.get("minimum_zesolver_version")),
        declared_status=declared_status,
        sources=sources,
        derived_indexes=indexes,
        coverage=coverage,
        integrity=dict(payload.get("integrity") or {}),
        provenance=dict(payload.get("provenance") or {}),
    )


def resolve_path_reference(payload: object, *, root: Path, field: str) -> PathReference:
    if not isinstance(payload, dict):
        raise CatalogManifestError(f"REQUIRED_FIELD_MISSING: {field}")
    kind_text = str(payload.get("kind") or "")
    value = str(payload.get("value") or "")
    try:
        kind = PathKind(kind_text)
    except ValueError as exc:
        raise CatalogManifestError(f"PATH_KIND_INVALID: {field}: {kind_text!r}") from exc
    if not value:
        raise CatalogManifestError(f"PATH_VALUE_MISSING: {field}")
    if "$" in value:
        raise CatalogManifestError(f"PATH_ENV_EXPANSION_NOT_ALLOWED: {field}")
    raw = Path(value)
    if kind is PathKind.RELATIVE:
        if raw.is_absolute() or "~" in raw.parts:
            raise CatalogManifestError(f"PATH_RELATIVE_INVALID: {field}: {value}")
        resolved = (root / raw).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError as exc:
            raise CatalogManifestError(f"PATH_ESCAPES_LIBRARY: {field}: {value}") from exc
    else:
        expanded = raw.expanduser()
        if not expanded.is_absolute():
            raise CatalogManifestError(f"EXTERNAL_PATH_NOT_ABSOLUTE: {field}: {value}")
        resolved = expanded.resolve()
    return PathReference(kind=kind, value=value, resolved=resolved)


def _source_from_payload(payload: object, *, root: Path) -> CatalogSource:
    if not isinstance(payload, dict):
        raise CatalogManifestError("SOURCE_INVALID: expected object")
    family = _as_nonempty_str(payload.get("family"), "source.family").lower()
    fmt = _as_nonempty_str(payload.get("format"), "source.format")
    status = _data_status(str(payload.get("status") or "MISSING"))
    return CatalogSource(
        id=_as_nonempty_str(payload.get("id"), "source.id"),
        kind=_as_nonempty_str(payload.get("kind"), "source.kind"),
        family=family,
        format=fmt,
        path=resolve_path_reference(payload.get("path"), root=root, field=f"source[{family}].path"),
        tile_count=_as_int(payload.get("tile_count"), "source.tile_count"),
        layout=_optional_str(payload.get("layout")),
        coverage=coverage_from_payload(payload.get("coverage"), provenance=f"source:{family}"),
        integrity_files=_integrity_files(payload.get("integrity"), root=root),
        provenance=dict(payload.get("provenance") or {}),
        status=status,
    )


def _index_from_payload(payload: object, *, root: Path) -> CatalogIndex:
    if not isinstance(payload, dict):
        raise CatalogManifestError("INDEX_INVALID: expected object")
    scale = payload.get("scale_range_arcsec")
    scale_tuple: tuple[float | None, float | None] | None = None
    if isinstance(scale, dict):
        scale_tuple = (_float_or_none(scale.get("min")), _float_or_none(scale.get("max")))
    manifest_ref = None
    if payload.get("manifest_path") is not None:
        manifest_ref = resolve_path_reference(payload.get("manifest_path"), root=root, field=f"index[{payload.get('id')}].manifest_path")
    return CatalogIndex(
        id=_as_nonempty_str(payload.get("id"), "index.id"),
        engine=_as_nonempty_str(payload.get("engine"), "index.engine"),
        schema=_as_nonempty_str(payload.get("schema"), "index.schema"),
        path=resolve_path_reference(payload.get("path"), root=root, field=f"index[{payload.get('id')}].path"),
        manifest_path=manifest_ref,
        source_ids=tuple(str(v) for v in _as_list(payload.get("source_ids"), "index.source_ids")),
        source_tiles=tuple(str(v) for v in _as_list(payload.get("source_tiles"), "index.source_tiles")),
        coverage=coverage_from_payload(payload.get("coverage"), provenance=f"index:{payload.get('id')}"),
        integrity_files=_integrity_files(payload.get("integrity"), root=root),
        status=_data_status(str(payload.get("status") or "MISSING")),
        algorithm_version=_optional_str(payload.get("algorithm_version")),
        parameters=dict(payload.get("parameters") or {}),
        scale_range_arcsec=scale_tuple,
        compatibility=dict(payload.get("compatibility") or {}),
    )


def _integrity_files(payload: object, *, root: Path) -> tuple[IntegrityFile, ...]:
    if not isinstance(payload, dict):
        return ()
    items = payload.get("files", [])
    if not isinstance(items, list):
        raise CatalogManifestError("INTEGRITY_FILES_INVALID")
    result: list[IntegrityFile] = []
    for item in items:
        if not isinstance(item, dict):
            raise CatalogManifestError("INTEGRITY_FILE_INVALID")
        rel = _as_nonempty_str(item.get("path"), "integrity.file.path")
        if Path(rel).is_absolute():
            resolved = Path(rel).expanduser().resolve()
        else:
            resolved = (root / rel).resolve()
            try:
                resolved.relative_to(root.resolve())
            except ValueError as exc:
                raise CatalogManifestError(f"PATH_ESCAPES_LIBRARY: integrity.file.path: {rel}") from exc
        result.append(
            IntegrityFile(
                path=rel,
                sha256=_optional_str(item.get("sha256")),
                size_bytes=_as_optional_int(item.get("size_bytes"), "integrity.file.size_bytes"),
                resolved_path=resolved,
            )
        )
    return tuple(result)


def _as_nonempty_str(value: object, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise CatalogManifestError(f"REQUIRED_FIELD_MISSING: {field}")
    return text


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _as_int(value: object, field: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise CatalogManifestError(f"FIELD_INVALID: {field}") from exc


def _as_optional_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    return _as_int(value, field)


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _as_list(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise CatalogManifestError(f"FIELD_INVALID: {field}: expected list")
    return value


def _catalog_status(value: str) -> CatalogStatus:
    try:
        return CatalogStatus(value)
    except ValueError as exc:
        raise CatalogManifestError(f"STATUS_INVALID: {value}") from exc


def _data_status(value: str) -> CatalogDataStatus:
    try:
        return CatalogDataStatus(value)
    except ValueError as exc:
        raise CatalogManifestError(f"DATA_STATUS_INVALID: {value}") from exc


def assert_supported_manifest_values(manifest: CatalogManifest) -> None:
    for source in manifest.sources:
        if source.family not in SUPPORTED_SOURCE_FAMILIES:
            raise CatalogManifestError(f"SOURCE_FAMILY_UNSUPPORTED: {source.id}: {source.family}")
        if source.format not in SUPPORTED_SOURCE_FORMATS:
            raise CatalogManifestError(f"SOURCE_FORMAT_UNSUPPORTED: {source.id}: {source.format}")
    for index in manifest.derived_indexes:
        if index.engine not in SUPPORTED_INDEX_ENGINES:
            raise CatalogManifestError(f"INDEX_ENGINE_UNSUPPORTED: {index.id}: {index.engine}")
