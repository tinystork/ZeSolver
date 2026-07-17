"""Read-only data models for ZeSolver catalogue libraries."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class CatalogStatus(str, Enum):
    READY_FULL = "READY_FULL"
    READY_PARTIAL = "READY_PARTIAL"
    NEAR_ONLY = "NEAR_ONLY"
    BLIND4D_ONLY = "BLIND4D_ONLY"
    SOURCE_ONLY = "SOURCE_ONLY"
    INDEX_BUILD_REQUIRED = "INDEX_BUILD_REQUIRED"
    INCOMPATIBLE = "INCOMPATIBLE"
    CORRUPT = "CORRUPT"
    MISSING = "MISSING"


class CoverageStatus(str, Enum):
    FULL = "FULL"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"
    INCOMPATIBLE = "INCOMPATIBLE"
    CORRUPT = "CORRUPT"
    UNKNOWN = "UNKNOWN"


class CatalogDataStatus(str, Enum):
    PRESENT = "PRESENT"
    READY_PARTIAL = "READY_PARTIAL"
    FULL = "FULL"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"
    INCOMPATIBLE = "INCOMPATIBLE"
    CORRUPT = "CORRUPT"
    EXTERNAL_REFERENCE = "EXTERNAL_REFERENCE"


class IssueSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class PathKind(str, Enum):
    RELATIVE = "relative"
    EXTERNAL_REFERENCE = "external_reference"


SUPPORTED_SOURCE_FAMILIES = frozenset({"d05", "d20", "d50", "d80", "v50", "g05"})
SUPPORTED_SOURCE_FORMATS = frozenset({"1476-5", "1476-6", "290-5"})
SUPPORTED_INDEX_ENGINES = frozenset({"blind4d", "historical"})
SUPPORTED_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class PathReference:
    kind: PathKind
    value: str
    resolved: Path

    @property
    def external_reference(self) -> bool:
        return self.kind is PathKind.EXTERNAL_REFERENCE


@dataclass(frozen=True, slots=True)
class IntegrityFile:
    path: str
    sha256: str | None = None
    size_bytes: int | None = None
    resolved_path: Path | None = None


@dataclass(frozen=True, slots=True)
class CatalogCoverage:
    status: CoverageStatus = CoverageStatus.UNKNOWN
    all_sky: bool = False
    families: tuple[str, ...] = ()
    tile_keys: tuple[str, ...] = ()
    dec_min_deg: float | None = None
    dec_max_deg: float | None = None
    ra_segments_deg: tuple[tuple[float, float], ...] = ()
    scale_min_arcsec: float | None = None
    scale_max_arcsec: float | None = None
    covered_tiles: int = 0
    total_tiles: int | None = None
    fraction: float | None = None
    provenance: str | None = None
    notes: str = ""


@dataclass(frozen=True, slots=True)
class CatalogCapabilities:
    near: bool = False
    blind4d: bool = False
    legacy_blind: bool = False
    all_sky_near: bool = False
    all_sky_blind4d: bool = False


@dataclass(frozen=True, slots=True)
class CatalogSource:
    id: str
    kind: str
    family: str
    format: str
    path: PathReference
    tile_count: int
    layout: str | None
    coverage: CatalogCoverage
    integrity_files: tuple[IntegrityFile, ...]
    status: CatalogDataStatus
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CatalogIndex:
    id: str
    engine: str
    schema: str
    path: PathReference
    manifest_path: PathReference | None
    source_ids: tuple[str, ...]
    source_tiles: tuple[str, ...]
    coverage: CatalogCoverage
    integrity_files: tuple[IntegrityFile, ...]
    status: CatalogDataStatus
    algorithm_version: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    scale_range_arcsec: tuple[float | None, float | None] | None = None
    compatibility: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CatalogManifest:
    schema_version: int
    library_id: str
    root: Path
    manifest_path: Path
    created_at: str | None
    created_by: str | None
    minimum_zesolver_version: str | None
    declared_status: CatalogStatus
    sources: tuple[CatalogSource, ...]
    derived_indexes: tuple[CatalogIndex, ...]
    coverage: CatalogCoverage
    integrity: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CatalogIssue:
    code: str
    severity: IssueSeverity
    message: str
    path: Path | None = None
    component_id: str | None = None


@dataclass(frozen=True, slots=True)
class CatalogValidationReport:
    status: CatalogStatus
    capabilities: CatalogCapabilities
    issues: tuple[CatalogIssue, ...]
    checked_sources: tuple[str, ...]
    checked_indexes: tuple[str, ...]
    coverage: CatalogCoverage


@dataclass(frozen=True, slots=True)
class NearCatalogDescriptor:
    root: Path
    families: tuple[str, ...]
    formats: tuple[str, ...]
    coverage: CatalogCoverage
    external_reference: bool


@dataclass(frozen=True, slots=True)
class Blind4DIndexDescriptor:
    id: str
    path: Path
    family: str | None
    tile_keys: tuple[str, ...]
    sha256: str | None
    coverage: CatalogCoverage
    schema: str
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class LegacyIndexDescriptor:
    root: Path
    manifest_path: Path | None
    external_reference: bool


@dataclass(frozen=True, slots=True)
class CatalogDiscoveryResult:
    astap_root: Path | None
    blind4d_manifest: Path | None
    legacy_index_root: Path | None
    families: tuple[str, ...]
    blind4d_indexes: tuple[Blind4DIndexDescriptor, ...]
    issues: tuple[CatalogIssue, ...]
    status: CatalogStatus
    candidate_manifest: dict[str, Any]
