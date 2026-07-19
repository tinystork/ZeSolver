"""Validation and status calculation for CatalogLibrary."""

from __future__ import annotations

import hashlib
from pathlib import Path

from zewcs290.catalog290 import FAMILY_SPECS

from .coverage import merge_coverages
from .models import (
    CatalogCapabilities,
    CatalogCoverage,
    CatalogDataStatus,
    CatalogIssue,
    CatalogStatus,
    CatalogValidationReport,
    CoverageStatus,
    IssueSeverity,
    SUPPORTED_INDEX_ENGINES,
    SUPPORTED_SOURCE_FAMILIES,
    SUPPORTED_SOURCE_FORMATS,
)


def validate_library(library) -> CatalogValidationReport:
    manifest = library.manifest
    issues: list[CatalogIssue] = []
    valid_sources = []
    valid_indexes = []
    checked_sources: list[str] = []
    checked_indexes: list[str] = []

    for source in manifest.sources:
        checked_sources.append(source.id)
        source_ok = True
        if source.family not in SUPPORTED_SOURCE_FAMILIES:
            issues.append(_issue("SOURCE_FAMILY_UNSUPPORTED", IssueSeverity.ERROR, source.id, source.path.resolved))
            source_ok = False
        if source.format not in SUPPORTED_SOURCE_FORMATS:
            issues.append(_issue("SOURCE_FORMAT_UNSUPPORTED", IssueSeverity.ERROR, source.id, source.path.resolved))
            source_ok = False
        if source.status in {CatalogDataStatus.MISSING, CatalogDataStatus.CORRUPT, CatalogDataStatus.INCOMPATIBLE, CatalogDataStatus.MISMATCH}:
            issues.append(_issue(f"SOURCE_STATUS_{source.status.value}", IssueSeverity.ERROR, source.id, source.path.resolved))
            source_ok = False
        if not source.path.resolved.exists():
            issues.append(_issue("SOURCE_PATH_MISSING", IssueSeverity.ERROR, source.id, source.path.resolved))
            source_ok = False
        elif not source.path.resolved.is_dir():
            issues.append(_issue("SOURCE_UNREADABLE", IssueSeverity.ERROR, source.id, source.path.resolved))
            source_ok = False
        elif source.family in FAMILY_SPECS:
            files = tuple(source.path.resolved.glob(FAMILY_SPECS[source.family].glob_pattern()))
            if source.tile_count > 0 and not files:
                issues.append(_issue("SOURCE_TILE_FILES_MISSING", IssueSeverity.ERROR, source.id, source.path.resolved))
                source_ok = False
        for integrity_file in source.integrity_files:
            if not _check_integrity_file(integrity_file.resolved_path, integrity_file.sha256, issues, source.id, "SOURCE"):
                source_ok = False
        for shard in source.shards:
            if not _check_integrity_file(shard.resolved_path, shard.sha256, issues, source.id, "SOURCE_SHARD"):
                source_ok = False
        if source_ok:
            valid_sources.append(source)

    source_ids = {source.id for source in manifest.sources}
    for index in manifest.derived_indexes:
        checked_indexes.append(index.id)
        index_ok = True
        if index.engine not in SUPPORTED_INDEX_ENGINES:
            issues.append(_issue("INDEX_SCHEMA_INCOMPATIBLE", IssueSeverity.ERROR, index.id, index.path.resolved))
            index_ok = False
        if index.status in {CatalogDataStatus.MISSING, CatalogDataStatus.CORRUPT, CatalogDataStatus.INCOMPATIBLE, CatalogDataStatus.MISMATCH}:
            issues.append(_issue(f"INDEX_STATUS_{index.status.value}", IssueSeverity.ERROR, index.id, index.path.resolved))
            index_ok = False
        if not index.path.resolved.exists():
            issues.append(_issue("INDEX_PATH_MISSING", IssueSeverity.ERROR, index.id, index.path.resolved))
            index_ok = False
        if index.manifest_path is not None and not index.manifest_path.resolved.exists():
            issues.append(_issue("INDEX_MANIFEST_PATH_MISSING", IssueSeverity.ERROR, index.id, index.manifest_path.resolved))
            index_ok = False
        for source_id in index.source_ids:
            if source_id not in source_ids:
                issues.append(_issue("INDEX_SOURCE_UNKNOWN", IssueSeverity.WARNING, index.id, index.path.resolved))
        for integrity_file in index.integrity_files:
            if not _check_integrity_file(integrity_file.resolved_path, integrity_file.sha256, issues, index.id, "INDEX"):
                index_ok = False
        if index.coverage.fraction is not None and not 0.0 <= index.coverage.fraction <= 1.0:
            issues.append(_issue("COVERAGE_INCONSISTENT", IssueSeverity.ERROR, index.id, index.path.resolved))
            index_ok = False
        if index.coverage.all_sky and index.coverage.status is not CoverageStatus.FULL:
            issues.append(_issue("COVERAGE_INCONSISTENT", IssueSeverity.ERROR, index.id, index.path.resolved))
            index_ok = False
        if index.coverage.status is CoverageStatus.FULL and not index.coverage.all_sky:
            issues.append(_issue("COVERAGE_INCONSISTENT", IssueSeverity.ERROR, index.id, index.path.resolved))
            index_ok = False
        if index_ok:
            valid_indexes.append(index)

    coverage = _validated_coverage(manifest.coverage, tuple(item.coverage for item in valid_sources), tuple(item.coverage for item in valid_indexes))
    capabilities = _capabilities(valid_sources, valid_indexes)
    status = _status_from(issues, capabilities, coverage, bool(manifest.sources), bool(manifest.derived_indexes))
    return CatalogValidationReport(
        status=status,
        capabilities=capabilities,
        issues=tuple(issues),
        checked_sources=tuple(checked_sources),
        checked_indexes=tuple(checked_indexes),
        coverage=coverage,
    )


def _capabilities(valid_sources, valid_indexes) -> CatalogCapabilities:
    near_sources = [source for source in valid_sources if source.kind == "astap_hnsky"]
    blind4d_indexes = [index for index in valid_indexes if index.engine == "blind4d"]
    legacy_indexes = [index for index in valid_indexes if index.engine == "historical"]
    return CatalogCapabilities(
        near=bool(near_sources),
        blind4d=bool(blind4d_indexes),
        legacy_blind=bool(legacy_indexes),
        all_sky_near=bool(near_sources) and any(source.coverage.all_sky for source in near_sources),
        all_sky_blind4d=bool(blind4d_indexes) and all(index.coverage.all_sky for index in blind4d_indexes),
    )


def _status_from(
    issues: list[CatalogIssue],
    capabilities: CatalogCapabilities,
    coverage: CatalogCoverage,
    declared_sources: bool,
    declared_indexes: bool,
) -> CatalogStatus:
    if any(issue.severity in {IssueSeverity.ERROR, IssueSeverity.FATAL} and ("SHA256_MISMATCH" in issue.code or "STATUS_MISMATCH" in issue.code) for issue in issues):
        return CatalogStatus.CORRUPT
    if any(issue.severity in {IssueSeverity.ERROR, IssueSeverity.FATAL} and "INCOMPATIBLE" in issue.code for issue in issues):
        return CatalogStatus.INCOMPATIBLE
    if any(issue.severity is IssueSeverity.FATAL for issue in issues):
        return CatalogStatus.CORRUPT
    if not capabilities.near and not capabilities.blind4d:
        if declared_sources or declared_indexes:
            return CatalogStatus.MISSING
        return CatalogStatus.MISSING
    if capabilities.near and capabilities.blind4d:
        return CatalogStatus.READY_FULL if coverage.all_sky and capabilities.all_sky_blind4d else CatalogStatus.READY_PARTIAL
    if capabilities.near:
        return CatalogStatus.NEAR_ONLY if declared_indexes else CatalogStatus.SOURCE_ONLY
    if capabilities.blind4d:
        return CatalogStatus.BLIND4D_ONLY
    return CatalogStatus.MISSING


def _validated_coverage(
    manifest_coverage: CatalogCoverage,
    source_coverages: tuple[CatalogCoverage, ...],
    index_coverages: tuple[CatalogCoverage, ...],
) -> CatalogCoverage:
    if index_coverages:
        merged = merge_coverages(index_coverages)
        if merged.status is not CoverageStatus.MISSING:
            return merged
    if source_coverages:
        return merge_coverages(source_coverages)
    return manifest_coverage


def _check_integrity_file(
    path: Path | None,
    expected_sha256: str | None,
    issues: list[CatalogIssue],
    component_id: str,
    prefix: str,
) -> bool:
    if path is None:
        return True
    if not path.exists():
        issues.append(_issue(f"{prefix}_PATH_MISSING", IssueSeverity.ERROR, component_id, path))
        return False
    if expected_sha256:
        actual = sha256_file(path)
        if actual.lower() != expected_sha256.lower():
            issues.append(_issue(f"{prefix}_SHA256_MISMATCH", IssueSeverity.ERROR, component_id, path))
            return False
    return True


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _issue(code: str, severity: IssueSeverity, component_id: str, path: Path | None) -> CatalogIssue:
    return CatalogIssue(
        code=code,
        severity=severity,
        message=f"{code}: {component_id}",
        path=path,
        component_id=component_id,
    )
