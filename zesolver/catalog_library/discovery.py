"""Read-only discovery of existing catalogue and index installations."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from zewcs290.catalog290 import FAMILY_SPECS
from zewcs290.layouts import get_layout

from .coverage import coverage_for_tiles
from .models import (
    Blind4DIndexDescriptor,
    CatalogDiscoveryResult,
    CatalogIssue,
    CatalogStatus,
    IssueSeverity,
)


def discover_existing(
    *,
    astap_root: str | Path | None = None,
    blind4d_manifest: str | Path | None = None,
    legacy_index_root: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> CatalogDiscoveryResult:
    env_map = os.environ if env is None else env
    issues: list[CatalogIssue] = []
    astap_path = _path_arg_or_env(astap_root, env_map, "ZESOLVER_ASTAP_ROOT")
    blind_path = _blind4d_manifest_arg_or_env(blind4d_manifest, env_map, issues)
    legacy_path = _path_arg_or_env(legacy_index_root, env_map, "ZESOLVER_LEGACY_INDEX_ROOT")

    families = _discover_astap_families(astap_path)
    indexes = _discover_4d_indexes(blind_path, issues)
    if families and indexes:
        status = CatalogStatus.READY_PARTIAL
    elif families:
        status = CatalogStatus.NEAR_ONLY
    elif indexes:
        status = CatalogStatus.BLIND4D_ONLY
    else:
        status = CatalogStatus.MISSING
    return CatalogDiscoveryResult(
        astap_root=astap_path,
        blind4d_manifest=blind_path,
        legacy_index_root=legacy_path,
        families=families,
        blind4d_indexes=indexes,
        issues=tuple(issues),
        status=status,
        candidate_manifest=_candidate_manifest(astap_path, blind_path, legacy_path, families, indexes, status),
    )


def _path_arg_or_env(value: str | Path | None, env: dict[str, str], key: str) -> Path | None:
    if value is not None:
        return Path(value).expanduser()
    text = env.get(key)
    return Path(text).expanduser() if text else None


def _blind4d_manifest_arg_or_env(
    value: str | Path | None,
    env: dict[str, str],
    issues: list[CatalogIssue],
) -> Path | None:
    if value is not None:
        return Path(value).expanduser()
    primary = env.get("ZESOLVER_BLIND4D_MANIFEST")
    legacy = env.get("ZEBLIND_4D_MANIFEST")
    if primary and legacy and Path(primary).expanduser() != Path(legacy).expanduser():
        issues.append(
            CatalogIssue(
                code="BLIND4D_ENV_CONFLICT",
                severity=IssueSeverity.WARNING,
                message="ZESOLVER_BLIND4D_MANIFEST overrides ZEBLIND_4D_MANIFEST",
                path=Path(primary).expanduser(),
                component_id="environment",
            )
        )
    chosen = primary or legacy
    return Path(chosen).expanduser() if chosen else None


def _discover_astap_families(root: Path | None) -> tuple[str, ...]:
    if root is None or not root.exists() or not root.is_dir():
        return ()
    found: list[str] = []
    candidates = [root]
    candidates.extend(path for path in sorted(root.iterdir()) if path.is_dir())
    for family, spec in sorted(FAMILY_SPECS.items()):
        if any(_has_family_files(candidate, spec) for candidate in candidates):
            found.append(family)
    return tuple(found)


def _has_family_files(root: Path, spec) -> bool:
    expected_prefix = f"{spec.prefix}_".lower()
    expected_suffix = f".{spec.extension}".lower()
    return any(
        path.is_file()
        and path.name.lower().startswith(expected_prefix)
        and path.name.lower().endswith(expected_suffix)
        for path in root.iterdir()
    )


def _discover_4d_indexes(
    manifest_path: Path | None,
    issues: list[CatalogIssue],
) -> tuple[Blind4DIndexDescriptor, ...]:
    if manifest_path is None:
        return ()
    if not manifest_path.exists():
        issues.append(_issue("BLIND4D_MANIFEST_MISSING", manifest_path, "blind4d"))
        return ()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        issues.append(
            CatalogIssue(
                code="MANIFEST_INVALID_JSON",
                severity=IssueSeverity.ERROR,
                message=f"Cannot read 4D manifest: {exc}",
                path=manifest_path,
                component_id="blind4d",
            )
        )
        return ()
    base = manifest_path.parent
    result: list[Blind4DIndexDescriptor] = []
    for entry in payload.get("indexes") or ():
        if not isinstance(entry, dict) or not bool(entry.get("enabled", True)):
            continue
        raw_path = Path(str(entry.get("path") or ""))
        path = raw_path if raw_path.is_absolute() else (base / raw_path)
        tile_keys = tuple(str(v) for v in entry.get("tile_keys") or ())
        family = _family_from_tiles(tile_keys)
        coverage = coverage_for_tiles(
            family=family or "unknown",
            tile_keys=tile_keys,
            total_tiles=_layout_tile_count(family) if family else None,
            dec_min_deg=None,
            dec_max_deg=None,
            provenance="4d-manifest",
        )
        if not path.exists():
            issues.append(_issue("INDEX_PATH_MISSING", path, str(entry.get("id") or "blind4d")))
            continue
        result.append(
            Blind4DIndexDescriptor(
                id=str(entry.get("id") or path.name),
                path=path.resolve(),
                family=family,
                tile_keys=tile_keys,
                sha256=str(entry.get("sha256") or "") or None,
                coverage=coverage,
                schema=str(entry.get("quad_schema") or payload.get("schema") or ""),
                enabled=True,
            )
        )
    return tuple(result)


def _candidate_manifest(
    astap_root: Path | None,
    blind4d_manifest: Path | None,
    legacy_index_root: Path | None,
    families: tuple[str, ...],
    indexes: tuple[Blind4DIndexDescriptor, ...],
    status: CatalogStatus,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "library_id": "discovered-existing",
        "status": status.value,
        "sources": [
            {"id": f"astap-{family}", "family": family, "path": str(astap_root)}
            for family in families
        ],
        "derived_indexes": [
            {"id": index.id, "engine": "blind4d", "path": str(index.path)}
            for index in indexes
        ],
        "legacy_index_root": str(legacy_index_root) if legacy_index_root else None,
        "blind4d_manifest": str(blind4d_manifest) if blind4d_manifest else None,
    }


def _family_from_tiles(tile_keys: tuple[str, ...]) -> str | None:
    for tile in tile_keys:
        if "_" in tile:
            return tile.split("_", 1)[0].lower()
    return None


def _layout_tile_count(family: str | None) -> int | None:
    if not family or family not in FAMILY_SPECS:
        return None
    spec = FAMILY_SPECS[family]
    layout_name = "hnsky_1476" if spec.extension == "1476" else "hnsky_290"
    return int(sum(ring.ra_cells for ring in get_layout(layout_name).iter_rings()))


def _issue(code: str, path: Path, component_id: str) -> CatalogIssue:
    return CatalogIssue(
        code=code,
        severity=IssueSeverity.ERROR,
        message=f"{code}: {path}",
        path=path,
        component_id=component_id,
    )
