#!/usr/bin/env python3
"""Read-only inventory helper for ZeSolver catalogue and 4D index assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None:
    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from zewcs290.catalog290 import FAMILY_SPECS
from zewcs290.layouts import get_layout
from zesolver.catalog_library import discover_existing


ASTAP_LAYOUT_BY_EXTENSION = {
    "1476": "hnsky_1476",
    "290": "hnsky_290",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tile_parts(tile_key_or_code: str) -> tuple[str | None, str]:
    text = str(tile_key_or_code)
    if "_" in text:
        family, code = text.split("_", 1)
        return family.lower(), code
    return None, text


def _tile_bounds(family: str, tile_code: str) -> dict[str, Any] | None:
    spec = FAMILY_SPECS.get(family.lower())
    if spec is None:
        return None
    if len(tile_code) < 4 or not tile_code.isdigit():
        return None
    ring_index = int(tile_code[:2])
    tile_index = int(tile_code[2:])
    layout_name = ASTAP_LAYOUT_BY_EXTENSION.get(spec.extension)
    if layout_name is None:
        return None
    layout = get_layout(layout_name)
    try:
        ring = layout.ring_for_index(ring_index)
    except KeyError:
        return None
    if tile_index < 1 or tile_index > int(ring.ra_cells):
        return None
    ra_step = 360.0 / float(ring.ra_cells)
    return {
        "family": spec.key,
        "tile_code": tile_code,
        "ring": ring_index,
        "tile": tile_index,
        "dec_min_deg": float(ring.dec_min_deg),
        "dec_max_deg": float(ring.dec_max_deg),
        "ra_min_deg": float((tile_index - 1) * ra_step),
        "ra_max_deg": float(tile_index * ra_step),
        "ring_ra_cells": int(ring.ra_cells),
    }


def _layout_tile_count_for_family(family: str) -> int | None:
    spec = FAMILY_SPECS.get(family.lower())
    if spec is None:
        return None
    layout_name = ASTAP_LAYOUT_BY_EXTENSION.get(spec.extension)
    if layout_name is None:
        return None
    return int(sum(ring.ra_cells for ring in get_layout(layout_name).iter_rings()))


def _scan_astap_root(astap_root: Path | None) -> dict[str, Any]:
    if astap_root is None:
        return {"status": "NOT_CONFIGURED", "root": None, "families": {}, "issues": []}
    root = astap_root.expanduser()
    result: dict[str, Any] = {"status": "MISSING", "root": str(root), "families": {}, "issues": []}
    if not root.exists():
        result["issues"].append({"severity": "missing", "code": "ASTAP_ROOT_MISSING", "path": str(root)})
        return result
    if not root.is_dir():
        result["status"] = "CORRUPT"
        result["issues"].append({"severity": "error", "code": "ASTAP_ROOT_NOT_DIRECTORY", "path": str(root)})
        return result

    installed = 0
    for family, spec in sorted(FAMILY_SPECS.items()):
        files = sorted(root.glob(spec.glob_pattern()))
        rings: dict[int, int] = defaultdict(int)
        dec_min: float | None = None
        dec_max: float | None = None
        total_size = 0
        bad_names: list[str] = []
        tile_codes: list[str] = []
        for path in files:
            total_size += int(path.stat().st_size)
            try:
                tile_code = path.stem.split("_", 1)[1]
                ring_index = int(tile_code[:2])
                tile_codes.append(tile_code)
                rings[ring_index] += 1
                bounds = _tile_bounds(family, tile_code)
                if bounds is not None:
                    dec_min = bounds["dec_min_deg"] if dec_min is None else min(dec_min, bounds["dec_min_deg"])
                    dec_max = bounds["dec_max_deg"] if dec_max is None else max(dec_max, bounds["dec_max_deg"])
            except Exception:
                bad_names.append(path.name)
        layout_total = _layout_tile_count_for_family(family)
        status = "MISSING"
        if files:
            installed += 1
            status = "FULL" if layout_total and len(files) >= layout_total else "PARTIAL"
        result["families"][family] = {
            "status": status,
            "format": spec.format_name,
            "glob": spec.glob_pattern(),
            "tile_count": len(files),
            "layout_tile_count": layout_total,
            "coverage_fraction_by_tile_count": (float(len(files)) / float(layout_total)) if layout_total else None,
            "rings_present": sorted(rings),
            "ring_tile_counts": {str(k): int(v) for k, v in sorted(rings.items())},
            "dec_min_deg": dec_min,
            "dec_max_deg": dec_max,
            "size_bytes": total_size,
            "sample_tiles": tile_codes[:10],
            "bad_filenames": bad_names[:20],
        }
    result["status"] = "READY_PARTIAL" if installed else "MISSING"
    return result


def _read_npz_metadata(path: Path) -> dict[str, Any] | None:
    with np.load(path, allow_pickle=False) as data:
        if "metadata" not in data:
            return None
        raw = data["metadata"][0]
        return json.loads(str(raw))


def _scan_4d_manifest(manifest_path: Path | None) -> dict[str, Any]:
    if manifest_path is None:
        return {"status": "NOT_CONFIGURED", "manifest_path": None, "indexes": [], "coverage": {}, "issues": []}
    path = manifest_path.expanduser()
    result: dict[str, Any] = {
        "status": "MISSING",
        "manifest_path": str(path),
        "schema": None,
        "manifest_version": None,
        "indexes": [],
        "coverage": {},
        "issues": [],
    }
    if not path.exists():
        result["issues"].append({"severity": "missing", "code": "BLIND4D_MANIFEST_MISSING", "path": str(path)})
        return result
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        result["status"] = "CORRUPT"
        result["issues"].append({"severity": "error", "code": "BLIND4D_MANIFEST_JSON_INVALID", "error": str(exc)})
        return result
    result["schema"] = payload.get("schema")
    result["manifest_version"] = payload.get("manifest_version")
    raw_indexes = payload.get("indexes")
    if not isinstance(raw_indexes, list):
        result["status"] = "INCOMPATIBLE"
        result["issues"].append({"severity": "error", "code": "BLIND4D_MANIFEST_INDEXES_INVALID"})
        return result

    seen_tiles: set[str] = set()
    family_tiles: dict[str, set[str]] = defaultdict(set)
    dec_min: float | None = None
    dec_max: float | None = None
    present = 0
    enabled = 0
    for entry in raw_indexes:
        if not isinstance(entry, dict):
            result["issues"].append({"severity": "error", "code": "BLIND4D_MANIFEST_ENTRY_INVALID"})
            continue
        entry_id = str(entry.get("id") or "")
        is_enabled = bool(entry.get("enabled", True))
        if is_enabled:
            enabled += 1
        raw_path = Path(str(entry.get("path") or "")).expanduser()
        index_path = raw_path if raw_path.is_absolute() else (path.parent / raw_path).resolve()
        tile_keys = [str(v) for v in entry.get("tile_keys") or []]
        index_report: dict[str, Any] = {
            "id": entry_id,
            "enabled": is_enabled,
            "path": str(index_path),
            "exists": index_path.exists(),
            "size_bytes": int(index_path.stat().st_size) if index_path.exists() else None,
            "sha256_expected": entry.get("sha256"),
            "sha256_actual": None,
            "sha256_ok": None,
            "quad_schema": entry.get("quad_schema"),
            "index_version": entry.get("index_version"),
            "level": entry.get("level"),
            "tile_keys": tile_keys,
            "star_count": entry.get("star_count"),
            "quad_count": entry.get("quad_count"),
            "sampler_tag": entry.get("sampler_tag"),
            "catalog_source": entry.get("catalog_source"),
            "metadata": {},
            "status": "MISSING",
            "coverage": [],
        }
        if not is_enabled:
            index_report["status"] = "DISABLED"
        elif not index_path.exists():
            result["issues"].append({"severity": "missing", "code": "BLIND4D_INDEX_MISSING", "id": entry_id, "path": str(index_path)})
        else:
            present += 1
            actual_sha = sha256_file(index_path)
            index_report["sha256_actual"] = actual_sha
            expected_sha = str(entry.get("sha256") or "").lower()
            index_report["sha256_ok"] = (not expected_sha) or (expected_sha == actual_sha.lower())
            if expected_sha and expected_sha != actual_sha.lower():
                index_report["status"] = "CORRUPT"
                result["issues"].append({"severity": "error", "code": "BLIND4D_INDEX_SHA256_MISMATCH", "id": entry_id})
            else:
                try:
                    metadata = _read_npz_metadata(index_path)
                    index_report["metadata"] = metadata or {}
                    index_report["status"] = "PRESENT"
                except Exception as exc:
                    index_report["status"] = "INCOMPATIBLE"
                    result["issues"].append({"severity": "error", "code": "BLIND4D_INDEX_METADATA_INVALID", "id": entry_id, "error": str(exc)})
        for tile_key in tile_keys:
            family, tile_code = _tile_parts(tile_key)
            if not family:
                continue
            family_tiles[family].add(tile_key)
            if tile_key in seen_tiles:
                result["issues"].append({"severity": "error", "code": "BLIND4D_DUPLICATE_TILE", "tile_key": tile_key})
            seen_tiles.add(tile_key)
            bounds = _tile_bounds(family, tile_code)
            if bounds is not None:
                index_report["coverage"].append(bounds)
                dec_min = bounds["dec_min_deg"] if dec_min is None else min(dec_min, bounds["dec_min_deg"])
                dec_max = bounds["dec_max_deg"] if dec_max is None else max(dec_max, bounds["dec_max_deg"])
        result["indexes"].append(index_report)

    by_family = {}
    for family, tiles in sorted(family_tiles.items()):
        layout_total = _layout_tile_count_for_family(family)
        by_family[family] = {
            "tile_count": len(tiles),
            "layout_tile_count": layout_total,
            "coverage_fraction_by_tile_count": (float(len(tiles)) / float(layout_total)) if layout_total else None,
            "tiles": sorted(tiles),
        }
    result["coverage"] = {
        "status": "PARTIAL" if seen_tiles else "MISSING",
        "enabled_index_count": enabled,
        "present_index_count": present,
        "enabled_tile_count": len(seen_tiles),
        "families": by_family,
        "dec_min_deg": dec_min,
        "dec_max_deg": dec_max,
        "all_sky": False,
    }
    if any(issue.get("severity") == "error" for issue in result["issues"]):
        result["status"] = "CORRUPT"
    elif enabled and present == enabled:
        result["status"] = "READY_PARTIAL"
    elif enabled and present:
        result["status"] = "PARTIAL"
    else:
        result["status"] = "MISSING"
    return result


def build_report(astap_root: Path | None, blind4d_manifest: Path | None, legacy_index_root: Path | None) -> dict[str, Any]:
    discovery = discover_existing(
        astap_root=astap_root,
        blind4d_manifest=blind4d_manifest,
        legacy_index_root=legacy_index_root,
    )
    legacy = {"status": "NOT_CONFIGURED", "root": None, "issues": []}
    if legacy_index_root is not None:
        root = legacy_index_root.expanduser()
        legacy = {
            "status": "PRESENT" if (root / "manifest.json").exists() else "MISSING",
            "root": str(root),
            "manifest_path": str(root / "manifest.json"),
            "issues": [] if (root / "manifest.json").exists() else [{"severity": "missing", "code": "LEGACY_INDEX_MANIFEST_MISSING"}],
        }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "astap_root": str(astap_root.expanduser()) if astap_root else None,
            "blind4d_manifest": str(blind4d_manifest.expanduser()) if blind4d_manifest else None,
            "legacy_index_root": str(legacy_index_root.expanduser()) if legacy_index_root else None,
        },
        "astap": _scan_astap_root(astap_root),
        "blind4d": _scan_4d_manifest(blind4d_manifest),
        "legacy_index": legacy,
        "catalog_library": {
            "status": discovery.status.value,
            "families": list(discovery.families),
            "blind4d_index_count": len(discovery.blind4d_indexes),
            "issues": [
                {
                    "code": issue.code,
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "path": str(issue.path) if issue.path else None,
                    "component_id": issue.component_id,
                }
                for issue in discovery.issues
            ],
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    blind = report["blind4d"]
    astap = report["astap"]
    lines = [
        "# Catalog Library Audit",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Summary",
        "",
        f"- ASTAP status: `{astap.get('status')}`",
        f"- Blind 4D status: `{blind.get('status')}`",
        f"- Legacy index status: `{report['legacy_index'].get('status')}`",
        "",
        "## ASTAP Families",
        "",
        "| Family | Status | Format | Tiles | Layout Tiles | Dec Range | Size |",
        "| --- | --- | --- | ---: | ---: | --- | ---: |",
    ]
    for family, info in sorted((astap.get("families") or {}).items()):
        dec = "-"
        if info.get("dec_min_deg") is not None:
            dec = f"{info['dec_min_deg']:.3f} .. {info['dec_max_deg']:.3f}"
        lines.append(
            f"| `{family}` | `{info.get('status')}` | `{info.get('format')}` | {info.get('tile_count', 0)} | "
            f"{info.get('layout_tile_count') or ''} | {dec} | {info.get('size_bytes', 0)} |"
        )
    lines.extend([
        "",
        "## Blind 4D Indexes",
        "",
        "| ID | Status | Tiles | Stars | Quads | SHA | Source |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ])
    for idx in blind.get("indexes") or []:
        sha = "ok" if idx.get("sha256_ok") else ("-" if idx.get("sha256_ok") is None else "mismatch")
        lines.append(
            f"| `{idx.get('id')}` | `{idx.get('status')}` | {len(idx.get('tile_keys') or [])} | "
            f"{idx.get('star_count') or ''} | {idx.get('quad_count') or ''} | {sha} | `{idx.get('catalog_source')}` |"
        )
    cov = blind.get("coverage") or {}
    lines.extend([
        "",
        "## Blind 4D Coverage",
        "",
        f"- Status: `{cov.get('status')}`",
        f"- Enabled indexes: `{cov.get('enabled_index_count')}`",
        f"- Present indexes: `{cov.get('present_index_count')}`",
        f"- Enabled tiles: `{cov.get('enabled_tile_count')}`",
        f"- Declination range: `{cov.get('dec_min_deg')}` to `{cov.get('dec_max_deg')}`",
        f"- All sky: `{cov.get('all_sky')}`",
        "",
        "## Issues",
        "",
    ])
    issues = list(astap.get("issues") or []) + list(blind.get("issues") or []) + list(report["legacy_index"].get("issues") or [])
    if not issues:
        lines.append("- None")
    else:
        for issue in issues:
            lines.append(f"- `{issue.get('severity')}` `{issue.get('code')}` {json.dumps(issue, sort_keys=True)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--astap-root", type=Path)
    parser.add_argument("--blind4d-manifest", type=Path)
    parser.add_argument("--legacy-index-root", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args(argv)

    report = build_report(args.astap_root, args.blind4d_manifest, args.legacy_index_root)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(report, args.output_md)

    issues = []
    for section in ("astap", "blind4d", "legacy_index"):
        issues.extend(report.get(section, {}).get("issues") or [])
    return 1 if any(issue.get("severity") == "error" for issue in issues) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
