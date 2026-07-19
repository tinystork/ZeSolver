#!/usr/bin/env python3
"""Preview or explicitly commit a CatalogLibrary adoption plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None:
    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from zesolver.catalog_library import CatalogLibraryAdoptionError, CatalogLibraryAdoptionPlan, CatalogLibraryAdoptionWriter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview or explicitly write a ZeSolver CatalogLibrary adoption manifest.")
    parser.add_argument("--library-root", required=True, help="Existing target library root. The tool does not create it.")
    parser.add_argument("--astap-root", action="append", default=[], help="Existing ASTAP/HNSKY root to adopt by reference. Can be repeated.")
    parser.add_argument("--blind4d-manifest", help="Existing strict Blind 4D manifest to adopt by reference.")
    parser.add_argument("--legacy-index-root", help="Existing historical index root to retain as compatibility.")
    parser.add_argument("--fingerprint-policy", choices=("fast", "full"), default="fast")
    parser.add_argument("--write", action="store_true", help="Commit catalog.json atomically. Default is preview-only.")
    parser.add_argument("--replace-existing", action="store_true", help="Explicitly replace an existing catalog.json.")
    parser.add_argument("--expected-existing-sha256", help="Required when replacing an existing catalog.json.")
    parser.add_argument("--report-json", help="Write a JSON report.")
    parser.add_argument("--report-md", help="Write a Markdown report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = CatalogLibraryAdoptionPlan.reference_existing(
        library_root=Path(args.library_root),
        astap_roots=[Path(item) for item in args.astap_root] or None,
        blind4d_manifest=Path(args.blind4d_manifest) if args.blind4d_manifest else None,
        legacy_index_root=Path(args.legacy_index_root) if args.legacy_index_root else None,
        fingerprint_policy=args.fingerprint_policy,
    )
    result = None
    error: CatalogLibraryAdoptionError | None = None
    if args.write:
        try:
            result = CatalogLibraryAdoptionWriter.commit(
                plan,
                mode="replace" if args.replace_existing else "create",
                expected_existing_sha256=args.expected_existing_sha256,
                create_backup=True,
            )
        except CatalogLibraryAdoptionError as exc:
            error = exc
    elif args.replace_existing or args.expected_existing_sha256:
        error = CatalogLibraryAdoptionError("CATALOG_ADOPTION_PLAN_INVALID", "--replace-existing and --expected-existing-sha256 require --write")

    report = _report(plan, result=result, error=error, write_requested=bool(args.write))
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.report_md:
        Path(args.report_md).write_text(_markdown(report), encoding="utf-8")
    if not args.report_json and not args.report_md:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if error is None else 2


def _report(plan, *, result, error: CatalogLibraryAdoptionError | None, write_requested: bool) -> dict[str, Any]:
    return {
        "mode": "write" if write_requested else "preview",
        "plan": {
            "status": plan.status.value,
            "library_root": str(plan.library_root),
            "sources": [
                {"id": source.id, "family": source.family, "tile_count": source.tile_count, "status": source.status.value}
                for source in plan.sources
            ],
            "indexes": [
                {
                    "id": index.id,
                    "engine": index.engine,
                    "tiles": list(index.source_tiles),
                    "coverage_status": index.coverage.status.value,
                    "all_sky": index.coverage.all_sky,
                }
                for index in plan.indexes
            ],
            "compatibility_resources": [
                {"id": item.id, "category": item.category, "status": item.status.value}
                for item in plan.compatibility_resources
            ],
            "coverage": {
                "status": plan.coverage.status.value,
                "all_sky": plan.coverage.all_sky,
                "covered_tiles": plan.coverage.covered_tiles,
                "total_tiles": plan.coverage.total_tiles,
                "tile_keys": list(plan.coverage.tile_keys),
            },
            "warnings": [issue.code for issue in plan.warnings],
            "errors": [issue.code for issue in plan.errors],
            "repair_actions": [
                {
                    "code": action.code,
                    "resource_id": action.resource_id,
                    "execution_phase": action.execution_phase,
                    "automatic": action.automatic,
                }
                for action in plan.repair_actions
            ],
            "telemetry": {
                "fingerprint_policy": plan.telemetry.fingerprint_policy.value,
                "source_file_count": plan.telemetry.source_file_count,
                "source_hashed_count": plan.telemetry.source_hashed_count,
                "index_file_count": plan.telemetry.index_file_count,
                "index_hashed_count": plan.telemetry.index_hashed_count,
                "builder_called": plan.telemetry.builder_called,
                "files_written": plan.telemetry.files_written,
            },
        },
        "commit": _result_payload(result),
        "error": _error_payload(error),
        "manifest_preview": plan.manifest_preview,
    }


def _result_payload(result) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "status": result.status.value,
        "mode": result.mode.value,
        "catalog_path": str(result.catalog_path),
        "created": result.created,
        "replaced": result.replaced,
        "unchanged": result.unchanged,
        "backup_path": str(result.backup_path) if result.backup_path else None,
        "manifest_sha256": result.manifest_sha256,
        "previous_sha256": result.previous_sha256,
        "lock_used": result.lock_used,
        "atomic_replace_used": result.atomic_replace_used,
        "post_write_validation": result.post_write_validation,
        "rollback_performed": result.rollback_performed,
        "files_written": result.files_written,
        "warnings": [issue.code for issue in result.warnings],
        "errors": [issue.code for issue in result.errors],
        "telemetry": dict(result.telemetry),
    }


def _error_payload(error: CatalogLibraryAdoptionError | None) -> dict[str, Any] | None:
    if error is None:
        return None
    return {
        "code": error.code,
        "message": str(error),
        "result": _result_payload(error.result),
    }


def _markdown(report: dict[str, Any]) -> str:
    plan = report["plan"]
    lines = [
        "# CatalogLibrary Adoption Report",
        "",
        f"- Mode: `{report['mode']}`",
        f"- Plan status: `{plan['status']}`",
        f"- Sources: `{len(plan['sources'])}`",
        f"- Indexes: `{len(plan['indexes'])}`",
        f"- Coverage: `{plan['coverage']['status']}` all_sky=`{plan['coverage']['all_sky']}` tiles=`{plan['coverage']['covered_tiles']}`/`{plan['coverage']['total_tiles']}`",
        f"- Commit: `{report['commit']['status'] if report['commit'] else 'not requested'}`",
        f"- Error: `{report['error']['code'] if report['error'] else 'none'}`",
        "",
    ]
    if plan["warnings"]:
        lines.append("## Warnings")
        lines.extend(f"- `{code}`" for code in plan["warnings"])
        lines.append("")
    if plan["repair_actions"]:
        lines.append("## Repair Actions")
        lines.extend(f"- `{item['code']}` for `{item['resource_id']}` ({item['execution_phase']})" for item in plan["repair_actions"])
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
