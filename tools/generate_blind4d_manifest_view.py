"""Generate or preview a CatalogLibrary-owned Blind 4D manifest view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from zesolver.catalog_library import CatalogBlind4DManifestViewError, build_blind4d_manifest_view


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-library", required=True, help="ZeSolver CatalogLibrary root")
    parser.add_argument("--output", help="Target strict manifest JSON path")
    parser.add_argument("--write", action="store_true", help="Actually materialize --output")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output file")
    parser.add_argument("--report-json", help="Optional JSON report path")
    return parser


def _report_payload(view, *, materialized_path: Path | None) -> dict[str, object]:
    return {
        "status": "READY" if view.valid else "INVALID",
        "materialized_path": str(materialized_path) if materialized_path is not None else None,
        "entry_count": len(view.entries),
        "source_library_id": view.source_library_id,
        "fingerprint": view.fingerprint,
        "telemetry": dict(view.telemetry),
        "warnings": [issue.code for issue in view.warnings],
        "errors": [issue.code for issue in view.errors],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    view = build_blind4d_manifest_view(Path(args.catalog_library))
    materialized_path: Path | None = None
    if args.output and args.write:
        try:
            materialized_path = view.materialize(args.output, overwrite=bool(args.overwrite))
        except CatalogBlind4DManifestViewError as exc:
            if args.report_json:
                payload = _report_payload(view, materialized_path=None)
                payload["status"] = "FAILED"
                payload["error"] = exc.code
                Path(args.report_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return 1
    if args.report_json:
        Path(args.report_json).write_text(
            json.dumps(_report_payload(view, materialized_path=materialized_path), indent=2),
            encoding="utf-8",
        )
    return 0 if view.valid else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
