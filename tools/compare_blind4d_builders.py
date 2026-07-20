#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

if __package__ is None:
    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from zeblindsolver.astap_4d_builder import Astap4DBuildConfig, build_4d_index_from_astap
from zeblindsolver.quad_index_4d import Quad4DIndex, build_experimental_4d_index, compare_4d_indexes, scientific_payload_fingerprint


def _ensure_output_dir(path: Path, *, allow_non_empty: bool) -> Path:
    out = path.expanduser().resolve()
    if out.exists():
        if not out.is_dir():
            raise RuntimeError(f"out_dir_not_directory: {out}")
        if any(out.iterdir()) and not allow_non_empty:
            raise RuntimeError(f"out_dir_not_empty: {out}")
    else:
        out.mkdir(parents=True)
    return out


def _family_from_tiles(tile_keys: list[str]) -> str:
    families = {key.split("_", 1)[0] for key in tile_keys if "_" in key}
    if len(families) != 1:
        raise RuntimeError(f"tile_family_ambiguous: {tile_keys}")
    return next(iter(families))


def _write_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is not None:
        path.expanduser().resolve().write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_markdown(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    lines = [
        "# Blind 4D Builder Comparison",
        "",
        f"- status: `{payload['status']}`",
        f"- tile_keys: `{', '.join(payload['tile_keys'])}`",
        f"- direct_index: `{payload['direct_index']}`",
        f"- direct_fingerprint: `{payload['direct_fingerprint']}`",
    ]
    for label, report in payload.get("comparisons", {}).items():
        lines.extend(
            [
                "",
                f"## {label}",
                "",
                f"- exact: `{report.get('exact')}`",
                f"- left_fingerprint: `{report.get('left_fingerprint')}`",
                f"- right_fingerprint: `{report.get('right_fingerprint')}`",
            ]
        )
        for name, array_report in report.get("arrays", {}).items():
            if array_report.get("equal"):
                continue
            lines.append(f"- divergence `{name}`: {array_report}")
    path.expanduser().resolve().write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare historical and direct ASTAP Blind 4D builders")
    parser.add_argument("--astap-root", required=True, help="ASTAP/HNSKY source root, read-only")
    parser.add_argument("--legacy-index-root", help="Historical ZeBlind index root with manifest.json and tiles/*.npz")
    parser.add_argument("--existing-4d-index", help="Existing Blind 4D NPZ to compare against")
    parser.add_argument("--tile-key", action="append", required=True, help="Tile key to build, e.g. d50_2822; repeatable")
    parser.add_argument("--out-dir", required=True, help="Output directory for generated comparison artifacts")
    parser.add_argument("--report-json", help="Optional JSON report path")
    parser.add_argument("--report-md", help="Optional Markdown report path")
    parser.add_argument("--allow-non-empty-out-dir", action="store_true", help="Allow writing into a non-empty output directory")
    parser.add_argument("--mag-cap", type=float, default=Astap4DBuildConfig().mag_cap)
    parser.add_argument("--source-max-stars", type=int, default=2000)
    parser.add_argument("--source-star-truncation-mode", choices=("native_prefix", "brightest_mag"), default="native_prefix")
    parser.add_argument("--max-stars-per-tile", type=int, default=400)
    parser.add_argument("--max-quads-per-tile", type=int, default=8000)
    parser.add_argument("--sampler-tag", default="catalog_ring_coverage")
    parser.add_argument("--dtype", default="float32")
    args = parser.parse_args(argv)

    try:
        tile_keys = [str(v) for v in args.tile_key]
        out_dir = _ensure_output_dir(Path(args.out_dir), allow_non_empty=bool(args.allow_non_empty_out_dir))
        config = Astap4DBuildConfig(
            family=_family_from_tiles(tile_keys),
            tile_keys=tuple(tile_keys),
            mag_cap=float(args.mag_cap),
            source_max_stars=int(args.source_max_stars),
            source_star_truncation_mode=str(args.source_star_truncation_mode),
            max_stars_per_tile=int(args.max_stars_per_tile),
            max_quads_per_tile=int(args.max_quads_per_tile),
            sampler_tag=str(args.sampler_tag),
            dtype=str(args.dtype),
        )
        started = time.perf_counter()
        direct_path = out_dir / "direct_astap_4d.npz"
        build_4d_index_from_astap(args.astap_root, direct_path, config=config)
        elapsed = time.perf_counter() - started
        direct_index = Quad4DIndex.load(direct_path)
        comparisons: dict[str, Any] = {}
        if args.legacy_index_root:
            historical_path = out_dir / "historical_tile_npz_4d.npz"
            build_experimental_4d_index(
                args.legacy_index_root,
                historical_path,
                tile_keys=tile_keys,
                max_stars_per_tile=config.max_stars_per_tile,
                max_quads_per_tile=config.max_quads_per_tile,
                sampler_tag=config.sampler_tag,
                dtype=config.dtype,
            )
            comparisons["historical_vs_direct"] = compare_4d_indexes(historical_path, direct_path)
        if args.existing_4d_index:
            comparisons["existing_vs_direct"] = compare_4d_indexes(args.existing_4d_index, direct_path)
        status = "EXACT" if comparisons and all(report.get("exact") for report in comparisons.values()) else "BUILT"
        if comparisons and not all(report.get("exact") for report in comparisons.values()):
            status = "DIVERGENT"
        payload = {
            "status": status,
            "tile_keys": tile_keys,
            "astap_root": str(Path(args.astap_root).expanduser().resolve()),
            "legacy_index_root": str(Path(args.legacy_index_root).expanduser().resolve()) if args.legacy_index_root else None,
            "existing_4d_index": str(Path(args.existing_4d_index).expanduser().resolve()) if args.existing_4d_index else None,
            "out_dir": str(out_dir),
            "direct_index": str(direct_path.resolve()),
            "direct_fingerprint": scientific_payload_fingerprint(direct_path),
            "direct_metadata": direct_index.metadata,
            "elapsed_s": elapsed,
            "config": asdict(config),
            "comparisons": comparisons,
        }
        _write_json(Path(args.report_json) if args.report_json else None, payload)
        _write_markdown(Path(args.report_md) if args.report_md else None, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if status != "DIVERGENT" else 2
    except Exception as exc:
        print(f"compare_blind4d_builders_failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
