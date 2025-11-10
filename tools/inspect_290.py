#!/usr/bin/env python3
"""
Binary inspection helper for ASTAP/HNSKY 1476/290 catalogues.

Example:
    python tools/inspect_290.py --db ./database --family d50 --limit 2 --dump-records 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import logging
import math
from typing import Iterable, List, Sequence

import numpy as np

from zewcs290 import CatalogDB, CatalogTile


def _format_segments(segments: Sequence[Sequence[float]]) -> str:
    if len(segments) == 1 and abs(segments[0][0]) < 1e-6 and abs(segments[0][1] - 360.0) < 1e-6:
        return "[0, 360)°"
    parts = [f"[{start:.2f}, {end:.2f})°" for start, end in segments]
    return ", ".join(parts)


def _select_tiles(tiles: Sequence[CatalogTile], limit: int | None, families: Sequence[str] | None) -> List[CatalogTile]:
    fam_filter = {fam.lower() for fam in families} if families else None
    picked: List[CatalogTile] = []
    per_family: dict[str, int] = {}
    for tile in tiles:
        if fam_filter and tile.spec.key not in fam_filter:
            continue
        if limit is not None and per_family.get(tile.spec.key, 0) >= limit:
            continue
        picked.append(tile)
        per_family[tile.spec.key] = per_family.get(tile.spec.key, 0) + 1
    return picked


def _sample_rows(block_stars: np.ndarray, limit: int) -> List[dict]:
    if block_stars.size == 0 or limit <= 0:
        return []
    if block_stars.size <= limit:
        indices = range(block_stars.size)
    else:
        indices = np.linspace(0, block_stars.size - 1, num=limit, dtype=int)
    samples = []
    for idx in indices:
        samples.append(
            {
                "ra_deg": float(block_stars["ra_deg"][idx]),
                "dec_deg": float(block_stars["dec_deg"][idx]),
                "mag": float(block_stars["mag"][idx]),
                "bp_rp": float(block_stars["bp_rp"][idx]),
            }
        )
    return samples


def inspect(
    db_root: Path,
    families: Sequence[str] | None,
    limit: int | None,
    dump_records: int,
    json_path: Path | None,
    cache_size: int,
) -> None:
    db = CatalogDB(db_root, families=families, cache_size=cache_size)
    summary = db.describe()
    family_info = ", ".join(f"{fam}:{count}" for fam, count in sorted(summary.items()))
    if not family_info:
        family_info = "none"
    ring_indices = sorted({tile.ring_index for tile in db.tiles})
    logging.info(
        "Loaded %d tile(s) across %d family(ies): %s",
        sum(summary.values()),
        len(summary),
        family_info,
    )
    if ring_indices:
        logging.info(
            "Rings covered: %d (min %02d, max %02d)",
            len(ring_indices),
            ring_indices[0],
            ring_indices[-1],
        )
    tiles = _select_tiles(db.tiles, limit=limit, families=families)
    if not tiles:
        print("No tiles matched your filters.")
        return
    json_records: List[dict] = []
    for tile in tiles:
        block = db._load_tile(tile)  # pylint: disable=protected-access
        info = {
            "file": str(tile.path),
            "family": tile.spec.key,
            "family_title": tile.spec.title,
            "format": tile.spec.format_name,
            "record_size": tile.spec.record_size,
            "header_records": block.header_records,
            "star_count": block.star_count,
            "description": block.description,
            "ra_segments": [list(seg) for seg in tile.bounds.ra_segments],
            "dec_range": [tile.bounds.dec_min, tile.bounds.dec_max],
            "mag_range": list(block.mag_range),
            "bp_rp_available": tile.spec.has_color,
            "sample_stars": _sample_rows(block.stars, dump_records),
        }
        json_records.append(info)
        print(f"{tile.path.name} — {tile.spec.title} ({tile.spec.format_name})")
        print(f"  Stars: {block.star_count:,} (header records: {block.header_records:,})")
        print(f"  RA tiles: {_format_segments(tile.bounds.ra_segments)}  DEC: [{tile.bounds.dec_min:.2f}, {tile.bounds.dec_max:.2f}]°")
        mag_min, mag_max = block.mag_range
        if not math.isnan(mag_min):
            print(f"  Magnitude range: {mag_min:.2f} … {mag_max:.2f} ({tile.spec.magnitude_band})")
        if block.stars.size:
            sample = info["sample_stars"]
            if sample:
                print("  Sample stars:")
                for row in sample:
                    color = "" if math.isnan(row["bp_rp"]) else f", bp-rp {row['bp_rp']:.2f}"
                    print(
                        f"    RA {row['ra_deg']:.6f}°  DEC {row['dec_deg']:.6f}°  mag {row['mag']:.2f}{color}"
                    )
        print("")
    if json_path:
        json_path.write_text(json.dumps(json_records, indent=2))
        print(f"JSON report saved to {json_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path, help="directory containing .1476/.290 files")
    parser.add_argument("--family", action="append", help="restrict to specific families (e.g. d50, g05)")
    parser.add_argument("--limit", type=int, help="limit the number of files per family to inspect")
    parser.add_argument("--dump-records", type=int, default=5, help="number of sample stars per tile")
    parser.add_argument("--json", type=Path, help="optional path to write the JSON summary")
    parser.add_argument("--cache-size", type=int, default=8, help="LRU cache size for decoded tiles (default: 8)")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: INFO",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(log_level, int):
        parser.error(f"invalid log level {args.log_level!r}")
    logging.basicConfig(level=log_level, format="%(levelname)s %(message)s")
    inspect(
        db_root=args.db,
        families=args.family,
        limit=args.limit,
        dump_records=args.dump_records,
        json_path=args.json,
        cache_size=args.cache_size,
    )


if __name__ == "__main__":
    main()
