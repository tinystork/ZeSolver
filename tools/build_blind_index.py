#!/usr/bin/env python3
"""CLI helper to pre-compute quad hashes from the ASTAP catalogue."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from zesolver.blindindex import BlindIndex


def _normalize_families(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in values:
        for chunk in str(raw).replace(";", ",").split(","):
            name = chunk.strip().lower()
            if not name or name in seen:
                continue
            seen.add(name)
            normalized.append(name)
    return normalized or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a quad-based blind index from ASTAP tiles.")
    parser.add_argument("--db-root", type=Path, required=True, help="Path to the ASTAP/HNSKY catalogue directory")
    parser.add_argument(
        "--family",
        action="append",
        help="Restrict to one or more catalogue families (e.g. --family d50 --family d20)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where the index will be written (default: <db-root>/.blind_index)",
    )
    parser.add_argument(
        "--max-tile-stars",
        type=int,
        default=40,
        help="Brightest stars to keep per tile before composing quads (default: %(default)s)",
    )
    parser.add_argument(
        "--max-quads-per-tile",
        type=int,
        default=256,
        help="Maximum quads to store per tile (default: %(default)s)",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging verbosity (default: %(default)s)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    families = _normalize_families(args.family)
    output_dir = args.output_dir or args.db_root / ".blind_index"
    output_path = BlindIndex.build_from_catalog(
        db_root=args.db_root,
        families=families,
        output_dir=output_dir,
        max_tile_stars=args.max_tile_stars,
        max_quads_per_tile=args.max_quads_per_tile,
    )
    logging.info("Blind index written to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
