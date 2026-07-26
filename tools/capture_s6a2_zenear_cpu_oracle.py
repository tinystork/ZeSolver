#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zeblindsolver.metadata_solver import astap_iso_image_for_solve, detect_stars_astap_strict


def _iter_paths(items: list[str], limit: int | None) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        path = Path(item).expanduser()
        if path.is_dir():
            paths.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in {".fit", ".fits", ".fts"}))
        elif path.is_file():
            paths.append(path)
    return paths[:limit] if limit else paths


def _source_signature(stars: np.ndarray) -> str:
    payload = np.column_stack((stars["x"], stars["y"], stars["flux"])).astype(np.float32, copy=False)
    return hashlib.sha256(payload.tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture CPU strict ZeNear detection snapshots for S6A-2 parity checks.")
    parser.add_argument("inputs", nargs="+", help="FITS files or directories")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-sources", type=int, default=500)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in _iter_paths(args.inputs, args.limit):
        with fits.open(path, memmap=False) as hdul:
            image = astap_iso_image_for_solve(hdul[0])
        result = detect_stars_astap_strict(image, backend="cpu", bin_factor=2, max_stars=args.max_sources)
        sample = [
            {
                "rank": int(i),
                "x": float(result.sources["x"][i]),
                "y": float(result.sources["y"][i]),
                "flux": float(result.sources["flux"][i]),
            }
            for i in range(min(int(result.sources.size), int(args.max_sources)))
        ]
        rows.append(
            {
                "path": str(path),
                "source_count": int(result.sources.size),
                "source_signature_sha256": _source_signature(result.sources),
                "duration_s": float(result.duration_total_s),
                "backend_requested": result.backend_requested,
                "backend_selected": result.backend_selected,
                "backend_used": result.backend_used,
                "sources": sample,
            }
        )
    payload = {"oracle": "near_astap_iso_strict=True near_detect_backend=cpu", "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "count": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
