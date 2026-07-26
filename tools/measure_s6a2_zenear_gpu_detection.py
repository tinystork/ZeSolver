#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zeblindsolver.metadata_solver import astap_iso_image_for_solve, detect_stars_astap_strict, reset_zenear_gpu_runtime_state


def _paths(args: argparse.Namespace) -> list[Path]:
    out: list[Path] = []
    for item in args.inputs:
        path = Path(item).expanduser()
        if path.is_dir():
            out.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in {".fit", ".fits", ".fts"}))
        elif path.is_file():
            out.append(path)
    if args.limit:
        out = out[: int(args.limit)]
    return out


def _synthetic() -> np.ndarray:
    image = np.ones((220, 220), dtype=np.float32) * 1000.0
    yy, xx = np.ogrid[:220, :220]
    for y, x, amp in ((60, 70, 5000), (120, 130, 7000), (160, 40, 6000), (80, 170, 6500)):
        image += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * 2.0**2)).astype(np.float32)
    return image


def _load_image(path: Path | None) -> np.ndarray:
    if path is None:
        return _synthetic()
    with fits.open(path, memmap=False) as hdul:
        return astap_iso_image_for_solve(hdul[0])


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "p95": None, "total": 0.0}
    ordered = sorted(values)
    p95_idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {"median": statistics.median(values), "p95": ordered[p95_idx], "total": sum(values)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure S6A-2 ZeNear strict detection CPU/CUDA timings.")
    parser.add_argument("inputs", nargs="*", help="FITS files or directories. Empty uses a synthetic field.")
    parser.add_argument("--backend", choices=["cpu", "cuda", "auto"], action="append", default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--gpu-slots", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    paths = _paths(args)
    images: list[tuple[str, np.ndarray]] = []
    if paths:
        for path in paths:
            images.append((str(path), _load_image(path)))
    else:
        images.append(("synthetic", _load_image(None)))
    backends = args.backend or ["cpu", "cuda", "auto"]
    report: dict[str, object] = {"inputs": [name for name, _image in images], "runs": []}

    for backend in backends:
        reset_zenear_gpu_runtime_state()
        rows = []
        for rep in range(max(1, int(args.repeat))):
            for name, image in images:
                t0 = time.perf_counter()
                result = detect_stars_astap_strict(
                    image,
                    backend=backend,
                    device=args.device,
                    gpu_slots=max(1, int(args.gpu_slots)),
                    bin_factor=2,
                    max_stars=500,
                    hfd_min=0.8,
                )
                wall = time.perf_counter() - t0
                rows.append(
                    {
                        "input": name,
                        "repeat": rep,
                        "sources": int(result.sources.size),
                        "backend_requested": result.backend_requested,
                        "backend_selected": result.backend_selected,
                        "backend_used": result.backend_used,
                        "fallback_used": result.fallback_used,
                        "fallback_reason": result.fallback_reason,
                        "duration_total_s": result.duration_total_s,
                        "wall_s": wall,
                        "transfer_to_gpu_s": result.duration_transfer_to_gpu_s,
                        "gpu_compute_s": result.duration_gpu_compute_s,
                        "transfer_to_cpu_s": result.duration_transfer_to_cpu_s,
                        "cpu_compute_s": result.duration_cpu_compute_s,
                        "gpu_slot_wait_s": result.gpu_slot_wait_s,
                        "vram_peak_bytes": result.vram_peak_bytes,
                        "cupy_pool_reserved_bytes": result.cupy_pool_reserved_bytes,
                    }
                )
        report["runs"].append(
            {
                "backend": backend,
                "count": len(rows),
                "fallbacks": sum(1 for row in rows if row["fallback_used"]),
                "backend_used": sorted({str(row["backend_used"]) for row in rows}),
                "wall_s": _summary([float(row["wall_s"]) for row in rows]),
                "detect_total_s": _summary([float(row["duration_total_s"]) for row in rows]),
                "transfer_to_gpu_s": _summary([float(row["transfer_to_gpu_s"]) for row in rows]),
                "gpu_compute_s": _summary([float(row["gpu_compute_s"]) for row in rows]),
                "transfer_to_cpu_s": _summary([float(row["transfer_to_cpu_s"]) for row in rows]),
                "rows": rows,
            }
        )

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
