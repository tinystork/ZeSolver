#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zesolver.catalog_resources import resolve_catalog_resources
from zesolver.gui_pipeline.pipeline_runner import PipelineGuiRunner
from zesolver.gui_pipeline.requests import GuiSettingsState
from zesolver.gui_pipeline.settings_adapter import build_gui_solve_request
from zesolver.resource_telemetry import current_rss_kib


def main() -> int:
    args = _parse_args()
    paths = _input_paths(args)
    if not paths:
        raise SystemExit("no FITS files selected")
    resources = resolve_catalog_resources(catalog_library=args.catalog_library)
    rows: list[dict[str, object]] = []
    combos = [(worker, stagger) for worker in args.workers for stagger in args.startup_stagger_ms]
    for repeat in range(max(1, int(args.repeat))):
        for workers, stagger in combos:
            rows.append(_run_once(args, paths, resources, workers=workers, stagger_ms=stagger, repeat=repeat))
    payload = {
        "machine": _machine_info(),
        "environment": _thread_environment(),
        "threadpools": _threadpool_info(),
        "input_count": len(paths),
        "inputs_sample": [str(path) for path in paths[:10]],
        "runs": rows,
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.csv_output:
        _write_csv(args.csv_output, rows)
    if args.trace_output:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        args.trace_output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_once(args, paths: tuple[Path, ...], resources, *, workers: int, stagger_ms: int, repeat: int) -> dict[str, object]:
    emitted: list[tuple[float, str, str]] = []
    progress_events: list[tuple[float, int, str | None]] = []
    started = time.perf_counter()
    rss_start = current_rss_kib()

    def on_result(result) -> None:
        emitted.append((time.perf_counter() - started, str(result.path), str(result.status)))

    def on_progress(progress) -> None:
        progress_events.append((time.perf_counter() - started, int(progress.completed), progress.current_phase))

    state = GuiSettingsState(
        workers=max(1, int(workers)),
        preserve_order=True,
        use_blind=False,
        catalog_resources=resources,
        catalog_library_path=args.catalog_library,
        legacy_config=object(),
        startup_stagger_ms=max(0, int(stagger_ms)),
    )
    request = build_gui_solve_request(paths, state)
    request = replace(request, product_settings=replace(request.product_settings, gpu_mode="cpu"))
    context = _native_thread_context(args.limit_native_threads)
    with context:
        summary = PipelineGuiRunner(progress_callback=on_progress, result_callback=on_result).run(request)
    duration = time.perf_counter() - started
    rss_end = current_rss_kib()
    statuses: dict[str, int] = {}
    for result in summary.results:
        statuses[str(result.status)] = statuses.get(str(result.status), 0) + 1
    scheduler = ((summary.telemetry or {}).get("scheduler") or {}).get("near", {})
    scheduler_summary = dict(scheduler.get("summary") or {}) if isinstance(scheduler, dict) else {}
    counters = dict(((summary.telemetry or {}).get("counters") or {}))
    intervals = _result_intervals_ms(emitted)
    row = {
        "repeat": repeat,
        "workers": int(workers),
        "startup_stagger_ms": int(stagger_ms),
        "limit_native_threads": bool(args.limit_native_threads),
        "input_count": len(paths),
        "duration_s": round(duration, 6),
        "first_result_s": round(emitted[0][0], 6) if emitted else None,
        "throughput_images_per_min": round((len(paths) / duration) * 60.0, 3) if duration > 0.0 else None,
        "rss_start_kib": rss_start,
        "rss_end_kib": rss_end,
        "rss_peak_kib": _ru_maxrss_kib(),
        "statuses": statuses,
        "scheduler": scheduler_summary,
        "counters": counters,
        "median_result_interval_ms": _percentile(intervals, 50),
        "p90_result_interval_ms": _percentile(intervals, 90),
        "result_intervals_lt_250ms": sum(1 for value in intervals if value < 250.0),
        "result_pauses_gt_1s": sum(1 for value in intervals if value > 1000.0),
        "result_pauses_gt_2s": sum(1 for value in intervals if value > 2000.0),
        "progress_events_sample": progress_events[:20],
    }
    return row


def _native_thread_context(enabled: bool):
    if not enabled:
        return nullcontext()
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        return nullcontext()
    return threadpool_limits(limits=1)


def _input_paths(args: argparse.Namespace) -> tuple[Path, ...]:
    values: list[Path] = []
    for item in args.inputs:
        path = Path(item).expanduser()
        if path.is_dir():
            values.extend(sorted(path.glob("*.fit")))
            values.extend(sorted(path.glob("*.fits")))
        else:
            values.append(path)
    unique = tuple(dict.fromkeys(path.resolve() for path in values if path.exists()))
    if args.max_files and args.max_files > 0:
        return unique[: int(args.max_files)]
    return unique


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "repeat",
        "workers",
        "startup_stagger_ms",
        "limit_native_threads",
        "input_count",
        "duration_s",
        "first_result_s",
        "throughput_images_per_min",
        "rss_start_kib",
        "rss_end_kib",
        "rss_peak_kib",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def _machine_info() -> dict[str, object]:
    return {
        "cpu_logical": os.cpu_count(),
        "cpu_physical_estimate": _physical_cpu_count(),
        "loadavg": os.getloadavg() if hasattr(os, "getloadavg") else None,
    }


def _physical_cpu_count() -> int | None:
    try:
        physical_ids: set[tuple[str, str]] = set()
        physical = core = None
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("physical id"):
                physical = line.split(":", 1)[1].strip()
            elif line.startswith("core id"):
                core = line.split(":", 1)[1].strip()
            elif not line.strip() and physical is not None and core is not None:
                physical_ids.add((physical, core))
                physical = core = None
        return len(physical_ids) or None
    except Exception:
        return None


def _thread_environment() -> dict[str, str | None]:
    keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    return {key: os.environ.get(key) for key in keys}


def _threadpool_info() -> object:
    try:
        from threadpoolctl import threadpool_info

        return threadpool_info()
    except Exception as exc:
        return {"unavailable": str(exc)}


def _result_intervals_ms(events: list[tuple[float, str, str]]) -> list[float]:
    return [max(0.0, (events[idx][0] - events[idx - 1][0]) * 1000.0) for idx in range(1, len(events))]


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(float(ordered[0]), 3)
    rank = (len(ordered) - 1) * max(0.0, min(100.0, float(pct))) / 100.0
    low = int(rank)
    high = min(len(ordered) - 1, low + 1)
    weight = rank - low
    return round(float(ordered[low] * (1.0 - weight) + ordered[high] * weight), 3)


def _ru_maxrss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure S6A-1B ZeNear CPU scheduler continuity.")
    parser.add_argument("inputs", nargs="+", help="FITS files or directories")
    parser.add_argument("--catalog-library", type=Path, default=Path("/home/tristan/ZeSolverCatalog/new"))
    parser.add_argument("--workers", type=int, action="append", default=None)
    parser.add_argument("--startup-stagger-ms", type=int, action="append", default=None)
    parser.add_argument("--max-files", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--limit-native-threads", action="store_true")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--trace-output", type=Path)
    args = parser.parse_args()
    args.workers = tuple(args.workers or (4, 5, 6))
    args.startup_stagger_ms = tuple(args.startup_stagger_ms or (0,))
    return args


if __name__ == "__main__":
    raise SystemExit(main())
