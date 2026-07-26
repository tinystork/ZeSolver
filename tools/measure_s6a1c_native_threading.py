#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import resource
import statistics
import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]

THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def main() -> int:
    args = _parse_args()
    if args.worker_run:
        print("S6A1C_JSON:" + json.dumps(_run_worker(args), sort_keys=True))
        return 0
    rows = _run_controller(args)
    payload = {
        "tool": "measure_s6a1c_native_threading",
        "machine": _machine_info(),
        "environment": _thread_environment(),
        "input_count": len(_input_paths(args)),
        "inputs_sample": [str(path) for path in _input_paths(args)[:10]],
        "runs": rows,
        "summaries": _summaries(rows),
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


def _run_controller(args: argparse.Namespace) -> list[dict[str, object]]:
    paths = _input_paths(args)
    if not paths:
        raise SystemExit("no FITS files selected")
    rows: list[dict[str, object]] = []
    combos = [(workers, native) for workers in args.workers for native in args.native_threads]
    for repeat in range(max(1, int(args.repeat))):
        ordered = _permuted(combos, repeat)
        for workers, native in ordered:
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-run",
                "--workers",
                str(workers),
                "--native-threads",
                str(native),
                "--native-user-api",
                str(args.native_user_api),
                "--max-files",
                str(args.max_files),
                "--max-loadavg1",
                str(args.max_loadavg1),
                "--catalog-library",
                str(args.catalog_library),
                "--repeat-index",
                str(repeat),
                *[str(path) for path in paths],
            ]
            if args.stub_near:
                cmd.append("--stub-near")
            if args.stop_after_results is not None:
                cmd.extend(["--stop-after-results", str(args.stop_after_results)])
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, text=True, capture_output=True)
            if proc.returncode != 0:
                rows.append(
                    {
                        "repeat": repeat,
                        "workers": workers,
                        "native_threads": native,
                        "native_user_api": args.native_user_api,
                        "valid": False,
                        "error": "subprocess_failed",
                        "returncode": proc.returncode,
                        "stdout": proc.stdout[-4000:],
                        "stderr": proc.stderr[-4000:],
                    }
                )
                continue
            parsed = _parse_worker_stdout(proc.stdout)
            if parsed is None:
                rows.append(
                    {
                        "repeat": repeat,
                        "workers": workers,
                        "native_threads": native,
                        "native_user_api": args.native_user_api,
                        "valid": False,
                        "error": "worker_json_missing",
                        "returncode": proc.returncode,
                        "stdout": proc.stdout[-4000:],
                        "stderr": proc.stderr[-4000:],
                    }
                )
                continue
            rows.append(parsed)
    return rows


def _parse_worker_stdout(stdout: str) -> dict[str, object] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith("S6A1C_JSON:"):
            return json.loads(line.split(":", 1)[1])
    return None


def _run_worker(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    threadpool = _ThreadpoolProbe()
    threadpool.capture("cold")
    import numpy  # noqa: F401

    threadpool.capture("after_numpy")
    try:
        import scipy  # noqa: F401

        threadpool.capture("after_scipy")
    except Exception as exc:
        threadpool.capture_error("after_scipy", exc)
    try:
        import skimage  # noqa: F401

        threadpool.capture("after_skimage")
    except Exception as exc:
        threadpool.capture_error("after_skimage", exc)

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from zesolver.catalog_resources import resolve_catalog_resources
    from zesolver.gui_pipeline.pipeline_runner import PipelineGuiRunner
    from zesolver.gui_pipeline.requests import GuiSettingsState
    from zesolver.gui_pipeline.settings_adapter import build_gui_solve_request

    threadpool.capture("after_zesolver_imports")
    paths = _input_paths(args)
    load_before = _loadavg()
    rejected = (
        load_before is not None
        and args.max_loadavg1 is not None
        and float(args.max_loadavg1) >= 0.0
        and float(load_before[0]) > float(args.max_loadavg1)
    )
    if rejected:
        return {
            "repeat": int(args.repeat_index),
            "workers": int(args.workers[0]),
            "native_threads": args.native_threads[0],
            "native_user_api": args.native_user_api,
            "input_count": len(paths),
            "valid": False,
            "rejected_by_load_guard": True,
            "loadavg_before": load_before,
            "max_loadavg1": float(args.max_loadavg1),
            "threadpool_info": threadpool.snapshots,
            "threadpoolctl_version": threadpool.version,
        }

    resources = resolve_catalog_resources(catalog_library=args.catalog_library)
    threadpool.capture("after_near_runtime_resource_resolution")
    emitted: list[tuple[float, str, str]] = []
    progress_events: list[tuple[float, int, str | None]] = []
    cancel_event = threading.Event()

    if args.stub_near:
        _install_stub_near(cancel_event)

    def on_result(result) -> None:
        emitted.append((time.perf_counter() - started, str(result.path), str(result.status)))
        if args.stop_after_results is not None and len(emitted) >= int(args.stop_after_results):
            cancel_event.set()

    def on_progress(progress) -> None:
        progress_events.append((time.perf_counter() - started, int(progress.completed), progress.current_phase))

    state = GuiSettingsState(
        workers=max(1, int(args.workers[0])),
        preserve_order=True,
        use_blind=False,
        catalog_resources=resources,
        catalog_library_path=args.catalog_library,
        legacy_config=object(),
        startup_stagger_ms=0,
    )
    request = build_gui_solve_request(paths, state, cancel_token=cancel_event)
    request = replace(request, product_settings=replace(request.product_settings, gpu_mode="cpu"))

    rss_start = _current_rss_kib()
    usage_start = resource.getrusage(resource.RUSAGE_SELF)
    sampler = _ProcessSampler(interval_s=0.25)
    sampler.start()
    context = native_thread_context(args.native_threads[0], args.native_user_api)
    summary = None
    error = None
    try:
        with context:
            threadpool.capture("during_limit")
            summary = PipelineGuiRunner(progress_callback=on_progress, result_callback=on_result).run(request)
    except Exception as exc:  # pragma: no cover - diagnostic path
        error = f"{type(exc).__name__}: {exc}"
    finally:
        sampler.stop()
    threadpool.capture("after_restore")
    usage_end = resource.getrusage(resource.RUSAGE_SELF)
    duration = time.perf_counter() - started
    rss_end = _current_rss_kib()
    load_after = _loadavg()

    statuses: dict[str, int] = {}
    result_signatures: dict[str, dict[str, object]] = {}
    telemetry: dict[str, object] = {}
    if summary is not None:
        telemetry = dict(summary.telemetry or {})
        for result in summary.results:
            statuses[str(result.status)] = statuses.get(str(result.status), 0) + 1
            result_signatures[str(result.path)] = _result_signature(result)
    scheduler = ((telemetry.get("scheduler") or {}).get("near") or {}) if isinstance(telemetry, dict) else {}
    scheduler_summary = dict(scheduler.get("summary") or {}) if isinstance(scheduler, dict) else {}
    counters = dict((telemetry.get("counters") or {})) if isinstance(telemetry, dict) else {}
    intervals = _result_intervals_ms(emitted)
    process_threads = sampler.thread_stats()
    process_rss = sampler.rss_stats()
    row = {
        "repeat": int(args.repeat_index),
        "workers": int(args.workers[0]),
        "native_threads": args.native_threads[0],
        "native_user_api": args.native_user_api,
        "valid": error is None,
        "error": error,
        "rejected_by_load_guard": False,
        "input_count": len(paths),
        "duration_s": round(duration, 6),
        "first_result_s": round(emitted[0][0], 6) if emitted else None,
        "throughput_images_per_min": round((len(paths) / duration) * 60.0, 3) if duration > 0.0 else None,
        "rss_start_kib": rss_start,
        "rss_end_kib": rss_end,
        "rss_peak_kib": max(_ru_maxrss_kib(), int(process_rss.get("peak", 0) or 0)),
        "rss_samples": process_rss,
        "threads_process_start": sampler.thread_start,
        "threads_process_median": process_threads.get("median"),
        "threads_process_p95": process_threads.get("p95"),
        "threads_process_peak": process_threads.get("peak"),
        "ru_nvcsw_delta": usage_end.ru_nvcsw - usage_start.ru_nvcsw,
        "ru_nivcsw_delta": usage_end.ru_nivcsw - usage_start.ru_nivcsw,
        "ru_utime_delta_s": round(usage_end.ru_utime - usage_start.ru_utime, 6),
        "ru_stime_delta_s": round(usage_end.ru_stime - usage_start.ru_stime, 6),
        "loadavg_before": load_before,
        "loadavg_after": load_after,
        "statuses": statuses,
        "scheduler": scheduler_summary,
        "counters": counters,
        "result_signatures": result_signatures,
        "median_result_interval_ms": _percentile(intervals, 50),
        "p90_result_interval_ms": _percentile(intervals, 90),
        "result_intervals_lt_250ms": sum(1 for value in intervals if value < 250.0),
        "result_pauses_gt_1s": sum(1 for value in intervals if value > 1000.0),
        "result_pauses_gt_2s": sum(1 for value in intervals if value > 2000.0),
        "progress_events_sample": progress_events[:20],
        "threadpoolctl_version": threadpool.version,
        "threadpool_info": threadpool.snapshots,
        "thread_environment": _thread_environment(),
    }
    return row


def native_thread_context(native_threads: str, user_api: str = "all"):
    if str(native_threads).strip().lower() == "default":
        return nullcontext()
    try:
        from threadpoolctl import threadpool_limits
    except Exception as exc:  # pragma: no cover - dependency checked by tests/tool output
        raise SystemExit(
            'threadpoolctl is required for native thread limiting; install with: .venv/bin/python -m pip install "threadpoolctl>=3.6,<4"'
        ) from exc
    limit = int(native_threads)
    api = str(user_api or "all").strip().lower()
    if api == "all":
        return threadpool_limits(limits=limit)
    return threadpool_limits(limits=limit, user_api=api)


class _ThreadpoolProbe:
    def __init__(self) -> None:
        try:
            import threadpoolctl

            self.version = str(threadpoolctl.__version__)
        except Exception:
            self.version = None
        self.snapshots: dict[str, object] = {}

    def capture(self, label: str) -> None:
        try:
            from threadpoolctl import threadpool_info

            self.snapshots[str(label)] = threadpool_info()
        except Exception as exc:
            self.capture_error(label, exc)

    def capture_error(self, label: str, exc: BaseException) -> None:
        self.snapshots[str(label)] = {"unavailable": f"{type(exc).__name__}: {exc}"}


class _ProcessSampler:
    def __init__(self, *, interval_s: float = 0.25) -> None:
        self.interval_s = max(0.05, float(interval_s))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.thread_samples: list[int] = []
        self.rss_samples: list[int] = []
        self.thread_start = _process_threads()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="s6a1c-process-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._sample()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def _sample(self) -> None:
        threads = _process_threads()
        rss = _current_rss_kib()
        if threads is not None:
            self.thread_samples.append(int(threads))
        if rss is not None:
            self.rss_samples.append(int(rss))

    def thread_stats(self) -> dict[str, int | float | None]:
        return _sample_stats(self.thread_samples)

    def rss_stats(self) -> dict[str, int | float | None]:
        return _sample_stats(self.rss_samples)


def _install_stub_near(cancel_event: threading.Event) -> None:
    from zesolver.core.models import SolveStatus

    def fake_near_solve(fits_path, index_root, *, catalog_provider=None, **kwargs):
        del index_root, catalog_provider, kwargs
        for _ in range(5):
            if cancel_event.is_set():
                return {"success": False, "message": "synthetic cancelled", "stats": {}, "wrote_wcs": False}
            time.sleep(0.01)
        return {
            "success": True,
            "status": SolveStatus.SOLVED,
            "message": "synthetic solved",
            "stats": {
                "inliers": 64,
                "rms_px": 0.5,
                "pix_scale_arcsec": 2.39,
                "orientation_deg": 0.0,
            },
            "wrote_wcs": False,
        }

    import zesolver.core.pipeline as pipeline

    pipeline.near_solve = fake_near_solve


def _result_signature(result) -> dict[str, object]:
    return {
        "status": str(result.status),
        "backend": result.backend,
        "inliers": result.inliers,
        "rms_px": _round_float(result.rms_px),
        "pixel_scale_arcsec": _round_float(result.pixel_scale_arcsec),
        "wcs_written": bool(result.wcs_written),
    }


def _input_paths(args: argparse.Namespace) -> tuple[Path, ...]:
    values: list[Path] = []
    for item in getattr(args, "input_dir", None) or ():
        path = Path(item).expanduser()
        values.extend(sorted(path.glob("*.fit")))
        values.extend(sorted(path.glob("*.fits")))
    for item in args.inputs:
        path = Path(item).expanduser()
        if path.is_dir():
            values.extend(sorted(path.glob("*.fit")))
            values.extend(sorted(path.glob("*.fits")))
        else:
            values.append(path)
    unique = tuple(dict.fromkeys(path.resolve() for path in values if path.exists()))
    max_files = int(getattr(args, "max_files", 0) or 0)
    if max_files > 0:
        return unique[:max_files]
    return unique


def _permuted(combos: list[tuple[int, str]], repeat: int) -> list[tuple[int, str]]:
    if not combos:
        return []
    shift = repeat % len(combos)
    rotated = combos[shift:] + combos[:shift]
    if repeat % 2:
        rotated = list(reversed(rotated))
    return rotated


def _summaries(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if not row.get("valid") or row.get("rejected_by_load_guard"):
            continue
        key = f"workers={row.get('workers')};native_threads={row.get('native_threads')};user_api={row.get('native_user_api')}"
        grouped.setdefault(key, []).append(row)
    out: dict[str, dict[str, object]] = {}
    for key, items in grouped.items():
        durations = [_float(row.get("duration_s")) for row in items]
        throughputs = [_float(row.get("throughput_images_per_min")) for row in items]
        out[key] = {
            "valid_runs": len(items),
            "duration_s_median": _percentile([x for x in durations if x is not None], 50),
            "throughput_images_per_min_median": _percentile([x for x in throughputs if x is not None], 50),
            "threads_process_peak_max": max((int(row.get("threads_process_peak") or 0) for row in items), default=None),
            "rss_peak_kib_max": max((int(row.get("rss_peak_kib") or 0) for row in items), default=None),
        }
    return out


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "repeat",
        "workers",
        "native_threads",
        "native_user_api",
        "valid",
        "rejected_by_load_guard",
        "input_count",
        "duration_s",
        "first_result_s",
        "throughput_images_per_min",
        "rss_start_kib",
        "rss_end_kib",
        "rss_peak_kib",
        "threads_process_median",
        "threads_process_p95",
        "threads_process_peak",
        "ru_nvcsw_delta",
        "ru_nivcsw_delta",
        "ru_utime_delta_s",
        "ru_stime_delta_s",
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
        "loadavg": _loadavg(),
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
    return {key: os.environ.get(key) for key in THREAD_ENV_KEYS}


def _loadavg() -> tuple[float, float, float] | None:
    try:
        return tuple(float(x) for x in os.getloadavg())
    except Exception:
        return None


def _process_threads() -> int | None:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("Threads:"):
                    return int(line.split()[1])
    except Exception:
        return None
    return None


def _current_rss_kib() -> int | None:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        return None
    return None


def _ru_maxrss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _sample_stats(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"median": None, "p95": None, "peak": None}
    return {
        "median": _percentile([float(value) for value in values], 50),
        "p95": _percentile([float(value) for value in values], 95),
        "peak": max(values),
    }


def _result_intervals_ms(events: list[tuple[float, str, str]]) -> list[float]:
    return [max(0.0, (events[idx][0] - events[idx - 1][0]) * 1000.0) for idx in range(1, len(events))]


def _percentile(values: list[float], pct: float) -> float | None:
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(float(ordered[0]), 3)
    rank = (len(ordered) - 1) * max(0.0, min(100.0, float(pct))) / 100.0
    low = int(math.floor(rank))
    high = min(len(ordered) - 1, int(math.ceil(rank)))
    if low == high:
        return round(float(ordered[low]), 3)
    weight = rank - low
    return round(float(ordered[low] * (1.0 - weight) + ordered[high] * weight), 3)


def _float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _round_float(value: object) -> float | None:
    value = _float(value)
    return round(value, 9) if value is not None else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure S6A-1C native BLAS/OpenMP threading for ZeNear CPU.")
    parser.add_argument("inputs", nargs="*", help="FITS files or directories")
    parser.add_argument("--input-dir", type=Path, action="append", default=None, help="Directory containing FITS files")
    parser.add_argument("--catalog-library", type=Path, default=Path("/home/tristan/ZeSolverCatalog/new"))
    parser.add_argument("--workers", type=int, action="append", default=None)
    parser.add_argument("--native-threads", action="append", default=None, choices=("default", "1", "2"))
    parser.add_argument("--native-user-api", default="all", choices=("all", "blas", "openmp"))
    parser.add_argument("--max-files", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-loadavg1", type=float, default=2.0)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--stub-near", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repeat-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--stop-after-results", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.workers = tuple(args.workers or (4, 5, 6))
    args.native_threads = tuple(args.native_threads or ("default", "1", "2"))
    return args


if __name__ == "__main__":
    raise SystemExit(main())
