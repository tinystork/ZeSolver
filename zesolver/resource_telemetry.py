from __future__ import annotations

import contextvars
import gc
import json
import logging
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path


BATCH_COUNTER_KEYS = (
    "catalog_resource_resolution_count",
    "catalog_library_open_count",
    "near_runtime_resolution_count",
    "blind_runtime_resolution_count",
    "blind_index_payload_load_count",
    "blind_kdtree_build_count",
    "solver_pipeline_constructor_count",
    "near_port_constructor_count",
    "blind_port_constructor_count",
    "catalog_provider_constructor_count",
    "worker_thread_count",
    "near_catalog_runtime_created",
    "near_catalog_runtime_reused",
    "near_catalog_runtime_closed",
    "near_catalog_inventory_load_count",
    "near_catalog_provider_created",
    "near_catalog_provider_reused",
    "near_catalog_db_created",
    "near_catalog_db_reused",
    "near_catalog_payload_cache_hits",
    "near_catalog_payload_cache_misses",
    "near_catalog_payload_cache_evictions",
    "near_catalog_payload_physical_loads",
    "near_catalog_payload_duplicate_loads",
    "near_catalog_payload_singleflight_waiters",
    "near_detect_backend_requested_cpu",
    "near_detect_backend_requested_cuda",
    "near_detect_backend_requested_auto",
    "near_detect_backend_selected_cpu",
    "near_detect_backend_selected_cuda",
    "near_detect_backend_used_cpu",
    "near_detect_backend_used_cuda",
    "near_detect_gpu_fallbacks",
    "near_detect_gpu_oom",
    "near_detect_gpu_errors",
    "near_detect_gpu_disabled_for_batch",
)


IMPORTANT_EVENT_PHASES = {
    "near_batch_runtime_created",
    "near_catalog_runtime_created",
    "near_batch_runtime_ready",
    "near_catalog_runtime_closed",
    "near_detection_active",
    "near_detection_fallback",
    "batch_complete",
}


_active_batch_telemetry: contextvars.ContextVar["BatchResourceTelemetry | None"] = contextvars.ContextVar(
    "zesolver_active_batch_telemetry",
    default=None,
)


@dataclass(slots=True)
class BatchResourceTelemetry:
    counters: dict[str, int] = field(default_factory=lambda: {key: 0 for key in BATCH_COUNTER_KEYS})
    rss_kib: dict[str, int | None] = field(default_factory=dict)
    events: list[dict[str, object]] = field(default_factory=list)
    scheduler: dict[str, dict[str, object]] = field(default_factory=dict)
    _worker_threads: set[int] = field(default_factory=set)
    _current_tasks: dict[int, tuple[str, int]] = field(default_factory=dict)
    _detect_windows: dict[tuple[str, int], dict[str, float]] = field(default_factory=dict)
    _near_detection: dict[str, object] = field(default_factory=lambda: _empty_near_detection())
    _cancel_requested_at: float | None = None
    _cancel_requested_wall: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def increment(self, key: str, amount: int = 1) -> None:
        if key not in self.counters:
            return
        with self._lock:
            self.counters[key] = int(self.counters.get(key, 0)) + int(amount)

    def note_worker_thread(self, ident: int | None = None) -> None:
        value = int(ident if ident is not None else threading.get_ident())
        with self._lock:
            self._worker_threads.add(value)
            self.counters["worker_thread_count"] = len(self._worker_threads)

    def mark_rss(self, label: str) -> int | None:
        value = current_rss_kib()
        with self._lock:
            self.rss_kib[label] = value
        return value

    def diagnostic_gc(self) -> int | None:
        gc.collect()
        return self.mark_rss("after_diagnostic_gc")

    def event(self, phase: str, **payload: object) -> None:
        item = {"phase": str(phase), "t_s": round(time.perf_counter(), 6)}
        item.update(payload)
        with self._lock:
            if len(self.events) < 128:
                self.events.append(item)
            elif str(phase) in IMPORTANT_EVENT_PHASES:
                for pos, existing in enumerate(self.events):
                    if str(existing.get("phase")) not in IMPORTANT_EVENT_PHASES:
                        self.events[pos] = item
                        break

    def mark_cancel_requested(self, *, source: str = "unknown") -> float:
        now = time.perf_counter()
        wall = _utc_now_iso()
        with self._lock:
            if self._cancel_requested_at is None:
                self._cancel_requested_at = now
                self._cancel_requested_wall = wall
                self.events.append({"phase": "cancel_requested", "t_s": round(now, 6), "source": str(source)})
            return float(self._cancel_requested_at)

    def cancel_requested_at(self) -> float | None:
        with self._lock:
            return self._cancel_requested_at

    def record_near_detection(self, record: Mapping[str, object]) -> dict[str, object]:
        now = time.perf_counter()
        payload = dict(record)
        payload.setdefault("recorded_at", now)
        with self._lock:
            summary = self._near_detection
            before_count = int(summary.get("image_count", 0) or 0)
            requested = _clean_text(payload.get("backend_requested")) or _clean_text(payload.get("requested")) or "unknown"
            selected = _clean_text(payload.get("backend_selected")) or _clean_text(payload.get("selected")) or "unknown"
            used = _clean_text(payload.get("backend_used")) or _clean_text(payload.get("used")) or "unknown"
            device = _clean_optional_int(payload.get("device_used"))
            fallback = bool(payload.get("fallback_used", False))
            fallback_reason = _clean_text(payload.get("fallback_reason")) if fallback else None

            summary["image_count"] = before_count + 1
            if summary.get("requested") is None:
                summary["requested"] = requested
            if summary.get("selected_initial") is None:
                summary["selected_initial"] = selected
            summary["selected_last"] = selected
            summary["used_last"] = used
            summary["device_last"] = device
            summary["gpu_slots"] = _clean_optional_int(payload.get("gpu_slots")) or summary.get("gpu_slots")

            backends = summary.setdefault("backends_used_counts", {})
            if isinstance(backends, dict):
                backends[used] = int(backends.get(used, 0) or 0) + 1
            if used == "cuda":
                summary["images_cuda"] = int(summary.get("images_cuda", 0) or 0) + 1
            elif used == "cpu":
                summary["images_cpu"] = int(summary.get("images_cpu", 0) or 0) + 1

            if device is not None:
                devices = summary.setdefault("devices_used_counts", {})
                if isinstance(devices, dict):
                    key = str(device)
                    devices[key] = int(devices.get(key, 0) or 0) + 1

            _append_metric(summary, "detect_duration_ms_values", _seconds_to_ms(payload.get("duration_total_s")))
            _append_metric(summary, "gpu_slot_wait_ms_values", _seconds_to_ms(payload.get("gpu_slot_wait_s")))
            summary["transfer_h2d_ms_total"] = float(summary.get("transfer_h2d_ms_total", 0.0) or 0.0) + (_seconds_to_ms(payload.get("duration_transfer_to_gpu_s")) or 0.0)
            summary["gpu_compute_ms_total"] = float(summary.get("gpu_compute_ms_total", 0.0) or 0.0) + (_seconds_to_ms(payload.get("duration_gpu_compute_s")) or 0.0)
            summary["transfer_d2h_ms_total"] = float(summary.get("transfer_d2h_ms_total", 0.0) or 0.0) + (_seconds_to_ms(payload.get("duration_transfer_to_cpu_s")) or 0.0)
            for src_key, dst_key in (
                ("vram_peak_bytes", "vram_peak_bytes"),
                ("cupy_pool_used_bytes", "cupy_pool_used_peak_bytes"),
                ("cupy_pool_reserved_bytes", "cupy_pool_reserved_peak_bytes"),
            ):
                value = _clean_optional_int(payload.get(src_key))
                if value is not None:
                    summary[dst_key] = max(int(summary.get(dst_key, 0) or 0), value)

            if fallback:
                summary["fallbacks"] = int(summary.get("fallbacks", 0) or 0) + 1
                reasons = summary.setdefault("fallback_reasons", {})
                if isinstance(reasons, dict):
                    key = fallback_reason or "unknown"
                    reasons[key] = int(reasons.get(key, 0) or 0) + 1
            if bool(payload.get("gpu_disabled_for_batch")):
                summary["gpu_disabled_for_batch"] = True
                summary["gpu_disabled_reason"] = _clean_text(payload.get("gpu_disabled_reason"))

            if _clean_text(payload.get("backend_used")) == "cpu" and fallback_reason:
                summary["last_fallback_reason"] = fallback_reason
            if _clean_text(payload.get("gpu_error")):
                summary["gpu_errors"] = int(summary.get("gpu_errors", 0) or 0) + 1
            if bool(payload.get("gpu_oom")):
                summary["gpu_oom"] = int(summary.get("gpu_oom", 0) or 0) + 1

            cancel_at = self._cancel_requested_at
            detect_started = _clean_optional_float(payload.get("detect_started_at"))
            gpu_started = _clean_optional_float(payload.get("gpu_section_started_at"))
            gpu_finished = _clean_optional_float(payload.get("gpu_section_finished_at"))
            if cancel_at is not None:
                if detect_started is not None and detect_started > cancel_at:
                    summary["detections_started_after_cancel"] = int(summary.get("detections_started_after_cancel", 0) or 0) + 1
                if gpu_started is not None and gpu_started > cancel_at:
                    summary["gpu_sections_started_after_cancel"] = int(summary.get("gpu_sections_started_after_cancel", 0) or 0) + 1
                if gpu_finished is not None and gpu_started is not None and gpu_started <= cancel_at < gpu_finished:
                    summary["gpu_sections_finished_after_cancel"] = int(summary.get("gpu_sections_finished_after_cancel", 0) or 0) + 1

            sample = {
                "backend_requested": requested,
                "backend_selected": selected,
                "backend_used": used,
                "device_used": device,
                "fallback_used": fallback,
                "duration_ms": _seconds_to_ms(payload.get("duration_total_s")),
            }
            _record_sample(summary, sample)
            transition = before_count == 0 or summary.get("_last_logged_used") != used or fallback
            if transition:
                self.events.append(
                    {
                        "phase": "near_detection_active" if not fallback else "near_detection_fallback",
                        "t_s": round(now, 6),
                        "requested": requested,
                        "selected": selected,
                        "used": used,
                        "device": device,
                        "fallback": fallback,
                        "reason": fallback_reason,
                    }
                )
                summary["_last_logged_used"] = used
        if transition:
            if fallback:
                logging.warning(
                    "ZeNear CUDA fallback: reason=%s continuing_on=cpu gpu_disabled_for_batch=%s",
                    fallback_reason or "unknown",
                    str(bool(payload.get("gpu_disabled_for_batch"))).lower(),
                )
            else:
                logging.info(
                    "ZeNear detection active: requested=%s selected=%s used=%s device=%s",
                    requested,
                    selected,
                    used,
                    device if device is not None else "-",
                )
        return self.near_detection_summary()

    def scheduler_phase(self, phase: str, *, summary: dict[str, object], tasks: tuple[dict[str, object], ...]) -> None:
        with self._lock:
            self.scheduler[str(phase)] = {
                "summary": dict(summary),
                "tasks": tuple(dict(item) for item in tasks),
            }

    def bind_scheduler_task(self, phase: str, index: int, ident: int | None = None) -> None:
        value = int(ident if ident is not None else threading.get_ident())
        with self._lock:
            self._current_tasks[value] = (str(phase), int(index))

    def unbind_scheduler_task(self, ident: int | None = None) -> None:
        value = int(ident if ident is not None else threading.get_ident())
        with self._lock:
            self._current_tasks.pop(value, None)

    def mark_near_detect_started(self, ident: int | None = None) -> None:
        value = int(ident if ident is not None else threading.get_ident())
        now = time.perf_counter()
        with self._lock:
            key = self._current_tasks.get(value)
            if key is not None:
                self._detect_windows.setdefault(key, {})["near_detect_started_at"] = now

    def mark_near_detect_finished(self, ident: int | None = None) -> None:
        value = int(ident if ident is not None else threading.get_ident())
        now = time.perf_counter()
        with self._lock:
            key = self._current_tasks.get(value)
            if key is not None:
                self._detect_windows.setdefault(key, {})["near_detect_finished_at"] = now

    def near_detection_summary(self, *, terminal_status: str | None = None) -> dict[str, object]:
        with self._lock:
            raw = dict(self._near_detection)
            cancel_requested_at = self._cancel_requested_at
            cancel_requested_wall = self._cancel_requested_wall
        return _near_detection_public_summary(raw, cancel_requested_at=cancel_requested_at, cancel_requested_wall=cancel_requested_wall, terminal_status=terminal_status)

    def detect_window(self, phase: str, index: int) -> dict[str, float]:
        with self._lock:
            return dict(self._detect_windows.get((str(phase), int(index)), {}))

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "counters": dict(self.counters),
                "rss_kib": dict(self.rss_kib),
                "events": tuple(dict(item) for item in self.events),
                "scheduler": {key: dict(value) for key, value in self.scheduler.items()},
                "near_detection": _near_detection_public_summary(
                    dict(self._near_detection),
                    cancel_requested_at=self._cancel_requested_at,
                    cancel_requested_wall=self._cancel_requested_wall,
                    terminal_status=None,
                ),
                "cancellation": {
                    "requested": self._cancel_requested_at is not None,
                    "requested_at": self._cancel_requested_wall,
                    "requested_monotonic_s": self._cancel_requested_at,
                    "detections_started_after_cancel": int(self._near_detection.get("detections_started_after_cancel", 0) or 0),
                    "gpu_sections_started_after_cancel": int(self._near_detection.get("gpu_sections_started_after_cancel", 0) or 0),
                },
            }


def current_rss_kib() -> int | None:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])
    except OSError:
        return None
    except Exception:
        return None
    return None


def active_batch_telemetry() -> BatchResourceTelemetry | None:
    return _active_batch_telemetry.get()


def set_active_batch_telemetry(telemetry: BatchResourceTelemetry | None):
    return _active_batch_telemetry.set(telemetry)


def reset_active_batch_telemetry(token) -> None:
    _active_batch_telemetry.reset(token)


def increment_batch_counter(key: str, amount: int = 1) -> None:
    telemetry = active_batch_telemetry()
    if telemetry is not None:
        telemetry.increment(key, amount)


def record_batch_event(phase: str, **payload: object) -> None:
    telemetry = active_batch_telemetry()
    if telemetry is not None:
        telemetry.event(phase, **payload)


def record_near_detection(payload: Mapping[str, object]) -> None:
    telemetry = active_batch_telemetry()
    if telemetry is not None:
        telemetry.record_near_detection(payload)


def mark_batch_cancel_requested(*, source: str = "unknown") -> None:
    telemetry = active_batch_telemetry()
    if telemetry is not None:
        telemetry.mark_cancel_requested(source=source)


def build_run_telemetry_payload(
    *,
    run_id: int | None,
    started_at: str | None,
    finished_at: str | None,
    duration_s: float | None,
    terminal_status: str,
    planned: int,
    solved: int,
    failed: int,
    cancelled: int,
    skipped: int = 0,
    telemetry: Mapping[str, object] | None = None,
) -> dict[str, object]:
    snap = dict(telemetry or {})
    near = snap.get("near_detection") if isinstance(snap.get("near_detection"), Mapping) else {}
    cancellation = snap.get("cancellation") if isinstance(snap.get("cancellation"), Mapping) else {}
    processed = int(solved) + int(failed) + int(cancelled) + int(skipped)
    return {
        "schema": "zesolver.run_telemetry.v1",
        "run": {
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_s": duration_s,
            "terminal_status": str(terminal_status),
        },
        "input": {
            "planned": int(planned),
            "processed": int(processed),
            "solved": int(solved),
            "failed": int(failed),
            "cancelled": int(cancelled),
            "skipped": int(skipped),
        },
        "near_detection": dict(near),
        "cancellation": dict(cancellation),
    }


def write_run_telemetry_sidecar(log_path: Path | str, payload: Mapping[str, object]) -> Path:
    dst = Path(log_path).with_suffix(".telemetry.json")
    tmp = dst.with_name(dst.name + ".tmp")
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True)
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n")
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    os.replace(tmp, dst)
    return dst


def _empty_near_detection() -> dict[str, object]:
    return {
        "requested": None,
        "selected_initial": None,
        "selected_last": None,
        "used_last": None,
        "device_last": None,
        "gpu_slots": None,
        "image_count": 0,
        "images_cuda": 0,
        "images_cpu": 0,
        "fallbacks": 0,
        "fallback_reasons": {},
        "gpu_errors": 0,
        "gpu_oom": 0,
        "gpu_disabled_for_batch": False,
        "gpu_disabled_reason": None,
        "backends_used_counts": {},
        "devices_used_counts": {},
        "detect_duration_ms_values": [],
        "gpu_slot_wait_ms_values": [],
        "transfer_h2d_ms_total": 0.0,
        "gpu_compute_ms_total": 0.0,
        "transfer_d2h_ms_total": 0.0,
        "vram_peak_bytes": None,
        "cupy_pool_used_peak_bytes": None,
        "cupy_pool_reserved_peak_bytes": None,
        "detections_started_after_cancel": 0,
        "gpu_sections_started_after_cancel": 0,
        "gpu_sections_finished_after_cancel": 0,
        "sample_count": 0,
        "first_samples": [],
        "last_samples": [],
        "sample_policy": "first_16_last_16",
        "samples_truncated": False,
    }


def _near_detection_public_summary(
    raw: Mapping[str, object],
    *,
    cancel_requested_at: float | None,
    cancel_requested_wall: str | None,
    terminal_status: str | None,
) -> dict[str, object]:
    backends = raw.get("backends_used_counts") if isinstance(raw.get("backends_used_counts"), Mapping) else {}
    devices = raw.get("devices_used_counts") if isinstance(raw.get("devices_used_counts"), Mapping) else {}
    return {
        "requested": raw.get("requested"),
        "selected_initial": raw.get("selected_initial"),
        "selected_last": raw.get("selected_last"),
        "used_last": raw.get("used_last"),
        "device_last": raw.get("device_last"),
        "backends_used": sorted(str(key) for key in backends.keys()),
        "devices_used": sorted(int(key) for key in devices.keys() if str(key).lstrip("-").isdigit()),
        "gpu_slots": raw.get("gpu_slots"),
        "images_cuda": int(raw.get("images_cuda", 0) or 0),
        "images_cpu": int(raw.get("images_cpu", 0) or 0),
        "fallbacks": int(raw.get("fallbacks", 0) or 0),
        "fallback_reasons": dict(raw.get("fallback_reasons") if isinstance(raw.get("fallback_reasons"), Mapping) else {}),
        "gpu_errors": int(raw.get("gpu_errors", 0) or 0),
        "gpu_oom": int(raw.get("gpu_oom", 0) or 0),
        "gpu_disabled_for_batch": bool(raw.get("gpu_disabled_for_batch", False)),
        "gpu_disabled_reason": raw.get("gpu_disabled_reason"),
        "detect_duration_ms": _metric_summary(raw.get("detect_duration_ms_values")),
        "gpu_slot_wait_ms": _metric_summary(raw.get("gpu_slot_wait_ms_values")),
        "transfer_h2d_ms_total": round(float(raw.get("transfer_h2d_ms_total", 0.0) or 0.0), 3),
        "gpu_compute_ms_total": round(float(raw.get("gpu_compute_ms_total", 0.0) or 0.0), 3),
        "transfer_d2h_ms_total": round(float(raw.get("transfer_d2h_ms_total", 0.0) or 0.0), 3),
        "vram_peak_bytes": raw.get("vram_peak_bytes"),
        "cupy_pool_used_peak_bytes": raw.get("cupy_pool_used_peak_bytes"),
        "cupy_pool_reserved_peak_bytes": raw.get("cupy_pool_reserved_peak_bytes"),
        "terminal": terminal_status,
        "samples": _public_samples(raw),
        "sample_count": int(raw.get("sample_count", 0) or 0),
        "sample_policy": str(raw.get("sample_policy", "first_16_last_16")),
        "samples_truncated": bool(raw.get("samples_truncated", False)),
        "cancel_requested_at": cancel_requested_wall,
        "cancel_requested_monotonic_s": cancel_requested_at,
        "detections_started_after_cancel": int(raw.get("detections_started_after_cancel", 0) or 0),
        "gpu_sections_started_after_cancel": int(raw.get("gpu_sections_started_after_cancel", 0) or 0),
        "gpu_sections_finished_after_cancel": int(raw.get("gpu_sections_finished_after_cancel", 0) or 0),
    }


def _public_samples(raw: Mapping[str, object]) -> list[dict[str, object]]:
    first = list(raw.get("first_samples") if isinstance(raw.get("first_samples"), list) else [])
    last = list(raw.get("last_samples") if isinstance(raw.get("last_samples"), list) else [])
    if not last:
        return [dict(item) for item in first if isinstance(item, Mapping)]
    merged = []
    seen = set()
    for item in first + last:
        if not isinstance(item, Mapping):
            continue
        marker = id(item)
        if marker in seen:
            continue
        seen.add(marker)
        merged.append(dict(item))
    return merged


def _record_sample(summary: dict[str, object], sample: dict[str, object]) -> None:
    count = int(summary.get("sample_count", 0) or 0) + 1
    summary["sample_count"] = count
    first = summary.setdefault("first_samples", [])
    last = summary.setdefault("last_samples", [])
    if isinstance(first, list) and len(first) < 16:
        first.append(dict(sample))
    if isinstance(last, list):
        last.append(dict(sample))
        del last[:-16]
    if count > 32:
        summary["samples_truncated"] = True


def _append_metric(summary: dict[str, object], key: str, value: float | None) -> None:
    if value is None:
        return
    values = summary.setdefault(key, [])
    if isinstance(values, list):
        values.append(float(value))


def _metric_summary(values_obj: object) -> dict[str, object]:
    if not isinstance(values_obj, list) or not values_obj:
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
    values = sorted(float(v) for v in values_obj if isinstance(v, (int, float)))
    if not values:
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
    count = len(values)
    mid = count // 2
    median = values[mid] if count % 2 else (values[mid - 1] + values[mid]) / 2.0
    p95_index = min(count - 1, max(0, int(round(0.95 * (count - 1)))))
    return {
        "count": count,
        "min": round(values[0], 3),
        "median": round(median, 3),
        "p95": round(values[p95_index], 3),
        "max": round(values[-1], 3),
    }


def _seconds_to_ms(value: object) -> float | None:
    number = _clean_optional_float(value)
    if number is None:
        return None
    return max(0.0, number * 1000.0)


def _clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _clean_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
