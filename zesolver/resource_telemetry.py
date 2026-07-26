from __future__ import annotations

import contextvars
import gc
import threading
import time
from dataclasses import dataclass, field


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
)


IMPORTANT_EVENT_PHASES = {
    "near_batch_runtime_created",
    "near_catalog_runtime_created",
    "near_batch_runtime_ready",
    "near_catalog_runtime_closed",
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
