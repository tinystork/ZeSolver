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
)


_active_batch_telemetry: contextvars.ContextVar["BatchResourceTelemetry | None"] = contextvars.ContextVar(
    "zesolver_active_batch_telemetry",
    default=None,
)


@dataclass(slots=True)
class BatchResourceTelemetry:
    counters: dict[str, int] = field(default_factory=lambda: {key: 0 for key in BATCH_COUNTER_KEYS})
    rss_kib: dict[str, int | None] = field(default_factory=dict)
    events: list[dict[str, object]] = field(default_factory=list)
    _worker_threads: set[int] = field(default_factory=set)
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

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "counters": dict(self.counters),
                "rss_kib": dict(self.rss_kib),
                "events": tuple(dict(item) for item in self.events),
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
