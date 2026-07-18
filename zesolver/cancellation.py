from __future__ import annotations

import contextlib
import multiprocessing
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Protocol


class CancellationToken(Protocol):
    def cancel(self) -> None:
        ...

    def is_cancelled(self) -> bool:
        ...

    def is_set(self) -> bool:
        ...


@dataclass(slots=True)
class ThreadCancellationToken:
    event: Any

    def cancel(self) -> None:
        setter = getattr(self.event, "set", None)
        if callable(setter):
            setter()

    def is_cancelled(self) -> bool:
        is_set = getattr(self.event, "is_set", None)
        return bool(is_set()) if callable(is_set) else bool(self.event)

    def is_set(self) -> bool:
        return self.is_cancelled()


@dataclass(slots=True)
class ProcessCancellationToken:
    event: Any
    worker_states: Any | None = None

    def cancel(self) -> None:
        setter = getattr(self.event, "set", None)
        if callable(setter):
            setter()

    def is_cancelled(self) -> bool:
        is_set = getattr(self.event, "is_set", None)
        return bool(is_set()) if callable(is_set) else bool(self.event)

    def is_set(self) -> bool:
        return self.is_cancelled()

    def set_worker_state(self, state: str) -> None:
        if self.worker_states is None:
            return
        try:
            self.worker_states[int(os.getpid())] = str(state)
        except Exception:
            pass

    def worker_state(self, pid: int) -> str | None:
        if self.worker_states is None:
            return None
        try:
            value = self.worker_states.get(int(pid))
        except Exception:
            return None
        return str(value) if value is not None else None

    @contextlib.contextmanager
    def critical_section(self, state: str) -> Iterator[None]:
        self.set_worker_state(state)
        try:
            yield
        finally:
            self.set_worker_state("active")


@dataclass(slots=True)
class CompositeCancellationToken:
    tokens: tuple[Any, ...]

    def cancel(self) -> None:
        for token in self.tokens:
            cancel = getattr(token, "cancel", None)
            if callable(cancel):
                cancel()
                continue
            setter = getattr(token, "set", None)
            if callable(setter):
                setter()

    def is_cancelled(self) -> bool:
        for token in self.tokens:
            is_cancelled = getattr(token, "is_cancelled", None)
            if callable(is_cancelled) and bool(is_cancelled()):
                return True
            is_set = getattr(token, "is_set", None)
            if callable(is_set) and bool(is_set()):
                return True
        return False

    def is_set(self) -> bool:
        return self.is_cancelled()

    @contextlib.contextmanager
    def critical_section(self, state: str) -> Iterator[None]:
        contexts = []
        try:
            for token in self.tokens:
                critical = getattr(token, "critical_section", None)
                if callable(critical):
                    ctx = critical(state)
                    ctx.__enter__()
                    contexts.append(ctx)
            yield
        finally:
            for ctx in reversed(contexts):
                try:
                    ctx.__exit__(None, None, None)
                except Exception:
                    pass


class ProcessCancellationController:
    def __init__(self) -> None:
        self._manager = multiprocessing.Manager()
        self.token = ProcessCancellationToken(self._manager.Event(), self._manager.dict())

    def cancel(self) -> None:
        self.token.cancel()

    def shutdown(self) -> None:
        try:
            self._manager.shutdown()
        except Exception:
            pass


@dataclass(slots=True)
class ExecutorShutdownStats:
    pending_cancelled: int = 0
    running_notified: int = 0
    already_completed: int = 0
    forced_terminated: int = 0
    forced_killed: int = 0
    protected_wcs_writers: int = 0


def shutdown_process_executor(
    executor: Any,
    futures: dict[Any, Any],
    *,
    token: ProcessCancellationToken | None = None,
    grace_period_s: float = 4.0,
    kill_grace_s: float = 1.0,
    log: Callable[[str], None] | None = None,
) -> ExecutorShutdownStats:
    stats = ExecutorShutdownStats()
    processes = dict(getattr(executor, "_processes", {}) or {})
    for future in list(futures):
        if future.done():
            stats.already_completed += 1
        elif future.cancel():
            stats.pending_cancelled += 1
        else:
            stats.running_notified += 1
    if log is not None:
        log(
            "PENDING_FUTURES_CANCELLED pending_cancelled=%d running_notified=%d already_completed=%d"
            % (stats.pending_cancelled, stats.running_notified, stats.already_completed)
        )
        log("ACTIVE_WORKERS_NOTIFIED workers=%d grace_period_s=%.2f" % (len(processes), grace_period_s))
        log("EXECUTOR_SHUTDOWN_STARTED")
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except TypeError:
        executor.shutdown(wait=False)

    deadline = time.monotonic() + max(0.0, float(grace_period_s))
    while time.monotonic() < deadline:
        alive = [proc for proc in processes.values() if proc.is_alive()]
        if not alive:
            break
        time.sleep(0.05)

    alive = [proc for proc in processes.values() if proc.is_alive()]
    if alive:
        for proc in alive:
            state = token.worker_state(proc.pid) if token is not None and proc.pid is not None else None
            if state == "wcs_write":
                stats.protected_wcs_writers += 1
                continue
            try:
                proc.terminate()
                stats.forced_terminated += 1
            except Exception:
                pass
        join_deadline = time.monotonic() + max(0.0, float(kill_grace_s))
        for proc in alive:
            timeout = max(0.0, join_deadline - time.monotonic())
            try:
                proc.join(timeout)
            except Exception:
                pass
        for proc in alive:
            if not proc.is_alive():
                continue
            state = token.worker_state(proc.pid) if token is not None and proc.pid is not None else None
            if state == "wcs_write":
                continue
            killer = getattr(proc, "kill", None)
            if callable(killer):
                try:
                    killer()
                    stats.forced_killed += 1
                except Exception:
                    pass
            try:
                proc.join(0.5)
            except Exception:
                pass
    if log is not None:
        remaining = sum(1 for proc in processes.values() if proc.is_alive())
        log(
            "EXECUTOR_SHUTDOWN_FINISHED forced_terminated=%d forced_killed=%d protected_wcs_writers=%d remaining=%d"
            % (stats.forced_terminated, stats.forced_killed, stats.protected_wcs_writers, remaining)
        )
    return stats
