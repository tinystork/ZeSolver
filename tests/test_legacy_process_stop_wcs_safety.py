from __future__ import annotations

import concurrent.futures
import multiprocessing
import time

from zesolver.cancellation import ProcessCancellationController, shutdown_process_executor


def _wcs_writer_worker(token, hold_s: float) -> str:
    token.set_worker_state("wcs_write")
    time.sleep(hold_s)
    token.set_worker_state("idle")
    return "written"


def _wait_for_wcs_writer(pool, future, token, *, timeout_s: float = 15.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if future.done():
            future.result(timeout=0)
        processes = dict(getattr(pool, "_processes", {}) or {})
        for proc in processes.values():
            pid = getattr(proc, "pid", None)
            if pid is not None and token.worker_state(pid) == "wcs_write":
                return
        time.sleep(0.02)
    raise AssertionError("worker did not enter wcs_write")


def test_forced_shutdown_does_not_kill_worker_in_wcs_critical_section() -> None:
    ctx = multiprocessing.get_context("spawn")
    controller = ProcessCancellationController(context=ctx)
    try:
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx)
        future = pool.submit(_wcs_writer_worker, controller.token, 2.0)
        _wait_for_wcs_writer(pool, future, controller.token)
        controller.cancel()
        stats = shutdown_process_executor(
            pool,
            {future: "frame.fit"},
            token=controller.token,
            grace_period_s=0.05,
            kill_grace_s=0.05,
        )
        assert stats.protected_wcs_writers == 1
        assert future.result(timeout=3) == "written"
    finally:
        controller.shutdown()
