from __future__ import annotations

import concurrent.futures
import time

from zesolver.cancellation import ProcessCancellationController, shutdown_process_executor


def _wcs_writer_worker(token, hold_s: float) -> str:
    token.set_worker_state("wcs_write")
    time.sleep(hold_s)
    token.set_worker_state("idle")
    return "written"


def test_forced_shutdown_does_not_kill_worker_in_wcs_critical_section() -> None:
    controller = ProcessCancellationController()
    try:
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=1)
        future = pool.submit(_wcs_writer_worker, controller.token, 0.4)
        time.sleep(0.1)
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
