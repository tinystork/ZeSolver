from __future__ import annotations

import multiprocessing

from zesolver.cancellation import ProcessCancellationController, ProcessCancellationToken


def _spawn_token_worker(token: ProcessCancellationToken, queue) -> None:
    queue.put(token.is_cancelled())
    token.cancel()
    queue.put(token.is_cancelled())


def test_process_cancellation_token_is_spawn_visible() -> None:
    ctx = multiprocessing.get_context("spawn")
    controller = ProcessCancellationController()
    queue = ctx.Queue()
    proc = ctx.Process(target=_spawn_token_worker, args=(controller.token, queue))
    try:
        proc.start()
        assert queue.get(timeout=5) is False
        assert queue.get(timeout=5) is True
        proc.join(timeout=5)
        assert proc.exitcode == 0
        assert controller.token.is_cancelled()
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2)
        controller.shutdown()
