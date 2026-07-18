from __future__ import annotations

from zesolver.gui_pipeline.lifecycle import RunLifecycle


def test_log_copy_attempt_is_once_per_run() -> None:
    lifecycle = RunLifecycle()
    run_id = lifecycle.start()

    assert lifecycle.mark_log_copy_once(run_id)
    assert not lifecycle.mark_log_copy_once(run_id)
    assert lifecycle.log_copy_count == 1


def test_log_copy_resets_for_next_run() -> None:
    lifecycle = RunLifecycle()
    first = lifecycle.start()
    assert lifecycle.mark_log_copy_once(first)
    assert lifecycle.finish_once(first)
    assert lifecycle.transition_idle_once(first)

    second = lifecycle.start()
    assert lifecycle.mark_log_copy_once(second)
    assert lifecycle.log_copy_count == 2
