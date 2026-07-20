from __future__ import annotations

from zesolver.gui_pipeline.lifecycle import RunLifecycle


def test_stale_completion_from_old_run_cannot_finish_current_run() -> None:
    lifecycle = RunLifecycle()
    first = lifecycle.start()
    assert lifecycle.finish_once(first)
    assert lifecycle.transition_idle_once(first)

    second = lifecycle.start()
    assert not lifecycle.finish_once(first)
    assert lifecycle.running
    assert lifecycle.active_run_id == second
    assert lifecycle.finish_once(second)
    assert lifecycle.terminal_count == 2


def test_synthetic_double_terminal_signal_is_ignored() -> None:
    lifecycle = RunLifecycle()
    run_id = lifecycle.start()

    accepted = [lifecycle.finish_once(run_id), lifecycle.finish_once(run_id)]

    assert accepted == [True, False]
    assert lifecycle.terminal_count == 1
