from __future__ import annotations

from zesolver.gui_pipeline.lifecycle import RunLifecycle


def test_progress_run_counters_reset_on_restart() -> None:
    lifecycle = RunLifecycle()

    first = lifecycle.start()
    assert lifecycle.finish_once(first, terminal_state="CANCELLED")
    assert lifecycle.run_terminal_count == 1
    assert lifecycle.transition_idle_once(first)

    second = lifecycle.start()
    assert lifecycle.run_terminal_count == 0
    assert lifecycle.run_log_copy_count == 0
    assert lifecycle.run_idle_transition_count == 0
    assert lifecycle.finish_once(second)
    assert lifecycle.run_terminal_count == 1
