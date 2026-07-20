from __future__ import annotations

from zesolver.gui_pipeline.lifecycle import RunLifecycle


def test_stale_run_completion_is_rejected_after_restart() -> None:
    lifecycle = RunLifecycle()
    first = lifecycle.start()
    assert lifecycle.finish_once(first, terminal_state="CANCELLED")
    assert lifecycle.transition_idle_once(first)

    second = lifecycle.start()
    assert not lifecycle.finish_once(first)
    assert lifecycle.finish_once(second)
    assert lifecycle.terminal_count == 2
