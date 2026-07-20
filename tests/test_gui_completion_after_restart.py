from __future__ import annotations

from zesolver.gui_pipeline.lifecycle import RunLifecycle


def test_two_successive_runs_each_complete_once() -> None:
    lifecycle = RunLifecycle()

    first = lifecycle.start()
    assert lifecycle.finish_once(first)
    assert lifecycle.run_terminal_count == 1
    assert lifecycle.transition_idle_once(first)
    assert lifecycle.run_idle_transition_count == 1

    second = lifecycle.start()
    assert second != first
    assert lifecycle.run_terminal_count == 0
    assert lifecycle.run_idle_transition_count == 0
    assert lifecycle.finish_once(second)
    assert lifecycle.run_terminal_count == 1
    assert lifecycle.transition_idle_once(second)
    assert lifecycle.run_idle_transition_count == 1

    assert lifecycle.terminal_count == 2
    assert lifecycle.idle_transition_count == 2
