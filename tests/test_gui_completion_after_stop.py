from __future__ import annotations

from zesolver.gui_pipeline.lifecycle import RunLifecycle


def test_cancelled_run_has_single_terminal_and_idle_transition() -> None:
    lifecycle = RunLifecycle()
    run_id = lifecycle.start()

    assert lifecycle.finish_once(run_id, terminal_state="CANCELLED")
    assert lifecycle.state == "CANCELLED"
    assert not lifecycle.finish_once(run_id, terminal_state="FINISHED")
    assert lifecycle.transition_idle_once(run_id)
    assert lifecycle.state == "IDLE"
    assert lifecycle.terminal_count == 1
    assert lifecycle.idle_transition_count == 1
