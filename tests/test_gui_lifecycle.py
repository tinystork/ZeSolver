from __future__ import annotations

import pytest

from zesolver.gui_pipeline.lifecycle import RunLifecycle


def test_finished_emitted_once_and_restart_possible() -> None:
    lifecycle = RunLifecycle()
    lifecycle.start()
    with pytest.raises(RuntimeError):
        lifecycle.start()
    assert lifecycle.finish_once()
    assert not lifecycle.finish_once()
    lifecycle.reset()
    lifecycle.start()
    assert lifecycle.finish_once()
