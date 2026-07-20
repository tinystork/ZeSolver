from __future__ import annotations

import pytest

from zesolver.gui_pipeline.lifecycle import RunLifecycle


def test_run_lifecycle_accepts_one_terminal_completion() -> None:
    lifecycle = RunLifecycle()
    run_id = lifecycle.start()

    assert lifecycle.finish_once(run_id)
    assert not lifecycle.finish_once(run_id)
    assert lifecycle.terminal_count == 1


def test_qthread_native_finished_is_single_terminal_signal() -> None:
    pytest.importorskip("PySide6")
    from PySide6 import QtCore

    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])

    class Worker(QtCore.QThread):
        def run(self) -> None:
            return

    calls = []
    worker = Worker()
    worker.finished.connect(lambda: calls.append("finished"))
    worker.start()
    assert worker.wait(2000)
    app.processEvents()

    assert calls == ["finished"]
