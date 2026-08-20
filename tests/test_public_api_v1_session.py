"""Tests for the opaque ``ConfigurationSession`` handle (API v1.2).

The handle must let a consumer observe the end of the configuration lifecycle
(``is_running`` / ``wait``) without ever exposing the underlying process, its
PID, or any filesystem path.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from zesolver.api.v1 import ConfigurationSession
from zesolver.api.v1.readiness import open_configuration
from zesolver.api.v1.session import ConfigurationSession as SessionModule


def _spawn(sleep_s: float = 0.0) -> "subprocess.Popen":
    return subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({sleep_s!r})"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def test_session_is_running_while_process_alive() -> None:
    proc = _spawn(0.5)
    try:
        session = ConfigurationSession(proc)
        assert session.is_running() is True
    finally:
        proc.wait()


def test_session_wait_timeout_zero_returns_false_while_running() -> None:
    proc = _spawn(0.5)
    try:
        session = ConfigurationSession(proc)
        assert session.wait(timeout=0) is False
        assert session.is_running() is True
    finally:
        proc.wait()


def test_session_wait_blocks_until_finished_and_returns_true() -> None:
    proc = _spawn(0.2)
    session = ConfigurationSession(proc)
    assert session.wait(timeout=5.0) is True
    assert session.is_running() is False


def test_session_wait_without_timeout_returns_true() -> None:
    proc = _spawn(0.0)
    session = ConfigurationSession(proc)
    assert session.wait() is True
    assert session.is_running() is False


def test_session_wait_rejects_negative_timeout() -> None:
    proc = _spawn(0.0)
    session = ConfigurationSession(proc)
    with pytest.raises(ValueError):
        session.wait(timeout=-1.0)


def test_session_does_not_expose_process_in_repr() -> None:
    proc = _spawn(0.0)
    try:
        session = ConfigurationSession(proc)
        assert "pid" not in repr(session)
        assert "Popen" not in repr(session)
    finally:
        proc.wait()


def test_open_configuration_returns_session_handle(monkeypatch) -> None:
    class _FakeProc:
        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    captured = {}

    def fake_popen(args, **kwargs):
        captured["kwargs"] = kwargs
        return _FakeProc()

    import importlib

    readiness_module = importlib.import_module("zesolver.api.v1.readiness")

    monkeypatch.setattr(readiness_module, "_gui_entry_point", lambda: None)
    monkeypatch.setattr(readiness_module.subprocess, "Popen", fake_popen)

    session = open_configuration()

    assert isinstance(session, SessionModule)
    assert session.is_running() is True
    assert session.wait(timeout=1.0) is True
    assert captured["kwargs"]["start_new_session"] is True
