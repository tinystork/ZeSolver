"""Public, thread-safe cancellation token for the ZeSolver v1 API."""

from __future__ import annotations

import threading


class CancellationToken:
    """A minimal, thread-safe cancellation token.

    Safe to call from a UI thread and from worker threads.  Passed to
    :meth:`zesolver.api.v1.SolverSession.solve` to request cooperative
    cancellation of an in-flight solve.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation.  Idempotent and safe from any thread."""
        self._event.set()

    def is_cancelled(self) -> bool:
        """Return ``True`` once :meth:`cancel` has been called."""
        return self._event.is_set()
