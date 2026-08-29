"""Opaque configuration-session handle for the ZeSolver v1 public API.

:class:`ConfigurationSession` is the only thing a consumer receives from
:func:`zesolver.api.v1.open_configuration`.  It hides the underlying process
entirely — no :class:`subprocess.Popen`, no PID and no filesystem path are ever
exposed — so consumers observe the end of the configuration lifecycle purely
through :meth:`is_running` and :meth:`wait`.

Import-boundary rule: this module imports only the standard library (plus the
frozen-dataclass machinery), so ``import zesolver.api.v1`` stays lightweight.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ConfigurationSession:
    """Opaque handle to a launched ZeSolver configuration subprocess.

    The process object is private and never exposed: consumers can only poll
    :meth:`is_running` or block on :meth:`wait` to observe lifecycle end.
    """

    _process: Any = field(repr=False, compare=False, hash=False)

    def is_running(self) -> bool:
        """Return ``True`` while the configuration process is still running."""
        return self._process.poll() is None

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the configuration process exits.

        Returns ``True`` when the process has finished, ``False`` when
        ``timeout`` (seconds) elapsed first.  A ``None`` timeout blocks
        indefinitely.
        """
        if timeout is None:
            self._process.wait()
            return True
        if timeout < 0:
            raise ValueError("timeout must be >= 0")
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        return True
