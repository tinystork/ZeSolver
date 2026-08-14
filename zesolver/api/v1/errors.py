"""Public exception hierarchy for the ZeSolver v1 API."""

from __future__ import annotations


class ZeSolverApiError(Exception):
    """Base class for every error raised by :mod:`zesolver.api.v1`.

    Operational, expected failures (no solution, missing catalog, timeout,
    write failures, cancellation, ...) are *not* raised; they are returned as
    :class:`zesolver.api.v1.SolveResult` objects.  This hierarchy is reserved
    for contract misuse and lifecycle violations.
    """


class InvalidRequestError(ZeSolverApiError, ValueError):
    """The request or its options violate the public contract."""


class SolverClosedError(ZeSolverApiError, RuntimeError):
    """An operation was attempted on a closed runtime or session."""
