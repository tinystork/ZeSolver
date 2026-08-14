"""Runtime and session lifecycle for the ZeSolver v1 API.

Architecture (process -> runtime -> session -> solve):

* One :class:`SolverRuntime` per process owns *process-scoped, shareable* heavy
  resources: the resolved catalog resources, the shared Near catalog provider,
  and the resolved (immutable) Blind-4D manifest selection.
* One :class:`SolverSession` per concurrent worker/thread owns the mutable,
  non-thread-safe solve context (a per-session Blind prep cache).  A session is
  not safe for concurrent ``solve`` calls and detects that misuse.

Heavy resources are resolved once per runtime and never rebuilt per image.
Mutable contexts stay session-scoped.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from . import _adapters
from ._adapters import (
    blind_solve_engine,
    build_runtime_context,
    near_solve_engine,
    run_solve,
)
from .errors import InvalidRequestError, SolverClosedError
from .models import (
    GpuPolicy,
    NetworkPolicy,
    RuntimeProbe,
    SolveRequest,
    SolveResult,
)
from .probe import probe as _probe


class SolverRuntime:
    """Process-scoped owner of shareable heavy solver resources."""

    def __init__(
        self,
        *,
        resources_path: Path | None = None,
        gpu_policy: GpuPolicy = GpuPolicy.AUTO,
        network_policy: NetworkPolicy = NetworkPolicy.DISABLED,
        _injected_resources=None,
        _near_solver: Callable | None = None,
        _blind_solver: Callable | None = None,
    ) -> None:
        if not isinstance(gpu_policy, GpuPolicy):
            raise InvalidRequestError("gpu_policy must be a GpuPolicy")
        if not isinstance(network_policy, NetworkPolicy):
            raise InvalidRequestError("network_policy must be a NetworkPolicy")
        if resources_path is not None and not isinstance(resources_path, Path):
            resources_path = Path(resources_path)
        self._resources_path = resources_path
        self._gpu_policy = gpu_policy
        self._network_policy = network_policy
        self._injected_resources = _injected_resources
        self._near_solver = _near_solver or near_solve_engine
        self._blind_solver = _blind_solver or blind_solve_engine
        self._ctx = None
        self._ctx_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = False
        self._sessions: set[SolverSession] = set()
        self._sessions_lock = threading.Lock()

    # -- public ------------------------------------------------------------

    def probe(
        self,
        *,
        check_catalogs: bool = True,
        check_gpu: bool = True,
        timeout_s: float = 2.0,
    ) -> RuntimeProbe:
        """Probe capabilities using this runtime's ``resources_path``."""
        return _probe(
            check_catalogs=check_catalogs,
            check_gpu=check_gpu,
            timeout_s=timeout_s,
            resources_path=self._resources_path,
        )

    def create_session(self) -> "SolverSession":
        """Create a new worker-scoped session sharing this runtime's resources."""
        self._ensure_open()
        session = SolverSession(self)
        with self._sessions_lock:
            self._sessions.add(session)
        return session

    def close(self) -> None:
        """Close the runtime.  Idempotent; invalidates all its sessions."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        ctx = self._ctx
        if ctx is not None and ctx.near_shared is not None:
            try:
                ctx.near_shared.close()
            except Exception:
                pass
        with self._sessions_lock:
            sessions = list(self._sessions)
            self._sessions.clear()
        for session in sessions:
            session._mark_closed()

    # -- internal ----------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._closed:
            raise SolverClosedError("runtime is closed")

    def _context(self):
        self._ensure_open()
        with self._ctx_lock:
            if self._ctx is None:
                self._ctx = build_runtime_context(
                    self._resources_path, self._injected_resources
                )
            return self._ctx

    def _drop_session(self, session: "SolverSession") -> None:
        with self._sessions_lock:
            self._sessions.discard(session)


class SolverSession:
    """One worker-scoped solve context.  Not safe for concurrent ``solve`` calls."""

    def __init__(self, runtime: SolverRuntime) -> None:
        self._runtime = runtime
        self._closed = False
        self._solve_lock = threading.Lock()
        self._solving = False
        self._prep_cache: dict = {}

    def solve(
        self,
        request: SolveRequest,
        *,
        cancellation=None,
        progress: Callable | None = None,
    ) -> SolveResult:
        """Solve a single request sequentially.

        Expected operational failures are returned as :class:`SolveResult`;
        contract misuse and unexpected engine bugs raise (see
        :mod:`zesolver.api.v1.errors`).
        """
        if not isinstance(request, SolveRequest):
            raise InvalidRequestError("request must be a SolveRequest")
        self._ensure_open()
        with self._solve_lock:
            if self._solving:
                raise InvalidRequestError("SolverSession does not support concurrent solve() calls")
            self._solving = True
        try:
            ctx = self._runtime._context()
            return run_solve(
                request,
                resources=ctx.resources,
                near_shared=ctx.near_shared,
                blind_selection=ctx.blind_selection,
                gpu_policy=self._runtime._gpu_policy,
                network_policy=request.options.network_policy,
                resources_path=self._runtime._resources_path,
                near_solver=self._runtime._near_solver,
                blind_solver=self._runtime._blind_solver,
                cancellation=cancellation,
                progress=progress,
                prep_cache=self._prep_cache,
            )
        finally:
            with self._solve_lock:
                self._solving = False

    def close(self) -> None:
        """Close the session.  Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._runtime._drop_session(self)

    # -- internal ----------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._closed:
            raise SolverClosedError("session is closed")
        self._runtime._ensure_open()

    def _mark_closed(self) -> None:
        self._closed = True


def create_solver_runtime(
    *,
    resources_path: Path | None = None,
    gpu_policy: GpuPolicy = GpuPolicy.AUTO,
    network_policy: NetworkPolicy = NetworkPolicy.DISABLED,
) -> SolverRuntime:
    """Create the single process-scoped solver runtime.

    Parameters
    ----------
    resources_path:
        Optional catalog library root / discovery hint.  It is a resource
        discovery hint only; it is never used to inject internal engine objects.
    gpu_policy:
        Runtime-wide GPU usage policy (``AUTO``/``DISABLED``/``REQUIRED``).
    network_policy:
        Runtime-wide network policy.  Only ``NetworkPolicy.DISABLED`` is valid
        in API 1.0 (local-only): the public API performs no network access.
    """
    return SolverRuntime(
        resources_path=resources_path,
        gpu_policy=gpu_policy,
        network_policy=network_policy,
    )
