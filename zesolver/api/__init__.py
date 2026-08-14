"""Public, versioned ZeSolver API package.

The stable contract lives under :mod:`zesolver.api.v1`. This top-level package
is a namespace only and intentionally re-exports nothing so that consumers must
select a versioned surface explicitly::

    from zesolver.api.v1 import create_solver_runtime, SolveRequest, probe

The historical :mod:`zesolver` package ``__init__`` is explicitly *not* the
public contract for this API.
"""

__all__: list[str] = []
