"""API v1 resource-error boundary: internal catalog resolution failures must
never leak ``CatalogResourceResolutionError`` through ``zesolver.api.v1``.

Contract under test (ZS-API-v1-RESOURCE-BOUNDARY):

* ``create_solver_runtime(resources_path=<invalid>)`` + ``session.solve(...)``
  returns ``SolveResult(status=FAILED, failure_code=MISSING_RESOURCE, ...)``.
* No ``CatalogResourceResolutionError`` escapes the public API.
* Unexpected programming bugs (e.g. ``RuntimeError``) stay visible — never a
  false ``SolveResult(MISSING_RESOURCE)``.
* The default ``resources_path=None`` degradation path is unchanged.

This module is a *consumer sentinel*: it imports only from ``zesolver.api.v1``
(plus stdlib / third-party test dependencies), never internal solver modules.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from zesolver.api.v1 import (
    BackendPolicy,
    FailureCode,
    SolveOptions,
    SolveRequest,
    SolveStatus,
    SolverClosedError,
    create_solver_runtime,
)

ROOT = Path(__file__).resolve().parents[1]

_ENV_CATALOG_HINTS = (
    "ZESOLVER_ASTAP_ROOT",
    "ZESOLVER_BLIND4D_MANIFEST",
    "ZEBLIND_4D_MANIFEST",
    "ZESOLVER_LEGACY_INDEX_ROOT",
)


def _write_clean_fits(path: Path) -> Path:
    """A valid 2D FITS image with *no* WCS keywords."""
    fits.writeto(path, np.zeros((100, 100), dtype=np.float32), overwrite=True)
    return path


@pytest.fixture(autouse=True)
def _clear_env_catalog_hints(monkeypatch) -> None:
    """Keep auto-discovery deterministic regardless of the host environment."""
    for key in _ENV_CATALOG_HINTS:
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# (a) Main leak reproduction: invalid resources_path must not leak
# ---------------------------------------------------------------------------


def test_invalid_resources_path_returns_missing_resource(tmp_path: Path) -> None:
    input_path = _write_clean_fits(tmp_path / "in.fits")
    rt = create_solver_runtime(resources_path=tmp_path / "no_such_catalog")
    session = rt.create_session()
    try:
        result = session.solve(SolveRequest(input_path))
    finally:
        session.close()
        rt.close()

    assert result.status is SolveStatus.FAILED
    assert result.failure_code is FailureCode.MISSING_RESOURCE
    # Internal detail is preserved in the non-stable diagnostic fields only.
    assert result.diagnostic_code
    assert "catalog_resources_invalid" in result.diagnostic_code
    assert result.message


# ---------------------------------------------------------------------------
# (b) Consumer sentinel: only zesolver.api.v1 is imported by the consumer
# ---------------------------------------------------------------------------


_CONSUMER_SCRIPT = r"""
import json
import os
from pathlib import Path

import numpy as np
from astropy.io import fits

# The consumer references ONLY the public zesolver.api.v1 surface.
from zesolver.api.v1 import (
    FailureCode,
    SolveRequest,
    SolveStatus,
    create_solver_runtime,
)

tmp = Path(os.environ["ZS_CONSUMER_TMP"])
input_path = tmp / "in.fits"
fits.writeto(input_path, np.zeros((100, 100), dtype=np.float32), overwrite=True)

rt = create_solver_runtime(resources_path=tmp / "no_such_catalog")
session = rt.create_session()
result = session.solve(SolveRequest(input_path))
session.close()
rt.close()

print(json.dumps({
    "status": result.status.value,
    "failure_code": result.failure_code.value,
}))
"""


def test_consumer_reproduces_boundary_without_internal_imports(tmp_path: Path) -> None:
    # Sentinel: the consumer script never imports an internal solver module.
    for forbidden in ("catalog_resources", "core.models", "core.pipeline"):
        assert forbidden not in _CONSUMER_SCRIPT, forbidden

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["ZS_CONSUMER_TMP"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", _CONSUMER_SCRIPT],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=env,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    payload = json.loads(result.stdout)
    assert payload["status"] == SolveStatus.FAILED.value
    assert payload["failure_code"] == FailureCode.MISSING_RESOURCE.value


# ---------------------------------------------------------------------------
# (c) Unexpected bug: RuntimeError stays visible, never a false MISSING_RESOURCE
# ---------------------------------------------------------------------------


def test_unexpected_bug_is_not_masked_as_missing_resource(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = _write_clean_fits(tmp_path / "in.fits")
    rt = create_solver_runtime(resources_path=tmp_path / "no_such_catalog")

    def _boom_context():
        raise RuntimeError("unexpected invariant failure")

    monkeypatch.setattr(rt, "_context", _boom_context)
    session = rt.create_session()
    try:
        with pytest.raises(RuntimeError, match="unexpected invariant failure"):
            session.solve(SolveRequest(input_path))
    finally:
        session.close()
        rt.close()


# ---------------------------------------------------------------------------
# (d) Default healthy path vs explicit invalid path: both respect the boundary
# ---------------------------------------------------------------------------


def test_default_and_invalid_resources_both_respect_boundary(tmp_path: Path) -> None:
    input_path = _write_clean_fits(tmp_path / "in.fits")

    # Auto-discovery with no resources available (the pre-existing healthy path).
    rt_default = create_solver_runtime()
    session_default = rt_default.create_session()
    try:
        result_default = session_default.solve(SolveRequest(input_path))
    finally:
        session_default.close()
        rt_default.close()
    assert result_default.status is SolveStatus.FAILED
    assert result_default.failure_code is FailureCode.MISSING_RESOURCE
    assert result_default.diagnostic_code == "catalog_resources_absent"

    # Explicit invalid resources_path (the previously leaking path).
    rt_invalid = create_solver_runtime(resources_path=tmp_path / "no_such_catalog")
    session_invalid = rt_invalid.create_session()
    try:
        result_invalid = session_invalid.solve(SolveRequest(input_path))
    finally:
        session_invalid.close()
        rt_invalid.close()
    assert result_invalid.status is SolveStatus.FAILED
    assert result_invalid.failure_code is FailureCode.MISSING_RESOURCE
    assert result_invalid.diagnostic_code.startswith("catalog_resources_invalid")


# ---------------------------------------------------------------------------
# (e) Cleanup: close() stays safe/idempotent after a resolution failure
# ---------------------------------------------------------------------------


def test_close_is_safe_and_idempotent_after_resolution_failure(tmp_path: Path) -> None:
    input_path = _write_clean_fits(tmp_path / "in.fits")
    rt = create_solver_runtime(resources_path=tmp_path / "no_such_catalog")
    session = rt.create_session()

    result = session.solve(SolveRequest(input_path))
    assert result.failure_code is FailureCode.MISSING_RESOURCE

    # Idempotent, exception-free double close on both objects.
    session.close()
    session.close()
    rt.close()
    rt.close()

    # A closed session still rejects further work cleanly (no internal leak).
    with pytest.raises(SolverClosedError):
        session.solve(SolveRequest(input_path))


# ---------------------------------------------------------------------------
# (guard) No internal catalog exception type is exposed by the public surface
# ---------------------------------------------------------------------------


def test_no_internal_catalog_exception_on_public_surface() -> None:
    import zesolver.api.v1 as v1

    for name in (
        "CatalogResourceResolutionError",
        "SolverCatalogResources",
        "catalog_resources",
    ):
        assert name not in v1.__all__
        assert not hasattr(v1, name)
