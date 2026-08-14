"""Tests for the public ZeSolver API v1 runtime/session lifecycle and solve path."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from astropy.io import fits

from zesolver.api.v1 import (
    BackendPolicy,
    CancellationToken,
    FailureCode,
    GpuPolicy,
    InvalidRequestError,
    NetworkPolicy,
    SolveOptions,
    SolveRequest,
    SolveStatus,
    SolverClosedError,
    SolverRuntime,
    SolverSession,
    WritePolicy,
    ZeSolverApiError,
    create_solver_runtime,
)
import zesolver.api.v1.runtime as runtime_module
from zesolver.core.models import EngineSolveResult, SolveStatus as InternalStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_clean_fits(path: Path) -> Path:
    """A valid 2D FITS image with *no* WCS keywords."""
    fits.writeto(path, np.zeros((100, 100), dtype=np.float32), overwrite=True)
    return path


def _write_solved_fits(path: Path) -> Path:
    """A valid 2D FITS image with a full CD-matrix celestial WCS."""
    hdr = fits.Header()
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRVAL1"] = 100.0
    hdr["CRVAL2"] = 30.0
    hdr["CRPIX1"] = 50.5
    hdr["CRPIX2"] = 50.5
    hdr["CD1_1"] = -0.0003
    hdr["CD1_2"] = 0.0
    hdr["CD2_1"] = 0.0
    hdr["CD2_2"] = 0.0003
    fits.writeto(path, np.zeros((100, 100), dtype=np.float32), header=hdr, overwrite=True)
    return path


def _full_cd_wcs():
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [100.0, 30.0]
    w.wcs.crpix = [50.5, 50.5]
    w.wcs.cd = [[-0.0003, 0.0], [0.0, 0.0003]]
    return w


def _fake_resources(*, near=True, blind=False) -> SimpleNamespace:
    return SimpleNamespace(near_available=near, blind4d_available=blind)


def _runtime_with_context(monkeypatch, resources, near_shared=None, blind_selection=None) -> SolverRuntime:
    fake_ctx = SimpleNamespace(
        resources=resources, near_shared=near_shared, blind_selection=blind_selection
    )
    monkeypatch.setattr(
        runtime_module, "build_runtime_context", lambda *a, **k: fake_ctx
    )
    return SolverRuntime()


def _unsolved_near(internal_req, resources, configuration, shared_near, cancel_check):
    return EngineSolveResult(status=InternalStatus.UNSOLVED, backend="NEAR", error="no_solution_fake")


def _solved_near(internal_req, resources, configuration, shared_near, cancel_check):
    w = _full_cd_wcs()
    with fits.open(internal_req.input_path, mode="update", memmap=False) as hdul:
        hdul[0].header.update(w.to_header())
        hdul[0].header["CD1_1"] = -0.0003
        hdul[0].header["CD1_2"] = 0.0
        hdul[0].header["CD2_1"] = 0.0
        hdul[0].header["CD2_2"] = 0.0003
        hdul.flush()
    return EngineSolveResult(
        status=InternalStatus.SOLVED,
        backend="NEAR",
        wcs=w,
        wcs_written=True,
        center_ra_deg=100.0,
        center_dec_deg=30.0,
        pixel_scale_arcsec=1.08,
        orientation_deg=0.0,
    )


def _run_solve(request, near_solver, *, gpu_policy=GpuPolicy.AUTO, cancellation=None):
    """Drive ``run_solve`` through the public session using a monkeypatched context."""
    import zesolver.api.v1._adapters as adapters

    resources = _fake_resources(near=True, blind=False)
    return adapters.run_solve(
        request,
        resources=resources,
        near_shared=None,
        blind_selection=None,
        gpu_policy=gpu_policy,
        network_policy=request.options.network_policy,
        resources_path=None,
        near_solver=near_solver,
        blind_solver=None,
        cancellation=cancellation,
        progress=None,
        prep_cache={},
    )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_create_solver_runtime_defaults() -> None:
    rt = create_solver_runtime()
    assert isinstance(rt, SolverRuntime)
    assert rt._gpu_policy is GpuPolicy.AUTO
    assert rt._network_policy is NetworkPolicy.DISABLED


def test_create_solver_runtime_rejects_bad_policy_types() -> None:
    with pytest.raises(InvalidRequestError):
        SolverRuntime(gpu_policy="AUTO")  # type: ignore[arg-type]
    with pytest.raises(InvalidRequestError):
        SolverRuntime(network_policy="DISABLED")  # type: ignore[arg-type]


def test_session_solve_rejects_non_solve_request(monkeypatch, tmp_path: Path) -> None:
    rt = _runtime_with_context(monkeypatch, _fake_resources())
    session = rt.create_session()
    with pytest.raises(InvalidRequestError):
        session.solve("not a request")  # type: ignore[arg-type]


def test_close_session_invalidates_it(monkeypatch, tmp_path: Path) -> None:
    rt = _runtime_with_context(monkeypatch, _fake_resources())
    session = rt.create_session()
    session.close()
    with pytest.raises(SolverClosedError):
        session.solve(SolveRequest(tmp_path / "x.fits"))
    # idempotent
    session.close()


def test_close_runtime_invalidates_sessions(monkeypatch, tmp_path: Path) -> None:
    rt = _runtime_with_context(monkeypatch, _fake_resources())
    session = rt.create_session()
    rt.close()
    with pytest.raises(SolverClosedError):
        session.solve(SolveRequest(tmp_path / "x.fits"))
    # create_session after close is rejected too
    with pytest.raises(SolverClosedError):
        rt.create_session()
    # idempotent
    rt.close()


def test_context_is_built_once_per_runtime(monkeypatch) -> None:
    calls = []

    def fake_build(*a, **k):
        calls.append(1)
        return SimpleNamespace(resources=_fake_resources(), near_shared=None, blind_selection=None)

    monkeypatch.setattr(runtime_module, "build_runtime_context", fake_build)
    rt = SolverRuntime()
    first = rt._context()
    second = rt._context()
    assert first is second
    assert len(calls) == 1


def test_multiple_sessions_share_runtime_context(monkeypatch) -> None:
    rt = _runtime_with_context(monkeypatch, _fake_resources())
    s1 = rt.create_session()
    s2 = rt.create_session()
    assert s1._runtime is rt
    assert s2._runtime is rt
    assert s1._runtime._context() is s2._runtime._context()


# ---------------------------------------------------------------------------
# Solve path outcomes (expected failures -> SolveResult, never raised)
# ---------------------------------------------------------------------------


def test_solve_returns_unsolved_result(monkeypatch, tmp_path: Path) -> None:
    req = SolveRequest(
        _write_clean_fits(tmp_path / "in.fits"),
        options=SolveOptions(backend_policy=BackendPolicy.NEAR_ONLY),
    )
    result = _run_solve(req, _unsolved_near)
    assert result.status is SolveStatus.FAILED
    assert result.failure_code is FailureCode.NO_SOLUTION


def test_solve_returns_solved_result(monkeypatch, tmp_path: Path) -> None:
    path = _write_clean_fits(tmp_path / "in.fits")
    req = SolveRequest(path, options=SolveOptions(backend_policy=BackendPolicy.NEAR_ONLY))
    result = _run_solve(req, _solved_near)
    assert result.status is SolveStatus.SOLVED
    assert result.backend_used == "NEAR"
    assert result.ra_deg == pytest.approx(100.0)
    assert result.dec_deg == pytest.approx(30.0)
    assert result.wcs_header is not None
    assert result.wcs_header.cards


def test_solve_missing_input_returns_invalid_input(monkeypatch, tmp_path: Path) -> None:
    req = SolveRequest(tmp_path / "missing.fits")
    result = _run_solve(req, _unsolved_near)
    assert result.status is SolveStatus.FAILED
    assert result.failure_code is FailureCode.INVALID_INPUT


def test_solve_existing_valid_wcs_skips(monkeypatch, tmp_path: Path) -> None:
    path = _write_solved_fits(tmp_path / "in.fits")
    req = SolveRequest(path, options=SolveOptions(backend_policy=BackendPolicy.NEAR_ONLY))
    result = _run_solve(req, _unsolved_near)
    assert result.status is SolveStatus.SKIPPED_EXISTING_WCS


def test_solve_existing_invalid_wcs_fails(monkeypatch, tmp_path: Path) -> None:
    path = _write_clean_fits(tmp_path / "in.fits")
    # Add partial WCS keywords (so it is "invalid", not "none").
    with fits.open(path, mode="update", memmap=False) as hdul:
        hdul[0].header["CTYPE1"] = "RA---TAN"
        hdul[0].header["CRVAL1"] = 0.0
        hdul.flush()
    req = SolveRequest(path, options=SolveOptions(backend_policy=BackendPolicy.NEAR_ONLY))
    result = _run_solve(req, _unsolved_near)
    assert result.status is SolveStatus.FAILED
    assert result.failure_code is FailureCode.EXISTING_WCS_INVALID


def test_solve_near_only_without_near_returns_backend_unavailable(monkeypatch, tmp_path: Path) -> None:
    req = SolveRequest(
        _write_clean_fits(tmp_path / "in.fits"),
        options=SolveOptions(backend_policy=BackendPolicy.NEAR_ONLY),
    )
    import zesolver.api.v1._adapters as adapters

    result = adapters.run_solve(
        req,
        resources=_fake_resources(near=False, blind=False),
        near_shared=None,
        blind_selection=None,
        gpu_policy=GpuPolicy.AUTO,
        network_policy=NetworkPolicy.DISABLED,
        resources_path=None,
        near_solver=_unsolved_near,
        blind_solver=None,
        cancellation=None,
        progress=None,
        prep_cache={},
    )
    assert result.status is SolveStatus.FAILED
    assert result.failure_code is FailureCode.BACKEND_UNAVAILABLE


def test_solve_gpu_required_without_gpu_returns_backend_unavailable(monkeypatch, tmp_path: Path) -> None:
    req = SolveRequest(_write_clean_fits(tmp_path / "in.fits"))
    result = _run_solve(req, _unsolved_near, gpu_policy=GpuPolicy.REQUIRED)
    assert result.status is SolveStatus.FAILED
    assert result.failure_code is FailureCode.BACKEND_UNAVAILABLE
    assert result.diagnostic_code == "gpu_required_unavailable"


def test_solve_write_copy_output_exists_fails(monkeypatch, tmp_path: Path) -> None:
    existing = tmp_path / "out.fits"
    existing.write_bytes(b"occupied")
    req = SolveRequest(
        _write_clean_fits(tmp_path / "in.fits"),
        options=SolveOptions(write_policy=WritePolicy.WRITE_COPY, output_path=existing),
    )
    result = _run_solve(req, _unsolved_near)
    assert result.status is SolveStatus.FAILED
    assert result.failure_code is FailureCode.WRITE_FAILED


# ---------------------------------------------------------------------------
# Error boundary: unexpected engine bugs are never swallowed
# ---------------------------------------------------------------------------


def test_unexpected_engine_bug_raises_api_error(monkeypatch, tmp_path: Path) -> None:
    def buggy_near(internal_req, resources, configuration, shared_near, cancel_check):
        raise ValueError("boom")

    req = SolveRequest(
        _write_clean_fits(tmp_path / "in.fits"),
        options=SolveOptions(backend_policy=BackendPolicy.NEAR_ONLY),
    )
    with pytest.raises(ZeSolverApiError) as excinfo:
        _run_solve(req, buggy_near)
    assert isinstance(excinfo.value.__cause__, ValueError)


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


def test_cancellation_token_basics() -> None:
    token = CancellationToken()
    assert not token.is_cancelled()
    token.cancel()
    assert token.is_cancelled()
    token.cancel()  # idempotent
    assert token.is_cancelled()


def test_solve_precancelled_returns_cancelled(monkeypatch, tmp_path: Path) -> None:
    token = CancellationToken()
    token.cancel()
    req = SolveRequest(_write_clean_fits(tmp_path / "in.fits"))
    result = _run_solve(req, _unsolved_near, cancellation=token)
    assert result.status is SolveStatus.CANCELLED


def test_solve_engine_cancelled_returns_cancelled(monkeypatch, tmp_path: Path) -> None:
    def cancelled_near(internal_req, resources, configuration, shared_near, cancel_check):
        return EngineSolveResult(status=InternalStatus.CANCELLED, backend="NEAR", error="cancelled")

    req = SolveRequest(
        _write_clean_fits(tmp_path / "in.fits"),
        options=SolveOptions(backend_policy=BackendPolicy.NEAR_ONLY),
    )
    result = _run_solve(req, cancelled_near)
    assert result.status is SolveStatus.CANCELLED


# ---------------------------------------------------------------------------
# Concurrency: a session is not safe for concurrent solve calls
# ---------------------------------------------------------------------------


def test_session_rejects_concurrent_solve(monkeypatch, tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocking_near(internal_req, resources, configuration, shared_near, cancel_check):
        entered.set()
        release.wait(timeout=10)
        return EngineSolveResult(status=InternalStatus.UNSOLVED, backend="NEAR", error="blocked")

    rt = _runtime_with_context(monkeypatch, _fake_resources())
    session = rt.create_session()
    req = SolveRequest(
        _write_clean_fits(tmp_path / "in.fits"),
        options=SolveOptions(backend_policy=BackendPolicy.NEAR_ONLY),
    )
    rt._near_solver = blocking_near

    first_result: list = []
    def run_first():
        first_result.append(session.solve(req))

    t = threading.Thread(target=run_first)
    t.start()
    assert entered.wait(timeout=15), "near solver never entered"
    try:
        with pytest.raises(InvalidRequestError):
            session.solve(req)
    finally:
        release.set()
        t.join(timeout=15)
    assert not t.is_alive()
    assert first_result[0].status is SolveStatus.FAILED


def test_solve_requires_solve_request_type(monkeypatch, tmp_path: Path) -> None:
    rt = _runtime_with_context(monkeypatch, _fake_resources())
    session = rt.create_session()
    with pytest.raises(InvalidRequestError):
        session.solve(None)  # type: ignore[arg-type]
