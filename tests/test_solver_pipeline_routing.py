from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from zesolver.core import EngineSolveResult, SolveRequest, SolveStatus, SolverPipeline
from zesolver.settings import ProductSettings, RuntimeOptions

from solver_pipeline_fixtures import empty_resources, near_resources, sample_wcs


class FakePort:
    def __init__(self, *results: EngineSolveResult) -> None:
        self.results = list(results)
        self.calls = 0

    def solve(self, request, *, resources, configuration):
        self.calls += 1
        if not self.results:
            return EngineSolveResult(status=SolveStatus.UNSOLVED, backend="FAKE", error="no fake result")
        return self.results.pop(0)


def _frame(tmp_path: Path) -> Path:
    path = tmp_path / "frame.fit"
    fits.PrimaryHDU(data=np.ones((16, 16), dtype=np.uint16)).writeto(path)
    return path


def _pipeline(tmp_path: Path, *, near, blind=None, resources=None, cancel_token=None, blind_enabled=True) -> SolverPipeline:
    return SolverPipeline(
        product_settings=ProductSettings(blind_enabled=blind_enabled),
        runtime_options=RuntimeOptions(cancel_token=cancel_token),
        catalog_resources=resources or near_resources(tmp_path, blind_count=6),
        near_solver=near,
        blind_solver=blind or FakePort(EngineSolveResult(status=SolveStatus.UNSOLVED, backend="BLIND4D")),
    )


def test_pipeline_near_success_skips_blind(tmp_path: Path) -> None:
    near = FakePort(
        EngineSolveResult(
            status=SolveStatus.SOLVED,
            backend="NEAR",
            wcs=sample_wcs(),
            center_ra_deg=184.6,
            center_dec_deg=47.2,
            inliers=39,
            rms_px=0.2,
        )
    )
    blind = FakePort(EngineSolveResult(status=SolveStatus.SOLVED, backend="BLIND4D", wcs=sample_wcs()))
    pipeline = _pipeline(tmp_path, near=near, blind=blind)

    result = pipeline.solve(SolveRequest(_frame(tmp_path), tmp_path / "out.fit", True, request_id="n1"))

    assert result.status is SolveStatus.SOLVED
    assert result.backend == "NEAR"
    assert result.wcs_written is True
    assert near.calls == 1
    assert blind.calls == 0
    assert pipeline.last_telemetry["near_attempted"] is True
    assert pipeline.last_telemetry["blind_attempted"] is False


def test_pipeline_near_failure_then_blind_success(tmp_path: Path) -> None:
    near = FakePort(EngineSolveResult(status=SolveStatus.UNSOLVED, backend="NEAR", error="near failed"))
    blind = FakePort(EngineSolveResult(status=SolveStatus.SOLVED, backend="BLIND4D", wcs=sample_wcs(), inliers=44, rms_px=0.5))
    pipeline = _pipeline(tmp_path, near=near, blind=blind)

    result = pipeline.solve(SolveRequest(_frame(tmp_path), tmp_path / "blind.fit", True))

    assert result.status is SolveStatus.SOLVED
    assert result.backend == "BLIND4D"
    assert near.calls == 1
    assert blind.calls == 1
    assert pipeline.last_telemetry["blind_result"] == "SOLVED"


def test_pipeline_near_failure_then_blind_failure(tmp_path: Path) -> None:
    near = FakePort(EngineSolveResult(status=SolveStatus.UNSOLVED, backend="NEAR", error="near failed"))
    blind = FakePort(EngineSolveResult(status=SolveStatus.UNSOLVED, backend="BLIND4D", error="blind failed"))
    pipeline = _pipeline(tmp_path, near=near, blind=blind)

    result = pipeline.solve(SolveRequest(_frame(tmp_path), None, True))

    assert result.status is SolveStatus.UNSOLVED
    assert result.error == "no_solver_produced_solution"
    assert near.calls == 1
    assert blind.calls == 1


def test_pipeline_catalog_unavailable_without_resources(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, near=FakePort(), resources=empty_resources())

    result = pipeline.solve(SolveRequest(_frame(tmp_path), None, True))

    assert result.status is SolveStatus.CATALOG_UNAVAILABLE
    assert result.error == "catalog_resources_unavailable"


def test_pipeline_cancellation_before_near(tmp_path: Path) -> None:
    near = FakePort(EngineSolveResult(status=SolveStatus.SOLVED, backend="NEAR", wcs=sample_wcs()))
    pipeline = _pipeline(tmp_path, near=near, cancel_token=lambda: True)

    result = pipeline.solve(SolveRequest(_frame(tmp_path), None, True))

    assert result.status is SolveStatus.CANCELLED
    assert near.calls == 0


def test_pipeline_cancellation_between_near_and_blind(tmp_path: Path) -> None:
    calls = {"count": 0}

    def cancel_after_near() -> bool:
        calls["count"] += 1
        return calls["count"] >= 2

    near = FakePort(EngineSolveResult(status=SolveStatus.UNSOLVED, backend="NEAR"))
    blind = FakePort(EngineSolveResult(status=SolveStatus.SOLVED, backend="BLIND4D", wcs=sample_wcs()))
    pipeline = _pipeline(tmp_path, near=near, blind=blind, cancel_token=cancel_after_near)

    result = pipeline.solve(SolveRequest(_frame(tmp_path), None, True))

    assert result.status is SolveStatus.CANCELLED
    assert near.calls == 1
    assert blind.calls == 0


def test_pipeline_rejected_false_solution_does_not_stop_blind(tmp_path: Path) -> None:
    near = FakePort(EngineSolveResult(status=SolveStatus.REJECTED_FALSE_SOLUTION, backend="NEAR", error="bad center"))
    blind = FakePort(EngineSolveResult(status=SolveStatus.SOLVED, backend="BLIND4D", wcs=sample_wcs()))
    pipeline = _pipeline(tmp_path, near=near, blind=blind)

    result = pipeline.solve(SolveRequest(_frame(tmp_path), None, True))

    assert result.status is SolveStatus.SOLVED
    assert result.backend == "BLIND4D"
