from __future__ import annotations

from pathlib import Path

from zesolver.core.models import EngineSolveResult, SolveRequest, SolveStatus
from zesolver.core.result_adapter import failure_result, result_from_engine

from solver_pipeline_fixtures import sample_wcs


def test_result_adapter_preserves_engine_fields_and_profiles() -> None:
    request = SolveRequest(Path("in.fit"), Path("out.fit"), True, request_id="r1")
    engine = EngineSolveResult(
        status=SolveStatus.SOLVED,
        backend="NEAR",
        wcs=sample_wcs(),
        wcs_written=True,
        center_ra_deg=184.6,
        center_dec_deg=47.2,
        inliers=39,
        rms_px=0.2,
        warnings=("partial_catalog",),
    )

    result = result_from_engine(
        request,
        engine,
        profile_ids={"near": "zenear-v1", "blind": "zeblind4d-v1", "pipeline": "pipeline-v1"},
        catalog_status="READY_PARTIAL",
        warnings=("blind4d_coverage_not_all_sky",),
    )

    assert result.status is SolveStatus.SOLVED
    assert result.backend == "NEAR"
    assert result.pixel_scale_arcsec is not None
    assert result.orientation_deg is not None
    assert result.parity in {"positive", "negative"}
    assert result.inliers == 39
    assert result.rms_px == 0.2
    assert result.profile_ids["near"] == "zenear-v1"
    assert result.catalog_status == "READY_PARTIAL"
    assert result.warnings == ("blind4d_coverage_not_all_sky", "partial_catalog")


def test_failure_result_normalizes_engine_exception() -> None:
    request = SolveRequest(Path("in.fit"), None, True)

    result = failure_result(
        request,
        status=SolveStatus.FAILED,
        profile_ids={"near": "zenear-v1", "blind": "zeblind4d-v1", "pipeline": "pipeline-v1"},
        error="engine exploded",
    )

    assert result.status is SolveStatus.FAILED
    assert result.backend is None
    assert result.wcs_written is False
    assert result.error == "engine exploded"
