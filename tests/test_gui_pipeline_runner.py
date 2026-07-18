from __future__ import annotations

from pathlib import Path

from zesolver.core.models import SolveRequest, SolveResult, SolveStatus
from zesolver.gui_pipeline.pipeline_runner import PipelineGuiRunner
from zesolver.gui_pipeline.requests import GuiSettingsState
from zesolver.gui_pipeline.settings_adapter import build_gui_solve_request


class _Pipeline:
    def __init__(self, phase: str) -> None:
        self.phase = phase

    def solve(self, request: SolveRequest) -> SolveResult:
        solved = self.phase == "near"
        return SolveResult(
            request_id=request.request_id,
            input_path=request.input_path,
            output_path=request.output_path,
            status=SolveStatus.SOLVED if solved else SolveStatus.UNSOLVED,
            backend="NEAR" if solved else None,
            wcs_written=solved,
            center_ra_deg=None,
            center_dec_deg=None,
            pixel_scale_arcsec=None,
            orientation_deg=None,
            parity=None,
            inliers=None,
            rms_px=None,
            profile_ids={},
            catalog_status="test",
            warnings=(),
            error=None if solved else "unsolved",
        )


def test_pipeline_runner_emits_results_once() -> None:
    emitted = []
    runner = PipelineGuiRunner(
        result_callback=emitted.append,
        solver_pipeline_factory=lambda phase, _request: _Pipeline(phase),
    )
    request = build_gui_solve_request([Path("a.fit")], GuiSettingsState())
    summary = runner.run(request)
    assert len(summary.results) == 1
    assert len(emitted) == 1
    assert summary.results[0].backend == "NEAR"
