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


class _ScriptedPipeline:
    def __init__(self, phase: str, calls: list[str]) -> None:
        self.phase = phase
        self.calls = calls

    def solve(self, request: SolveRequest) -> SolveResult:
        self.calls.append(f"{self.phase}:{request.input_path.suffix.lower()}:{request.request_id}")
        near_solved = {"0", "3"}
        blind_solved = {"1", "4"}
        if self.phase == "near" and request.request_id in near_solved:
            status = SolveStatus.SOLVED
            backend = "NEAR"
        elif self.phase == "blind" and request.request_id in blind_solved:
            status = SolveStatus.SOLVED
            backend = "BLIND4D"
        else:
            status = SolveStatus.UNSOLVED
            backend = self.phase.upper()
        return SolveResult(
            request_id=request.request_id,
            input_path=request.input_path,
            output_path=request.output_path,
            status=status,
            backend=backend,
            wcs_written=status is SolveStatus.SOLVED,
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
            error=None if status is SolveStatus.SOLVED else "unsolved",
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


def test_pipeline_runner_mixed_fits_batch_sends_near_failures_to_blind() -> None:
    calls: list[str] = []
    runner = PipelineGuiRunner(
        solver_pipeline_factory=lambda phase, _request: _ScriptedPipeline(phase, calls),
    )
    request = build_gui_solve_request(
        [Path("a.fit"), Path("b.fits"), Path("c.fts"), Path("d.fit"), Path("e.fits")],
        GuiSettingsState(use_blind=True, workers=2),
    )

    summary = runner.run(request)

    assert [result.backend for result in summary.results] == ["NEAR", "BLIND4D", "BLIND", "NEAR", "BLIND4D"]
    assert calls.count("near:.fit:0") == 1
    assert calls.count("near:.fits:1") == 1
    assert calls.count("near:.fts:2") == 1
    assert calls.count("near:.fit:3") == 1
    assert calls.count("near:.fits:4") == 1
    assert "blind:.fit:0" not in calls
    assert "blind:.fit:3" not in calls
    assert {call for call in calls if call.startswith("blind:")} == {
        "blind:.fits:1",
        "blind:.fts:2",
        "blind:.fits:4",
    }
