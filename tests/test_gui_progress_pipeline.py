from __future__ import annotations

from zesolver.core.models import SolveResult, SolveStatus
from zesolver.gui_pipeline.pipeline_runner import PipelineGuiRunner
from zesolver.gui_pipeline.requests import GuiSettingsState
from zesolver.gui_pipeline.settings_adapter import build_gui_solve_request


def test_pipeline_progress_emits_each_result_once(tmp_path) -> None:
    files = tuple(tmp_path / f"{idx}.fit" for idx in range(3))
    request = build_gui_solve_request(files, GuiSettingsState(workers=1))
    callback_results = []

    class _Pipeline:
        def solve(self, solve_request):
            return SolveResult(
                request_id=solve_request.request_id,
                input_path=solve_request.input_path,
                output_path=None,
                status=SolveStatus.SOLVED,
                backend="NEAR",
                wcs_written=True,
                center_ra_deg=None,
                center_dec_deg=None,
                pixel_scale_arcsec=None,
                orientation_deg=None,
                parity=None,
                inliers=12,
                rms_px=0.5,
                profile_ids={},
                catalog_status=None,
                warnings=(),
                error=None,
            )

    summary = PipelineGuiRunner(
        solver_pipeline_factory=lambda _phase, _request: _Pipeline(),
        result_callback=callback_results.append,
    ).run(request)

    assert len(callback_results) == 3
    assert len(summary.results) == 3
    assert [item.path for item in callback_results] == list(files)
