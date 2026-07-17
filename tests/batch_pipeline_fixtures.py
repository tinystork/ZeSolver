from __future__ import annotations

import time
from pathlib import Path

from zesolver.core.models import SolveRequest, SolveResult, SolveStatus


def request(name: str) -> SolveRequest:
    return SolveRequest(Path(f"{name}.fit"), None, True, request_id=name)


def result_for(req: SolveRequest, status: SolveStatus, backend: str | None, error: str | None = None) -> SolveResult:
    return SolveResult(
        request_id=req.request_id,
        input_path=req.input_path,
        output_path=req.output_path,
        status=status,
        backend=backend,
        wcs_written=(status is SolveStatus.SOLVED),
        center_ra_deg=None,
        center_dec_deg=None,
        pixel_scale_arcsec=None,
        orientation_deg=None,
        parity=None,
        inliers=None,
        rms_px=None,
        profile_ids={"near": "zenear-v1", "blind": "zeblind4d-v1", "pipeline": "pipeline-v1"},
        catalog_status="legacy",
        warnings=(),
        error=error,
    )


class ScriptedPipeline:
    def __init__(self, phase: str, script: dict[str, dict[str, SolveStatus]], calls: list[str], delay: float = 0.0) -> None:
        self.phase = phase
        self.script = script
        self.calls = calls
        self.delay = delay

    def solve(self, req: SolveRequest) -> SolveResult:
        if self.delay:
            time.sleep(self.delay)
        self.calls.append(f"{self.phase}:{req.request_id}")
        status = self.script.get(self.phase, {}).get(str(req.request_id), SolveStatus.UNSOLVED)
        if status is SolveStatus.FAILED:
            raise RuntimeError(f"{self.phase} boom")
        backend = "NEAR" if self.phase == "near" and status is SolveStatus.SOLVED else None
        if self.phase == "blind" and status is SolveStatus.SOLVED:
            backend = "BLIND4D"
        return result_for(req, status, backend, None if status is SolveStatus.SOLVED else f"{self.phase} failed")


def factory(script: dict[str, dict[str, SolveStatus]], calls: list[str], delay: float = 0.0):
    return lambda phase: ScriptedPipeline(phase, script, calls, delay=delay)
