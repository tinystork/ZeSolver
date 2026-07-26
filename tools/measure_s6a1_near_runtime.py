#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zesolver.catalog_resources import resolve_catalog_resources
from zesolver.gui_pipeline.pipeline_runner import PipelineGuiRunner
from zesolver.gui_pipeline.requests import GuiSettingsState
from zesolver.gui_pipeline.settings_adapter import build_gui_solve_request


def main() -> int:
    args = _parse_args()
    paths = _input_paths(args)
    if not paths:
        raise SystemExit("no FITS files selected")
    resources = resolve_catalog_resources(catalog_library=args.catalog_library)

    if args.stub_near:
        import zesolver.core.pipeline as pipeline_module

        def fake_near_solve(fits_path, index_root, *, catalog_provider=None, **kwargs):
            return {
                "success": True,
                "message": "stub_near_solved",
                "stats": {"inliers": 0, "rms_px": 0.0, "pix_scale_arcsec": None, "strict_acceptance": {}},
                "wrote_wcs": False,
            }

        pipeline_module.near_solve = fake_near_solve
    if args.stub_blind:
        import zesolver.core.blind_port as blind_port_module
        from zesolver.core.models import EngineSolveResult, SolveStatus

        def fake_blind_solve(self, request, *, resources, configuration):
            return EngineSolveResult(status=SolveStatus.UNSOLVED, backend="BLIND4D", error="stub_blind_disabled")

        blind_port_module.ProductionBlindSolverPort.solve = fake_blind_solve

    emitted: list[tuple[float, str, str]] = []
    progress_events: list[tuple[float, int, str | None]] = []
    started = time.perf_counter()

    def on_result(result) -> None:
        emitted.append((time.perf_counter() - started, str(result.path), str(result.status)))

    def on_progress(progress) -> None:
        progress_events.append((time.perf_counter() - started, int(progress.completed), progress.current_phase))

    state = GuiSettingsState(
        workers=max(1, int(args.workers)),
        preserve_order=True,
        use_blind=False,
        catalog_resources=resources,
        catalog_library_path=args.catalog_library,
        legacy_config=object(),
    )
    request = build_gui_solve_request(paths, state)
    summary = PipelineGuiRunner(progress_callback=on_progress, result_callback=on_result).run(request)
    duration = time.perf_counter() - started
    statuses: dict[str, int] = {}
    for result in summary.results:
        statuses[str(result.status)] = statuses.get(str(result.status), 0) + 1
    payload = {
        "input_count": len(paths),
        "workers": int(args.workers),
        "stub_near": bool(args.stub_near),
        "duration_s": round(duration, 6),
        "first_result_s": round(emitted[0][0], 6) if emitted else None,
        "statuses": statuses,
        "telemetry": dict(summary.telemetry or {}),
        "result_events": emitted[:10],
        "progress_events": progress_events[:20],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _input_paths(args: argparse.Namespace) -> tuple[Path, ...]:
    values: list[Path] = []
    for item in args.inputs:
        path = Path(item).expanduser()
        if path.is_dir():
            values.extend(sorted(path.glob("*.fit")))
            values.extend(sorted(path.glob("*.fits")))
        else:
            values.append(path)
    unique = tuple(dict.fromkeys(path.resolve() for path in values if path.exists()))
    if args.max_files and args.max_files > 0:
        return unique[: int(args.max_files)]
    return unique


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure S6A-1 ZeNear catalog runtime lifecycle counters.")
    parser.add_argument("inputs", nargs="+", help="FITS files or directories")
    parser.add_argument("--catalog-library", type=Path, default=Path("/home/tristan/ZeSolverCatalog/new"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-files", type=int, default=30)
    parser.add_argument("--stub-near", action="store_true", help="Bypass scientific Near solve and measure catalog lifecycle only.")
    parser.add_argument("--stub-blind", action="store_true", help="Return immediately from Blind fallback so Near measurements stay bounded.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
