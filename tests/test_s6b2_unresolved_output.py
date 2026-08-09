from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

from zesolver.core.models import SolveResult, SolveStatus
from zesolver.core.terminal_reasons import TerminalReasonCode
from zesolver.output_contract import UNRESOLVED_DIRECTORY_NAME
from zesolver.unresolved_output import move_unresolved_results


def _load_app_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("zesolver_app_s6b2_scan", root / "zesolver/_app.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fits")
    return path


def _result(path: Path, reason: str, *, status: SolveStatus = SolveStatus.UNSOLVED) -> SolveResult:
    return SolveResult(
        request_id=path.name,
        input_path=path,
        output_path=None,
        status=status,
        backend=None,
        wcs_written=False,
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
        error="failed",
        terminal_reason_code=reason,
    )


def test_move_unresolved_disabled_keeps_files_in_place(tmp_path):
    root = tmp_path / "input"
    image = _write(root / "a.fit")

    summary = move_unresolved_results(
        input_root=root,
        results=(_result(image, TerminalReasonCode.ALL_ENABLED_SOLVERS_EXHAUSTED.value),),
        terminal_status="completed",
        requested=False,
    )

    assert summary.eligible == 1
    assert summary.moved == 0
    assert image.exists()
    assert not (root / UNRESOLVED_DIRECTORY_NAME).exists()


def test_completed_batch_moves_scientific_unresolved_manifest_and_sidecars(tmp_path):
    root = tmp_path / "input"
    image = _write(root / "session_A" / "a.fit")
    known = _write(Path(str(image) + ".wcs.json"))
    unknown = _write(image.with_suffix(".json"))

    summary = move_unresolved_results(
        input_root=root,
        results=(_result(image, TerminalReasonCode.NEAR_UNRESOLVED_BLIND_UNAVAILABLE.value),),
        terminal_status="completed",
        requested=True,
        run_id=7,
    )

    moved = root / UNRESOLVED_DIRECTORY_NAME / "session_A" / "a.fit"
    assert summary.eligible == 1
    assert summary.moved == 1
    assert moved.exists()
    assert (root / UNRESOLVED_DIRECTORY_NAME / "session_A" / "a.fit.wcs.json").exists()
    assert not image.exists()
    assert not known.exists()
    assert unknown.exists()
    assert summary.manifest_path is not None
    payload = json.loads(summary.manifest_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "zesolver.unresolved.v1"
    assert payload["summary"] == {"eligible": 1, "moved": 1, "move_failed": 0}
    assert payload["files"][0]["original_relative_path"] == "session_A/a.fit"
    assert payload["files"][0]["destination_relative_path"] == "unresolved_by_zesolver/session_A/a.fit"


def test_cancelled_batch_never_moves(tmp_path):
    root = tmp_path / "input"
    image = _write(root / "a.fit")

    summary = move_unresolved_results(
        input_root=root,
        results=(_result(image, TerminalReasonCode.ALL_ENABLED_SOLVERS_EXHAUSTED.value),),
        terminal_status="cancelled",
        requested=True,
    )

    assert summary.moved == 0
    assert image.exists()


def test_collision_never_overwrites_existing_destination(tmp_path):
    root = tmp_path / "input"
    image = _write(root / "a.fit")
    _write(root / UNRESOLVED_DIRECTORY_NAME / "a.fit")

    summary = move_unresolved_results(
        input_root=root,
        results=(_result(image, TerminalReasonCode.ALL_ENABLED_SOLVERS_EXHAUSTED.value),),
        terminal_status="completed",
        requested=True,
    )

    assert (root / UNRESOLVED_DIRECTORY_NAME / "a.fit").read_bytes() == b"fits"
    assert (root / UNRESOLVED_DIRECTORY_NAME / "a__2.fit").exists()
    assert summary.records[0].destination_relative_path == "unresolved_by_zesolver/a__2.fit"


def test_technical_failures_are_not_moved(tmp_path):
    root = tmp_path / "input"
    image = _write(root / "a.fit")

    summary = move_unresolved_results(
        input_root=root,
        results=(_result(image, TerminalReasonCode.RUNTIME_ERROR.value, status=SolveStatus.FAILED),),
        terminal_status="completed",
        requested=True,
    )

    assert summary.eligible == 0
    assert image.exists()


def test_scanner_ignores_unresolved_directory_at_any_depth(tmp_path):
    root = tmp_path / "input"
    keep = _write(root / "a.fit")
    _write(root / UNRESOLVED_DIRECTORY_NAME / "b.fit")
    _write(root / "nested" / UNRESOLVED_DIRECTORY_NAME / "c.fit")

    appmod = _load_app_module()
    found = tuple(appmod._iter_image_files(root, {".fit"}))

    assert found == (keep,)
