import importlib.util
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("zesolver_app", REPO_ROOT / "zesolver.py")
assert SPEC is not None and SPEC.loader is not None
zesolver = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = zesolver
SPEC.loader.exec_module(zesolver)


def _solver(tmp_path: Path) -> zesolver.ImageSolver:
    solver = object.__new__(zesolver.ImageSolver)
    solver.config = zesolver.SolveConfig(
        db_root=tmp_path,
        input_dir=tmp_path,
        families=("d50",),
        blind_index_path=tmp_path,
        workers=6,
        blind_enabled=True,
        near_defer_blind_fallback=False,
    )
    solver._cancel_event = None
    solver._near_seed = None
    solver._near_warmstart_parallel_notified = False
    return solver


def _metadata(path: Path) -> zesolver.ImageMetadata:
    return zesolver.ImageMetadata(
        path=path,
        kind="fits",
        width=1080,
        height=1920,
        ra_deg=184.0,
        dec_deg=47.0,
        source="header",
        has_wcs=False,
    )


def test_batch_near_phase_defers_blind_when_fallback_is_disallowed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fallback_flags: list[bool] = []

    def fake_near_solve(*args, fallback_to_blind: bool, **kwargs):
        fallback_flags.append(fallback_to_blind)
        return {"success": False, "message": "near failed"}

    monkeypatch.setattr(zesolver, "near_solve", fake_near_solve)
    path = tmp_path / "frame.fit"

    result = _solver(tmp_path)._run_index_near_solver(
        path,
        _metadata(path),
        allow_blind_fallback=False,
    )

    assert result is None
    assert fallback_flags == [False, False]


def test_sequential_near_path_defers_4d_blind_fallback_to_batch_phase(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fallback_flags: list[bool] = []

    def fake_near_solve(*args, fallback_to_blind: bool, **kwargs):
        fallback_flags.append(fallback_to_blind)
        return {"success": False, "message": "near failed"}

    monkeypatch.setattr(zesolver, "near_solve", fake_near_solve)
    path = tmp_path / "frame.fit"

    result = _solver(tmp_path)._run_index_near_solver(
        path,
        _metadata(path),
        allow_blind_fallback=True,
    )

    assert result is None
    assert fallback_flags == [False, False]


def test_sequential_near_path_keeps_immediate_historical_blind_fallback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fallback_flags: list[bool] = []

    def fake_near_solve(*args, fallback_to_blind: bool, **kwargs):
        fallback_flags.append(fallback_to_blind)
        return {"success": False, "message": "near failed"}

    monkeypatch.setattr(zesolver, "near_solve", fake_near_solve)
    path = tmp_path / "frame.fit"
    solver = _solver(tmp_path)
    solver.config.blind_backend_profile = zesolver.HISTORICAL_PROFILE

    result = solver._run_index_near_solver(
        path,
        _metadata(path),
        allow_blind_fallback=True,
    )

    assert result is None
    assert fallback_flags == [False, False, True]


def test_blind_worker_count_is_memory_bounded(monkeypatch) -> None:
    monkeypatch.delenv("ZE_BLIND_WORKERS", raising=False)

    assert zesolver._auto_blind_worker_count(6, ram_gb=7.52) == 2
    assert zesolver._auto_blind_worker_count(6, ram_gb=12.0) == 3
    assert zesolver._auto_blind_worker_count(6, ram_gb=24.0) == 4
    assert zesolver._auto_blind_worker_count(6, ram_gb=48.0) == 6


def test_blind_worker_count_honors_safe_override(monkeypatch) -> None:
    monkeypatch.setenv("ZE_BLIND_WORKERS", "1")

    assert zesolver._auto_blind_worker_count(6, ram_gb=64.0) == 1
