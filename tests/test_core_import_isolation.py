from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_core_import_does_not_load_root_entrypoint_or_pyside6() -> None:
    script = r"""
import builtins
import json
import sys
from pathlib import Path

root = Path.cwd() / "zesolver.py"
real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "PySide6" or name.startswith("PySide6."):
        raise RuntimeError("PySide6 import blocked for core isolation test")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from zesolver.core import SolverPipeline
from zesolver.core.blind_port import ProductionBlindSolverPort

modules = list(sys.modules.items())
root_modules = [
    name for name, mod in modules
    if getattr(mod, "__file__", None) and Path(mod.__file__).resolve() == root
]
payload = {
    "pipeline": SolverPipeline.__name__,
    "blind_port": ProductionBlindSolverPort.__name__,
    "entrypoint_loaded": "zesolver_entrypoint_blind_port" in sys.modules,
    "root_modules": root_modules,
    "pyside_loaded": any(name == "PySide6" or name.startswith("PySide6.") for name in sys.modules),
    "image_solver_loaded": any(getattr(mod, "ImageSolver", None) is not None for _, mod in modules),
    "batch_solver_loaded": any(getattr(mod, "BatchSolver", None) is not None for _, mod in modules),
}
print(json.dumps(payload, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        env=_clean_env(),
    )
    payload = json.loads(result.stdout)
    assert payload["pipeline"] == "SolverPipeline"
    assert payload["blind_port"] == "ProductionBlindSolverPort"
    assert payload["entrypoint_loaded"] is False
    assert payload["root_modules"] == []
    assert payload["pyside_loaded"] is False
    assert payload["image_solver_loaded"] is False
    assert payload["batch_solver_loaded"] is False


def test_shared_blind_builder_works_with_synthetic_manifest(tmp_path: Path) -> None:
    script = f"""
from pathlib import Path
from types import SimpleNamespace

from zesolver.solver_config import build_blind_solve_config

manifest = SimpleNamespace(
    manifest_path=Path({str(tmp_path / "manifest.json")!r}),
    enabled_index_paths=(Path({str(tmp_path / "a.npz")!r}), Path({str(tmp_path / "b.npz")!r})),
)
inputs = SimpleNamespace(
    blind_backend_profile="zeblind_4d_experimental",
    blind_4d_manifest_path=manifest.manifest_path,
    blind_4d_loaded_manifest=manifest,
    blind_max_candidates=10,
    blind_max_stars=500,
    blind_max_quads=8000,
    dev_detect_k_sigma=3.0,
    dev_detect_min_area=5,
    dev_bucket_cap_S=0,
    dev_bucket_cap_M=0,
    dev_bucket_cap_L=0,
    blind_quality_rms=1.2,
    blind_quality_inliers=40,
    blind_pixel_tolerance=2.5,
    blind_fast_mode=True,
    log_level="INFO",
    dev_bucket_limit_override=0,
    dev_vote_percentile=40,
    dev_collect_matches_vectorized_experimental=False,
    hint_ra_deg=None,
    hint_dec_deg=None,
    hint_radius_deg=None,
    hint_focal_mm=None,
    hint_pixel_um=None,
    hint_resolution_arcsec=None,
    hint_resolution_min_arcsec=None,
    hint_resolution_max_arcsec=None,
    downsample=1,
    dev_verify_logodds_enabled=False,
    dev_verify_logodds_bail=-24.0,
    dev_verify_logodds_stoplooking=24.0,
    dev_verify_logodds_min_validations=8,
    dev_hard_max_candidates_tried=0,
    dev_hard_max_validations=0,
    dev_depth_ladder_enabled=False,
    dev_depth_ladder_caps=(80, 160, 500),
    blind_index_scale_overlap_prefilter_enabled=False,
    blind_index_scale_overlap_proxy_lo_frac=0.05,
    blind_index_scale_overlap_proxy_hi_frac=0.95,
)
config = build_blind_solve_config(inputs, loaded_manifest=manifest)
assert config.blind_astrometry_4d_index_paths == tuple(str(path.resolve()) for path in manifest.enabled_index_paths)
assert config.blind_astrometry_4d_search_budget_s == 45.0
"""
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True, env=_clean_env())


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    return env
