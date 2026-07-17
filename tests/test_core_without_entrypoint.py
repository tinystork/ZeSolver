from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_core_imports_from_package_without_root_entrypoint(tmp_path: Path) -> None:
    package_root = tmp_path / "packages"
    for name in ("zesolver", "zeblindsolver", "zewcs290"):
        shutil.copytree(
            ROOT / name,
            package_root / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.zip"),
        )
    assert not (package_root / "zesolver.py").exists()

    script = r"""
from pathlib import Path
from types import SimpleNamespace

from zesolver.core import SolverPipeline
from zesolver.core.blind_port import ProductionBlindSolverPort
from zesolver.solver_config import build_blind_solve_config

root_entrypoint = Path.cwd() / "zesolver.py"
assert not root_entrypoint.exists()

manifest = SimpleNamespace(
    manifest_path=Path("manifest.json"),
    enabled_index_paths=(Path("d50_2822_S_q40000.npz"),),
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
assert SolverPipeline.__name__ == "SolverPipeline"
assert ProductionBlindSolverPort.__name__ == "ProductionBlindSolverPort"
assert config.blind_astrometry_4d_index_paths == tuple(str(path.resolve()) for path in manifest.enabled_index_paths)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(package_root)
    subprocess.run([sys.executable, "-c", script], cwd=package_root, env=env, check=True)
