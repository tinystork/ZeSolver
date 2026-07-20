from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from astropy.io import fits

from tools import diagnose_zn35_replay as replay
from zeblindsolver.metadata_solver import NearSolveConfig


def _mini_case(tmp_path: Path) -> dict[str, object]:
    src = tmp_path / "Light_M_31_11_30.0s_IRCUT_20250922-230409.fit"
    hdr = fits.Header()
    hdr["OBJECT"] = "M 31"
    hdr["RA"] = 10.6125
    hdr["DEC"] = 40.865278
    hdr["FOCALLEN"] = 250.0
    hdr["XPIXSZ"] = 2.9
    hdr["YPIXSZ"] = 2.9
    fits.PrimaryHDU(data=np.zeros((16, 24), dtype=np.float32), header=hdr).writeto(src)
    return {
        "case_id": "230409",
        "role": "m31_positive",
        "source_path": str(src),
        "SHA256": replay.sha256_file(src),
        "expected_astap": True,
        "expected_near": True,
        "expected_4d": False,
        "reason_for_selection": "unit test mini case",
    }


def test_zn35b_gate_defaults_to_diagnostic() -> None:
    assert NearSolveConfig().strict_acceptance_mode == "diagnostic"


def test_zn35b_prepare_creates_independent_pixel_identical_branches(tmp_path: Path) -> None:
    case = _mini_case(tmp_path)
    out = replay.prepare_case(case, tmp_path / "runs")

    assert out["success"] is True
    provenance = out["outputs"]
    assert provenance["pixels_identical_before_resolution"] is True
    assert set(provenance["branches"]) == {"clean_base", "astap_branch", "zenear_branch", "chain_4d_branch"}
    assert provenance["clean_wcs_present"] is False
    assert provenance["hint_header"]["RA"] == 10.6125


def test_zn35b_stage_resume_keeps_success_without_rerun(tmp_path: Path) -> None:
    case_root = tmp_path / "case"
    calls = {"n": 0}

    def fn() -> dict[str, object]:
        calls["n"] += 1
        return {"success": True, "outputs": {"value": calls["n"]}}

    first = replay.run_stage_wrapper(case_root=case_root, stage="prepare", timeout_s=10, resume=True, force=False, fn=fn)
    second = replay.run_stage_wrapper(case_root=case_root, stage="prepare", timeout_s=10, resume=True, force=False, fn=fn)

    assert first["status"] == "SUCCESS"
    assert second["status"] == "SUCCESS"
    assert calls["n"] == 1
    assert replay.read_json(case_root / "stages" / "prepare.json")["outputs"]["value"] == 1


def test_zn35b_stage_timeout_is_persisted(tmp_path: Path) -> None:
    case_root = tmp_path / "case"

    def slow() -> dict[str, object]:
        time.sleep(1.0)
        return {"success": True}

    result = replay.run_stage_wrapper(case_root=case_root, stage="near", timeout_s=0.01, resume=False, force=False, fn=slow)

    saved = replay.read_json(case_root / "stages" / "near.json")
    assert result["status"] == "TIMEOUT"
    assert saved["status"] == "TIMEOUT"
    assert saved["timeout"] is True

