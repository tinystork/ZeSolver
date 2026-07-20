from __future__ import annotations

import json
from pathlib import Path

import pytest
from astropy.io import fits

from tools.diagnose_zn31_astap_oracles import REPO_ROOT
from zeblindsolver.metadata_solver import (
    NearSolveConfig,
    _extract_angle,
    _extract_near_center_angle,
    choose_astap_compatible_bin_factor,
)


def _load_report(name: str):
    path = REPO_ROOT / "reports" / name
    if not path.exists():
        pytest.skip(f"ZN3.4 report missing: {name}")
    return json.loads(path.read_text(encoding="utf-8"))


def test_zn34_m31_frozen_runtime_baseline_is_8_of_8_conformant() -> None:
    rows = _load_report("zenear_zn34_m31_frozen_baseline.json")

    assert len(rows) == 8
    assert all(row["solve"]["success"] for row in rows)
    assert {row["wcs"]["classification"] for row in rows} == {"WCS_CONFORMANT"}
    assert all(row["solve"]["stats"]["reason"] == "strict_astap_iso_mirror" for row in rows)


def test_zn34_broad_matrix_does_not_hide_wcs_failures() -> None:
    matrix = _load_report("zenear_zn34_final_matrix.json")
    promotion = (REPO_ROOT / "reports" / "zenear_zn34_promotion_gate.md").read_text(encoding="utf-8")

    wrong_wcs = sum(group.get("wrong_WCS", 0) for group in matrix.values())
    assert wrong_wcs > 0
    assert "Verdict: E - Faux positif ou WCS incorrect" in promotion
    assert "Promouvable: non" in promotion


def test_zn34_negative_controls_have_no_false_positive() -> None:
    rows = _load_report("zenear_zn34_negative_controls.json")

    assert rows
    assert all(not row["success"] for row in rows)
    assert all(not row["false_positive"] for row in rows)


def test_zn34_flux_order_is_explicit_not_synthetic_flux() -> None:
    audit = _load_report("zenear_zn34_flux_order_audit.json")

    assert audit["synthetic_flux_removed"] is True
    assert audit["strict_order_representation"] == "img_ranks = np.arange(stars.size) in strict ASTAP-ISO"
    assert audit["native_flux_source"] == "ASTAP HFD measured flux"
    assert audit["can_affect_non_strict"] is False


def test_zn34_detector_path_records_m31_30sigma_and_local_sections() -> None:
    coverage = _load_report("zenear_zn34_detector_path_coverage.json")
    m31_230409 = [diag for path, diag in coverage.items() if "230409" in path and path.endswith("_runtime.fit")]

    assert m31_230409
    assert all(diag["global_candidates"] == 131 for diag in m31_230409)
    assert all(diag["local_candidates"] == 252 for diag in m31_230409)
    assert all(diag["final_image_stars"] == 252 for diag in m31_230409)


def test_zn34_binning_policy_covers_bin1_and_reported_bin2() -> None:
    assert choose_astap_compatible_bin_factor(width=200, height=200, requested=None) == 1
    assert choose_astap_compatible_bin_factor(width=1920, height=1080, requested=2) == 2

    report = _load_report("zenear_zn34_binning_policy_validation.json")
    assert any(row.get("bin_factor") == 2 for row in report.values())


def test_zn34_strict_ra_numeric_degrees_and_text_hourangle() -> None:
    numeric = fits.Header()
    numeric["RA"] = 10.6125
    assert _extract_angle(numeric, ("RA",), is_ra=True) == pytest.approx(159.1875)
    assert _extract_near_center_angle(numeric, ("RA",), is_ra=True, strict_astap_iso=True) == pytest.approx(10.6125)

    text = fits.Header()
    text["OBJCTRA"] = "00:42:27"
    assert _extract_near_center_angle(text, ("OBJCTRA",), is_ra=True, strict_astap_iso=True) == pytest.approx(10.6125, abs=1e-3)


def test_zn34_non_strict_detector_defaults_remain_historical() -> None:
    cfg = NearSolveConfig(astap_iso_strict=False)

    assert cfg.detect_backend == "auto"
    assert cfg.detect_k_sigma == 4.0
    assert cfg.detect_min_area == 8


def test_zn34_determinism_probe_is_functionally_stable() -> None:
    data = _load_report("zenear_zn34_determinism.json")

    assert data["repetitions"] == 3
    assert data["cases"]
    assert all(case["stable_success"] for case in data["cases"].values())
    assert all(case["stable_inliers"] for case in data["cases"].values())
    assert all(case["stable_rms"] for case in data["cases"].values())


def test_zn34_runtime_dependency_audit_has_no_astap_runtime() -> None:
    audit = _load_report("zenear_zn34_packaging_audit.json")

    assert audit["astap_runtime_required"] is False
    assert audit["lazarus_runtime_required"] is False
    assert audit["dump_runtime_required"] is False
