from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from tools import diagnose_zn38_image_selector as zn38
from zeblindsolver.fits_utils import to_luminance_for_solve
from zeblindsolver.metadata_solver import NearSolveConfig, astap_iso_image_for_solve


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_report(name: str) -> dict:
    path = REPO_ROOT / "reports" / name
    if not path.exists():
        pytest.skip(f"{name} not generated")
    return json.loads(path.read_text(encoding="utf-8"))


def test_zn38_instrumentation_defaults_remain_opt_in() -> None:
    cfg = NearSolveConfig()

    assert cfg.diagnostic_iso_trace is False
    assert cfg.strict_acceptance_mode == "diagnostic"


def test_zn38_strict_image_uses_native_adu_not_normalized_luminance() -> None:
    path = zn38.CASE_FITS["232102"]
    if not path.exists():
        pytest.skip("232102 FITS fixture missing")

    with fits.open(path, mode="readonly", memmap=False) as hdul:
        native = astap_iso_image_for_solve(hdul[0])
        normalized = to_luminance_for_solve(hdul[0])

    assert native.shape == normalized.shape
    assert float(np.nanmax(native)) > 1000.0
    assert float(np.nanmax(normalized)) <= 1.0
    assert float(np.nanmedian(native)) > 100.0
    assert float(np.nanmedian(normalized)) < 1.0


def test_zn38_detector_record_exposes_first_pixel_type_divergence() -> None:
    path = zn38.CASE_FITS["232102"]
    if not path.exists():
        pytest.skip("232102 FITS fixture missing")

    record = zn38.detector_record("232102")

    assert record["native_vs_astap_binned"]["verdict"] == "BINNED_PIXELS_IDENTICAL"
    assert record["normalized_vs_astap_binned"]["verdict"] == "PIXEL_TYPE_DIVERGENCE"
    assert record["native_detector"]["count"] == 58
    assert record["normalized_detector"]["count"] == 0


def test_zn38_reports_show_near_restored_without_catalog_patch() -> None:
    first = _read_report("zenear_zn38_first_image_cause.json")
    downstream = _read_report("zenear_zn38_downstream_catalog_effect.json")

    assert first["authorized_verdict"].startswith("A -")
    assert first["first_divergent_stage"] == "solve_near strict image input conversion"
    assert downstream["232102"]["nrstars_image"] == 58
    assert downstream["232102"]["catalog_final"] in {248, 249}
    assert downstream["232102"]["near_success"] is True


def test_zn38_no_empirical_caps_or_object_exceptions_in_patch() -> None:
    source = (REPO_ROOT / "zeblindsolver" / "metadata_solver.py").read_text(encoding="utf-8")

    assert "[:58]" not in source
    assert "[:100]" not in source
    assert "[:146]" not in source
    assert "232102" not in source
    assert "Light_mosaic_M 106" not in source


def test_zn38_legacy_fallback_does_not_replace_strict_detector() -> None:
    source = (REPO_ROOT / "zeblindsolver" / "metadata_solver.py").read_text(encoding="utf-8")

    assert "if (not strict_astap_iso) and detect_backend != \"astap\" and stars.size < 12:" in source
    assert "if (not strict_astap_iso) and detect_backend != \"astap\" and stars.size < 8:" in source
