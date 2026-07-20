from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from scipy.spatial import cKDTree

from tools.diagnose_zn31_astap_oracles import REPO_ROOT
from tools.diagnose_zn33_image_detector import matrix_case, points_from_rows
from zeblindsolver.metadata_solver import (
    NearSolveConfig,
    astap_adaptive_image_detection,
    astap_binned_to_full_coords,
    astap_section_grid,
    estimate_astap_global_background,
)


STEM = "Light_M_31_11_30.0s_IRCUT_20250922-230409"


def _dump_path(suffix: str) -> Path:
    return REPO_ROOT / "reports" / "zn31_astap_dumps" / f"{STEM}_runtime{suffix}"


def test_astap_section_grid_matches_m31_shape() -> None:
    sections = astap_section_grid(width=540, height=960)

    assert len(sections) == 104
    assert sections[0] == (0, 1, 68, 1, 74)
    assert sections[-1] == (103, 473, 538, 887, 958)


def test_astap_global_background_matches_oracle_if_available() -> None:
    binned = _dump_path("_astap_binned.fits")
    if not binned.exists():
        pytest.skip("ZN3.1 binned dump not present")

    image = np.asarray(fits.getdata(binned), dtype=np.float32)
    stats = estimate_astap_global_background(image, max_stars=500)

    assert stats["background"] == pytest.approx(1202.0)
    assert stats["noise"] == pytest.approx(83.0950273256029, abs=0.02)
    assert stats["star_level"] == pytest.approx(988.0, abs=0.1)
    assert stats["star_level2"] == pytest.approx(290.83258056640625, abs=0.1)


def test_astap_adaptive_detector_matches_230409_oracle_if_available() -> None:
    runtime = REPO_ROOT / "reports" / "zn1_runtime" / f"{STEM}_runtime.fit"
    raw_candidates = _dump_path("_astap_raw_image_candidates.csv")
    if not runtime.exists() or not raw_candidates.exists():
        pytest.skip("ZN3.1 M31 runtime/dumps not present")

    image = np.asarray(fits.getdata(runtime), dtype=np.float32)
    stars, diag = astap_adaptive_image_detection(image, bin_factor=2, max_stars=500, hfd_min=0.8)

    assert int(stars.size) == 252
    assert diag["raw_candidates"] == 252
    assert [(p["retry_index"], p["candidate_count"], p["reason"]) for p in diag["passes"]] == [
        (4, 0, "skipped_star_level"),
        (3, 0, "skipped_star_level2"),
        (2, 131, "thirty_sigma"),
        (1, 252, "section_sigma_clip"),
    ]

    rows = list(csv.DictReader(raw_candidates.open(newline="", encoding="utf-8")))
    astap = np.asarray([(float(r["x_full"]), float(r["y_full"])) for r in rows], dtype=float)
    python = np.column_stack((stars["x"], stars["y"])).astype(float)
    d, _ = cKDTree(python).query(astap, k=1)

    assert int((d <= 0.25).sum()) == 252
    assert float(np.max(d)) < 0.001


def test_astap_adaptive_detector_preserves_full_coord_mapping() -> None:
    x, y = astap_binned_to_full_coords(27.385792302565651, 65.219002603529873, 2)

    assert x == pytest.approx(55.271584605131302)
    assert y == pytest.approx(130.93800520705975)


def test_image_gate_with_corrected_catalog_reaches_transform_if_available() -> None:
    runtime = REPO_ROOT / "reports" / "zn1_runtime" / f"{STEM}_runtime.fit"
    catalog = REPO_ROOT / "reports" / "zn32cat_matrix_runs" / f"{STEM}_CATP_zenear_catalog_stars.csv"
    if not runtime.exists() or not catalog.exists():
        pytest.skip("ZN3.2 corrected catalog dump not present")

    image = np.asarray(fits.getdata(runtime), dtype=np.float32)
    stars, _diag = astap_adaptive_image_detection(image, bin_factor=2, max_stars=500, hfd_min=0.8)
    image_points = np.column_stack((stars["x"], stars["y"])).astype(np.float64)
    catalog_points = points_from_rows(list(csv.DictReader(catalog.open(newline="", encoding="utf-8"))))

    result = matrix_case(image_points, catalog_points)

    assert result["success"] is True
    assert result["matches_raw"] > 0
    assert result["refs"] >= 4


def test_non_strict_config_still_has_historical_detector_defaults() -> None:
    cfg = NearSolveConfig(astap_iso_strict=False)

    assert cfg.detect_backend == "auto"
    assert cfg.detect_k_sigma == 4.0
    assert cfg.detect_min_area == 8
