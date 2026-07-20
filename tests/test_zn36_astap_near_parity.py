from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools import diagnose_zn36_parity as zn36
from zeblindsolver.metadata_solver import (
    NearSolveConfig,
    _astap_iso_collect_quad_hits,
    _astap_iso_find_quads,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_zn36_diagnostic_trace_is_opt_in_and_gate_default_is_diagnostic() -> None:
    cfg = NearSolveConfig()
    assert cfg.strict_acceptance_mode == "diagnostic"
    assert cfg.diagnostic_iso_trace is False


def test_zn36_spatial_parity_distinguishes_count_from_position() -> None:
    astap = np.asarray([[0.0, 0.0], [10.0, 10.0]], dtype=float)
    near = np.asarray([[0.01, 0.0], [10.1, 10.0], [50.0, 50.0]], dtype=float)
    out = zn36.spatial_parity(astap, near)

    assert out["count_astap"] == 2
    assert out["count_near"] == 3
    assert out["common_0.25px"] == 2
    assert out["extra_in_near_1px"] == 1


def test_zn36_quad_hits_are_deterministic_for_identical_quad_sets() -> None:
    pts = np.asarray(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 11.0],
            [10.0, 11.0],
            [25.0, 0.0],
            [25.0, 11.0],
            [40.0, 0.0],
            [40.0, 11.0],
        ],
        dtype=float,
    )
    quads = _astap_iso_find_quads(pts, pts.shape[0])
    hits = _astap_iso_collect_quad_hits(quads, quads, minimum_count=3, quad_tolerance=0.007)

    assert hits["matches_raw"] >= 3
    assert hits["matches_kept"] >= 3
    assert hits["hits"][0]["retained"] is True


def test_zn36_reports_expose_h00_and_no_correctif_when_generated() -> None:
    first_path = REPO_ROOT / "reports" / "zenear_zn36_first_causal_divergence.json"
    matrix_path = REPO_ROOT / "reports" / "zenear_zn36_list_cross_matrix.json"
    if not first_path.exists() or not matrix_path.exists():
        pytest.skip("ZN3.6 reports not generated")

    first = json.loads(first_path.read_text(encoding="utf-8"))
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))

    assert first["case_id"] == "232102"
    assert first["correctif_applied"] is False
    assert "H00" in matrix["232102"]
    assert matrix["232102"]["H00"]["success"] is True


def test_zn36_selected_trace_stage_is_recorded_when_generated() -> None:
    counts_path = REPO_ROOT / "reports" / "zenear_zn36_near_stage_counts.json"
    if not counts_path.exists():
        pytest.skip("ZN3.6 reports not generated")
    counts = json.loads(counts_path.read_text(encoding="utf-8"))

    assert counts["232102"]["near_selected_trace_stage"]
    assert counts["232102"]["image_final_for_quads_count"] >= 1
    assert counts["232102"]["catalog_final_for_quads_count"] >= 1
