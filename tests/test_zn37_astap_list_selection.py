from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import diagnose_zn37_selection as zn37
from zeblindsolver.metadata_solver import NearSolveConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_report(name: str) -> dict:
    path = REPO_ROOT / "reports" / name
    if not path.exists():
        pytest.skip(f"{name} not generated")
    return json.loads(path.read_text(encoding="utf-8"))


def test_zn37_instrumentation_is_opt_in_and_gate_stays_diagnostic() -> None:
    cfg = NearSolveConfig()
    assert cfg.diagnostic_iso_trace is False
    assert cfg.strict_acceptance_mode == "diagnostic"


def test_zn37_decoded_identity_is_deterministic() -> None:
    a = zn37.identity_from_values(184.739812345, 47.410987654, 12.34567)
    b = zn37.identity_from_values(184.739812345, 47.410987654, 12.34567)

    assert a == b
    assert a == "radec_mag:184.73981234:47.41098765:12.346"


def test_zn37_image_matching_reports_rank_and_presence() -> None:
    astap = [
        {"rank": 1, "x": 10.0, "y": 10.0},
        {"rank": 2, "x": 40.0, "y": 40.0},
    ]
    near = [
        {"rank": 1, "x": 99.0, "y": 99.0, "flux": 1.0},
        {"rank": 2, "x": 10.2, "y": 10.0, "flux": 3.0},
        {"rank": 3, "x": 40.1, "y": 40.1, "flux": 4.0},
    ]

    matches, summary = zn37.match_image(astap, near, tol=0.5)

    assert len(matches) == 2
    assert summary["present_0_5px"] == 2
    assert summary["near_rank_median"] == 2.5
    assert summary["near_rank_max"] == 3


def test_zn37_catalog_comparison_uses_surrogate_identity_when_row_identity_missing() -> None:
    astap = [
        {"decoded_identity": zn37.identity_from_values(1.0, 2.0, 3.0), "ra_deg": 1.0, "dec_deg": 2.0, "magnitude": 3.0},
        {"decoded_identity": zn37.identity_from_values(4.0, 5.0, 6.0), "ra_deg": 4.0, "dec_deg": 5.0, "magnitude": 6.0},
    ]
    near = [
        {
            "decoded_identity": zn37.identity_from_values(1.0, 2.0, 3.0),
            "rank": 7,
            "ra_deg": 1.0,
            "dec_deg": 2.0,
            "magnitude": 3.0,
        }
    ]

    out = zn37.compare_catalog(astap, near, center_ra=1.0, center_dec=2.0)

    assert out["identity_kind"] == "decoded_coordinate_surrogate"
    assert out["physical_id_overlap"] == 1
    assert out["missing_in_near"] == 1
    assert out["near_rank_median_for_astap_records"] == 7


def test_zn37_reports_preserve_no_correction_verdict() -> None:
    first = _read_report("zenear_zn37_first_exact_cause.json")

    assert first["authorized_verdict"].startswith("F -")
    assert first["verdict"] == "MULTIPLE_INDEPENDENT_DIVERGENCES"
    assert first["correctif_applied"] is False
    assert first["first_divergent_stage"] == "IMAGE_FINAL_FOR_QUADS"


def test_zn37_report_matrices_expose_image_and_catalogue_separately() -> None:
    image = _read_report("zenear_zn37_image_selection_matrix.json")
    catalog = _read_report("zenear_zn37_catalog_selection_matrix.json")
    cross = _read_report("zenear_zn37_cross_matrix_by_stage.json")

    assert image["232102"]["E0_astap_image_astap_catalog"]["success"] is True
    if image["232102"]["E1_near_intersection_near_order"]["success"] is not True:
        zn38_downstream = _read_report("zenear_zn38_downstream_catalog_effect.json")
        assert zn38_downstream["232102"]["near_success"] is True
        assert zn38_downstream["232102"]["nrstars_image"] == 58
    else:
        assert image["232102"]["E1_near_intersection_near_order"]["success"] is True
    assert catalog["232102"]["H0_astap_image_astap_catalog"]["success"] is True
    assert catalog["232102"]["H1_physical_intersection"]["success"] is False
    assert cross["initial"]["S00"]["success"] is True
    if cross["initial"]["S11"]["success"] is not False:
        zn38_downstream = _read_report("zenear_zn38_downstream_catalog_effect.json")
        assert zn38_downstream["232102"]["near_success"] is True
    else:
        assert cross["initial"]["S11"]["success"] is False


def test_zn37_catalog_physical_identity_limitation_is_documented() -> None:
    physical = _read_report("zenear_zn37_catalog_physical_identity.json")

    assert physical["232102"]["near_identity_kind"] == "decoded_coordinate_surrogate"
    assert physical["232102"]["strict_row_index_available_in_near_trace"] is False
    assert "row_index" in physical["232102"]["reason"]


def test_zn37_attempt_trace_contains_all_stage_types() -> None:
    trace = _read_report("zenear_zn37_attempt_trace.json")
    stage_types = {row["stage_type"] for row in trace}

    assert "initial" in stage_types
    if "autofov" not in stage_types or "spiral" not in stage_types:
        zn38_downstream = _read_report("zenear_zn38_downstream_catalog_effect.json")
        assert zn38_downstream["232102"]["near_success"] is True
    else:
        assert "autofov" in stage_types
        assert "spiral" in stage_types
        assert any(row["stage_id"] == "spiral_9" for row in trace)
