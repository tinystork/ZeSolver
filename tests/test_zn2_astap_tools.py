import csv
import json
from pathlib import Path

import numpy as np

from tools import astap_zn2_build_and_compare as build_tool
from tools import diagnose_zn2_astap_internal_parity as parity_tool


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_compare_points_counts_overlap() -> None:
    a = np.asarray([(1.0, 1.0), (10.0, 10.0), (100.0, 100.0)])
    b = np.asarray([(1.2, 1.1), (10.3, 10.3), (200.0, 200.0)])

    result = parity_tool.compare_points(a, b)

    assert result["a_count"] == 3
    assert result["b_count"] == 3
    assert result["overlap_0.5px"] == 2
    assert result["overlap_1px"] == 2


def test_load_trace_validates_internal_dump_shape(tmp_path: Path) -> None:
    stem = "Light_M_31_test"
    prefix = tmp_path / stem
    (prefix.with_name(prefix.name + "_astap_internal_metadata.json")).write_text(
        json.dumps({"bin_factor": 2}),
        encoding="utf-8",
    )
    _write_csv(
        prefix.with_name(prefix.name + "_astap_internal_image_stars.csv"),
        [{"internal_id": "i0", "rank": 1, "x_full_resolution": 2.5, "y_full_resolution": 4.5, "bin_factor": 2}],
    )
    _write_csv(
        prefix.with_name(prefix.name + "_astap_internal_catalog_stars.csv"),
        [{"internal_id": "c0", "rank": 1, "ra_deg": 10.0, "dec_deg": 40.0, "x_projected": 1.0, "y_projected": 2.0}],
    )
    _write_csv(
        prefix.with_name(prefix.name + "_astap_internal_image_quads.csv"),
        [{"quad_id": "iq0", "star_id_0": "i0", "star_id_1": "i1", "star_id_2": "i2", "star_id_3": "i3", "signature_component_0": 1.0}],
    )
    _write_csv(
        prefix.with_name(prefix.name + "_astap_internal_catalog_quads.csv"),
        [{"quad_id": "cq0", "star_id_0": "c0", "star_id_1": "c1", "star_id_2": "c2", "star_id_3": "c3", "signature_component_0": 1.0}],
    )
    _write_csv(
        prefix.with_name(prefix.name + "_astap_internal_matches.csv"),
        [{"match_rank": 1, "image_quad_id": "iq0", "catalog_quad_id": "cq0", "signature_delta": 0.0, "accepted_by_hash": "true"}],
    )
    (prefix.with_name(prefix.name + "_astap_internal_solution.json")).write_text(
        json.dumps({"winning_image_quad_id": "iq0", "winning_catalog_quad_id": "cq0", "transform": {"a": 1.0}, "inliers": 4, "rms": 0.1}),
        encoding="utf-8",
    )

    trace = parity_tool.load_trace(stem, tmp_path)

    assert trace is not None
    assert len(trace.image_stars) == 1
    assert len(trace.catalog_stars) == 1
    assert len(trace.image_quads) == 1
    assert len(trace.catalog_quads) == 1
    assert len(trace.matches) == 1
    assert trace.solution is not None
    assert trace.solution.winning_image_quad_id == "iq0"


def test_try_build_reference_blocks_without_lazbuild(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(build_tool, "which_all", lambda names: {name: None for name in names})

    result = build_tool.try_build_reference(tmp_path / "ASTAP-main", tmp_path / "build")

    assert result["status"] == "blocked"
    assert "lazbuild not found" in result["reason"]
