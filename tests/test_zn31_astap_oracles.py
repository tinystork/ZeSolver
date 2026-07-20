from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tools.diagnose_zn31_astap_oracles import (
    REPO_ROOT,
    compare_binned,
    prefixed_path,
    spherical_match,
)


def test_prefixed_path_prefers_runtime_dump(tmp_path: Path) -> None:
    stem = "frame"
    plain = tmp_path / "frame_astap_background.json"
    runtime = tmp_path / "frame_runtime_astap_background.json"
    plain.write_text("{}", encoding="utf-8")
    runtime.write_text("{}", encoding="utf-8")

    assert prefixed_path(tmp_path, stem, "_astap_background.json") == runtime


def test_spherical_match_direct_and_ra_divisor() -> None:
    astap = [
        {"ra_deg": "10.0", "dec_deg": "41.0"},
        {"ra_deg": "10.01", "dec_deg": "41.0"},
    ]
    zenear = [
        {"ra_deg": "150.0", "dec_deg": "41.0"},
        {"ra_deg": "151.0", "dec_deg": "41.0"},
    ]

    direct = spherical_match(astap, zenear)
    normalized = spherical_match(astap, zenear, zenear_ra_divisor=15.0)

    assert direct["common_2arcsec"] == 0
    assert normalized["common_2arcsec"] == 1


def test_real_astap_binned_dump_pixel_parity_if_available() -> None:
    stem = "Light_M_31_11_30.0s_IRCUT_20250922-230409"
    runtime = REPO_ROOT / "reports" / "zn1_runtime" / f"{stem}_runtime.fit"
    dump_dir = REPO_ROOT / "reports" / "zn31_astap_dumps"
    if not runtime.exists() or not prefixed_path(dump_dir, stem, "_astap_binned.fits").exists():
        pytest.skip("ZN3.1 ASTAP binned dump not present")

    parity = compare_binned(runtime, dump_dir, stem)

    assert parity["astap_dump_available"]
    assert parity["same_shape"]
    assert parity["mean_abs_error"] == 0.0
    assert parity["max_abs_error"] == 0.0
    assert parity["pixel_equivalent"] is True


def test_detection_passes_dump_is_parseable_if_available() -> None:
    stem = "Light_M_31_11_30.0s_IRCUT_20250922-230409"
    path = prefixed_path(REPO_ROOT / "reports" / "zn31_astap_dumps", stem, "_astap_detection_passes.csv")
    if not path.exists():
        pytest.skip("ZN3.1 detection passes dump not present")

    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))

    assert rows
    assert {"pass_id", "retry_index", "threshold", "candidate_count", "accepted_count"} <= set(rows[0])
    assert any(row["reason_for_next_pass"] == "thirty_sigma" for row in rows)


def test_catalog_radec_dump_is_parseable_if_available() -> None:
    stem = "Light_M_31_11_30.0s_IRCUT_20250922-230409"
    path = prefixed_path(REPO_ROOT / "reports" / "zn31_astap_dumps", stem, "_astap_catalog_raw.csv")
    if not path.exists():
        pytest.skip("ZN3.1 ASTAP catalog RA/Dec dump not present")

    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    ra = np.asarray([float(row["ra_deg"]) for row in rows[:10]], dtype=float)
    dec = np.asarray([float(row["dec_deg"]) for row in rows[:10]], dtype=float)

    assert len(rows) > 400
    assert np.all(np.isfinite(ra))
    assert np.all(np.isfinite(dec))
    assert float(np.median(ra)) < 20.0


def test_gate_reports_keep_oracle_success_if_available() -> None:
    path = REPO_ROOT / "reports" / "zenear_zn31_image_gate.json"
    if not path.exists():
        pytest.skip("ZN3.1 image gate report not present")

    data = json.loads(path.read_text(encoding="utf-8"))
    first = data["Light_M_31_11_30.0s_IRCUT_20250922-230409"]

    assert first["IMG-O0"]["success"] is True
    assert first["IMG-P"]["success"] is False
    assert first["IMG-P"]["failure_stage"] == "NO_SIGNATURE_MATCHES"
