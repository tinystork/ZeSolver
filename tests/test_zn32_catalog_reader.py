from __future__ import annotations

import json
from pathlib import Path

import pytest
from astropy.io import fits

from tools.diagnose_zn32_catalog_reader import (
    REPO_ROOT,
    astap_area_to_filename_code,
    decode_d50_records,
    parse_astap_filenames1476,
)
from zeblindsolver.metadata_solver import _extract_angle, _extract_near_center_angle


def test_strict_near_numeric_ra_keyword_is_degrees() -> None:
    h = fits.Header()
    h["RA"] = 10.6125

    legacy = _extract_angle(h, ("RA",), is_ra=True)
    strict = _extract_near_center_angle(h, ("RA",), is_ra=True, strict_astap_iso=True)
    non_strict = _extract_near_center_angle(h, ("RA",), is_ra=True, strict_astap_iso=False)

    assert legacy == pytest.approx(159.1875)
    assert non_strict == pytest.approx(159.1875)
    assert strict == pytest.approx(10.6125)


def test_strict_near_textual_ra_still_accepts_hourangle() -> None:
    h = fits.Header()
    h["OBJCTRA"] = "00:42:27"

    strict = _extract_near_center_angle(h, ("OBJCTRA",), is_ra=True, strict_astap_iso=True)

    assert strict == pytest.approx(10.6125, abs=1e-3)


def test_astap_area_id_maps_to_physical_filename() -> None:
    source = REPO_ROOT / "ASTAP-main" / "command-line_version" / "unit_command_line_star_database.pas"
    if not source.exists():
        pytest.skip("ASTAP source checkout not present")

    files = parse_astap_filenames1476(source)

    assert astap_area_to_filename_code(files, 1188) == "2602"
    assert astap_area_to_filename_code(files, 1240) == "2702"


def test_d50_record_decode_known_first_m31_tile_record() -> None:
    path = Path("/opt/astap/d50_2702.1476")
    if not path.exists():
        pytest.skip("ASTAP D50 tile d50_2702.1476 not installed")

    records = decode_d50_records(path)
    first = records[0]

    assert first.raw_record_hex == "8f8906e145"
    assert first.raw_ra == 428431
    assert first.ra_deg == pytest.approx(9.1931324716)
    assert first.dec_deg == pytest.approx(44.4888084517)
    assert first.mag == pytest.approx(5.4)


def test_zn32_gate_report_catp_8_of_8_if_available() -> None:
    path = REPO_ROOT / "reports" / "zenear_zn32cat_gate.json"
    if not path.exists():
        pytest.skip("ZN3.2-CAT gate report not present")

    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["CAT-P"]["successes"] == 8
    assert data["CAT-P"]["total"] == 8


def test_zn32_record_identity_report_all_common_if_available() -> None:
    path = REPO_ROOT / "reports" / "zenear_zn32cat_record_identity_parity.json"
    if not path.exists():
        pytest.skip("ZN3.2-CAT identity report not present")

    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["counts"]["SAME_RAW_SAME_DECODE"] == 448
    assert data["total_astap_rows"] == 448
    assert data["total_zenear_rows"] == 448
