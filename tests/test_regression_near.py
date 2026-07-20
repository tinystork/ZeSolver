from __future__ import annotations

import json
from pathlib import Path

import pytest
from astropy.wcs import WCS

from corpus_loader import CorpusDataMissing, iter_cases, validate_wcs_against_case


ROOT = Path(__file__).resolve().parents[1]


def _reference(name: str):
    return json.loads((ROOT / "tests" / "corpus" / "oracles" / name).read_text(encoding="utf-8"))


def _wcs_for_case(case_id: str) -> WCS:
    case = next(case for case in iter_cases(mode="near") if case.id == case_id)
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [case.raw["expected_center_ra_deg"], case.raw["expected_center_dec_deg"]]
    wcs.wcs.crpix = [540.0, 960.0]
    scale_deg = case.raw["expected_pixel_scale_arcsec"] / 3600.0
    wcs.wcs.cd = [[-scale_deg, 0.0], [0.0, scale_deg]]
    return wcs


def test_zenear_reference_oracle_records_current_positive_corpus() -> None:
    oracle = _reference("zenear_reference.json")

    assert oracle["summary"]["total_unique"] == 142
    assert oracle["summary"]["near_wcs_confirmed"] == 142
    assert oracle["summary"]["wrong_field"] == 0
    assert oracle["summary"]["generic_fallback_called"] == 0
    assert oracle["summary"]["m31_canonical"] == "8/8"


def test_zenear_wcs_validation_rejects_success_with_wrong_center() -> None:
    case = next(case for case in iter_cases(mode="near") if case.id == "near_m106_232102")
    wcs = _wcs_for_case(case.id)
    wcs.wcs.crval = [case.raw["expected_center_ra_deg"] + 1.0, case.raw["expected_center_dec_deg"]]

    with pytest.raises(AssertionError, match="CENTER_OUT_OF_TOLERANCE"):
        validate_wcs_against_case(wcs, case, inliers=39, rms_px=0.2)


def test_zenear_wcs_validation_accepts_reference_geometry() -> None:
    case = next(case for case in iter_cases(mode="near") if case.id == "near_m106_232102")
    validate_wcs_against_case(_wcs_for_case(case.id), case, inliers=39, rms_px=0.2)


@pytest.mark.corpus
@pytest.mark.slow
def test_zenear_corpus_files_resolve_or_skip_explicitly() -> None:
    for case in iter_cases(mode="near"):
        try:
            path = case.resolve_path()
        except CorpusDataMissing as exc:
            pytest.skip(str(exc))
        assert path.exists()
