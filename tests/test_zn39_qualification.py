from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from tools import diagnose_zn39_qualification as zn39
from zeblindsolver.fits_utils import to_luminance_for_solve
from zeblindsolver.metadata_solver import NearSolveConfig, astap_iso_image_for_solve


REPO_ROOT = Path(__file__).resolve().parents[1]


def _report(name: str):
    path = REPO_ROOT / "reports" / name
    if not path.exists():
        pytest.skip(f"{name} not generated")
    return json.loads(path.read_text(encoding="utf-8"))


def test_zn39_strict_defaults_remain_diagnostic_and_quiet() -> None:
    cfg = NearSolveConfig()

    assert cfg.strict_acceptance_mode == "diagnostic"
    assert cfg.diagnostic_iso_trace is False


def test_zn39_astap_iso_keeps_native_adu_range_for_uint16() -> None:
    arr = (np.arange(100, dtype=np.uint16).reshape(10, 10) * 512)
    hdu = fits.PrimaryHDU(data=arr)

    native = astap_iso_image_for_solve(hdu)
    normalized = to_luminance_for_solve(hdu)

    assert native.dtype == np.float32
    assert float(np.nanmax(native)) > 1000.0
    assert float(np.nanmax(normalized)) <= 1.0


def test_zn39_astap_iso_supports_2d_and_channel_first_3d() -> None:
    mono = astap_iso_image_for_solve(fits.PrimaryHDU(data=np.ones((5, 6), dtype=np.uint16) * 1234))
    cube = astap_iso_image_for_solve(
        fits.PrimaryHDU(data=np.stack([np.ones((5, 6)), np.ones((5, 6)) * 3, np.ones((5, 6)) * 5]).astype(np.float32))
    )

    assert mono.shape == (5, 6)
    assert cube.shape == (5, 6)
    assert float(np.nanmedian(cube)) == pytest.approx(3.0)


def test_zn39_astap_iso_rejects_incompatible_shape_explicitly() -> None:
    with pytest.raises(ValueError, match="Unsupported FITS data shape"):
        astap_iso_image_for_solve(fits.PrimaryHDU(data=np.arange(10, dtype=np.float32)))


def test_zn39_adu_audit_records_channel_last_as_unqualified_not_silent_fallback() -> None:
    audit = _report("zenear_zn39_native_adu_matrix.json")

    assert audit["matrix"]["uint16_2d"]["status"] == "SUPPORTED"
    assert audit["matrix"]["unsupported_1d"]["status"] == "EXPLICIT_ERROR"
    assert audit["matrix"]["cube_channel_last"]["status"] == "UNSUPPORTED_CHANNEL_LAST_SHAPE"
    assert audit["matrix"]["cube_channel_last"]["normalization_0_1_reaches_strict"] is False


def test_zn39_corpus_manifest_contains_required_groups_and_is_deduplicated() -> None:
    manifest = _report("zenear_zn39_corpus_manifest.json")
    groups = {}
    for row in manifest:
        groups[row["group"]] = groups.get(row["group"], 0) + 1

    assert groups["M31_canonical"] == 8
    assert groups["M106"] == 60
    assert groups["NGC3628"] == 2
    assert groups["NGC6888"] >= 17
    assert len({row["SHA256"] for row in manifest}) == len(manifest)


def test_zn39_near_replay_used_native_adu_and_no_generic_fallback() -> None:
    replay = _report("zenear_zn39_near_replay.json")
    ran = [row for row in replay.values() if row.get("status") == "RAN"]
    if not ran:
        pytest.skip("Near replay not executed")

    assert all(row["strict_detector_used_native_adu"] for row in ran)
    assert not any(row["generic_fallback_called"] for row in ran)
    assert not any(row["historical_blind_called"] for row in ran)


def test_zn39_near_matrix_confirms_replayed_positive_corpus() -> None:
    matrix = _report("zenear_zn39_near_matrix.json")

    assert matrix["M31_canonical"]["Near_WCS_CONFIRMED"] == 8
    assert matrix["M106"]["Near_WCS_CONFIRMED"] == matrix["M106"]["images_uniques"]
    assert matrix["NGC6888"]["Near_WRONG_FIELD"] == 0
    assert matrix["NGC3628"]["Near_WCS_CONFIRMED"] == 2


def test_zn39_wcs_validation_has_no_wrong_field_in_replay() -> None:
    validation = _report("zenear_zn39_independent_wcs_validation.json")
    confirmed = [row for row in validation.values() if row.get("classification") == "WCS_CONFIRMED"]

    assert len(confirmed) == len(validation)
    assert not [row for row in validation.values() if row.get("classification") == "WRONG_FIELD"]


def test_zn39_final_matrix_has_no_wrong_accepted_or_historical_backend() -> None:
    final = _report("zenear_zn39_final_matrix.json")
    summary = final["summary"]

    assert summary["counts"].get("NEAR_WRONG_ACCEPTED", 0) == 0
    assert summary["counts"].get("BLIND4D_WRONG_ACCEPTED", 0) == 0
    assert summary["historical_blind_called"] == 0
    assert summary["generic_fallback_called_in_strict"] == 0


def test_zn39_gate_and_chain_are_not_promoted_without_holdout_fixture() -> None:
    gate = _report("zenear_zn39_gate_calibration.json")
    fixture = _report("zenear_zn39_fallback_fixture.json")
    summary = (REPO_ROOT / "reports" / "zenear_zn39_summary.md").read_text(encoding="utf-8")

    assert gate["mode"] == "diagnostic"
    assert fixture["controlled_routing_fixture"]["status"] == "NOT_CREATED_IN_THIS_RUN"
    assert "Promotion stricte: non" in summary


def test_zn39_no_empirical_caps_or_object_exceptions_in_solver() -> None:
    source = (REPO_ROOT / "zeblindsolver" / "metadata_solver.py").read_text(encoding="utf-8")

    assert "[:58]" not in source
    assert "[:100]" not in source
    assert "[:146]" not in source
    assert "232102" not in source
    assert "OBJECT == " not in source

