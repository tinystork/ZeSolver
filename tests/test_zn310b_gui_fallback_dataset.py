from __future__ import annotations

import json
from pathlib import Path

import pytest
from astropy.io import fits
from astropy.wcs import WCS

from tools import diagnose_zn310b_gui_fallback as zn310b_log
from zeblindsolver.metadata_solver import NearSolveConfig
from zesolver.settings_store import PersistentSettings, _migrate_settings_if_needed


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
HINT_RA_KEYS = ("RA", "OBJCTRA", "OBJRA", "OBJ_RA", "TELRA", "CENTRA", "CRVAL1")
HINT_DEC_KEYS = ("DEC", "OBJCTDEC", "OBJDEC", "OBJ_DEC", "TELDEC", "CENTDEC", "CRVAL2")
ZEBLIND_4D_EXPERIMENTAL_PROFILE = "zeblind_4d_experimental"


def _json(name: str):
    path = REPORTS / name
    if not path.exists():
        pytest.skip(f"{name} not generated")
    return json.loads(path.read_text(encoding="utf-8"))


def _has_wcs(path: Path) -> bool:
    try:
        return bool(WCS(fits.getheader(path)).has_celestial)
    except Exception:
        return False


def test_zn310b_manifest_has_eight_deterministic_gui_cases() -> None:
    manifest = _json("zenear_zn310b_gui_manifest.json")

    assert len(manifest["items"]) == 8
    assert [item["variant"] for item in manifest["items"]] == [
        "CONTROL",
        "CONTROL",
        "CONTROL",
        "NOHINT",
        "NOHINT",
        "NOHINT",
        "BADHINT",
        "BADHINT",
    ]
    assert Path(manifest["gui_mixed"]).is_dir()


def test_zn310b_originals_remain_unmodified_by_source_sha() -> None:
    manifest = _json("zenear_zn310b_gui_manifest.json")

    for item in manifest["items"]:
        assert zn310b_log.sha256_file(Path(item["source_path"])) == item["source_SHA256"]


def test_zn310b_pixels_identical_between_all_variants() -> None:
    integrity = _json("zenear_zn310b_pixel_integrity.json")

    assert integrity
    assert all(row["pixels_identical"] for row in integrity.values())


def test_zn310b_all_generated_copies_have_no_old_wcs() -> None:
    manifest = _json("zenear_zn310b_gui_manifest.json")
    run_dir = Path(manifest["run_dir"])

    for sub in ("control_clean", "no_hints", "wrong_hints", "gui_mixed"):
        for path in (run_dir / sub).glob("*.fit"):
            assert not _has_wcs(path), path


def test_zn310b_control_keeps_near_hints_when_source_has_them() -> None:
    manifest = _json("zenear_zn310b_gui_manifest.json")
    run_dir = Path(manifest["run_dir"])
    controls = sorted((run_dir / "control_clean").glob("*.fit"))

    assert controls
    for path in controls:
        header = fits.getheader(path)
        assert any(key in header for key in ("RA", "OBJCTRA", "OBJRA", "OBJ_RA"))
        assert any(key in header for key in ("DEC", "OBJCTDEC", "OBJDEC", "OBJ_DEC"))


def test_zn310b_nohint_removes_all_near_hint_aliases_and_object() -> None:
    manifest = _json("zenear_zn310b_gui_manifest.json")
    run_dir = Path(manifest["run_dir"])

    for path in (run_dir / "no_hints").glob("*.fit"):
        header = fits.getheader(path)
        for key in (*HINT_RA_KEYS, *HINT_DEC_KEYS, "OBJECT"):
            assert key not in header, (path, key)
        assert "ZN310B_NOHINT" in path.name


def test_zn310b_badhint_has_wrong_center_hints() -> None:
    manifest = _json("zenear_zn310b_gui_manifest.json")
    run_dir = Path(manifest["run_dir"])

    for path in (run_dir / "wrong_hints").glob("*.fit"):
        header = fits.getheader(path)
        assert "RA" in header and "DEC" in header
        assert header.get("OBJECT") == "ZN310B_BADHINT"


def test_zn310b_oracle_sidecars_exist_for_every_gui_case() -> None:
    manifest = _json("zenear_zn310b_gui_manifest.json")

    for item in manifest["items"]:
        sidecar = Path(item["oracle_sidecar"])
        assert sidecar.exists()
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        assert data["classification"] == "ASTAP_ORACLE_VALID"
        assert data["center_ra_deg"] is not None
        assert data["center_dec_deg"] is not None


def test_zn310b_gate_diagnostic_and_4d_defaults() -> None:
    source = (ROOT / "zesolver.py").read_text(encoding="utf-8")

    assert NearSolveConfig().strict_acceptance_mode == "diagnostic"
    assert "blind_backend_profile: str = ZEBLIND_4D_EXPERIMENTAL_PROFILE" in source
    assert PersistentSettings().blind_backend_profile == ZEBLIND_4D_EXPERIMENTAL_PROFILE


def test_zn310b_settings_migrates_historical_to_4d_default() -> None:
    settings = PersistentSettings(blind_backend_profile="historical")
    migrated, changed = _migrate_settings_if_needed(settings)

    assert changed is True
    assert migrated.blind_backend_profile == ZEBLIND_4D_EXPERIMENTAL_PROFILE


def test_zn310b_seed_scale_not_written_to_fits_headers() -> None:
    source = (ROOT / "zeblindsolver" / "metadata_solver.py").read_text(encoding="utf-8")

    assert 'header_updates["SEED_SCALE"]' not in source
    assert 'final_stats["seed_scale"]' in source


def test_zn310b_gui_text_no_longer_claims_historical_default() -> None:
    text = (ROOT / "zesolver.py").read_text(encoding="utf-8")
    docs = (ROOT / "docs" / "zeblind_astrometry_4d_experimental.md").read_text(encoding="utf-8")

    assert "remains the default" not in text
    assert "reste le défaut" not in text
    assert "remains the default" not in docs
    assert "Historical remains the default" not in docs
    assert "Coverage is limited to installed 4D indexes" in text


def test_zn310b_astrometry_web_can_be_disabled_for_manual_test() -> None:
    instructions = (REPORTS / "zenear_zn310b_gui_test_instructions.md").read_text(encoding="utf-8")

    assert "Astrometry.net web: disabled" in instructions
    assert "Disable Astrometry.net web" in instructions


def test_zn310b_log_parser_detects_forbidden_backends(tmp_path: Path) -> None:
    log = tmp_path / "gui.log"
    log.write_text(
        'INFO ZN310B_EVENT {"event":"blind4d_result","case_filename":"A.fit","historical_blind_called":true}\n'
        'INFO ZN310B_EVENT {"event":"near_result","case_filename":"B.fit","astrometry_web_called":true}\n',
        encoding="utf-8",
    )
    events = zn310b_log.parse_events(log)

    assert events["A.fit"][0]["historical_blind_called"] is True
    assert events["B.fit"][0]["astrometry_web_called"] is True


def test_zn310b_log_parser_counts_double_4d_call(tmp_path: Path) -> None:
    log = tmp_path / "gui.log"
    log.write_text(
        'INFO ZN310B_EVENT {"event":"blind4d_result","case_filename":"A.fit","blind4d_called":true,"blind4d_call_count":1}\n'
        'INFO ZN310B_EVENT {"event":"blind4d_result","case_filename":"A.fit","blind4d_called":true,"blind4d_call_count":1}\n',
        encoding="utf-8",
    )
    events = zn310b_log.parse_events(log)

    assert sum(int(e.get("blind4d_call_count") or 0) for e in events["A.fit"]) == 2


def test_zn310b_generator_is_deterministic_for_existing_manifest() -> None:
    manifest = _json("zenear_zn310b_gui_manifest.json")

    assert [item["gui_filename"] for item in manifest["items"]] == [
        f"ZN310B_CONTROL_{i:03d}.fit" for i in range(1, 4)
    ] + [
        f"ZN310B_NOHINT_{i:03d}.fit" for i in range(4, 7)
    ] + [
        f"ZN310B_BADHINT_{i:03d}.fit" for i in range(7, 9)
    ]
