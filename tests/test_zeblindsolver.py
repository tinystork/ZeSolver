# """
# STANDARDIZED_PROJECT_HEADER_V1
# ╔═══════════════════════════════════════════════════════════════════════════════════╗
# ║ ZeSolver Project (ZeMosaic / ZeSeestarStacker ecosystem)                         ║
# ║                                                                                   ║
# ║ Auteur principal : Tinystork (Tristan Nauleau)                                   ║
# ║ Partenaire IA   : J.A.R.V.I.S. (OpenAI ChatGPT)                                  ║
# ║                                                                                   ║
# ║ Licence du dépôt : MIT (voir pyproject.toml / repository metadata)               ║
# ║                                                                                   ║
# ║ Remerciements amont :                                                             ║
# ║ - ASTAP, par Han Kleijn                                                           ║
# ║ - Astrometry.net, par Dustin Lang, David W. Hogg, Keir Mierle, et al.            ║
# ║                                                                                   ║
# ║ Description FR :                                                                  ║
# ║ Ce code sert à transformer des nuages de photons en solutions WCS et en images   ║
# ║ astronomiques exploitables. Merci de créditer les auteurs et projets amont lors   ║
# ║ de toute réutilisation.                                                           ║
# ║                                                                                   ║
# ║ EN Description:                                                                    ║
# ║ This code helps turn clouds of photons into usable WCS solutions and astronomical ║
# ║ imagery outputs. Please credit both project authors and upstream references when  ║
# ║ reusing this work.                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════════════╝
# """

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from zesolver import zeblindsolver
from zeblindsolver import zeblindsolver as core_solver
from zeblindsolver import metadata_solver as near_solver
from zeblindsolver.astrometry_startree import AstrometryStartree
from zeblindsolver.image_io import load_raster_image
from zeblindsolver.image_prep import downsample_image
from zeblindsolver.quad_index_builder import select_tiles_in_cone
from types import SimpleNamespace
from typing import Any, Optional


def _populate_valid_wcs(header: fits.Header) -> None:
    header["CRVAL1"] = 120.5
    header["CRVAL2"] = -15.2
    header["CRPIX1"] = 512.0
    header["CRPIX2"] = 512.0
    header["CD1_1"] = -2.3e-4
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 2.3e-4
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["RADESYS"] = "ICRS"


def test_has_valid_wcs_accepts_cd_matrix_even_with_cdelt_cards() -> None:
    header = fits.Header()
    _populate_valid_wcs(header)
    header["CDELT1"] = 1.0
    header["CDELT2"] = 1.0
    assert zeblindsolver.has_valid_wcs(header)


def test_sanitize_removes_wcs_keys() -> None:
    header = fits.Header()
    for key in ("CTYPE1", "CRVAL1", "CD1_1", "RADESYS"):
        header[key] = 1.0
    removed = zeblindsolver.sanitize_wcs(header)
    assert removed == 4
    for key in ("CTYPE1", "CRVAL1", "CD1_1", "RADESYS"):
        assert key not in header


def test_blind_solve_skips_valid_header(tmp_path) -> None:
    path = tmp_path / "valid.fits"
    hdu = fits.PrimaryHDU(data=np.zeros((4, 4), dtype=np.float32))
    _populate_valid_wcs(hdu.header)
    hdu.writeto(path)
    result = zeblindsolver.blind_solve(
        fits_path=str(path),
        index_root=str(tmp_path),
        skip_if_valid=True,
    )
    assert result["success"]
    assert "skipped" in result["message"]
    assert not result["wrote_wcs"]


def test_blind_solve_delegates_to_internal(monkeypatch, tmp_path) -> None:
    path = tmp_path / "input.fits"
    fits.PrimaryHDU(data=np.zeros((2, 2), dtype=np.float32)).writeto(path)
    index_root = tmp_path / "index"
    index_root.mkdir()

    captured: list[tuple[str, str]] = []

    def fake_internal(
        input_fits: str,
        index_root_arg: str,
        *,
        config: Optional[Any],
        cancel_check: Optional[Any] = None,
        prep_cache: Optional[dict[str, Any]] = None,
    ) -> SimpleNamespace:
        captured.append((input_fits, index_root_arg))
        return SimpleNamespace(
            success=True,
            message="ok",
            tile_key="tile42",
            header_updates={"SOLVED": 1},
            stats={},
        )

    monkeypatch.setattr(zeblindsolver, "_internal_solve_blind", fake_internal)
    result = zeblindsolver.blind_solve(
        fits_path=str(path),
        index_root=str(index_root),
        skip_if_valid=False,
    )
    assert result["success"]
    assert result["used_db"] == "tile42"
    assert result["tried_dbs"] == [str(Path(index_root).expanduser())]
    assert result["updated_keywords"]["SOLVED"] == 1
    assert captured and captured[0][0] == str(path)


def test_manifest_cone_filter():
    manifest = {
        "tiles": [
            {
                "tile_key": "near",
                "bounds": {"dec_min": -5.0, "dec_max": 5.0, "ra_segments": [[10.0, 15.0]]},
            },
            {
                "tile_key": "far",
                "bounds": {"dec_min": 50.0, "dec_max": 60.0, "ra_segments": [[200.0, 205.0]]},
            },
        ]
    }
    selected = select_tiles_in_cone(manifest, ra_deg=12.0, dec_deg=0.0, radius_deg=6.0)
    assert selected == [0]
    selected_all = select_tiles_in_cone(manifest, ra_deg=0.0, dec_deg=80.0, radius_deg=50.0)
    assert set(selected_all) == {1}


def test_downsample_image_reduces_shape():
    img = np.arange(64, dtype=np.float32).reshape(8, 8)
    reduced = downsample_image(img, 2)
    assert reduced.shape == (4, 4)
    reduced_four = downsample_image(img, 4)
    assert reduced_four.shape == (2, 2)


def test_load_raster_image(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    data = (np.linspace(0, 255, num=25, dtype=np.uint8).reshape(5, 5))
    path = tmp_path / "sample.png"
    Image.fromarray(data).save(path)
    array, meta = load_raster_image(path)
    assert array.shape == (5, 5)
    assert array.dtype == np.float32
    assert 0.0 <= float(array.max()) <= 1.0
    assert meta.get("backend")


def test_detection_params_forwarded(monkeypatch, tmp_path):
    """Ensure SolveConfig forwards developer detection knobs into detect_stars."""
    fits_path = tmp_path / "scene.fits"
    fits.PrimaryHDU(data=np.zeros((16, 16), dtype=np.float32)).writeto(fits_path)
    manifest = {
        "levels": [{"name": "S"}, {"name": "M"}, {"name": "L"}],
        "tiles": [],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    hash_dir = tmp_path / "hash_tables"
    hash_dir.mkdir()
    for level in ("S", "M", "L"):
        (hash_dir / f"quads_{level}.npz").write_bytes(b"")
    recorded: dict[str, float | int] = {}

    def fake_detect_stars(
        img,
        *,
        min_fwhm_px,
        max_fwhm_px,
        k_sigma,
        min_area,
        backend="auto",
        device=None,
    ):
        recorded["k_sigma"] = k_sigma
        recorded["min_area"] = min_area
        raise RuntimeError("stop_after_detection")

    monkeypatch.setattr(core_solver, "detect_stars", fake_detect_stars)
    cfg = core_solver.SolveConfig(detect_k_sigma=1.7, detect_min_area=7)
    with pytest.raises(RuntimeError, match="stop_after_detection"):
        core_solver.solve_blind(str(fits_path), str(tmp_path), config=cfg)
    assert recorded["k_sigma"] == pytest.approx(1.7)
    assert recorded["min_area"] == 7


def _build_simple_tan_wcs(*, crval1: float, crval2: float, scale_arcsec: float = 2.37) -> WCS:
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [float(crval1), float(crval2)]
    w.wcs.crpix = [540.5, 960.5]
    sdeg = float(scale_arcsec) / 3600.0
    w.wcs.cd = np.array([[-sdeg, 0.0], [0.0, sdeg]], dtype=float)
    w.wcs.radesys = "ICRS"
    return w


def test_near_conformance_rejects_large_center_offset() -> None:
    w = _build_simple_tan_wcs(crval1=260.0, crval2=-10.0)
    ok, reason, diag = near_solver._near_conformance_check(
        w,
        width=1080,
        height=1920,
        ra_hint_deg=184.9,
        dec_hint_deg=47.3,
        search_radius_deg=2.0,
        approx_fov_deg=1.2,
        approx_scale_arcsec=2.37,
    )
    assert not ok
    assert "center_offset_too_large" in reason
    assert isinstance(diag, dict)


def test_near_conformance_accepts_consistent_solution() -> None:
    w = _build_simple_tan_wcs(crval1=184.62, crval2=47.30)
    ok, reason, diag = near_solver._near_conformance_check(
        w,
        width=1080,
        height=1920,
        ra_hint_deg=184.9,
        dec_hint_deg=47.3,
        search_radius_deg=2.0,
        approx_fov_deg=1.2,
        approx_scale_arcsec=2.37,
    )
    assert ok
    assert reason == "ok"
    assert isinstance(diag, dict)


def test_onefield_logodds_score_priority() -> None:
    row = {"prob_logodds": 1.0, "accept_logodds": 2.0, "solve_logodds": 3.0, "onefield_final_logodds": 4.0}
    assert core_solver._onefield_logodds_score_from_row(row) == pytest.approx(4.0)
    row2 = {"prob_logodds": 1.5, "accept_logodds": 2.5}
    assert core_solver._onefield_logodds_score_from_row(row2) == pytest.approx(2.5)
    row3 = {"prob_logodds": -0.5}
    assert core_solver._onefield_logodds_score_from_row(row3) == pytest.approx(-0.5)


def test_unsupported_verify_reject_signature_stable_across_reentry_contexts() -> None:
    stats = {
        "inliers": 38,
        "rms_px": 6.1662,
        "model_scale_arcsec": 12.1234,
        "onefield_final_logodds": -1.38629,
        "verify_entry_provenance_path": "resolve_hit_direct",
    }
    sig1 = core_solver._unsupported_verify_reject_signature(
        stats,
        {"phase": "hinted", "level": "S", "parity": "nominal"},
        source="resolve_hit_direct",
        candidate_key="d50_2725",
        level_name="S",
        parity_label="nominal",
    )
    sig2 = core_solver._unsupported_verify_reject_signature(
        dict(stats),
        {"phase": "blind", "level": "S", "parity": "nominal"},
        source="resolve_hit_direct",
        candidate_key="d50_2725",
        level_name="S",
        parity_label="nominal",
    )
    assert sig1 == sig2


def test_should_reject_duplicate_no_positive_verify_match_requires_seen_signature() -> None:
    stats = {
        "inliers": 38,
        "rms_px": 6.1662,
        "model_scale_arcsec": 12.1234,
        "onefield_final_logodds": -1.38629,
        "prob_matches": 0,
        "prob_theta_match_total": 0,
        "verify_entry_provenance_path": "resolve_hit_direct",
    }
    reject, sig, counts = core_solver._should_reject_duplicate_no_positive_verify_match(
        stats,
        {"phase": "blind", "level": "S", "parity": "nominal"},
        source="resolve_hit_direct",
        candidate_key="d50_2725",
        level_name="S",
        parity_label="nominal",
        seen_signatures=set(),
        strict_astrometry_accept_mode=True,
        require_positive_match_support=True,
    )
    assert not reject
    assert counts == {"prob_matches": 0, "prob_theta_match_total": 0}

    reject_seen, sig_seen, counts_seen = core_solver._should_reject_duplicate_no_positive_verify_match(
        stats,
        {"phase": "hinted", "level": "S", "parity": "nominal"},
        source="resolve_hit_direct",
        candidate_key="d50_2725",
        level_name="S",
        parity_label="nominal",
        seen_signatures={sig},
        strict_astrometry_accept_mode=True,
        require_positive_match_support=True,
    )
    assert reject_seen
    assert sig_seen == sig
    assert counts_seen == counts


def test_should_reject_duplicate_no_positive_verify_match_ignores_positive_support() -> None:
    stats = {
        "inliers": 38,
        "rms_px": 6.1662,
        "model_scale_arcsec": 12.1234,
        "onefield_final_logodds": -1.38629,
        "prob_matches": 1,
        "prob_theta_match_total": 0,
        "verify_entry_provenance_path": "resolve_hit_direct",
    }
    sig = core_solver._unsupported_verify_reject_signature(
        stats,
        {"phase": "blind", "level": "S", "parity": "nominal"},
        source="resolve_hit_direct",
        candidate_key="d50_2725",
        level_name="S",
        parity_label="nominal",
    )
    reject, _, counts = core_solver._should_reject_duplicate_no_positive_verify_match(
        stats,
        {"phase": "blind", "level": "S", "parity": "nominal"},
        source="resolve_hit_direct",
        candidate_key="d50_2725",
        level_name="S",
        parity_label="nominal",
        seen_signatures={sig},
        strict_astrometry_accept_mode=True,
        require_positive_match_support=True,
    )
    assert not reject
    assert counts == {"prob_matches": 1, "prob_theta_match_total": 0}


def test_prevalidate_duplicate_signature_stable_for_identical_reentry() -> None:
    img = np.array([[1.1111, 2.2222], [3.3333, 4.4444], [5.5555, 6.6666], [7.7777, 8.8888]], dtype=np.float64)
    tile = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float64)
    err = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    sig1 = core_solver._prevalidate_duplicate_signature(
        candidate_key="d50_2725",
        level_name="S",
        parity_label="nominal",
        source="resolve_hit",
        model_scale_arcsec=12.1234,
        pair_err_px=err,
        img_points_subset=img,
        tile_points_subset=tile,
    )
    sig2 = core_solver._prevalidate_duplicate_signature(
        candidate_key="d50_2725",
        level_name="S",
        parity_label="nominal",
        source="resolve_hit",
        model_scale_arcsec=12.12341,
        pair_err_px=np.array([1.0001, 1.9999, 3.0001, 4.0], dtype=np.float64),
        img_points_subset=np.array(img, copy=True),
        tile_points_subset=np.array(tile, copy=True),
    )
    assert sig1 == sig2


def test_prevalidate_duplicate_signature_changes_when_pairset_changes() -> None:
    img1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    img2 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [9.0, 10.0]], dtype=np.float64)
    tile = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float64)
    err = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    sig1 = core_solver._prevalidate_duplicate_signature(
        candidate_key="d50_2725",
        level_name="S",
        parity_label="nominal",
        source="resolve_hit",
        model_scale_arcsec=12.1234,
        pair_err_px=err,
        img_points_subset=img1,
        tile_points_subset=tile,
    )
    sig2 = core_solver._prevalidate_duplicate_signature(
        candidate_key="d50_2725",
        level_name="S",
        parity_label="nominal",
        source="resolve_hit",
        model_scale_arcsec=12.1234,
        pair_err_px=err,
        img_points_subset=img2,
        tile_points_subset=tile,
    )
    assert sig1 != sig2


def test_should_reject_duplicate_prevalidate_validation_requires_seen_signature() -> None:
    sig = core_solver._prevalidate_duplicate_signature(
        candidate_key="d50_2725",
        level_name="S",
        parity_label="nominal",
        source="resolve_hit",
        model_scale_arcsec=12.1234,
        pair_err_px=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        img_points_subset=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float64),
        tile_points_subset=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float64),
    )
    assert not core_solver._should_reject_duplicate_prevalidate_validation(
        sig,
        seen_signatures=set(),
        strict_astrometry_accept_mode=True,
        source="resolve_hit",
    )
    assert core_solver._should_reject_duplicate_prevalidate_validation(
        sig,
        seen_signatures={sig},
        strict_astrometry_accept_mode=True,
        source="resolve_hit",
    )
    assert not core_solver._should_reject_duplicate_prevalidate_validation(
        sig,
        seen_signatures={sig},
        strict_astrometry_accept_mode=False,
        source="resolve_hit",
    )
    assert not core_solver._should_reject_duplicate_prevalidate_validation(
        sig,
        seen_signatures={sig},
        strict_astrometry_accept_mode=True,
        source="spread_repair",
    )


def test_astrometry_verify_positive_match_support_requires_real_match_when_present() -> None:
    ok_missing, counts_missing = core_solver._astrometry_verify_has_positive_match_support({})
    assert ok_missing
    assert counts_missing == {"prob_matches": 0, "prob_theta_match_total": 0}

    ok_zero, counts_zero = core_solver._astrometry_verify_has_positive_match_support(
        {"prob_matches": 0, "prob_theta_match_total": 0}
    )
    assert not ok_zero
    assert counts_zero == {"prob_matches": 0, "prob_theta_match_total": 0}

    ok_prob, _ = core_solver._astrometry_verify_has_positive_match_support({"prob_matches": 1})
    ok_theta, _ = core_solver._astrometry_verify_has_positive_match_support({"prob_theta_match_total": 1})
    assert ok_prob
    assert ok_theta


def test_onefield_dedup_rows_field_identity_and_sort() -> None:
    rows = [
        {"fieldfile": "a.fit", "fieldnum": 7, "solve_logodds": 1.0, "inliers": 10, "rms_px": 2.0},
        {"fieldfile": "a.fit", "fieldnum": 7, "solve_logodds": 2.0, "inliers": 8, "rms_px": 3.0},
        {"fieldfile": "b.fit", "fieldnum": 3, "accept_logodds": 1.5, "inliers": 12, "rms_px": 1.0},
        {"fieldfile": "", "fieldnum": -1, "prob_logodds": 0.3, "inliers": 4, "rms_px": 5.0},
    ]
    out = core_solver._onefield_dedup_rows(rows, sep_arcsec=3.0)
    assert len(out) == 3
    keys = [(str(r.get("fieldfile") or ""), int(r.get("fieldnum", -1) or -1)) for r in out]
    assert keys.count(("a.fit", 7)) == 1
    # Best score for duplicate key kept.
    kept_a = [r for r in out if str(r.get("fieldfile")) == "a.fit" and int(r.get("fieldnum")) == 7][0]
    assert float(kept_a.get("solve_logodds")) == pytest.approx(2.0)
    # Valid rows follow Astrometry-like onefield ordering: fieldfile, fieldnum, logodds desc.
    assert keys[:2] == [("a.fit", 7), ("b.fit", 3)]


def test_onefield_field_identity_uses_canonical_defaults() -> None:
    fieldfile, fieldnum = core_solver._onefield_field_identity_from_row(
        {"fieldfile": "", "fieldnum": None},
        default_fieldfile="/tmp/m106.fit",
        default_fieldnum=0,
    )
    assert fieldfile == "/tmp/m106.fit"
    assert fieldnum == 0


def test_onefield_insert_sorted_row_uses_astrometry_compare_order() -> None:
    rows: list[dict[str, object]] = []
    core_solver._onefield_insert_sorted_row(rows, {"fieldfile": "b.fit", "fieldnum": 0, "solve_logodds": 10.0})
    core_solver._onefield_insert_sorted_row(rows, {"fieldfile": "a.fit", "fieldnum": 0, "solve_logodds": 1.0})
    core_solver._onefield_insert_sorted_row(rows, {"fieldfile": "a.fit", "fieldnum": 0, "solve_logodds": 3.0})
    keys = [(str(r.get("fieldfile") or ""), int(r.get("fieldnum", -1) if r.get("fieldnum", None) is not None else -1)) for r in rows]
    scores = [float(r.get("solve_logodds", float("nan")) or float("nan")) for r in rows]
    assert keys == [("a.fit", 0), ("a.fit", 0), ("b.fit", 0)]
    assert scores[:2] == [3.0, 1.0]


def test_resolve_scale_arcsec_prefers_local_hint_in_native_mode() -> None:
    cfg = core_solver.SolveConfig(
        pixel_scale_arcsec=9.56,
        pixel_scale_min_arcsec=2.0,
        pixel_scale_max_arcsec=2.8,
    )
    default_scale, default_source = core_solver._resolve_scale_arcsec(cfg, 2.39, return_source=True)
    native_scale, native_source = core_solver._resolve_scale_arcsec(
        cfg,
        2.39,
        prefer_local_hint=True,
        return_source=True,
    )
    assert default_scale == pytest.approx(9.56)
    assert default_source == "config_pixel_scale_arcsec"
    assert native_scale == pytest.approx(2.39)
    assert native_source == "header_scale_arcsec"


def test_estimate_pairset_local_scale_summary_returns_local_arcsec() -> None:
    img_points = np.asarray(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [5.0, 15.0],
        ],
        dtype=np.float64,
    )
    scale_arcsec = 2.392674
    deg_per_px = scale_arcsec / 3600.0
    tile_points = img_points * deg_per_px
    summary = core_solver._estimate_pairset_local_scale_summary(img_points, tile_points)
    assert float(summary["median_arcsec"]) == pytest.approx(scale_arcsec, rel=1e-6)
    assert float(summary["span_arcsec"]) == pytest.approx(scale_arcsec, rel=1e-6)


def test_empty_inliers_fallback_disallowed_in_native_mode() -> None:
    cfg = core_solver.SolveConfig(blind_astrometry_empty_inliers_fallback_enabled=True)
    assert core_solver._empty_inliers_fallback_allowed(
        cfg,
        strict_astrometry_now=True,
        astrometry_native_verify_semantics_mode=False,
    )
    assert not core_solver._empty_inliers_fallback_allowed(
        cfg,
        strict_astrometry_now=True,
        astrometry_native_verify_semantics_mode=True,
    )
    assert not core_solver._empty_inliers_fallback_allowed(
        cfg,
        strict_astrometry_now=False,
        astrometry_native_verify_semantics_mode=False,
    )


def test_pairset_scale_gate_does_not_hard_reject_on_strict_astrometry_lookup_ready() -> None:
    assert not core_solver._pairset_scale_gate_can_hard_reject(
        strict_verify_path_enabled=True,
        astrometry_lookup_ready=True,
        astrometry_native_verify_semantics_mode=False,
        pairset_scale_gate_enabled=True,
    )


def test_pairset_scale_gate_still_allows_noncanonical_hard_reject() -> None:
    assert core_solver._pairset_scale_gate_can_hard_reject(
        strict_verify_path_enabled=False,
        astrometry_lookup_ready=True,
        astrometry_native_verify_semantics_mode=False,
        pairset_scale_gate_enabled=True,
    )
    assert core_solver._pairset_scale_gate_can_hard_reject(
        strict_verify_path_enabled=True,
        astrometry_lookup_ready=False,
        astrometry_native_verify_semantics_mode=False,
        pairset_scale_gate_enabled=True,
    )
    assert not core_solver._pairset_scale_gate_can_hard_reject(
        strict_verify_path_enabled=True,
        astrometry_lookup_ready=True,
        astrometry_native_verify_semantics_mode=True,
        pairset_scale_gate_enabled=True,
    )
    assert not core_solver._pairset_scale_gate_can_hard_reject(
        strict_verify_path_enabled=True,
        astrometry_lookup_ready=False,
        astrometry_native_verify_semantics_mode=False,
        pairset_scale_gate_enabled=False,
    )


def test_native_verify_ref_pool_hard_cap_disabled_on_canonical_paths() -> None:
    assert not core_solver._native_verify_ref_pool_can_hard_cap(
        astrometry_native_verify_semantics_mode=True,
        mirror_scope_enabled=False,
    )
    assert not core_solver._native_verify_ref_pool_can_hard_cap(
        astrometry_native_verify_semantics_mode=False,
        mirror_scope_enabled=True,
    )
    assert core_solver._native_verify_ref_pool_can_hard_cap(
        astrometry_native_verify_semantics_mode=False,
        mirror_scope_enabled=False,
    )


def test_astrometry_startree_query_preserves_leaf_order_and_native_ids() -> None:
    scale = float((1 << 31) - 1)
    minval = np.asarray([-1.0, -1.0, -1.0], dtype=np.float64)
    xyz = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.9999, 0.01, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    data = np.rint((xyz - minval[None, :]) * scale).astype(np.uint32)
    tree = AstrometryStartree(
        data=data,
        lr=np.asarray([2], dtype=np.uint32),
        split=np.empty((0,), dtype=np.uint32),
        sweep=np.asarray([7, 3, 1], dtype=np.uint8),
        minval=minval,
        scale=scale,
        ninterior=0,
    )
    result = tree.query_radec(0.0, 0.0, 1.0)
    assert result.star_ids.tolist() == [0, 1]
    assert result.sweep.tolist() == [7, 3]
    assert result.xyz[:, :2] == pytest.approx(xyz[:2, :2], abs=2e-9)


def test_resolve_native_mo_scale_prefers_wcs_scale_on_native_path() -> None:
    scale, source = core_solver._resolve_native_mo_scale_arcsec_px(
        astrometry_native_verify_semantics_mode=True,
        model_scale_arcsec=4.433,
        pix_scale_arcsec=4.500,
        fallback_pixel_geom_scale=180.689,
        probe_model_scale_enabled=False,
    )
    assert scale == pytest.approx(4.433)
    assert source == "model_scale_arcsec_native"


def test_resolve_native_mo_scale_keeps_pixel_geom_on_noncanonical_path() -> None:
    scale, source = core_solver._resolve_native_mo_scale_arcsec_px(
        astrometry_native_verify_semantics_mode=False,
        model_scale_arcsec=4.433,
        pix_scale_arcsec=4.500,
        fallback_pixel_geom_scale=180.689,
        probe_model_scale_enabled=False,
    )
    assert scale == pytest.approx(180.689)
    assert source == "quadpix_median_px"


def test_resolve_native_mo_scale_prefers_pix_scale_if_model_missing_on_native_path() -> None:
    scale, source = core_solver._resolve_native_mo_scale_arcsec_px(
        astrometry_native_verify_semantics_mode=True,
        model_scale_arcsec=None,
        pix_scale_arcsec=4.500,
        fallback_pixel_geom_scale=180.689,
        probe_model_scale_enabled=False,
    )
    assert scale == pytest.approx(4.500)
    assert source == "pix_scale_arcsec_native_fallback"


def test_resolve_verify_quad_geometry_uses_ab_midpoint_on_native_path() -> None:
    center, q2, source = core_solver._resolve_verify_quad_geometry_px(
        astrometry_native_verify_semantics_mode=True,
        quadpix_points=[
            [1000.323974609375, 494.0926208496094],
            [1041.4422607421875, 738.7149658203125],
            [978.0613403320312, 462.7736511230469],
            [993.9321899414062, 433.05499267578125],
        ],
    )
    assert center == pytest.approx([1020.8831176757812, 616.4037933349609])
    assert q2 == pytest.approx(15382.701278366381)
    assert source == "matchobj_ab_midpoint_native"


def test_resolve_verify_quad_geometry_keeps_centroid_on_noncanonical_path() -> None:
    center, q2, source = core_solver._resolve_verify_quad_geometry_px(
        astrometry_native_verify_semantics_mode=False,
        quadpix_points=[
            [1000.323974609375, 494.0926208496094],
            [1041.4422607421875, 738.7149658203125],
            [978.0613403320312, 462.7736511230469],
            [993.9321899414062, 433.05499267578125],
        ],
    )
    assert center == pytest.approx([1003.43994140625, 532.1590576171875])
    assert q2 == pytest.approx(15234.6758496142)
    assert source == "quad_centroid_mean_r2"


def test_resolve_verify_quad_geometry_requires_two_points() -> None:
    center, q2, source = core_solver._resolve_verify_quad_geometry_px(
        astrometry_native_verify_semantics_mode=True,
        quadpix_points=[[1.0, 2.0]],
    )
    assert center is None
    assert q2 is None
    assert source == "none"


def test_apply_verify_ror_filter_preserves_test_ids_with_mask_order() -> None:
    test_xy = np.asarray(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [100.0, 0.0],
            [200.0, 0.0],
        ],
        dtype=np.float64,
    )
    test_ids = np.asarray([11, 12, 13, 14], dtype=np.int64)
    ref_xy = np.asarray(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [100.0, 0.0],
            [200.0, 0.0],
        ],
        dtype=np.float64,
    )
    ref_ids = np.asarray([21, 22, 23, 24], dtype=np.int64)
    tkeep, tid_keep, rkeep, rid_keep, effa, stats = core_solver._apply_verify_ror_filter(
        test_xy_px=test_xy,
        teststarid_px=test_ids,
        ref_xy_px=ref_xy,
        refstarid_px=ref_ids,
        image_shape=(256, 256),
        sigma2_px=1.0,
        distractor_rate=0.25,
        enabled=False,
    )
    assert tkeep.shape == (4, 2)
    assert rkeep.shape == (4, 2)
    assert tid_keep.tolist() == [11, 12, 13, 14]
    assert rid_keep.tolist() == [21, 22, 23, 24]
    assert effa == pytest.approx(256.0 * 256.0)
    assert stats["ror_applied"] is False


def test_apply_verify_ror_filter_uses_bin_area_fallback_at_cutnside_zero() -> None:
    test_xy = np.asarray(
        [
            [120.0, 120.0],
            [132.0, 128.0],
            [140.0, 136.0],
            [148.0, 144.0],
        ],
        dtype=np.float64,
    )
    ref_xy = np.asarray(
        [
            [118.0, 122.0],
            [134.0, 126.0],
            [142.0, 138.0],
            [150.0, 146.0],
        ],
        dtype=np.float64,
    )
    _, _, _, _, effa, stats = core_solver._apply_verify_ror_filter(
        test_xy_px=test_xy,
        teststarid_px=None,
        ref_xy_px=ref_xy,
        refstarid_px=None,
        image_shape=(256, 256),
        sigma2_px=1.0,
        distractor_rate=0.25,
        enabled=True,
        center_xy=np.asarray([128.0, 128.0], dtype=np.float64),
        quad_q2_px2=64.0,
        verify_uniformize=True,
        index_cutnside=0,
        mo_scale_arcsec_px=4.433,
    )
    assert stats["ror_applied"] is True
    assert stats["ror_uniformize_nw"] == 1
    assert stats["ror_uniformize_nh"] == 1
    assert stats["ror_uniformize_goodbins_n"] == 1
    assert effa == pytest.approx(256.0 * 256.0)


def test_astrometry_verify_dedup_teststar_indices_drops_later_sigma_neighbor() -> None:
    test_xy = np.asarray(
        [
            [18.121782302856445, 1908.493408203125],
            [6.658976078033447, 1902.537109375],
            [200.0, 200.0],
        ],
        dtype=np.float64,
    )
    keep, removed = core_solver._astrometry_verify_dedup_teststar_indices(
        test_xy,
        sigma2_px=1.0270715472637384,
        quad_center_px=np.asarray([1020.8831176757812, 616.4037933349609], dtype=np.float64),
        quad_q2_px2=15382.701278366381,
    )
    assert keep.tolist() == [0, 2]
    assert removed.tolist() == [1]


def test_sort_stars_for_astrometry_parity_orders_by_flux_desc_stably() -> None:
    stars = np.array(
        [
            (1.0, 1.0, 10.0, 2.0),
            (2.0, 2.0, 30.0, 2.0),
            (3.0, 3.0, 30.0, 2.0),
            (4.0, 4.0, 20.0, 2.0),
        ],
        dtype=[("x", "f8"), ("y", "f8"), ("flux", "f8"), ("fwhm", "f8")],
    )
    out = core_solver._sort_stars_for_astrometry_parity(stars)
    assert out["flux"].tolist() == [30.0, 30.0, 20.0, 10.0]
    assert out["x"].tolist()[:2] == [2.0, 3.0]


def test_reset_scale_anchor_for_candidate_scopes_native_mode_to_initial_anchor() -> None:
    arcsec, source = core_solver._reset_scale_anchor_for_candidate(
        24.21388441049848,
        "candidate_pairset_local",
        2.3926740786838514,
        "header_scale_arcsec",
        candidate_scoped_enabled=True,
    )
    assert arcsec == pytest.approx(2.3926740786838514)
    assert source == "header_scale_arcsec"


def test_reset_scale_anchor_for_candidate_keeps_state_when_disabled() -> None:
    arcsec, source = core_solver._reset_scale_anchor_for_candidate(
        24.21388441049848,
        "candidate_pairset_local",
        2.3926740786838514,
        "header_scale_arcsec",
        candidate_scoped_enabled=False,
    )
    assert arcsec == pytest.approx(24.21388441049848)
    assert source == "candidate_pairset_local"
