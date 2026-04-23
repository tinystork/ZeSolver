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
from astropy.io import fits

from zeblindsolver.metadata_solver import NearSolveConfig, solve_near
from zeblindsolver.projections import project_tan


def _build_test_index(index_root: Path, ra_vals: np.ndarray, dec_vals: np.ndarray, center_ra: float, center_dec: float) -> None:
    index_root.mkdir(parents=True, exist_ok=True)
    tiles_dir = index_root / "tiles"
    tiles_dir.mkdir(exist_ok=True)
    x_deg, y_deg = project_tan(ra_vals, dec_vals, center_ra, center_dec)
    mag = np.linspace(8.5, 11.0, ra_vals.size, dtype=np.float32)
    tile_path = tiles_dir / "test01.npz"
    np.savez(
        tile_path,
        ra_deg=ra_vals.astype(np.float64),
        dec_deg=dec_vals.astype(np.float64),
        mag=mag,
        x_deg=x_deg.astype(np.float32),
        y_deg=y_deg.astype(np.float32),
    )
    manifest = {
        "tiles": [
            {
                "tile_key": "TEST01",
                "tile_file": f"tiles/{tile_path.name}",
                "center_ra_deg": center_ra,
                "center_dec_deg": center_dec,
                "bounds": {
                    "dec_min": center_dec - 2.0,
                    "dec_max": center_dec + 2.0,
                    "ra_segments": [[center_ra - 2.0, center_ra + 2.0]],
                },
            }
        ]
    }
    (index_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _make_synthetic_fits(path: Path, star_px: np.ndarray, center_ra: float, center_dec: float) -> None:
    image = np.zeros((200, 200), dtype=np.float32)
    for x, y in star_px:
        xi = int(round(x))
        yi = int(round(y))
        image[max(0, yi - 1) : yi + 2, max(0, xi - 1) : xi + 2] = 5000.0
    header = fits.Header()
    header["RA"] = center_ra
    header["DEC"] = center_dec
    header["FOCALLEN"] = 150.0
    header["XPIXSZ"] = 3.76
    header["YPIXSZ"] = 3.76
    fits.PrimaryHDU(data=image, header=header).writeto(path)


def test_metadata_solver_solves_synthetic_frame(tmp_path):
    center_ra = 33.0
    center_dec = 12.0
    scale_arcsec = 5.0
    scale_deg = scale_arcsec / 3600.0
    base_pixels = np.array([[100, 100], [120, 105], [82, 123], [140, 140]], dtype=np.float64)
    offsets = (base_pixels - np.array([100.0, 100.0])) * scale_deg
    ra_offsets = offsets[:, 0] / np.cos(np.radians(center_dec))
    ra_vals = center_ra + ra_offsets
    dec_vals = center_dec + offsets[:, 1]
    index_root = tmp_path / "index"
    _build_test_index(index_root, ra_vals, dec_vals, center_ra, center_dec)
    fits_path = tmp_path / "frame.fits"
    _make_synthetic_fits(fits_path, base_pixels, center_ra, center_dec)
    config = NearSolveConfig(
        max_img_stars=50,
        max_cat_stars=50,
        pixel_tolerance=4.0,
        quality_inliers=3,
    )
    result = solve_near(fits_path, index_root, config=config)
    assert result.success, result.message
    assert result.stats.get("quality") == "GOOD"
    with fits.open(fits_path) as hdul:
        header = hdul[0].header
        assert header["SOLVED"] == 1
        assert "NEAR_VER" in header
        assert abs(header["CRVAL1"] - center_ra) < 0.1

import zeblindsolver.metadata_solver as metadata_solver


def _build_dense_test_index(index_root: Path, center_ra: float, center_dec: float, n: int = 320) -> None:
    rng = np.random.default_rng(42)
    # Keep stars inside tile bounds while providing enough spread for window changes.
    off_x = rng.uniform(-1.7, 1.7, size=n)
    off_y = rng.uniform(-1.7, 1.7, size=n)
    ra_vals = center_ra + off_x / np.cos(np.radians(center_dec))
    dec_vals = center_dec + off_y
    _build_test_index(index_root, ra_vals.astype(np.float64), dec_vals.astype(np.float64), center_ra, center_dec)


def _capture_debug_records(monkeypatch):
    records: list[dict] = []

    def _collector(record: dict) -> None:
        records.append(dict(record))

    monkeypatch.setattr(metadata_solver, "_emit_near_debug_record", _collector)
    return records


def test_strict_fov_hint_source_override_priority(tmp_path, monkeypatch):
    center_ra = 33.0
    center_dec = 12.0
    base_pixels = np.array([[100, 100], [120, 105], [82, 123], [140, 140]], dtype=np.float64)
    scale_deg = (5.0 / 3600.0)
    offsets = (base_pixels - np.array([100.0, 100.0])) * scale_deg
    ra_vals = center_ra + offsets[:, 0] / np.cos(np.radians(center_dec))
    dec_vals = center_dec + offsets[:, 1]

    index_root = tmp_path / "index"
    _build_test_index(index_root, ra_vals, dec_vals, center_ra, center_dec)
    fits_path = tmp_path / "frame_override.fits"
    _make_synthetic_fits(fits_path, base_pixels, center_ra, center_dec)
    with fits.open(fits_path, mode="update") as hdul:
        hdul[0].header["FOV"] = 2.5
        hdul.flush()

    def _iso_fail(*args, **kwargs):
        diag = kwargs.get("diag")
        if isinstance(diag, dict):
            diag.update(
                {
                    "stars_img": 4,
                    "quads_img": 1,
                    "selected": {"best_refs": 0},
                    "tolerances": [{"matches_raw": 0, "matches_kept": 0, "ok": False}],
                }
            )
        return None, None, None, 0

    monkeypatch.setattr(metadata_solver, "_astap_iso_hypothesis", _iso_fail)
    records = _capture_debug_records(monkeypatch)

    cfg = NearSolveConfig(astap_iso_strict=True, fov_override_deg=1.1, quality_inliers=3, quality_rms=10.0)
    result = solve_near(fits_path, index_root, config=cfg)
    assert not result.success
    assert records, "expected debug records"
    diag = records[-1].get("astap_iso_diag") or {}
    assert diag.get("fov_hint_source") == "override"
    assert diag.get("auto_fov_retries") is None


def test_strict_fov_hint_source_header_without_override(tmp_path, monkeypatch):
    center_ra = 33.0
    center_dec = 12.0
    base_pixels = np.array([[100, 100], [120, 105], [82, 123], [140, 140]], dtype=np.float64)
    scale_deg = (5.0 / 3600.0)
    offsets = (base_pixels - np.array([100.0, 100.0])) * scale_deg
    ra_vals = center_ra + offsets[:, 0] / np.cos(np.radians(center_dec))
    dec_vals = center_dec + offsets[:, 1]

    index_root = tmp_path / "index"
    _build_test_index(index_root, ra_vals, dec_vals, center_ra, center_dec)
    fits_path = tmp_path / "frame_header.fits"
    _make_synthetic_fits(fits_path, base_pixels, center_ra, center_dec)
    with fits.open(fits_path, mode="update") as hdul:
        hdul[0].header["FOVDEG"] = 1.8
        hdul.flush()

    def _iso_fail(*args, **kwargs):
        diag = kwargs.get("diag")
        if isinstance(diag, dict):
            diag.update(
                {
                    "stars_img": 4,
                    "quads_img": 1,
                    "selected": {"best_refs": 0},
                    "tolerances": [{"matches_raw": 0, "matches_kept": 0, "ok": False}],
                }
            )
        return None, None, None, 0

    monkeypatch.setattr(metadata_solver, "_astap_iso_hypothesis", _iso_fail)
    records = _capture_debug_records(monkeypatch)

    cfg = NearSolveConfig(astap_iso_strict=True, quality_inliers=3, quality_rms=10.0)
    result = solve_near(fits_path, index_root, config=cfg)
    assert not result.success
    diag = (records[-1].get("astap_iso_diag") or {})
    assert diag.get("fov_hint_source") == "header"
    assert diag.get("auto_fov_retries") is None


def test_strict_scale_source_low_support_skips_autofov(tmp_path, monkeypatch):
    center_ra = 33.0
    center_dec = 12.0
    base_pixels = np.array([[100, 100], [120, 105], [82, 123], [140, 140]], dtype=np.float64)
    scale_deg = (5.0 / 3600.0)
    offsets = (base_pixels - np.array([100.0, 100.0])) * scale_deg
    ra_vals = center_ra + offsets[:, 0] / np.cos(np.radians(center_dec))
    dec_vals = center_dec + offsets[:, 1]

    index_root = tmp_path / "index"
    _build_test_index(index_root, ra_vals, dec_vals, center_ra, center_dec)
    fits_path = tmp_path / "frame_scale.fits"
    _make_synthetic_fits(fits_path, base_pixels, center_ra, center_dec)

    def _iso_fail(*args, **kwargs):
        diag = kwargs.get("diag")
        if isinstance(diag, dict):
            diag.update(
                {
                    "stars_img": 4,
                    "quads_img": 1,
                    "selected": {"best_refs": 0},
                    "tolerances": [{"matches_raw": 0, "matches_kept": 0, "ok": False}],
                }
            )
        return None, None, None, 0

    monkeypatch.setattr(metadata_solver, "_astap_iso_hypothesis", _iso_fail)
    records = _capture_debug_records(monkeypatch)

    cfg = NearSolveConfig(astap_iso_strict=True, quality_inliers=3, quality_rms=10.0)
    result = solve_near(fits_path, index_root, config=cfg)
    assert not result.success
    diag = (records[-1].get("astap_iso_diag") or {})
    assert diag.get("fov_hint_source") == "scale"
    retries = diag.get("auto_fov_retries")
    assert isinstance(retries, list) and retries
    assert retries[0].get("skip") == "low_support"


def test_strict_contextual_retry_zero_ref_patience(tmp_path, monkeypatch):
    center_ra = 33.0
    center_dec = 12.0
    index_root = tmp_path / "index_dense"
    _build_dense_test_index(index_root, center_ra, center_dec, n=320)

    # Image content is irrelevant, we patch detect_stars to return rich support.
    fits_path = tmp_path / "frame_dense.fits"
    _make_synthetic_fits(fits_path, np.array([[100, 100]], dtype=np.float64), center_ra, center_dec)

    stars = np.zeros(40, dtype=[("x", "f4"), ("y", "f4"), ("flux", "f4"), ("fwhm", "f4")])
    stars["x"] = np.linspace(20, 180, 40, dtype=np.float32)
    stars["y"] = np.linspace(25, 175, 40, dtype=np.float32)
    stars["flux"] = np.linspace(1000, 4000, 40, dtype=np.float32)
    stars["fwhm"] = np.float32(2.0)

    def _detect_stub(*args, **kwargs):
        return stars.copy()

    def _iso_fail(*args, **kwargs):
        diag = kwargs.get("diag")
        if isinstance(diag, dict):
            diag.update(
                {
                    "stars_img": 40,
                    "quads_img": 12,
                    "selected": {"best_refs": 0},
                    "tolerances": [{"matches_raw": 0, "matches_kept": 0, "ok": False}],
                    "path_used": "find_fit_using_hash",
                }
            )
        return None, None, None, 0

    monkeypatch.setattr(metadata_solver, "detect_stars", _detect_stub)
    monkeypatch.setattr(metadata_solver, "_astap_iso_hypothesis", _iso_fail)
    records = _capture_debug_records(monkeypatch)

    cfg = NearSolveConfig(astap_iso_strict=True, quality_inliers=3, quality_rms=10.0)
    result = solve_near(fits_path, index_root, config=cfg)
    assert not result.success

    diag = (records[-1].get("astap_iso_diag") or {})
    retries = diag.get("auto_fov_retries") or []
    calls = [r for r in retries if isinstance(r, dict) and "refs" in r]
    assert len(calls) <= 3  # context budget/patience should bound retries
    assert any((r.get("skip") == "zero_refs_patience") for r in retries)
