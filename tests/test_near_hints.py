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

"""Deterministic Near hint resolution and real preset-consumption tests (Phase E).

Covers: the pure resolver priority table, the GUI claim decision function, the
fov-override-only rule, real preset consumption through `solve_near` (no external
fixture), and non-regression when the FITS carries RA/DEC and no hint is supplied.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from zesolver.core.near_hints import (
    NearHintResolution,
    instrument_hint_applied,
    preset_scale_from_focal_pixel,
    resolve_near_hints,
)
from zeblindsolver.metadata_solver import NearSolveConfig, solve_near
from zeblindsolver.projections import project_tan


# ---------------------------------------------------------------------------
# 1. Resolver priority table (override > fits > preset > fallback), center + scale.
# ---------------------------------------------------------------------------

def test_center_priority_override_wins():
    r = resolve_near_hints(
        override_ra_deg=1.0, override_dec_deg=2.0,
        fits_ra_deg=10.0, fits_dec_deg=20.0,
        preset_ra_deg=100.0, preset_dec_deg=200.0,
    )
    assert r.center_ra_deg == 1.0 and r.center_dec_deg == 2.0
    assert r.center_source == "override"


def test_center_priority_fits_beats_preset():
    r = resolve_near_hints(fits_ra_deg=10.0, fits_dec_deg=20.0, preset_ra_deg=100.0, preset_dec_deg=200.0)
    assert r.center_ra_deg == 10.0 and r.center_dec_deg == 20.0
    assert r.center_source == "fits"


def test_center_priority_preset_fallback():
    r = resolve_near_hints(preset_ra_deg=100.0, preset_dec_deg=200.0)
    assert r.center_ra_deg == 100.0 and r.center_dec_deg == 200.0
    assert r.center_source == "preset"


def test_center_priority_nothing():
    r = resolve_near_hints()
    assert r.center_ra_deg is None and r.center_dec_deg is None
    assert r.center_source == "none"


def test_center_partial_pairs_are_ignored():
    # A half pair (only RA, or only DEC) is not a usable centre.
    r = resolve_near_hints(override_ra_deg=1.0, fits_dec_deg=20.0, preset_ra_deg=100.0, preset_dec_deg=200.0)
    assert r.center_source == "preset"  # override half-pair + fits half-pair are ignored


def test_scale_priority_override_wins():
    r = resolve_near_hints(override_scale_arcsec=1.0, fits_scale_arcsec=2.0, preset_focal_mm=100, preset_pixel_um=3.0)
    assert r.scale_arcsec == 1.0
    assert r.scale_source == "override"


def test_scale_priority_fits_beats_preset():
    r = resolve_near_hints(fits_scale_arcsec=2.0, preset_focal_mm=100, preset_pixel_um=3.0)
    assert r.scale_arcsec == 2.0
    assert r.scale_source == "fits"


def test_scale_priority_preset_from_focal_pixel():
    r = resolve_near_hints(preset_focal_mm=1829.0, preset_pixel_um=4.63)
    assert r.scale_arcsec == pytest.approx(206.26480624709636 * 4.63 / 1829.0, rel=1e-12)
    assert r.scale_source == "preset"


def test_scale_priority_nothing():
    r = resolve_near_hints()
    assert r.scale_arcsec is None
    assert r.scale_source == "none"


def test_radius_override_only():
    r = resolve_near_hints(override_radius_deg=3.0)
    assert r.radius_deg == 3.0 and r.radius_source == "override"
    r2 = resolve_near_hints()
    assert r2.radius_deg is None and r2.radius_source == "none"


def test_fov_override_only_not_silent():
    r = resolve_near_hints(override_fov_deg=0.6)
    assert r.fov_deg == 0.6 and r.fov_source == "override"
    r2 = resolve_near_hints()
    assert r2.fov_deg is None and r2.fov_source == "none"


def test_preset_scale_binning():
    assert preset_scale_from_focal_pixel(1829.0, 4.63, binning=2.0) == pytest.approx(
        206.26480624709636 * 4.63 / 1829.0 * 2.0, rel=1e-12
    )


# ---------------------------------------------------------------------------
# 2. GUI claim decision function (derived from real consumption, not instrument_mode).
# ---------------------------------------------------------------------------

def test_instrument_hint_applied_true_when_preset_consumed():
    assert instrument_hint_applied(
        center_source="preset", scale_source="none", radius_source="none", fov_source="none"
    ) is True


def test_instrument_hint_applied_false_when_only_fits():
    assert instrument_hint_applied(
        center_source="fits", scale_source="fits", radius_source="none", fov_source="none"
    ) is False


def test_instrument_hint_applied_false_when_nothing():
    assert instrument_hint_applied(
        center_source="none", scale_source="none", radius_source="none", fov_source="none"
    ) is False


# ---------------------------------------------------------------------------
# 3. Real consumption proof: FITS WITHOUT RA/DEC + preset center/scale -> Near
#    passes the metadata gate (fails elsewhere, not "metadata RA/DEC missing").
# ---------------------------------------------------------------------------

def _build_index(index_root: Path, center_ra: float, center_dec: float) -> None:
    index_root.mkdir(parents=True, exist_ok=True)
    (index_root / "tiles").mkdir(exist_ok=True)
    ra_vals = np.array([center_ra, center_ra + 0.01, center_ra - 0.01, center_ra + 0.02], dtype=np.float64)
    dec_vals = np.array([center_dec, center_dec + 0.01, center_dec - 0.01, center_dec], dtype=np.float64)
    x_deg, y_deg = project_tan(ra_vals, dec_vals, center_ra, center_dec)
    mag = np.linspace(8.5, 11.0, ra_vals.size, dtype=np.float32)
    tile_path = index_root / "tiles" / "t01.npz"
    np.savez(tile_path, ra_deg=ra_vals, dec_deg=dec_vals, mag=mag,
             x_deg=x_deg.astype(np.float32), y_deg=y_deg.astype(np.float32))
    manifest = {
        "tiles": [{
            "tile_key": "T01", "tile_file": f"tiles/{tile_path.name}",
            "center_ra_deg": center_ra, "center_dec_deg": center_dec,
            "bounds": {"dec_min": center_dec - 2.0, "dec_max": center_dec + 2.0,
                       "ra_segments": [[center_ra - 2.0, center_ra + 2.0]]},
        }]
    }
    (index_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_preset_hint_consumption_passes_metadata_gate(tmp_path):
    # FITS with image data but NO RA/DEC and NO scale cards.
    rng = np.random.default_rng(0)
    image = (rng.normal(100.0, 5.0, size=(120, 120))).astype(np.float32)
    header = fits.Header()
    fits_path = tmp_path / "nometa.fits"
    fits.PrimaryHDU(data=image, header=header).writeto(fits_path)
    index_root = tmp_path / "index"
    _build_index(index_root, 33.0, 12.0)

    # Without hints -> fails at the metadata gate.
    res_no = solve_near(fits_path, index_root, config=NearSolveConfig())
    assert not res_no.success
    assert "metadata RA/DEC missing" in res_no.message

    # With preset center + scale hints -> passes the metadata gate.
    cfg = NearSolveConfig(
        hint_ra_deg=33.0, hint_dec_deg=12.0, hint_scale_arcsec=5.0,
        hint_source={"center": "preset", "scale": "preset"},
    )
    res = solve_near(fits_path, index_root, config=cfg)
    assert not res.success  # still fails (synthetic frame), but NOT at metadata
    assert "metadata RA/DEC missing" not in res.message
    # observability surfaced in stats
    assert res.stats.get("near_hint_center_source") == "preset"
    assert res.stats.get("near_hint_scale_source") == "preset"


# ---------------------------------------------------------------------------
# 4. Non-regression: FITS with RA/DEC and no hint -> unchanged behaviour.
# ---------------------------------------------------------------------------

def test_fits_center_without_hint_unchanged(tmp_path):
    image = (np.random.default_rng(1).normal(100.0, 5.0, size=(120, 120))).astype(np.float32)
    header = fits.Header()
    header["RA"] = 33.0
    header["DEC"] = 12.0
    header["FOCALLEN"] = 150.0
    header["XPIXSZ"] = 3.76
    header["YPIXSZ"] = 3.76
    fits_path = tmp_path / "withmeta.fits"
    fits.PrimaryHDU(data=image, header=header).writeto(fits_path)
    index_root = tmp_path / "index"
    _build_index(index_root, 33.0, 12.0)

    res = solve_near(fits_path, index_root, config=NearSolveConfig())
    assert not res.success
    assert "metadata RA/DEC missing" not in res.message  # RA/DEC present -> past metadata
