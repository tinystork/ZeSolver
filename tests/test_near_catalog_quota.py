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

"""Unit tests for the strict ASTAP-ISO catalogue star quota (surface law).

These tests exercise only the pure helpers `_near_catalog_star_quota` and
`_near_catalog_oversize` from `zeblindsolver.metadata_solver`.  They require no
external fixture, no catalogue, and no FITS file.
"""

from __future__ import annotations

import math

import pytest

from zeblindsolver.metadata_solver import (
    _near_catalog_oversize,
    _near_catalog_quota_effective,
    _near_catalog_star_quota,
)


# ---------------------------------------------------------------------------
# 1. Orientation-generality: the surface law must be invariant to a 90-degree
#    transposition of the frame (portrait <-> landscape) for the same physical
#    geometry, and it must reproduce the concrete Seestar / ASI294 values.
# ---------------------------------------------------------------------------

def _seestar_geometry():
    """Seestar S50: 1080 x 1920 @ 2.392675 arcsec/px (portrait)."""
    scale_deg = 2.392675 / 3600.0
    width, height = 1080, 1920
    window = max(scale_deg * width, scale_deg * height)
    return scale_deg, width, height, window


def _asi_geometry():
    """ASI294MC: 4144 x 2822 @ 0.522147 arcsec/px (landscape)."""
    scale_deg = 0.522147 / 3600.0
    width, height = 4144, 2822
    window = max(scale_deg * width, scale_deg * height)
    return scale_deg, width, height, window


def test_orientation_generality_transposition_invariance():
    # Same physical sensor rotated 90 degrees: the surface law must give the same
    # quota regardless of which axis is "width" vs "height".
    scale_deg = 1.0 / 3600.0  # arbitrary but fixed pixel scale
    nstars = 100
    # portrait 1080 x 1920
    p_win = max(scale_deg * 1080.0, scale_deg * 1920.0)
    p_quota = _near_catalog_star_quota(
        nstars_image=nstars,
        window_w_deg=p_win, window_h_deg=p_win,
        footprint_w_deg=scale_deg * 1080.0, footprint_h_deg=scale_deg * 1920.0,
    )
    # transposed landscape 1920 x 1080 (same pixel scale, same footprint area)
    l_win = max(scale_deg * 1920.0, scale_deg * 1080.0)
    l_quota = _near_catalog_star_quota(
        nstars_image=nstars,
        window_w_deg=l_win, window_h_deg=l_win,
        footprint_w_deg=scale_deg * 1920.0, footprint_h_deg=scale_deg * 1080.0,
    )
    assert p_quota == l_quota
    # The invariant value is Nimg * max(w,h)/min(w,h).
    assert p_quota == round(nstars * 1920.0 / 1080.0)


def test_orientation_generality_concrete_seestar():
    scale_deg, width, height, window = _seestar_geometry()
    quota = _near_catalog_star_quota(
        nstars_image=254,
        window_w_deg=window, window_h_deg=window,
        footprint_w_deg=scale_deg * width, footprint_h_deg=scale_deg * height,
    )
    assert quota == 452


def test_orientation_generality_concrete_asi():
    scale_deg, width, height, window = _asi_geometry()
    quota = _near_catalog_star_quota(
        nstars_image=164,
        window_w_deg=window, window_h_deg=window,
        footprint_w_deg=scale_deg * width, footprint_h_deg=scale_deg * height,
    )
    assert quota == 241


# ---------------------------------------------------------------------------
# 2. Non-regression (portrait): the new law must reduce EXACTLY to the old
#    formula round(Nimg * height/width) for every portrait frame.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "width,height,nstars",
    [
        (1080, 1920, 254),
        (1080, 1920, 100),
        (1080, 1920, 40),
        (900, 1600, 120),
        (2048, 3072, 300),
    ],
)
def test_portrait_non_regression_matches_old_formula(width, height, nstars):
    scale_deg = 2.0 / 3600.0  # arbitrary; cancels out in the ratio
    window = max(scale_deg * width, scale_deg * height)
    quota = _near_catalog_star_quota(
        nstars_image=nstars,
        window_w_deg=window, window_h_deg=window,
        footprint_w_deg=scale_deg * width, footprint_h_deg=scale_deg * height,
    )
    expected = max(32, int(round(float(nstars) * (float(height) / float(width)))))
    assert quota == expected


# ---------------------------------------------------------------------------
# 3. Robustness: degenerate inputs (0, negative, inf, nan) must return the
#    documented floor (32) without raising.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("-inf"), float("nan")])
def test_robustness_degenerate_window(bad):
    assert _near_catalog_star_quota(
        nstars_image=100, window_w_deg=bad, window_h_deg=1.0,
        footprint_w_deg=1.0, footprint_h_deg=1.0,
    ) == 32


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("-inf"), float("nan")])
def test_robustness_degenerate_footprint(bad):
    assert _near_catalog_star_quota(
        nstars_image=100, window_w_deg=1.0, window_h_deg=1.0,
        footprint_w_deg=bad, footprint_h_deg=1.0,
    ) == 32


def test_robustness_negative_nstars():
    assert _near_catalog_star_quota(
        nstars_image=-5, window_w_deg=1.0, window_h_deg=1.0,
        footprint_w_deg=1.0, footprint_h_deg=1.0,
    ) == 32


def test_robustness_zero_nstars_floor():
    # 0 image stars -> 0 * ratio = 0 -> floored at 32.
    assert _near_catalog_star_quota(
        nstars_image=0, window_w_deg=1.0, window_h_deg=1.0,
        footprint_w_deg=1.0, footprint_h_deg=1.0,
    ) == 32


# ---------------------------------------------------------------------------
# 4. Invariant nrstars_required2: the oversize factor is unchanged
#    (Nimg < 35 -> 2.0 ; Nimg > 140 -> 1.0 ; else 2*sqrt(35/Nimg)).
# ---------------------------------------------------------------------------

def test_oversize_below_35():
    assert _near_catalog_oversize(1) == 2.0
    assert _near_catalog_oversize(34) == 2.0


def test_oversize_above_140():
    assert _near_catalog_oversize(141) == 1.0
    assert _near_catalog_oversize(1000) == 1.0


def test_oversize_middle_formula():
    # 35 <= Nimg <= 140 -> 2 * sqrt(35 / Nimg)
    for n in (35, 50, 100, 140):
        assert _near_catalog_oversize(n) == pytest.approx(2.0 * math.sqrt(35.0 / n), rel=1e-12)


def test_oversize_boundaries():
    # Nimg=35 -> 2.0 (formula: 2*sqrt(1) = 2.0) ; Nimg=140 -> 1.0 (formula: 2*sqrt(0.25)=1.0)
    assert _near_catalog_oversize(35) == pytest.approx(2.0, rel=1e-12)
    assert _near_catalog_oversize(140) == pytest.approx(1.0, rel=1e-12)


# ---------------------------------------------------------------------------
# 5. Pipeline composition (window + oversize + quota) — the R4 regression hole:
#    portrait + Nimg <= 140 must reproduce the OLD formula EXACTLY (oversize is
#    applied once, squared, not folded into the window).
# ---------------------------------------------------------------------------

def _old_quota_pipeline(nstars, width, height):
    """Historical inline pipeline: base = round(Nimg*h/w), then oversize^2 once."""
    base = max(32, int(round(float(nstars) * (float(height) / max(1.0, float(width))))))
    oversize = _near_catalog_oversize(nstars)
    effective = max(64, int(round(float(base) * oversize * oversize)))
    return base, effective


@pytest.mark.parametrize("nstars", [1, 10, 30, 34, 35, 50, 100, 139, 140, 141, 200, 254])
def test_portrait_pipeline_composition_matches_old_formula(nstars):
    # Seestar portrait geometry (1080 x 1920 @ 2.392675"/px).
    scale_deg = 2.392675 / 3600.0
    width, height = 1080, 1920
    requested, effective = _near_catalog_quota_effective(
        nstars_image=nstars, scale_deg=scale_deg, width=width, height=height)
    old_base, old_effective = _old_quota_pipeline(nstars, width, height)
    assert requested == old_base
    assert effective == old_effective


def test_portrait_pipeline_composition_non_square_portrait():
    # A different portrait geometry must also reproduce the old formula exactly.
    scale_deg = 1.5 / 3600.0
    width, height = 900, 1600
    for nstars in (30, 50, 100, 254):
        requested, effective = _near_catalog_quota_effective(
            nstars_image=nstars, scale_deg=scale_deg, width=width, height=height)
        old_base, old_effective = _old_quota_pipeline(nstars, width, height)
        assert requested == old_base
        assert effective == old_effective


def test_asi_pipeline_composition_landscape_correction_preserved():
    # ASI294 landscape (4144 x 2822 @ 0.522147"/px), Nimg=164 -> 241 (oversize=1).
    scale_deg = 0.522147 / 3600.0
    requested, effective = _near_catalog_quota_effective(
        nstars_image=164, scale_deg=scale_deg, width=4144, height=2822)
    assert requested == 241
    assert effective == 241


def test_pipeline_composition_transposition_invariance():
    # Same physical sensor rotated 90 degrees: the composed quota is invariant.
    scale_deg = 1.0 / 3600.0
    nstars = 100
    p_req, p_eff = _near_catalog_quota_effective(
        nstars_image=nstars, scale_deg=scale_deg, width=1080, height=1920)
    l_req, l_eff = _near_catalog_quota_effective(
        nstars_image=nstars, scale_deg=scale_deg, width=1920, height=1080)
    assert p_req == l_req
    assert p_eff == l_eff


def test_pipeline_composition_no_oversize4():
    # R4 regression guard: for Nimg < 35 (oversize=2.0), the effective quota must
    # carry oversize^2 (x4), NOT oversize^4 (x16).  Portrait 1080x1920, Nimg=30:
    # old base = round(30*1920/1080)=53 -> effective = max(64, round(53*4)) = 212.
    scale_deg = 2.392675 / 3600.0
    requested, effective = _near_catalog_quota_effective(
        nstars_image=30, scale_deg=scale_deg, width=1080, height=1920)
    assert requested == 53
    assert effective == 212
