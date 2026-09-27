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

from zeblindsolver.metadata_solver import _near_catalog_oversize, _near_catalog_star_quota


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
