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

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class InstrumentPreset:
    id: str
    label: str
    focal_mm: float
    reducer: float
    pixel_um: float
    res_w: int
    res_h: int
    spec_confidence: str = "high"  # high|approx|unknown
    notes: str = ""


def list_presets() -> List[InstrumentPreset]:
    """Return built-in instrument/camera presets.

    These are intentionally few and conservative; they serve as starting points
    and can be edited by the user in the GUI.
    """
    return [
        InstrumentPreset(
            id="c11_0p63_asi533",
            label="C11 + 0.63x + ASI533MC",
            focal_mm=2800.0,
            reducer=0.63,
            pixel_um=3.76,
            res_w=3008,
            res_h=3008,
            spec_confidence="high",
        ),
        InstrumentPreset(
            id="c11_0p63_asi294",
            label="C11 + 0.63x + ASI294MC",
            focal_mm=2800.0,
            reducer=0.63,
            pixel_um=4.63,
            res_w=4144,
            res_h=2822,
            spec_confidence="high",
        ),
        InstrumentPreset(
            id="seestar_s50",
            label="Seestar S50",
            focal_mm=250.0,
            reducer=1.0,
            pixel_um=2.9,
            res_w=1080,
            res_h=1920,
            spec_confidence="high",
            notes="Seestar S50 (250mm, 2.9um, 1080x1920 portrait, validated P2.19b scale ~2.37\"/px; experimental range 1.90..2.85\"/px)",
        ),
        InstrumentPreset(
            id="seestar_s30",
            label="Seestar S30",
            focal_mm=150.0,
            reducer=1.0,
            pixel_um=2.9,
            res_w=1920,
            res_h=1080,
            spec_confidence="high",
            notes="Seestar S30 / IMX462-class (2.9um), 1920x1080; validated P2.19b focal ~150mm and scale ~3.99\"/px; experimental range 3.19..4.79\"/px",
        ),
    ]


def compute_scale_and_fov(
    focal_mm: float,
    pixel_um: float,
    res_w: int,
    res_h: int,
    *,
    reducer: float = 1.0,
    binning: int = 1,
) -> Dict[str, float]:
    """Compute effective pixel scale (arcsec/px) and FOV (deg).

    Returns keys: eff_focal_mm, scale_arcsec_per_px, fov_w_deg, fov_h_deg, fov_diag_deg,
    sensor_w_mm, sensor_h_mm.
    """
    if focal_mm <= 0 or pixel_um <= 0 or res_w <= 0 or res_h <= 0:
        raise ValueError("invalid optical/sensor inputs")
    reducer = float(reducer or 1.0)
    if reducer <= 0:
        reducer = 1.0
    binning = int(binning or 1)
    if binning <= 0:
        binning = 1

    eff_focal = focal_mm * reducer
    eff_pixel_um = pixel_um * binning
    # Plate scale in arcsec/px
    scale = 206.265 * (eff_pixel_um / eff_focal)
    sensor_w_mm = (res_w * eff_pixel_um) / 1000.0
    sensor_h_mm = (res_h * eff_pixel_um) / 1000.0
    # Small-angle approx to degrees
    fov_w_deg = (sensor_w_mm / eff_focal) * 57.2957795
    fov_h_deg = (sensor_h_mm / eff_focal) * 57.2957795
    # Diagonal
    fov_diag_deg = ( (sensor_w_mm ** 2 + sensor_h_mm ** 2) ** 0.5 / eff_focal ) * 57.2957795
    return {
        "eff_focal_mm": eff_focal,
        "scale_arcsec_per_px": scale,
        "sensor_w_mm": sensor_w_mm,
        "sensor_h_mm": sensor_h_mm,
        "fov_w_deg": fov_w_deg,
        "fov_h_deg": fov_h_deg,
        "fov_diag_deg": fov_diag_deg,
    }


def recommend_params(scale_arcsec_per_px: float, fov_diag_deg: float) -> Dict[str, object]:
    """Suggest index-build params based on scale and FOV heuristics.

    Returns a dict with: mag_cap (float), levels (dict L/M/S bools),
    local_brightness_for (list of level keys), max_quads_per_tile (int), notes (str).
    """
    if scale_arcsec_per_px <= 0 or fov_diag_deg <= 0:
        raise ValueError("invalid scale/FOV inputs")

    levels = {"L": False, "M": False, "S": False}
    local_brightness_for: List[str] = []
    mag_cap: float
    max_quads_per_tile: int
    notes: List[str] = []

    if scale_arcsec_per_px > 3.0 and fov_diag_deg > 3.0:
        mag_cap = 13.8
        levels["L"] = True
        levels["M"] = True
        max_quads_per_tile = 2000
        notes.append("Wide/undersampled: favor L/M for speed")
    elif 1.0 <= scale_arcsec_per_px <= 3.0 and 0.5 <= fov_diag_deg <= 3.0:
        mag_cap = 15.0
        levels["M"] = True
        levels["S"] = True
        max_quads_per_tile = 3500
        local_brightness_for = ["M"]
        notes.append("Moderate scale/FOV: M with some S")
    elif scale_arcsec_per_px < 1.0 and fov_diag_deg < 0.7:
        mag_cap = 16.5
        levels["M"] = True
        levels["S"] = True
        max_quads_per_tile = 4500
        local_brightness_for = ["M", "S"]
        notes.append("Narrow/oversampled: dense M/S with local brightness")
    else:
        # Transitional cases
        mag_cap = 15.5
        levels["M"] = True
        max_quads_per_tile = 3000
        local_brightness_for = ["M"]
        notes.append("Transitional: start from M, tune as needed")

    return {
        "mag_cap": mag_cap,
        "levels": levels,
        "local_brightness_for": local_brightness_for,
        "max_quads_per_tile": max_quads_per_tile,
        "notes": "; ".join(notes),
    }


def describe_quads_profile(levels: Dict[str, bool]) -> str:
    order = [k for k in ("L", "M", "S") if levels.get(k)]
    return "/".join(order) if order else "(none)"
