# """
# STANDARDIZED_PROJECT_HEADER_V1
# ╔═══════════════════════════════════════════════════════════════════════════════════╗
# ║ ZeSolver Project (ZeMosaic / ZeSeestarStacker ecosystem)                         ║
# ║                                                                                   ║
# ║ Auteur principal : Tinystork (Tristan Nauleau)                                   ║
# ║ Partenaire IA   : J.A.R.V.I.S. (OpenAI ChatGPT)                                  ║
# ║                                                                                   ║
# ║ Licence du dépôt : GPL V3 (voir pyproject.toml / repository metadata)               ║
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
from typing import Mapping, Sequence

@dataclass(frozen=True)
class QuadLevelSpec:
    name: str
    min_area: float
    max_area: float
    min_diameter: float | None
    max_diameter: float | None
    bucket_cap: int

    def to_manifest(self) -> dict[str, float | int | None]:
        return {
            "name": self.name,
            "min_area": self.min_area,
            "max_area": self.max_area,
            "min_diameter": self.min_diameter,
            "max_diameter": self.max_diameter,
            "bucket_cap": self.bucket_cap,
        }


LEVEL_SPECS: Sequence[QuadLevelSpec] = (
    QuadLevelSpec(name="L", min_area=0.5, max_area=16.0, min_diameter=0.2, max_diameter=6.0, bucket_cap=8192),
    QuadLevelSpec(name="M", min_area=0.15, max_area=4.0, min_diameter=0.1, max_diameter=3.5, bucket_cap=4096),
    # M106 oracle probes contain valid compact quads around 0.0044–0.0057
    # square degrees. Keep the S level below that floor so the catalogue side
    # does not discard geometries that are present in the observed sampler.
    QuadLevelSpec(name="S", min_area=0.002, max_area=1.5, min_diameter=0.05, max_diameter=2.0, bucket_cap=6000),
)

LEVEL_MAP: dict[str, QuadLevelSpec] = {level.name: level for level in LEVEL_SPECS}

_BUCKET_CAP_OVERRIDES: dict[str, int] = {}


def set_bucket_cap_overrides(overrides: Mapping[str, int] | None) -> None:
    """Override per-level bucket caps at runtime (e.g. from GUI dev settings)."""
    _BUCKET_CAP_OVERRIDES.clear()
    if not overrides:
        return
    for level, value in overrides.items():
        try:
            cap = max(0, int(value))
        except (TypeError, ValueError):
            continue
        if cap <= 0:
            continue
        _BUCKET_CAP_OVERRIDES[level.upper()] = cap


def bucket_cap_for(level: str, default: int) -> int:
    return max(0, int(_BUCKET_CAP_OVERRIDES.get(level.upper(), default)))
