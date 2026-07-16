from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

from .quad_index_4d import ASTROMETRY_AB_CODE_4D_SCHEMA
from .zeblindsolver import SolveConfig


HISTORICAL_PROFILE = "historical"
ZEBLIND_4D_EXPERIMENTAL_PROFILE = "zeblind_4d_experimental"


@dataclass(frozen=True)
class SolverProfile:
    name: str
    experimental: bool
    parameters: Mapping[str, object]

    def apply_to_config(self, config: SolveConfig, *, index_paths: Sequence[Path | str] = ()) -> SolveConfig:
        if self.name == HISTORICAL_PROFILE:
            return config
        if self.name != ZEBLIND_4D_EXPERIMENTAL_PROFILE:
            raise KeyError(f"unknown solver profile: {self.name}")
        resolved_paths = tuple(str(Path(path).expanduser().resolve()) for path in index_paths)
        if not resolved_paths:
            raise ValueError("zeblind_4d_experimental requires at least one 4D index path")
        return replace(
            config,
            quad_hash_schema=str(self.parameters["quad_hash_schema"]),
            max_stars=int(self.parameters["quad_sources"]),
            max_quads=int(self.parameters["max_quads"]),
            quality_inliers=int(self.parameters["quality_inliers"]),
            quality_rms=float(self.parameters["quality_rms"]),
            blind_global_hard_budget_s=0.0,
            blind_astrometry_4d_search_budget_s=float(self.parameters["max_wall_s"]),
            blind_astrometry_4d_index_paths=resolved_paths,
            blind_astrometry_4d_validation_catalog_policy=str(self.parameters["validation_catalog_policy"]),
            blind_astrometry_4d_accept_policy=str(self.parameters["accept_policy"]),
            blind_astrometry_4d_max_hypotheses=int(self.parameters["max_hypotheses"]),
            blind_astrometry_4d_max_accepts=int(self.parameters["max_accepts"]),
            blind_astrometry_4d_match_radius_px=float(self.parameters["match_radius_px"]),
            blind_astrometry_4d_source_policy=str(self.parameters["source_policy"]),
            blind_astrometry_4d_image_strategy=str(self.parameters["image_strategy"]),
            blind_astrometry_4d_code_tol=float(self.parameters["code_tol"]),
            blind_astrometry_4d_max_hits=int(self.parameters["max_hits"]),
            blind_astrometry_4d_max_hits_per_image_quad=int(self.parameters["max_hits_per_image_quad"]),
            blind_reuse_existing_solved_wcs=False,
            ra_hint_deg=None,
            dec_hint_deg=None,
            radius_hint_deg=None,
        )


_PROFILES: dict[str, SolverProfile] = {
    HISTORICAL_PROFILE: SolverProfile(
        name=HISTORICAL_PROFILE,
        experimental=False,
        parameters={},
    ),
    ZEBLIND_4D_EXPERIMENTAL_PROFILE: SolverProfile(
        name=ZEBLIND_4D_EXPERIMENTAL_PROFILE,
        experimental=True,
        parameters={
            "quad_hash_schema": ASTROMETRY_AB_CODE_4D_SCHEMA,
            "quad_sources": 120,
            "verification_sources": "full",
            "validation_catalog_policy": "union_candidate_tiles",
            "accept_policy": "best_within_budget",
            "quality_inliers": 40,
            "quality_rms": 1.2,
            "match_radius_px": 3.0,
            "max_quads": 2500,
            "max_hypotheses": 2000,
            "max_accepts": 64,
            "max_wall_s": 45.0,
            "source_policy": "diagnostic_unfiltered",
            "image_strategy": "log_spaced",
            "code_tol": 0.015,
            "max_hits": 2000,
            "max_hits_per_image_quad": 8,
        },
    ),
}


def get_solver_profile(name: str | None) -> SolverProfile:
    key = str(name or HISTORICAL_PROFILE).strip().lower()
    try:
        return _PROFILES[key]
    except KeyError as exc:
        raise KeyError(f"unknown solver profile: {name!r}") from exc


def list_solver_profiles() -> tuple[str, ...]:
    return tuple(_PROFILES)


__all__ = [
    "HISTORICAL_PROFILE",
    "SolverProfile",
    "ZEBLIND_4D_EXPERIMENTAL_PROFILE",
    "get_solver_profile",
    "list_solver_profiles",
]
