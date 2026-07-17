from __future__ import annotations

import pytest

from zeblindsolver.profiles import ZEBLIND_4D_EXPERIMENTAL_PROFILE, get_solver_profile
from zeblindsolver.quad_index_4d import ASTROMETRY_AB_CODE_4D_SCHEMA
from zesolver.settings import get_blind_profile, get_near_profile


def test_zenear_v1_is_immutable() -> None:
    profile = get_near_profile("zenear-v1")

    with pytest.raises(Exception):
        profile.profile_id = "zenear-v2"  # type: ignore[misc]


def test_zeblind4d_v1_is_immutable() -> None:
    profile = get_blind_profile("zeblind4d-v1")

    with pytest.raises(Exception):
        profile.profile_version = 2  # type: ignore[misc]


def test_unknown_profile_rejected() -> None:
    with pytest.raises(KeyError):
        get_near_profile("missing")
    with pytest.raises(KeyError):
        get_blind_profile("missing")


def test_zeblind4d_v1_matches_current_solver_profile_values() -> None:
    current = get_solver_profile(ZEBLIND_4D_EXPERIMENTAL_PROFILE).parameters
    profile = get_blind_profile("zeblind4d-v1").values

    assert profile["quad_hash_schema"] == ASTROMETRY_AB_CODE_4D_SCHEMA
    assert profile["quad_hash_schema"] == current["quad_hash_schema"]
    assert profile["max_quads"] == current["max_quads"]
    assert profile["max_hypotheses"] == current["max_hypotheses"]
    assert profile["max_accepts"] == current["max_accepts"]
    assert profile["max_wall_s"] == current["max_wall_s"]
