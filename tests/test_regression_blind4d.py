from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from corpus_loader import iter_cases
from zeblindsolver.index_manifest_4d import load_4d_index_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_zeblind4d_reference_oracle_has_bounded_validation_contract() -> None:
    oracle = json.loads((ROOT / "tests" / "corpus" / "oracles" / "zeblind4d_reference.json").read_text(encoding="utf-8"))

    assert oracle["profile"] == "zeblind_4d_experimental bounded multi-index union"
    assert oracle["summary"]["known_case"] == "232329"
    assert oracle["summary"]["minimum_inliers_for_reference_case"] == 40


def test_blind4d_manifest_path_is_explicit_for_corpus_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZESOLVER_BLIND4D_MANIFEST", raising=False)

    assert os.environ.get("ZESOLVER_BLIND4D_MANIFEST") is None


@pytest.mark.corpus
@pytest.mark.slow
@pytest.mark.blind4d
def test_blind4d_manifest_loads_when_configured() -> None:
    manifest = os.environ.get("ZESOLVER_BLIND4D_MANIFEST")
    if not manifest:
        pytest.skip("ZESOLVER_BLIND4D_MANIFEST is not set")

    loaded = load_4d_index_manifest(Path(manifest))
    assert loaded.tiles


@pytest.mark.corpus
@pytest.mark.slow
@pytest.mark.blind4d
def test_blind4d_cases_are_mapped_before_execution() -> None:
    cases = list(iter_cases(mode="blind4d"))
    assert cases
    unmapped = [case.id for case in cases if not case.raw.get("relative_path")]
    if unmapped:
        pytest.skip(f"blind4d source FITS paths not mapped yet: {', '.join(unmapped)}")
