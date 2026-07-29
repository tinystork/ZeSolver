from __future__ import annotations

from dataclasses import replace

import numpy as np

from catalog_resource_helpers import write_catalog_library, write_fake_4d_index
from near_catalog_provider_helpers import write_astap_1476_tile
from zesolver.catalog_resources import resolve_catalog_resources
from zesolver.core.batch import BatchSolverPipeline, BatchSolveRequest
from zesolver.core.terminal_reasons import TerminalReasonCode
from zesolver.core.models import SolveStatus
from zesolver.settings import ProductSettings
from zesolver.simplified_capability import (
    SimplifiedSolveCapability,
    evaluate_simplified_capability,
    product_settings_for_simplified_run,
)

from batch_pipeline_fixtures import factory, request


def _astap(root):
    write_astap_1476_tile(
        root,
        family="d50",
        tile_code="1501",
        ra_deg=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        dec_deg=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        mag=np.asarray([8.0, 9.0, 10.0], dtype=np.float32),
    )
    return root


def test_complete_library_maps_to_full_local(tmp_path):
    idx = write_fake_4d_index(tmp_path / "d50_TEST_S_q.npz", "d50_TEST")
    library = write_catalog_library(tmp_path / "library", index_paths=[idx])

    decision = evaluate_simplified_capability(resolve_catalog_resources(catalog_library=library))

    assert decision.capability is SimplifiedSolveCapability.FULL_LOCAL
    assert decision.near_available is True
    assert decision.blind4d_available is True
    assert decision.effective_blind_enabled is True


def test_near_only_library_disables_blind_for_run_without_persisting(tmp_path):
    library = write_catalog_library(tmp_path / "library")
    product = ProductSettings(blind_enabled=True, interface_mode="easy")

    decision = evaluate_simplified_capability(resolve_catalog_resources(catalog_library=library))
    effective = product_settings_for_simplified_run(product, decision)

    assert decision.capability is SimplifiedSolveCapability.NEAR_ONLY
    assert product.blind_enabled is True
    assert effective.blind_enabled is False


def test_blind_only_library_is_unavailable_in_simplified_mode(tmp_path):
    idx = write_fake_4d_index(tmp_path / "d50_TEST_S_q.npz", "d50_TEST")
    library = write_catalog_library(tmp_path / "library", include_source=False, index_paths=[idx])

    decision = evaluate_simplified_capability(resolve_catalog_resources(catalog_library=library))

    assert decision.capability is SimplifiedSolveCapability.UNAVAILABLE
    assert decision.near_available is False
    assert decision.blind4d_available is True


def test_astap_only_maps_to_near_only(tmp_path):
    resources = resolve_catalog_resources(legacy_db_root=_astap(tmp_path / "astap"), enable_environment_discovery=False)

    decision = evaluate_simplified_capability(resources)

    assert decision.capability is SimplifiedSolveCapability.NEAR_ONLY
    assert decision.catalog_source_used == "legacy"


def test_invalid_library_can_fall_back_to_astap_near_only(tmp_path):
    resources = resolve_catalog_resources(
        catalog_library=tmp_path / "missing-library",
        legacy_db_root=_astap(tmp_path / "astap"),
        enable_environment_discovery=False,
        allow_legacy_fallback_on_invalid_library=True,
    )

    decision = evaluate_simplified_capability(resources)

    assert decision.capability is SimplifiedSolveCapability.NEAR_ONLY
    assert "catalog_library_invalid_fell_back_to_legacy" in " ".join(resources.warnings)


def test_no_source_is_unavailable():
    decision = evaluate_simplified_capability(resolve_catalog_resources(enable_environment_discovery=False))

    assert decision.capability is SimplifiedSolveCapability.UNAVAILABLE
    assert decision.preflight_error == "NO_LOCAL_CATALOG_SOURCE_AVAILABLE"


def test_near_only_batch_does_not_submit_blind_phase():
    reqs = (request("a"),)
    calls: list[str] = []
    script = {"near": {"a": SolveStatus.UNSOLVED}, "blind": {"a": SolveStatus.SOLVED}}

    result = BatchSolverPipeline(solver_pipeline_factory=factory(script, calls)).solve(
        BatchSolveRequest(requests=reqs, workers=1, blind_enabled=False)
    )

    assert calls == ["near:a"]
    assert result.results[0].status is SolveStatus.UNSOLVED
    assert result.results[0].terminal_reason_code == TerminalReasonCode.NEAR_UNRESOLVED_BLIND_UNAVAILABLE.value

