from __future__ import annotations

import importlib.util
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

from zesolver.core.blind_models import BlindSolveRequest
from zesolver.core.blind_port import ProductionBlindSolverPort
from zesolver.settings import DeveloperOverrides, ProductSettings, RuntimeOptions, build_solver_configuration
from zesolver.solver_config import build_blind_config_inputs, build_blind_solve_config

from solver_pipeline_fixtures import near_resources


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("zesolver_entrypoint_blind_config_builder_parity", ROOT / "zesolver.py")
assert SPEC is not None and SPEC.loader is not None
zesolver_app = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = zesolver_app
SPEC.loader.exec_module(zesolver_app)


class FakeManifest:
    def __init__(self, root: Path) -> None:
        self.manifest_path = root / "manifest.json"
        self.enabled_index_paths = (root / "d50_2822_S_q40000.npz", root / "d50_2823_S_q40000.npz")
        self.entries = ()


def _field_values(config) -> dict[str, object]:
    return {field.name: getattr(config, field.name) for field in fields(config)}


def _request(tmp_path: Path, product: ProductSettings) -> tuple[BlindSolveRequest, object]:
    configuration = build_solver_configuration(
        product_settings=product,
        runtime_options=RuntimeOptions(),
        developer_overrides=DeveloperOverrides(
            enabled=True,
            values={
                "dev_bucket_limit_override": 123,
                "dev_vote_percentile": 55,
                "dev_collect_matches_vectorized_experimental": True,
                "dev_bucket_cap_S": 11,
                "dev_bucket_cap_M": 22,
                "dev_bucket_cap_L": 33,
                "dev_detect_k_sigma": 4.0,
                "dev_detect_min_area": 7,
            },
        ),
    )
    request = BlindSolveRequest.from_solve_request(
        SimpleNamespace(input_path=tmp_path / "in.fit", output_path=None, overwrite_wcs=True, request_id="p2b2c"),
        configuration=configuration,
    )
    return request, configuration


def test_shared_builder_is_root_entrypoint_source_of_truth() -> None:
    assert zesolver_app.build_blind_solve_config is build_blind_solve_config


def test_port_builder_matches_shared_builder_with_full_hints_and_overrides(tmp_path: Path) -> None:
    manifest = FakeManifest(tmp_path)
    product = ProductSettings(
        hint_ra_deg=184.0,
        hint_dec_deg=47.0,
        hint_radius_deg=1.5,
        hint_focal_mm=250.0,
        hint_pixel_um=2.9,
        hint_resolution_arcsec=2.37,
        hint_resolution_min_arcsec=2.0,
        hint_resolution_max_arcsec=2.8,
        log_level="DEBUG",
    )
    request, configuration = _request(tmp_path, product)
    resources = near_resources(tmp_path, blind_count=2)

    inputs = build_blind_config_inputs(request, resources=resources, configuration=configuration, loaded_manifest=manifest)
    expected = build_blind_solve_config(inputs, ra_hint=request.ra_hint_deg, dec_hint=request.dec_hint_deg, loaded_manifest=manifest)
    produced = ProductionBlindSolverPort().build_config(request, resources=resources, configuration=configuration, loaded_manifest=manifest)

    assert _field_values(produced) == _field_values(expected)
    assert produced.blind_astrometry_4d_index_paths == tuple(str(path.resolve()) for path in manifest.enabled_index_paths)
    assert produced.log_level == "DEBUG"
    assert produced.bucket_limit_override == 123
    assert produced.vote_percentile == 55
    assert produced.detect_k_sigma == 4.0
    assert produced.detect_min_area == 7


def test_port_builder_matches_shared_builder_without_hints(tmp_path: Path) -> None:
    manifest = FakeManifest(tmp_path)
    request, configuration = _request(tmp_path, ProductSettings())
    resources = near_resources(tmp_path, blind_count=2)

    inputs = build_blind_config_inputs(request, resources=resources, configuration=configuration, loaded_manifest=manifest)
    expected = build_blind_solve_config(inputs, loaded_manifest=manifest)
    produced = ProductionBlindSolverPort().build_config(request, resources=resources, configuration=configuration, loaded_manifest=manifest)

    assert _field_values(produced) == _field_values(expected)
    assert produced.ra_hint_deg is None
    assert produced.dec_hint_deg is None
    assert produced.downsample == 1
    assert produced.blind_astrometry_4d_search_budget_s == 45.0
