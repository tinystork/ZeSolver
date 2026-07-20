from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from zesolver.core.blind_models import BlindSolveRequest
from zesolver.core.blind_port import ProductionBlindSolverPort
from zesolver.settings import ProductSettings, RuntimeOptions, build_solver_configuration

from solver_pipeline_fixtures import near_resources


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("zesolver_entrypoint_blind_config_parity", ROOT / "zesolver.py")
assert SPEC is not None and SPEC.loader is not None
zesolver_app = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = zesolver_app
SPEC.loader.exec_module(zesolver_app)


class FakeManifest:
    def __init__(self, root: Path) -> None:
        self.manifest_path = root / "manifest.json"
        self.enabled_index_paths = (root / "d50_2822_S_q40000.npz", root / "d50_2823_S_q40000.npz")


def test_blind_port_config_matches_legacy_builder_for_v1(tmp_path: Path) -> None:
    manifest = FakeManifest(tmp_path)
    configuration = build_solver_configuration(
        product_settings=ProductSettings(
            hint_ra_deg=184.0,
            hint_dec_deg=47.0,
            hint_radius_deg=1.5,
            hint_focal_mm=250.0,
            hint_pixel_um=2.9,
            hint_resolution_arcsec=2.37,
        ),
        runtime_options=RuntimeOptions(),
    )
    request = BlindSolveRequest.from_solve_request(
        SimpleNamespace(input_path=tmp_path / "in.fit", output_path=None, overwrite_wcs=True, request_id=None),
        configuration=configuration,
    )

    produced = ProductionBlindSolverPort().build_config(
        request,
        resources=near_resources(tmp_path, blind_count=2),
        configuration=configuration,
        loaded_manifest=manifest,
    )
    legacy_config = zesolver_app.SolveConfig(
        db_root=tmp_path,
        input_dir=tmp_path,
        families=("d50",),
        blind_index_path=tmp_path,
        blind_4d_manifest_path=manifest.manifest_path,
        hint_ra_deg=184.0,
        hint_dec_deg=47.0,
        hint_radius_deg=1.5,
        hint_focal_mm=250.0,
        hint_pixel_um=2.9,
        hint_resolution_arcsec=2.37,
    )
    expected = zesolver_app.build_blind_solve_config(legacy_config, loaded_manifest=manifest)

    fields = (
        "quad_hash_schema",
        "max_stars",
        "max_quads",
        "quality_inliers",
        "quality_rms",
        "blind_astrometry_4d_search_budget_s",
        "blind_astrometry_4d_index_paths",
        "blind_astrometry_4d_validation_catalog_policy",
        "blind_astrometry_4d_accept_policy",
        "blind_astrometry_4d_max_hypotheses",
        "blind_astrometry_4d_max_accepts",
        "blind_astrometry_4d_match_radius_px",
        "blind_astrometry_4d_source_policy",
        "blind_astrometry_4d_image_strategy",
        "blind_astrometry_4d_code_tol",
        "blind_astrometry_4d_max_hits",
        "blind_astrometry_4d_max_hits_per_image_quad",
        "ra_hint_deg",
        "dec_hint_deg",
        "radius_hint_deg",
        "focal_length_mm",
        "pixel_size_um",
        "pixel_scale_arcsec",
    )
    for field in fields:
        assert getattr(produced, field) == getattr(expected, field), field

    assert produced.ra_hint_deg is None
    assert produced.dec_hint_deg is None
    assert produced.radius_hint_deg is None
