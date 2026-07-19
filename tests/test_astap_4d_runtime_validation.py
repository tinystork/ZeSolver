from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from zeblindsolver.quad_index_4d import Quad4DPayloadTile, build_4d_index_from_payload_tiles


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "validate_direct_blind4d_runtime.py"
SPEC = importlib.util.spec_from_file_location("validate_direct_blind4d_runtime", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
runtime = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runtime
SPEC.loader.exec_module(runtime)


def _write_index(path: Path, tile_key: str) -> Path:
    ra = np.asarray([10.00, 10.05, 10.11, 10.16, 10.20, 10.24], dtype=np.float64)
    dec = np.asarray([1.00, 1.04, 0.98, 1.07, 1.02, 1.10], dtype=np.float64)
    mag = np.asarray([8.0, 9.0, 9.5, 10.0, 10.5, 11.0], dtype=np.float32)
    x = np.asarray([-0.12, -0.07, -0.01, 0.04, 0.08, 0.13], dtype=np.float64)
    y = np.asarray([-0.05, 0.02, -0.03, 0.06, 0.01, 0.08], dtype=np.float64)
    return build_4d_index_from_payload_tiles(
        path,
        tiles=[Quad4DPayloadTile(tile_key=tile_key, ra_deg=ra, dec_deg=dec, mag=mag, x_deg=x, y_deg=y)],
        max_stars_per_tile=6,
        max_quads_per_tile=4,
        sampler_tag="legacy_brightness",
        source_catalog="test",
    )


def test_p1d3b_build_config_is_explicit_and_matches_gate_values():
    cfg = runtime.assert_p1d3b_build_config()

    assert cfg["mag_cap"] == 15.0
    assert cfg["source_max_stars"] == 2000
    assert cfg["source_star_truncation_mode"] == "native_prefix"
    assert cfg["max_stars_per_tile"] == 2000
    assert cfg["max_quads_per_tile"] == 40000
    assert cfg["sampler_tag"] == "catalog_ring_coverage"
    assert cfg["dtype"] == "float32"


def test_runtime_config_is_blind4d_isolated(tmp_path):
    index_path = tmp_path / "d50_2822.npz"
    cfg = runtime.build_runtime_config([index_path], accept_policy="best_within_budget")

    assert cfg.quad_hash_schema == runtime.ASTROMETRY_AB_CODE_4D_SCHEMA
    assert cfg.blind_astrometry_4d_index_paths == (str(index_path.resolve()),)
    assert cfg.blind_astrometry_4d_index_enabled is False
    assert cfg.blind_astrometry_4d_index_path == ""
    assert cfg.blind_astrometry_4d_accept_policy == "best_within_budget"
    assert cfg.blind_global_hard_budget_s == 0.0
    assert cfg.blind_astrometry_4d_search_budget_s == 45.0
    assert cfg.blind_astrometry_4d_validation_catalog_policy == "union_candidate_tiles"
    assert cfg.blind_astrometry_4d_source_policy == "diagnostic_unfiltered"


def test_comparison_manifests_keep_order_and_only_swap_paths(tmp_path):
    product_dir = tmp_path / "product"
    direct_dir = tmp_path / "direct"
    entries = []
    direct_paths = {}
    for priority, tile_key in enumerate(runtime.P1D3B_PRODUCT_ORDER):
        product_index = _write_index(product_dir / f"{tile_key}_S_q40000.npz", tile_key)
        direct_index = _write_index(direct_dir / f"{tile_key}_direct_S_q40000.npz", tile_key)
        entries.append(
            runtime.manifest_entry_from_index(
                f"{tile_key}_S_q40000",
                product_index,
                priority=priority,
                source_label="tile_npz:x_deg/y_deg/ra_deg/dec_deg",
            )
        )
        direct_paths[tile_key] = direct_index
    product_manifest = tmp_path / "product_manifest.json"
    runtime.write_strict_manifest(product_manifest, entries, description="test product")

    made = runtime.make_comparison_manifests(product_manifest, direct_paths, tmp_path / "out")
    baseline = runtime.load_4d_index_manifest(made["baseline_manifest"])
    direct = runtime.load_4d_index_manifest(made["direct_manifest"])

    assert baseline.tile_keys == runtime.P1D3B_PRODUCT_ORDER
    assert direct.tile_keys == runtime.P1D3B_PRODUCT_ORDER
    assert baseline.index_ids == direct.index_ids
    assert [entry.path for entry in baseline.entries] != [entry.path for entry in direct.entries]
    assert all(entry.catalog_source == "astap_raw" for entry in direct.entries)


def _row(success: bool, *, inliers: int = 42, rms: float = 1.0, scale: float = 2.39, compare_center=None, compare_corner=None):
    return {
        "success": success,
        "failure_reason": "" if success else "candidate_exhausted",
        "inliers": inliers,
        "rms_px": rms,
        "pix_scale_arcsec": scale,
        "wcs_metrics": {
            "oracle_center_sep_arcsec": 1.0,
            "oracle_corner_max_sep_arcsec": 5.0,
            "compare_center_sep_arcsec": compare_center,
            "compare_corner_max_sep_arcsec": compare_corner,
        },
    }


def test_classification_flags_losses_invalids_and_equivalent_success():
    assert runtime.classify_pair(_row(True, compare_center=1.0, compare_corner=2.0), _row(True, compare_center=1.0, compare_corner=2.0)) == "SAME_SUCCESS_EQUIVALENT_WCS"
    assert runtime.classify_pair(_row(True), _row(False)) == "DIRECT_LOSS"
    assert runtime.classify_pair(_row(False), _row(True)) == "DIRECT_GAIN_VALIDATED"
    assert runtime.classify_pair(_row(True), _row(True, inliers=12)) == "INVALID_DIRECT_SOLUTION"
    assert runtime.classify_pair(_row(True, rms=5.0), _row(True)) == "INVALID_BASELINE_SOLUTION"


def test_solution_valid_uses_wcs_metric_scale_when_stats_scale_is_absent():
    row = _row(True, scale=float("nan"))
    row["wcs_metrics"]["scale_arcsec_px"] = 2.39

    assert runtime.solution_valid(row) is True


def test_solution_valid_ignores_unusable_external_oracle():
    row = _row(True)
    row["wcs_metrics"].update(
        {
            "oracle_usable": False,
            "oracle_unusable_reason": "reference_pixel_scale_incompatible",
            "oracle_center_sep_arcsec": 30000.0,
            "oracle_corner_max_sep_arcsec": 50000.0,
        }
    )

    assert runtime.solution_valid(row) is True


def test_classification_prefers_equivalent_runtime_wcs_over_noncanonical_oracle():
    baseline = _row(True)
    direct = _row(True, compare_center=0.0, compare_corner=0.0)
    for row in (baseline, direct):
        row["wcs_metrics"].update(
            {
                "oracle_usable": True,
                "oracle_center_sep_arcsec": 30000.0,
                "oracle_corner_max_sep_arcsec": 50000.0,
            }
        )

    assert runtime.classify_pair(baseline, direct) == "SAME_SUCCESS_EQUIVALENT_WCS"
