from __future__ import annotations

from pathlib import Path

from zesolver.settings import ProductSettings, RuntimeOptions, build_solver_configuration


def test_profile_v1_assembly_exposes_baseline_values() -> None:
    resolved = build_solver_configuration(
        product_settings=ProductSettings(catalog_library_path=Path("/tmp/lib"), input_formats=(".fit",)),
        runtime_options=RuntimeOptions(worker_count_resolved=3),
    )
    values = resolved.legacy_solve_config_values

    assert values["catalog_library_path"] == Path("/tmp/lib")
    assert values["workers"] == 3
    assert values["near_quality_inliers"] == 60
    assert values["near_quality_rms"] == 1.0
    assert values["blind_backend_profile"] == "zeblind_4d_experimental"
    assert values["blind_quality_inliers"] == 40
    assert values["blind_quality_rms"] == 1.2


def test_profiles_appear_in_reports() -> None:
    resolved = build_solver_configuration(
        product_settings=ProductSettings(),
        runtime_options=RuntimeOptions(),
    )

    assert resolved.report_metadata["near_profile"] == "zenear-v1"
    assert resolved.report_metadata["blind_profile"] == "zeblind4d-v1"
    assert resolved.report_metadata["pipeline_profile"] == "pipeline-v1"


def test_catalog_library_path_and_legacy_absence_are_compatible() -> None:
    with_library = build_solver_configuration(
        product_settings=ProductSettings(catalog_library_path=Path("/catalog")),
        runtime_options=RuntimeOptions(),
    )
    without_library = build_solver_configuration(
        product_settings=ProductSettings(catalog_library_path=None),
        runtime_options=RuntimeOptions(),
    )

    assert with_library.legacy_solve_config_values["catalog_library_path"] == Path("/catalog")
    assert without_library.legacy_solve_config_values["catalog_library_path"] is None
