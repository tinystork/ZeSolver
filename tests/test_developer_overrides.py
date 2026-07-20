from __future__ import annotations

from zesolver.settings import DeveloperOverrides, ProductSettings, RuntimeOptions, build_solver_configuration


def test_developer_overrides_disabled_have_no_effect() -> None:
    resolved = build_solver_configuration(
        product_settings=ProductSettings(),
        runtime_options=RuntimeOptions(),
        developer_overrides=DeveloperOverrides(enabled=False, values={"blind_max_quads": 1}),
    )

    assert resolved.developer_overrides_active is False
    assert resolved.legacy_solve_config_values["blind_max_quads"] == 8000


def test_developer_overrides_enabled_are_explicit_and_reported() -> None:
    resolved = build_solver_configuration(
        product_settings=ProductSettings(),
        runtime_options=RuntimeOptions(),
        developer_overrides=DeveloperOverrides(enabled=True, values={"blind_max_quads": 1}),
    )

    assert resolved.developer_overrides_active is True
    assert resolved.report_metadata["developer_overrides_active"] is True
    assert resolved.legacy_solve_config_values["blind_max_quads"] == 1
