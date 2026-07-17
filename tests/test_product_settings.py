from __future__ import annotations

from dataclasses import fields

from zesolver.settings import ProductSettings, RuntimeOptions


def test_product_settings_excludes_internal_solver_parameters() -> None:
    names = {field.name for field in fields(ProductSettings)}

    assert "blind_max_quads" not in names
    assert "near_quality_rms" not in names
    assert "dev_bucket_limit_override" not in names
    assert "blind_4d_manifest_path" not in names


def test_product_settings_v2_payload_references_profiles_only() -> None:
    payload = ProductSettings().to_v2_payload()

    assert payload["settings_schema_version"] == 2
    assert payload["profiles"] == {
        "near": "zenear-v1",
        "blind": "zeblind4d-v1",
        "pipeline": "pipeline-v1",
    }
    assert "near_quality_rms" not in str(payload["product"])


def test_runtime_options_are_separate_from_product_payload() -> None:
    runtime = RuntimeOptions(worker_count_resolved=4, diagnostic_capture=True)
    payload = ProductSettings().to_v2_payload()

    assert runtime.worker_count_resolved == 4
    assert "worker_count_resolved" not in str(payload)
    assert "diagnostic_capture" not in str(payload)
