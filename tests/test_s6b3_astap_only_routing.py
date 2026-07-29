from __future__ import annotations

import numpy as np

from near_catalog_provider_helpers import write_astap_1476_tile
from zesolver.catalog_resources import resolve_catalog_resources
from zesolver.settings import ProductSettings
from zesolver.simplified_capability import evaluate_simplified_capability, product_settings_for_simplified_run


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


def test_simplified_astap_only_forces_astap_native_without_legacy_index(tmp_path):
    resources = resolve_catalog_resources(
        legacy_db_root=_astap(tmp_path / "astap"),
        legacy_index_root=None,
        enable_environment_discovery=False,
    )
    decision = evaluate_simplified_capability(resources)
    product = ProductSettings(interface_mode="easy", near_catalog_mode="auto", blind_enabled=True)

    effective = product_settings_for_simplified_run(product, decision)

    assert effective.near_catalog_mode == "astap-native"
    assert effective.blind_enabled is False
    assert product.near_catalog_mode == "auto"
    assert product.blind_enabled is True

