from __future__ import annotations

import ast

import numpy as np
import pytest

from near_catalog_provider_helpers import write_astap_1476_tile
from zeblindsolver.astap_4d_builder import Astap4DBuildConfig, build_4d_index_from_astap


def _write_catalog(root):
    write_astap_1476_tile(
        root,
        family="d50",
        tile_code="1501",
        ra_deg=np.asarray([55.0, 55.03, 55.07, 55.12, 55.18, 55.25], dtype=np.float64),
        dec_deg=np.asarray([4.0, 4.06, 3.97, 4.11, 4.02, 4.17], dtype=np.float64),
        mag=np.asarray([8.0, 9.0, 9.5, 10.0, 8.7, 10.3], dtype=np.float32),
    )


def _config() -> Astap4DBuildConfig:
    return Astap4DBuildConfig(
        family="d50",
        tile_keys=("d50_1501",),
        mag_cap=11.0,
        source_max_stars=6,
        max_stars_per_tile=6,
        max_quads_per_tile=4,
        sampler_tag="legacy_brightness",
    )


def test_direct_builder_module_does_not_import_product_or_gui():
    source = __import__("pathlib").Path("zeblindsolver/astap_4d_builder.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")

    assert not any(name == "zesolver" or name.startswith("zesolver.") for name in imported)
    assert "CatalogLibrary" not in source
    assert "CatalogLibraryAdoptionWriter" not in source
    assert "PySide" not in source
    assert "tkinter" not in source


def test_direct_builder_does_not_modify_astap_source(tmp_path):
    astap = tmp_path / "astap"
    _write_catalog(astap)
    source = astap / "d50_1501.1476"
    before = (source.read_bytes(), source.stat().st_mtime_ns)

    build_4d_index_from_astap(astap, tmp_path / "out.npz", config=_config())

    after = (source.read_bytes(), source.stat().st_mtime_ns)
    assert after == before


def test_direct_builder_cancellation_happens_before_write(tmp_path):
    astap = tmp_path / "astap"
    _write_catalog(astap)
    out = tmp_path / "out.npz"

    with pytest.raises(RuntimeError, match="build_cancelled"):
        build_4d_index_from_astap(astap, out, config=_config(), cancel_callback=lambda: True)

    assert not out.exists()


def test_direct_builder_requires_explicit_tile_keys(tmp_path):
    astap = tmp_path / "astap"
    _write_catalog(astap)

    with pytest.raises(ValueError, match="tile_keys"):
        build_4d_index_from_astap(astap, tmp_path / "out.npz", config=Astap4DBuildConfig(family="d50"))
