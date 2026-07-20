from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from near_catalog_provider_helpers import write_astap_1476_tile
from zeblindsolver.db_convert import DEFAULT_MAG_CAP


def _load_cli_module():
    path = Path("tools/compare_blind4d_builders.py").resolve()
    spec = importlib.util.spec_from_file_location("compare_blind4d_builders", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_catalog(root: Path) -> None:
    write_astap_1476_tile(
        root,
        family="d50",
        tile_code="1501",
        ra_deg=np.asarray([62.0, 62.03, 62.07, 62.12, 62.18, 62.25], dtype=np.float64),
        dec_deg=np.asarray([5.0, 5.06, 4.97, 5.11, 5.02, 5.17], dtype=np.float64),
        mag=np.asarray([8.0, 9.0, 9.5, 10.0, 8.7, 10.3], dtype=np.float32),
    )


def test_compare_blind4d_builders_cli_preview_artifacts(tmp_path, capsys):
    module = _load_cli_module()
    astap = tmp_path / "astap"
    out_dir = tmp_path / "out"
    report = tmp_path / "report.json"
    _write_catalog(astap)

    code = module.main(
        [
            "--astap-root",
            str(astap),
            "--tile-key",
            "d50_1501",
            "--out-dir",
            str(out_dir),
            "--report-json",
            str(report),
            "--sampler-tag",
            "legacy_brightness",
            "--max-stars-per-tile",
            "6",
            "--max-quads-per-tile",
            "4",
            "--mag-cap",
            "15.0",
        ]
    )

    assert code == 0
    assert (out_dir / "direct_astap_4d.npz").exists()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "BUILT"
    assert payload["tile_keys"] == ["d50_1501"]
    assert payload["direct_fingerprint"]
    assert payload["config"]["mag_cap"] == 15.0
    assert payload["direct_metadata"]["build_parameters"]["mag_cap"] == 15.0
    assert payload["config"]["mag_cap"] == payload["direct_metadata"]["build_parameters"]["mag_cap"]
    assert "direct_astap_4d.npz" in capsys.readouterr().out


def test_compare_blind4d_builders_cli_default_mag_cap_matches_engine_config():
    module = _load_cli_module()
    assert module.Astap4DBuildConfig().mag_cap == DEFAULT_MAG_CAP


def test_compare_blind4d_builders_cli_refuses_non_empty_out_dir(tmp_path):
    module = _load_cli_module()
    astap = tmp_path / "astap"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "existing.txt").write_text("x", encoding="utf-8")
    _write_catalog(astap)

    code = module.main(["--astap-root", str(astap), "--tile-key", "d50_1501", "--out-dir", str(out_dir)])

    assert code == 1
    assert not (out_dir / "direct_astap_4d.npz").exists()
