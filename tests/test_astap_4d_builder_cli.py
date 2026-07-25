from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from zeblindsolver.astap_4d_builder import Astap4DBuildConfig, build_sharded_4d_indexes_from_astap
from zeblindsolver.index_manifest_4d import load_4d_index_manifest

from near_catalog_provider_helpers import write_astap_1476_tile


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


def _write_multi_tile_catalog(root: Path, tile_codes: tuple[str, ...] = ("1501", "1502", "1503")) -> None:
    for offset, tile_code in enumerate(tile_codes):
        write_astap_1476_tile(
            root,
            family="d50",
            tile_code=tile_code,
            ra_deg=np.asarray([62.0 + offset, 62.03 + offset, 62.07 + offset, 62.12 + offset, 62.18 + offset, 62.25 + offset], dtype=np.float64),
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


def test_compare_blind4d_builders_cli_default_mag_cap_matches_qualified_product_config():
    module = _load_cli_module()
    assert module.Astap4DBuildConfig().mag_cap == 15.0


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


def test_native_sharded_builder_writes_fixed_shards_and_manifest(tmp_path):
    astap = tmp_path / "astap"
    out_dir = tmp_path / "shards"
    events: list[dict[str, object]] = []
    _write_multi_tile_catalog(astap)

    result = build_sharded_4d_indexes_from_astap(
        astap,
        out_dir,
        config=Astap4DBuildConfig(tile_keys=("d50_1501", "d50_1502", "d50_1503"), max_stars_per_tile=6, max_quads_per_tile=4),
        tiles_per_shard=2,
        progress_callback=events.append,
    )

    assert [path.name for path in result.shard_paths] == ["direct-d50-fixed2-00.npz", "direct-d50-fixed2-01.npz"]
    loaded = load_4d_index_manifest(result.manifest_path)
    assert [entry.id for entry in loaded.entries] == ["direct-d50-fixed2-00", "direct-d50-fixed2-01"]
    assert [entry.tile_keys for entry in loaded.entries] == [("d50_1501", "d50_1502"), ("d50_1503",)]
    assert events[-1]["stage"] == "manifest_written"


def test_native_sharded_builder_resume_and_repair_one_shard(tmp_path):
    astap = tmp_path / "astap"
    out_dir = tmp_path / "shards"
    _write_multi_tile_catalog(astap)
    cfg = Astap4DBuildConfig(tile_keys=("d50_1501", "d50_1502", "d50_1503"), max_stars_per_tile=6, max_quads_per_tile=4)
    first = build_sharded_4d_indexes_from_astap(astap, out_dir, config=cfg, tiles_per_shard=2)
    mtimes = {path.name: path.stat().st_mtime_ns for path in first.shard_paths}
    missing = first.shard_paths[1]
    missing.unlink()

    repaired = build_sharded_4d_indexes_from_astap(
        astap,
        out_dir,
        config=cfg,
        tiles_per_shard=2,
        resume=True,
        repair_shards=(missing.stem,),
    )

    assert repaired.skipped_shards == ("direct-d50-fixed2-00",)
    assert repaired.repaired_shards == ("direct-d50-fixed2-01",)
    assert repaired.shard_paths[0].stat().st_mtime_ns == mtimes["direct-d50-fixed2-00.npz"]
    assert repaired.shard_paths[1].exists()


def test_native_sharded_builder_cancellation_before_first_shard(tmp_path):
    astap = tmp_path / "astap"
    _write_multi_tile_catalog(astap)

    try:
        build_sharded_4d_indexes_from_astap(
            astap,
            tmp_path / "shards",
            config=Astap4DBuildConfig(tile_keys=("d50_1501", "d50_1502"), max_stars_per_tile=6, max_quads_per_tile=4),
            tiles_per_shard=1,
            cancel_callback=lambda: True,
        )
    except RuntimeError as exc:
        assert "build_cancelled" in str(exc)
    else:  # pragma: no cover - assertion aid
        raise AssertionError("builder should have honoured cancellation")
