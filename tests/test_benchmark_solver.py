from __future__ import annotations

import json
from pathlib import Path

from tools import benchmark_solver as bench


def test_normalize_overrides_is_case_insensitive() -> None:
    overrides = bench.normalize_overrides({"MAX_STARS": 123, "bucket_cap_m": 2})
    assert overrides["max_stars"] == 123
    assert overrides["bucket_cap_M"] == 2


def test_expand_input_token_reads_list_file(tmp_path: Path) -> None:
    img_dir = tmp_path / "data"
    img_dir.mkdir()
    (img_dir / "sub").mkdir()
    target = img_dir / "sub" / "frame.fit"
    target.touch()
    list_file = tmp_path / "images.lst"
    list_file.write_text(f"# comment line\n./data/sub/frame.fit\n", encoding="utf-8")
    entries = bench.expand_input_token(f"@{list_file}")
    assert str(target.resolve()) in entries


def test_looks_like_fits_handles_compressed_suffixes(tmp_path: Path) -> None:
    path = tmp_path / "sample.fit.gz"
    path.touch()
    assert bench.looks_like_fits(path)
    assert not bench.looks_like_fits(path.with_suffix(".jpg"))


def test_load_sweeps_parses_inline_overrides(tmp_path: Path) -> None:
    grid = tmp_path / "grid.json"
    payload = [
        {"label": "dense", "max_stars": 1000, "detect_k_sigma": 2.5},
        {"name": "caps", "overrides": {"bucket_cap_l": 4}, "description": "cap override"},
    ]
    grid.write_text(json.dumps(payload), encoding="utf-8")
    profiles = bench.load_sweeps(grid)
    assert len(profiles) == 2
    assert profiles[0].overrides["max_stars"] == 1000
    assert profiles[1].overrides["bucket_cap_L"] == 4
    assert profiles[1].description == "cap override"


def test_resolve_inputs_scans_directories_for_fits(tmp_path: Path) -> None:
    fits_file = tmp_path / "frame.fits"
    fits_file.touch()
    (tmp_path / "ignore.jpg").touch()
    resolved = bench.resolve_inputs([str(tmp_path)], None)
    assert fits_file.resolve() in resolved
    assert len(resolved) == 1
