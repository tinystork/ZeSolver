from __future__ import annotations

import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from catalog_resource_helpers import write_catalog_library
from near_catalog_provider_helpers import write_astap_1476_tile
from zeblindsolver.near_catalog_provider import AstapNearCatalogProvider
from zesolver.catalog_resources import resolve_catalog_resources
from zesolver.gui_pipeline.pipeline_runner import PipelineGuiRunner
from zesolver.gui_pipeline.requests import GuiSettingsState
from zesolver.gui_pipeline.settings_adapter import build_gui_solve_request


def _fits(path: Path) -> Path:
    fits.PrimaryHDU(data=np.ones((16, 16), dtype=np.uint16)).writeto(path)
    return path


def _library_with_astap_tile(tmp_path: Path) -> Path:
    root = write_catalog_library(tmp_path / "library", include_source=True, index_paths=[])
    write_astap_1476_tile(
        root / "sources" / "astap" / "d50",
        family="d50",
        tile_code="2823",
        ra_deg=np.asarray([184.6, 184.7], dtype=np.float64),
        dec_deg=np.asarray([47.2, 47.3], dtype=np.float64),
    )
    return root


def _write_three_tiles(root: Path) -> None:
    write_astap_1476_tile(
        root,
        family="d50",
        tile_code="0501",
        ra_deg=np.asarray([1.0, 2.0], dtype=np.float64),
        dec_deg=np.asarray([-70.0, -69.5], dtype=np.float64),
    )
    write_astap_1476_tile(
        root,
        family="d50",
        tile_code="1501",
        ra_deg=np.asarray([1.0, 1.2], dtype=np.float64),
        dec_deg=np.asarray([-18.2, -18.0], dtype=np.float64),
    )
    write_astap_1476_tile(
        root,
        family="d50",
        tile_code="2823",
        ra_deg=np.asarray([184.6, 184.7], dtype=np.float64),
        dec_deg=np.asarray([47.2, 47.3], dtype=np.float64),
    )


def _state(resources, *, workers: int = 6) -> GuiSettingsState:
    return GuiSettingsState(
        workers=workers,
        preserve_order=True,
        use_blind=False,
        catalog_resources=resources,
        legacy_config=object(),
    )


def _counters(summary) -> dict[str, int]:
    assert summary.telemetry is not None
    counters = summary.telemetry["counters"]
    assert isinstance(counters, dict)
    return counters


def test_s6a1_baseline_near_catalog_runtime_is_shared_by_batch(tmp_path: Path, monkeypatch) -> None:
    library = _library_with_astap_tile(tmp_path)
    resources = resolve_catalog_resources(catalog_library=library)
    files = tuple(_fits(tmp_path / f"frame_{idx}.fit") for idx in range(6))
    provider_ids: list[int] = []
    provider_lock = threading.Lock()

    def fake_near_solve(fits_path, index_root, *, catalog_provider=None, **kwargs):
        time.sleep(0.01)
        with provider_lock:
            provider_ids.append(id(catalog_provider))
        return {"success": False, "message": "synthetic miss", "stats": {}, "wrote_wcs": False}

    monkeypatch.setattr("zesolver.core.pipeline.near_solve", fake_near_solve)

    request = build_gui_solve_request(files, _state(resources, workers=6))
    summary = PipelineGuiRunner().run(request)
    counters = _counters(summary)

    assert len(provider_ids) == 6
    assert len(set(provider_ids)) == 1
    assert counters["near_runtime_resolution_count"] == 1
    assert counters["catalog_provider_constructor_count"] == 1
    assert counters["near_catalog_runtime_created"] == 1
    assert counters["near_catalog_runtime_reused"] == 6
    assert counters["near_catalog_inventory_load_count"] == 1
    assert counters["near_catalog_provider_created"] == 1
    assert counters["near_catalog_provider_reused"] == 6
    assert counters["near_catalog_runtime_closed"] == 1


def test_s6a1_astap_cache_single_flight_for_concurrent_same_tile(tmp_path: Path, monkeypatch) -> None:
    _write_three_tiles(tmp_path)
    counters: defaultdict[str, int] = defaultdict(int)

    def metric(name: str, amount: int = 1) -> None:
        counters[name] += amount

    provider = AstapNearCatalogProvider(tmp_path, families=("d50",), cache_size=2, metrics_callback=metric)
    tile = provider.select_tiles(184.6, 47.3, 2.0, 1)[0]
    original_loader = provider._catalog_db._load_tile
    load_count = 0
    load_lock = threading.Lock()
    start = threading.Barrier(4)
    results = []
    errors = []

    def slow_loader(*args, **kwargs):
        nonlocal load_count
        with load_lock:
            load_count += 1
        time.sleep(0.05)
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(provider._catalog_db, "_load_tile", slow_loader)

    def worker() -> None:
        try:
            start.wait(timeout=2.0)
            results.append(provider.load_stars(tile))
        except BaseException as exc:  # pragma: no cover - assertion aid
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)

    assert not errors
    assert len(results) == 4
    assert load_count == 1
    assert counters["near_catalog_payload_cache_misses"] == 1
    assert counters["near_catalog_payload_physical_loads"] == 1
    assert counters["near_catalog_payload_duplicate_loads"] >= 1
    assert counters["near_catalog_payload_singleflight_waiters"] >= 1
    assert len({id(item.ra_deg) for item in results}) == 1
    assert not results[0].ra_deg.flags.writeable
    with pytest.raises(ValueError):
        results[0].ra_deg[0] = 123.0


def test_s6a1_astap_cache_hits_and_evictions_are_bounded(tmp_path: Path) -> None:
    _write_three_tiles(tmp_path)
    counters: defaultdict[str, int] = defaultdict(int)

    def metric(name: str, amount: int = 1) -> None:
        counters[name] += amount

    provider = AstapNearCatalogProvider(tmp_path, families=("d50",), cache_size=1, metrics_callback=metric)
    tiles = provider.select_tiles(1.0, -20.0, 180.0, 3)

    first = provider.load_stars(tiles[0])
    second = provider.load_stars(tiles[0])
    provider.load_stars(tiles[1])
    third = provider.load_stars(tiles[0])

    assert first.ra_deg is second.ra_deg
    assert third.ra_deg is not first.ra_deg
    assert counters["near_catalog_payload_cache_misses"] == 3
    assert counters["near_catalog_payload_cache_hits"] == 1
    assert counters["near_catalog_payload_physical_loads"] == 3
    assert counters["near_catalog_payload_cache_evictions"] >= 2


def test_s6a1_two_batches_do_not_share_near_runtime_or_provider(tmp_path: Path, monkeypatch) -> None:
    library_a = _library_with_astap_tile(tmp_path / "a")
    library_b = _library_with_astap_tile(tmp_path / "b")
    resources_a = resolve_catalog_resources(catalog_library=library_a)
    resources_b = resolve_catalog_resources(catalog_library=library_b)
    files_a = (_fits(tmp_path / "a.fit"),)
    files_b = (_fits(tmp_path / "b.fit"),)
    seen_roots: list[Path] = []
    seen_provider_ids: list[int] = []
    seen_providers: list[object] = []

    def fake_near_solve(fits_path, index_root, *, catalog_provider=None, **kwargs):
        seen_roots.append(catalog_provider.db_root)
        seen_providers.append(catalog_provider)
        seen_provider_ids.append(id(catalog_provider))
        return {"success": False, "message": "synthetic miss", "stats": {}, "wrote_wcs": False}

    monkeypatch.setattr("zesolver.core.pipeline.near_solve", fake_near_solve)

    first = PipelineGuiRunner().run(build_gui_solve_request(files_a, _state(resources_a, workers=1)))
    second = PipelineGuiRunner().run(build_gui_solve_request(files_b, _state(resources_b, workers=1)))

    assert seen_roots == [resources_a.near.root.resolve(), resources_b.near.root.resolve()]
    assert len(seen_provider_ids) == 2
    assert seen_provider_ids[0] != seen_provider_ids[1]
    assert _counters(first)["near_runtime_resolution_count"] == 1
    assert _counters(second)["near_runtime_resolution_count"] == 1
    assert _counters(first)["near_catalog_runtime_closed"] == 1
    assert _counters(second)["near_catalog_runtime_closed"] == 1


def test_s6a1_near_runtime_closes_after_worker_exception(tmp_path: Path, monkeypatch) -> None:
    library = _library_with_astap_tile(tmp_path)
    resources = resolve_catalog_resources(catalog_library=library)
    files = (_fits(tmp_path / "boom.fit"),)

    def exploding_near_solve(*args, **kwargs):
        raise RuntimeError("synthetic worker failure")

    monkeypatch.setattr("zesolver.core.pipeline.near_solve", exploding_near_solve)

    summary = PipelineGuiRunner().run(build_gui_solve_request(files, _state(resources, workers=1)))
    counters = _counters(summary)

    assert summary.results[0].status in {"FAILED", "UNSOLVED"}
    assert counters["near_runtime_resolution_count"] == 1
    assert counters["near_catalog_runtime_closed"] == 1
