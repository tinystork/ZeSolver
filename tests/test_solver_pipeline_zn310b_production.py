from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from zesolver.catalog_resources import resolve_catalog_resources
from zesolver.core import SolveRequest, SolveStatus, SolverPipeline
from zesolver.core.blind_port import ProductionBlindSolverPort
from zesolver.settings import ProductSettings, RuntimeOptions


ROOT = Path(__file__).resolve().parents[1]


class CountingProductionBlindPort(ProductionBlindSolverPort):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def solve(self, request, *, resources, configuration):
        self.calls.append(request.input_path.name)
        return super().solve(request, resources=resources, configuration=configuration)


def _pixel_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is None:
                continue
            arr = np.ascontiguousarray(hdu.data)
            h.update(str(arr.dtype).encode("ascii"))
            h.update(str(tuple(arr.shape)).encode("ascii"))
            h.update(arr.tobytes())
    return h.hexdigest()


def _load_items() -> list[dict[str, object]]:
    path = ROOT / "reports" / "zenear_zn310b_gui_manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items")
    if not isinstance(items, list) or len(items) != 8:
        raise AssertionError("ZN3.10B manifest must contain eight GUI cases")
    return list(items)


def _required_path(env_key: str) -> Path:
    raw = os.environ.get(env_key)
    if not raw:
        pytest.skip(f"{env_key} is not set")
    path = Path(raw).expanduser()
    if not path.exists():
        pytest.skip(f"{env_key} path is absent: {path}")
    return path


@pytest.mark.corpus
@pytest.mark.slow
@pytest.mark.blind4d
def test_zn310b_eight_cases_through_solver_pipeline_production_blind(tmp_path: Path) -> None:
    zn_root = _required_path("ZESOLVER_ZN310B_ROOT")
    astap_root = _required_path("ZESOLVER_ASTAP_ROOT")
    manifest_path = _required_path("ZESOLVER_BLIND4D_MANIFEST")
    legacy_root = Path(os.environ.get("ZESOLVER_LEGACY_INDEX_ROOT") or manifest_path.parent).expanduser()
    resources = resolve_catalog_resources(
        legacy_db_root=astap_root,
        legacy_families=("d50",),
        legacy_blind4d_manifest=manifest_path,
        legacy_index_root=legacy_root,
        enable_environment_discovery=False,
    )
    blind_port = CountingProductionBlindPort()
    counts = {
        "CONTROL_NEAR_CORRECT": 0,
        "NOHINT_4D_CORRECT": 0,
        "BADHINT_4D_CORRECT": 0,
    }

    for item in _load_items():
        source = zn_root / "gui_mixed" / str(item["gui_filename"])
        if not source.exists():
            pytest.skip(f"ZN3.10B case missing: {source}")
        before = _pixel_fingerprint(source)
        output = tmp_path / str(item["gui_filename"])
        pipeline = SolverPipeline(
            product_settings=ProductSettings(blind_enabled=True, web_fallback=False),
            runtime_options=RuntimeOptions(),
            catalog_resources=resources,
            blind_solver=blind_port,
        )

        result = pipeline.solve(SolveRequest(source, output, True, request_id=str(item["case_id"])))

        assert result.status is SolveStatus.SOLVED, (item["case_id"], result.error)
        assert output.exists()
        assert _pixel_fingerprint(source) == before
        with fits.open(output, memmap=False) as hdul:
            wcs = WCS(hdul[0].header, naxis=2, relax=True)
            assert bool(wcs.has_celestial), item["case_id"]

        variant = str(item["variant"])
        if variant == "CONTROL":
            assert result.backend == "NEAR", item["case_id"]
            assert pipeline.last_telemetry["blind_attempted"] is False
            counts["CONTROL_NEAR_CORRECT"] += 1
        elif variant == "NOHINT":
            assert result.backend == "BLIND4D", item["case_id"]
            assert pipeline.last_telemetry["blind_attempted"] is True
            counts["NOHINT_4D_CORRECT"] += 1
        elif variant == "BADHINT":
            assert result.backend == "BLIND4D", item["case_id"]
            assert pipeline.last_telemetry["blind_attempted"] is True
            counts["BADHINT_4D_CORRECT"] += 1
        else:
            raise AssertionError(f"unknown ZN3.10B variant: {variant}")

    assert counts == {
        "CONTROL_NEAR_CORRECT": 3,
        "NOHINT_4D_CORRECT": 3,
        "BADHINT_4D_CORRECT": 2,
    }
    assert len(blind_port.calls) == 5
