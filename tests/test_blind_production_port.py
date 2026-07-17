from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from zeblindsolver.index_manifest_4d import IndexManifestError
from zeblindsolver.wcs_header import apply_wcs_solution_to_header
from zesolver.core.blind_models import BlindSolveRequest
from zesolver.core.blind_port import ProductionBlindSolverPort
from zesolver.core.models import SolveRequest, SolveStatus
from zesolver.settings import ProductSettings, RuntimeOptions, build_solver_configuration

from solver_pipeline_fixtures import near_resources, sample_wcs


class FakeManifest:
    def __init__(self, root: Path, count: int = 6) -> None:
        self.manifest_path = root / "manifest.json"
        self.entries = tuple(SimpleNamespace(path=root / f"index-{idx}.npz") for idx in range(count))
        self.enabled_index_paths = tuple(entry.path for entry in self.entries)
        self.tile_keys = tuple(f"d50_{idx}" for idx in range(count))
        self.schema = "zeblind.astrometry_4d_index_manifest.v1"


def _frame(path: Path) -> Path:
    fits.PrimaryHDU(data=np.arange(64, dtype=np.uint16).reshape(8, 8)).writeto(path)
    return path


def _has_celestial_wcs(path: Path) -> bool:
    with fits.open(path, memmap=False) as hdul:
        return bool(WCS(hdul[0].header, naxis=2, relax=True).has_celestial)


def _pixel_hash(path: Path) -> str:
    with fits.open(path, memmap=False) as hdul:
        arr = np.ascontiguousarray(hdul[0].data)
    import hashlib

    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(str(tuple(arr.shape)).encode("ascii"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _configuration(**product_kwargs):
    return build_solver_configuration(
        product_settings=ProductSettings(**product_kwargs),
        runtime_options=RuntimeOptions(),
    )


def test_production_blind_port_solves_on_copy_and_preserves_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = _frame(tmp_path / "input.fit")
    before = _pixel_hash(source)
    manifest = FakeManifest(tmp_path)
    calls: list[Path] = []

    def fake_load(path):
        assert Path(path) == manifest.manifest_path
        return manifest

    def fake_blind_solve(*, fits_path, index_root, config, log, skip_if_valid, cancel_check, prep_cache):
        temp = Path(fits_path)
        calls.append(temp)
        assert temp != source
        assert Path(index_root) == manifest.manifest_path.parent
        assert skip_if_valid is False
        with fits.open(temp, mode="update", memmap=False) as hdul:
            apply_wcs_solution_to_header(
                hdul[0].header,
                sample_wcs(),
                header_updates={"SOLVED": 1, "SOLVER": "ZeSolver", "SOLVMODE": "BLIND4D"},
            )
            hdul.flush()
        return {
            "success": True,
            "message": "solved",
            "elapsed_sec": 0.1,
            "tried_dbs": [str(index_root)],
            "used_db": "d50_2822",
            "wrote_wcs": True,
            "updated_keywords": {"SOLVED": 1, "SOLVER": "ZeSolver", "SOLVMODE": "BLIND4D"},
            "output_path": str(temp),
            "stats": {
                "astrometry_4d_runtime_accepted": True,
                "astrometry_4d_best_accepted_validation": {"inliers": 47, "rms_px": 1.106},
            },
        }

    monkeypatch.setattr("zesolver.core.blind_port.load_4d_index_manifest", fake_load)
    monkeypatch.setattr("zesolver.core.blind_port.blind_solve", fake_blind_solve)
    monkeypatch.setattr(ProductionBlindSolverPort, "build_config", lambda *args, **kwargs: object())

    result = ProductionBlindSolverPort().solve(
        SolveRequest(input_path=source, output_path=tmp_path / "out.fit", overwrite_wcs=True),
        resources=near_resources(tmp_path, blind_count=6),
        configuration=_configuration(),
    )

    assert result.status is SolveStatus.SOLVED
    assert result.backend == "BLIND4D"
    assert result.wcs is not None
    assert result.wcs_written is False
    assert result.inliers == 47
    assert result.rms_px == pytest.approx(1.106)
    assert result.raw["header_updates"]["SOLVMODE"] == "BLIND4D"
    assert len(calls) == 1
    assert not calls[0].exists()
    assert _pixel_hash(source) == before
    assert not _has_celestial_wcs(source)


@pytest.mark.parametrize(
    ("message", "expected_status", "expected_error"),
    [
        ("manifest_absent: missing.json", SolveStatus.CATALOG_UNAVAILABLE, "BLIND4D_MANIFEST_INVALID"),
        ("manifest_sha256_mismatch: d50_2822", SolveStatus.CATALOG_UNAVAILABLE, "BLIND4D_MANIFEST_INVALID"),
        ("manifest_index_incompatible: d50_2822", SolveStatus.CATALOG_UNAVAILABLE, "BLIND4D_MANIFEST_INVALID"),
    ],
)
def test_production_blind_port_normalizes_manifest_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    message: str,
    expected_status: SolveStatus,
    expected_error: str,
) -> None:
    source = _frame(tmp_path / "input.fit")

    def fake_load(path):
        raise IndexManifestError(message)

    monkeypatch.setattr("zesolver.core.blind_port.load_4d_index_manifest", fake_load)

    result = ProductionBlindSolverPort().solve(
        SolveRequest(input_path=source, output_path=None, overwrite_wcs=True),
        resources=near_resources(tmp_path, blind_count=6),
        configuration=_configuration(),
    )

    assert result.status is expected_status
    assert expected_error in str(result.error)


def test_blind_solve_request_carries_hints() -> None:
    configuration = _configuration(
        hint_ra_deg=12.5,
        hint_dec_deg=45.0,
        hint_radius_deg=2.0,
        hint_focal_mm=250.0,
        hint_pixel_um=2.9,
        hint_resolution_arcsec=2.1,
        hint_resolution_min_arcsec=1.8,
        hint_resolution_max_arcsec=2.4,
    )

    request = BlindSolveRequest.from_solve_request(
        SolveRequest(Path("input.fit"), None, True, request_id="hints"),
        configuration=configuration,
    )

    assert request.ra_hint_deg == pytest.approx(12.5)
    assert request.dec_hint_deg == pytest.approx(45.0)
    assert request.radius_hint_deg == pytest.approx(2.0)
    assert request.focal_length_mm == pytest.approx(250.0)
    assert request.pixel_size_um == pytest.approx(2.9)
    assert request.pixel_scale_arcsec == pytest.approx(2.1)
    assert request.pixel_scale_min_arcsec == pytest.approx(1.8)
    assert request.pixel_scale_max_arcsec == pytest.approx(2.4)
    assert request.profile_id == "zeblind4d-v1"


def test_unknown_blind_profile_rejected_without_engine_call(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = _frame(tmp_path / "input.fit")
    configuration = SimpleNamespace(blind_profile=SimpleNamespace(profile_id="unknown"))

    result = ProductionBlindSolverPort().solve_blind(
        BlindSolveRequest(
            input_path=source,
            output_path=None,
            overwrite_wcs=True,
            ra_hint_deg=None,
            dec_hint_deg=None,
            radius_hint_deg=None,
            focal_length_mm=None,
            pixel_size_um=None,
            pixel_scale_arcsec=None,
            pixel_scale_min_arcsec=None,
            pixel_scale_max_arcsec=None,
            profile_id="unknown",
        ),
        resources=near_resources(tmp_path, blind_count=6),
        configuration=configuration,
    )

    assert result.status is SolveStatus.FAILED
    assert "unknown_or_unsupported_blind_profile" in str(result.error)
