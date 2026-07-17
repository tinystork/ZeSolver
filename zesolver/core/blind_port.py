from __future__ import annotations

import importlib.util
import logging
import shutil
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

from zeblindsolver.index_manifest_4d import IndexManifestError, Loaded4DManifest, load_4d_index_manifest
from zeblindsolver.profiles import ZEBLIND_4D_EXPERIMENTAL_PROFILE

from zesolver.catalog_resources import SolverCatalogResources
from zesolver.zeblindsolver import BlindSolverRuntimeError, blind_solve

from .blind_models import BlindSolveRequest
from .blind_result_adapter import engine_result_from_blind_result
from .models import EngineSolveResult, SolveRequest, SolveStatus


class ProductionBlindSolverPort:
    """Production Blind 4D adapter for SolverPipeline.

    The existing Blind wrapper writes WCS cards to its input FITS. To keep final
    output ownership in ``wcs_io.py``, this port solves a temporary copy and
    returns the produced WCS as an engine result.
    """

    def solve(self, request: SolveRequest, *, resources: SolverCatalogResources, configuration) -> EngineSolveResult:
        blind_request = BlindSolveRequest.from_solve_request(request, configuration=configuration)
        return self.solve_blind(blind_request, resources=resources, configuration=configuration)

    def solve_blind(
        self,
        request: BlindSolveRequest,
        *,
        resources: SolverCatalogResources,
        configuration,
    ) -> EngineSolveResult:
        if configuration.blind_profile.profile_id != "zeblind4d-v1":
            return EngineSolveResult(
                status=SolveStatus.FAILED,
                backend="BLIND4D",
                error=f"unknown_or_unsupported_blind_profile: {configuration.blind_profile.profile_id}",
            )
        if resources.blind4d_manifest_path is None:
            return EngineSolveResult(status=SolveStatus.CATALOG_UNAVAILABLE, backend="BLIND4D", error="blind4d_manifest_unavailable")
        try:
            loaded_manifest = load_4d_index_manifest(resources.blind4d_manifest_path)
        except IndexManifestError as exc:
            return EngineSolveResult(status=SolveStatus.CATALOG_UNAVAILABLE, backend="BLIND4D", error=f"BLIND4D_MANIFEST_INVALID: {exc}")

        try:
            blind_cfg = self.build_config(request, resources=resources, configuration=configuration, loaded_manifest=loaded_manifest)
        except Exception as exc:
            return EngineSolveResult(status=SolveStatus.FAILED, backend="BLIND4D", error=f"blind_config_failed: {exc}")

        index_root = loaded_manifest.manifest_path.parent
        try:
            with tempfile.TemporaryDirectory(prefix="zesolver-blind4d-") as tmp:
                temp_path = Path(tmp) / request.input_path.name
                shutil.copyfile(request.input_path, temp_path)
                result = blind_solve(
                    fits_path=str(temp_path),
                    index_root=str(index_root),
                    config=blind_cfg,
                    log=logging.info,
                    skip_if_valid=False,
                    cancel_check=_cancel_check(configuration),
                    prep_cache={},
                )
                engine = engine_result_from_blind_result(result, solved_path=temp_path)
                if engine.raw:
                    raw = dict(engine.raw)
                    raw["manifest_path"] = str(loaded_manifest.manifest_path)
                    raw["blind4d_index_count"] = len(loaded_manifest.entries)
                    engine = EngineSolveResult(
                        status=engine.status,
                        backend=engine.backend,
                        wcs=engine.wcs,
                        wcs_written=engine.wcs_written,
                        center_ra_deg=engine.center_ra_deg,
                        center_dec_deg=engine.center_dec_deg,
                        pixel_scale_arcsec=engine.pixel_scale_arcsec,
                        orientation_deg=engine.orientation_deg,
                        parity=engine.parity,
                        inliers=engine.inliers,
                        rms_px=engine.rms_px,
                        warnings=engine.warnings,
                        error=engine.error,
                        raw=raw,
                    )
                return engine
        except BlindSolverRuntimeError as exc:
            return EngineSolveResult(status=SolveStatus.FAILED, backend="BLIND4D", error=str(exc))
        except Exception as exc:
            return EngineSolveResult(status=SolveStatus.FAILED, backend="BLIND4D", error=f"blind_port_failed: {exc}")

    def build_config(
        self,
        request: BlindSolveRequest,
        *,
        resources: SolverCatalogResources,
        configuration,
        loaded_manifest: Loaded4DManifest | None = None,
    ):
        manifest = loaded_manifest
        if manifest is None:
            if resources.blind4d_manifest_path is None:
                raise IndexManifestError("blind_4d_manifest_required")
            manifest = load_4d_index_manifest(resources.blind4d_manifest_path)
        shim = _legacy_config_shim(request, resources=resources, configuration=configuration, loaded_manifest=manifest)
        app = _load_entrypoint_module()
        # The validated 4D product profile clears RA/Dec hints in the profile
        # application itself; pass them through here so config parity stays owned
        # by the existing builder.
        return app.build_blind_solve_config(
            shim,
            ra_hint=request.ra_hint_deg,
            dec_hint=request.dec_hint_deg,
            loaded_manifest=manifest,
        )


def _legacy_config_shim(
    request: BlindSolveRequest,
    *,
    resources: SolverCatalogResources,
    configuration,
    loaded_manifest: Loaded4DManifest,
) -> SimpleNamespace:
    values = dict(configuration.legacy_solve_config_values)
    return SimpleNamespace(
        blind_backend_profile=ZEBLIND_4D_EXPERIMENTAL_PROFILE,
        blind_4d_manifest_path=loaded_manifest.manifest_path,
        blind_4d_loaded_manifest=loaded_manifest,
        blind_index_path=loaded_manifest.manifest_path.parent,
        blind_max_candidates=values.get("blind_max_candidates", 10),
        blind_max_stars=values.get("blind_max_stars", 500),
        blind_max_quads=values.get("blind_max_quads", 8000),
        blind_pixel_tolerance=values.get("blind_pixel_tolerance", 2.5),
        blind_quality_inliers=values.get("blind_quality_inliers", 40),
        blind_quality_rms=values.get("blind_quality_rms", 1.2),
        blind_fast_mode=values.get("blind_fast_mode", True),
        blind_index_scale_overlap_prefilter_enabled=values.get("blind_index_scale_overlap_prefilter_enabled", False),
        blind_index_scale_overlap_proxy_lo_frac=values.get("blind_index_scale_overlap_proxy_lo_frac", 0.05),
        blind_index_scale_overlap_proxy_hi_frac=values.get("blind_index_scale_overlap_proxy_hi_frac", 0.95),
        log_level=values.get("log_level", "INFO"),
        downsample=values.get("downsample", 1),
        hint_ra_deg=request.ra_hint_deg,
        hint_dec_deg=request.dec_hint_deg,
        hint_radius_deg=request.radius_hint_deg,
        hint_focal_mm=request.focal_length_mm,
        hint_pixel_um=request.pixel_size_um,
        hint_resolution_arcsec=request.pixel_scale_arcsec,
        hint_resolution_min_arcsec=request.pixel_scale_min_arcsec,
        hint_resolution_max_arcsec=request.pixel_scale_max_arcsec,
        dev_bucket_limit_override=values.get("dev_bucket_limit_override", 0),
        dev_vote_percentile=values.get("dev_vote_percentile", 40),
        dev_collect_matches_vectorized_experimental=values.get("dev_collect_matches_vectorized_experimental", False),
        dev_bucket_cap_S=values.get("dev_bucket_cap_S", 0),
        dev_bucket_cap_M=values.get("dev_bucket_cap_M", 0),
        dev_bucket_cap_L=values.get("dev_bucket_cap_L", 0),
        dev_detect_k_sigma=values.get("dev_detect_k_sigma", 3.0),
        dev_detect_min_area=values.get("dev_detect_min_area", 5),
        catalog_library_path=resources.library_path,
    )


def _cancel_check(configuration):
    token = configuration.runtime_options.cancel_token
    if token is None:
        return None

    def _check() -> bool:
        if callable(token):
            return bool(token())
        is_set = getattr(token, "is_set", None)
        if callable(is_set):
            return bool(is_set())
        return bool(token)

    return _check


@lru_cache(maxsize=1)
def _load_entrypoint_module():
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "zesolver.py"
    name = "zesolver_entrypoint_blind_port"
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
