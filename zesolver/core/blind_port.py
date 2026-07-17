from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from zeblindsolver.index_manifest_4d import IndexManifestError, Loaded4DManifest, load_4d_index_manifest

from zesolver.catalog_resources import SolverCatalogResources
from zesolver.solver_config import build_blind_config_inputs, build_blind_solve_config
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
        inputs = build_blind_config_inputs(request, resources=resources, configuration=configuration, loaded_manifest=manifest)
        # The validated 4D product profile clears RA/Dec hints in the profile
        # application itself; pass them through here so config parity stays owned
        # by the existing builder.
        return build_blind_solve_config(
            inputs,
            ra_hint=request.ra_hint_deg,
            dec_hint=request.dec_hint_deg,
            loaded_manifest=manifest,
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
