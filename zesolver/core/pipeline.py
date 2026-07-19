from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol

from astropy.io import fits
from astropy.wcs import WCS

from zeblindsolver.metadata_solver import NearSolveConfig

from zesolver.catalog_resources import (
    CatalogResourceResolutionError,
    NearCatalogMode,
    NearCatalogRuntimeError,
    SolverCatalogResources,
    resolve_catalog_resources,
    resolve_near_catalog_runtime,
)
from zesolver.settings import ProductSettings, RuntimeOptions, build_solver_configuration
from zesolver.zeblindsolver import near_solve

from .blind_port import ProductionBlindSolverPort
from .models import EngineSolveResult, SolveRequest, SolveResult, SolveStatus
from .preflight import run_preflight
from .result_adapter import failure_result, result_from_engine
from .telemetry import PipelineTelemetry
from .wcs_io import write_wcs_safely


class NearSolverPort(Protocol):
    def solve(self, request: SolveRequest, *, resources: SolverCatalogResources, configuration) -> EngineSolveResult:
        ...


class BlindSolverPort(Protocol):
    def solve(self, request: SolveRequest, *, resources: SolverCatalogResources, configuration) -> EngineSolveResult:
        ...


class ExistingNearSolverPort:
    """Thin adapter over the existing Near wrapper."""

    def solve(self, request: SolveRequest, *, resources: SolverCatalogResources, configuration) -> EngineSolveResult:
        values = configuration.legacy_solve_config_values
        try:
            runtime = resolve_near_catalog_runtime(
                resources,
                mode=str(values.get("near_catalog_mode", "auto") or "auto"),
                legacy_index_root=resources.legacy_index_root,
                blind_only=bool(configuration.product_settings.blind_only),
                legacy_cache_size=int(values.get("near_tile_cache_size", 128) or 128),
            )
        except NearCatalogRuntimeError as exc:
            return EngineSolveResult(status=SolveStatus.CATALOG_UNAVAILABLE, backend="NEAR", error=f"{exc.code}: {exc}")
        if runtime.provider is None:
            error = runtime.error_message or runtime.error_code or "near_catalog_provider_unavailable"
            return EngineSolveResult(status=SolveStatus.CATALOG_UNAVAILABLE, backend="NEAR", error=str(error))
        index_root = runtime.legacy_index_root if runtime.effective_mode is NearCatalogMode.LEGACY_INDEX else None
        target = request.output_path or request.input_path
        if target != request.input_path:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(request.input_path, target)
        near_cfg = NearSolveConfig(
            family=(resources.near.families[0] if resources.near and resources.near.families else "d50"),
            max_tile_candidates=int(values.get("near_max_tile_candidates", 48) or 48),
            tile_cache_size=int(values.get("near_tile_cache_size", 128) or 128),
            detect_backend=str(values.get("near_detect_backend") or "auto"),
            detect_k_sigma=float(values.get("near_detect_k_sigma", 4.5) or 4.5),
            detect_min_area=int(values.get("near_detect_min_area", 8) or 8),
            detect_max_labels=int(values.get("near_detect_max_labels", 1200) or 1200),
            ransac_trials=int(values.get("near_ransac_trials", 1200) or 1200),
            search_margin=float(values.get("near_search_margin", 1.2) or 1.2),
            pixel_tolerance=float(values.get("near_pixel_tolerance", 3.0) or 3.0),
            quality_inliers=int(values.get("near_quality_inliers", 60) or 60),
            quality_rms=float(values.get("near_quality_rms", 1.0) or 1.0),
            max_img_stars=int(values.get("near_max_img_stars", 800) or 800),
            max_cat_stars=int(values.get("near_max_cat_stars", 2000) or 2000),
            try_parity_flip=bool(values.get("near_try_parity_flip", True)),
            astap_iso_strict=bool(values.get("near_astap_iso_strict", True)),
        )
        result = near_solve(
            str(target),
            str(index_root) if index_root is not None else None,
            catalog_provider=runtime.provider,
            config=near_cfg,
            skip_if_valid=False,
            fallback_to_blind=False,
        )
        stats = result.get("stats") if isinstance(result, dict) else {}
        stats = stats if isinstance(stats, dict) else {}
        stats.update(runtime.telemetry(include_paths=False))
        if isinstance(result, dict):
            result["stats"] = stats
        wcs_obj = None
        if result.get("success"):
            with fits.open(target, memmap=False) as hdul:
                wcs_obj = WCS(hdul[0].header, naxis=2, relax=True)
        strict = stats.get("strict_acceptance") if isinstance(stats.get("strict_acceptance"), dict) else {}
        return EngineSolveResult(
            status=SolveStatus.SOLVED if result.get("success") else SolveStatus.UNSOLVED,
            backend="NEAR",
            wcs=wcs_obj,
            wcs_written=bool(result.get("wrote_wcs")),
            center_ra_deg=strict.get("center_ra_deg"),
            center_dec_deg=strict.get("center_dec_deg"),
            pixel_scale_arcsec=stats.get("pix_scale_arcsec"),
            inliers=stats.get("inliers"),
            rms_px=stats.get("rms_px"),
            error=None if result.get("success") else str(result.get("message") or "near_failed"),
            raw=dict(result),
        )


class UnconfiguredBlindSolverPort:
    def solve(self, request: SolveRequest, *, resources: SolverCatalogResources, configuration) -> EngineSolveResult:
        return EngineSolveResult(status=SolveStatus.CATALOG_UNAVAILABLE, backend="BLIND4D", error="blind_port_unconfigured")


class SolverPipeline:
    def __init__(
        self,
        *,
        product_settings: ProductSettings,
        runtime_options: RuntimeOptions,
        near_profile: str = "zenear-v1",
        blind_profile: str = "zeblind4d-v1",
        pipeline_profile: str = "pipeline-v1",
        catalog_resources: SolverCatalogResources | None = None,
        near_solver: NearSolverPort | None = None,
        blind_solver: BlindSolverPort | None = None,
    ) -> None:
        self.configuration = build_solver_configuration(
            product_settings=product_settings,
            runtime_options=runtime_options,
            near_profile=near_profile,
            blind_profile=blind_profile,
            pipeline_profile=pipeline_profile,
        )
        self.catalog_resources = catalog_resources
        self.near_solver = near_solver or ExistingNearSolverPort()
        self.blind_solver = blind_solver or ProductionBlindSolverPort()
        self.last_telemetry: dict[str, object] | None = None

    @property
    def profile_ids(self) -> dict[str, str]:
        return {
            "near": self.configuration.near_profile.profile_id,
            "blind": self.configuration.blind_profile.profile_id,
            "pipeline": self.configuration.pipeline_profile.profile_id,
        }

    def solve(self, request: SolveRequest) -> SolveResult:
        telemetry = PipelineTelemetry(
            request_id=request.request_id,
            pipeline_profile=self.configuration.pipeline_profile.profile_id,
            near_profile=self.configuration.near_profile.profile_id,
            blind_profile=self.configuration.blind_profile.profile_id,
        )
        resources = self._resources()
        telemetry.catalog_source = resources.source
        telemetry.catalog_status = resources.library_status.value if resources.library_status else resources.source
        telemetry.catalog_coverage_fraction = resources.blind4d_coverage_fraction
        telemetry.warnings.extend(resources.warnings)
        catalog_status = telemetry.catalog_status

        if self._cancelled():
            return self._finish_failure(request, telemetry, SolveStatus.CANCELLED, catalog_status, "cancelled_before_near")

        preflight = run_preflight(request, catalog_resources=resources)
        telemetry.warnings.extend(preflight.warnings)
        if not preflight.ok:
            return self._finish_failure(
                request,
                telemetry,
                preflight.status or SolveStatus.FAILED,
                catalog_status,
                preflight.error,
            )

        near_mode = str(getattr(self.configuration.product_settings, "near_catalog_mode", "auto") or "auto").strip().lower()
        should_attempt_near = (
            not self.configuration.product_settings.blind_only
            and (
                resources.near_available
                or resources.legacy_index_root is not None
                or resources.source == "library"
                or near_mode != "auto"
            )
        )
        if should_attempt_near:
            telemetry.near_attempted = True
            try:
                near_result = self.near_solver.solve(request, resources=resources, configuration=self.configuration)
            except Exception as exc:
                near_result = EngineSolveResult(status=SolveStatus.FAILED, backend="NEAR", error=str(exc))
            telemetry.near_result = near_result.status.value
            if near_result.solved:
                final = self._finalize_success(request, near_result, resources, telemetry)
                self.last_telemetry = dict(telemetry.finish(final_status=final.status.value, wcs_written=final.wcs_written))
                return final

        if self._cancelled():
            return self._finish_failure(request, telemetry, SolveStatus.CANCELLED, catalog_status, "cancelled_between_near_and_blind")

        if self.configuration.product_settings.blind_enabled and resources.blind4d_available:
            telemetry.blind_attempted = True
            try:
                blind_result = self.blind_solver.solve(request, resources=resources, configuration=self.configuration)
            except Exception as exc:
                blind_result = EngineSolveResult(status=SolveStatus.FAILED, backend="BLIND4D", error=str(exc))
            telemetry.blind_result = blind_result.status.value
            if blind_result.solved:
                final = self._finalize_success(request, blind_result, resources, telemetry)
                self.last_telemetry = dict(telemetry.finish(final_status=final.status.value, wcs_written=final.wcs_written))
                return final

        status = SolveStatus.UNSOLVED if (should_attempt_near or resources.blind4d_available) else SolveStatus.CATALOG_UNAVAILABLE
        return self._finish_failure(request, telemetry, status, catalog_status, "no_solver_produced_solution")

    def _resources(self) -> SolverCatalogResources:
        if self.catalog_resources is not None:
            return self.catalog_resources
        product = self.configuration.product_settings
        try:
            return resolve_catalog_resources(catalog_library=product.catalog_library_path, enable_environment_discovery=True)
        except CatalogResourceResolutionError:
            raise

    def _cancelled(self) -> bool:
        token = self.configuration.runtime_options.cancel_token
        if token is None:
            return False
        if callable(token):
            return bool(token())
        is_set = getattr(token, "is_set", None)
        if callable(is_set):
            return bool(is_set())
        return bool(token)

    def _finalize_success(
        self,
        request: SolveRequest,
        engine_result: EngineSolveResult,
        resources: SolverCatalogResources,
        telemetry: PipelineTelemetry,
    ) -> SolveResult:
        wcs_written = engine_result.wcs_written
        output_path = request.output_path
        error = engine_result.error
        if engine_result.wcs is not None and not engine_result.wcs_written:
            written = write_wcs_safely(
                input_path=request.input_path,
                output_path=request.output_path,
                wcs=engine_result.wcs,
                overwrite_wcs=request.overwrite_wcs,
                header_updates=_header_updates_from_engine(engine_result),
            )
            wcs_written = written.wcs_written and written.ok
            output_path = written.path
            error = written.error
            if not written.ok:
                failed = EngineSolveResult(
                    status=SolveStatus.FAILED,
                    backend=engine_result.backend,
                    warnings=engine_result.warnings,
                    error=written.error,
                )
                return result_from_engine(
                    request,
                    failed,
                    profile_ids=self.profile_ids,
                    catalog_status=telemetry.catalog_status,
                    warnings=tuple(telemetry.warnings),
                    output_path=output_path,
                )
        normalized = EngineSolveResult(
            status=engine_result.status,
            backend=engine_result.backend,
            wcs=engine_result.wcs,
            wcs_written=wcs_written,
            center_ra_deg=engine_result.center_ra_deg,
            center_dec_deg=engine_result.center_dec_deg,
            pixel_scale_arcsec=engine_result.pixel_scale_arcsec,
            orientation_deg=engine_result.orientation_deg,
            parity=engine_result.parity,
            inliers=engine_result.inliers,
            rms_px=engine_result.rms_px,
            warnings=engine_result.warnings,
            error=error,
            raw=engine_result.raw,
        )
        return result_from_engine(
            request,
            normalized,
            profile_ids=self.profile_ids,
            catalog_status=telemetry.catalog_status,
            warnings=tuple(telemetry.warnings),
            output_path=output_path,
        )

    def _finish_failure(
        self,
        request: SolveRequest,
        telemetry: PipelineTelemetry,
        status: SolveStatus,
        catalog_status: str | None,
        error: str | None,
    ) -> SolveResult:
        result = failure_result(
            request,
            status=status,
            profile_ids=self.profile_ids,
            catalog_status=catalog_status,
            warnings=tuple(telemetry.warnings),
            error=error,
        )
        self.last_telemetry = dict(telemetry.finish(final_status=result.status.value, wcs_written=result.wcs_written))
        return result


def _header_updates_from_engine(engine_result: EngineSolveResult) -> dict[str, object] | None:
    raw = engine_result.raw
    value = raw.get("header_updates") if isinstance(raw, dict) else None
    if isinstance(value, dict):
        return dict(value)
    return None
