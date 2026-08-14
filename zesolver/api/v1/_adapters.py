"""Private adapters between the public v1 API and the existing solver engine.

This module is intentionally *not* part of the public contract.  It contains
all of the coupling to the internal engine (catalog resolution, near/blind
adapters, WCS write helpers) so the public models stay engine-free.

Import-boundary rule: importing :mod:`zesolver.api.v1` must stay lightweight.
All imports of the heavy solver engine, Astropy, and the optional GPU stack are
therefore performed lazily inside the functions that need them, never at module
import time.

Error-boundary rule: expected operational failures (no solution, missing
catalog, backend unavailable, timeout, WCS rejected, write impossible,
cancellation) are returned as :class:`SolveResult` objects.  Unexpected
exceptions raised by the underlying engine are re-raised wrapped in
:class:`ZeSolverApiError` (preserving ``__cause__``) so programming bugs never
silently become ``SolveResult(FAILED)``.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .errors import InvalidRequestError, ZeSolverApiError
from .models import (
    BackendPolicy,
    CanonicalWcsHeader,
    FailureCode,
    GpuPolicy,
    NetworkPolicy,
    ProgressEvent,
    ProgressPhase,
    SolveHints,
    SolveRequest,
    SolveResult,
    SolveStatus,
    WritePolicy,
)

log = logging.getLogger("zesolver.api.v1")


# ---------------------------------------------------------------------------
# Runtime context
# ---------------------------------------------------------------------------


@dataclass
class _RuntimeContext:
    resources: object
    near_shared: object | None
    blind_selection: object | None
    blind_selection_error: object | None
    resources_error: str | None = None


def build_runtime_context(
    resources_path: Path | None,
    injected_resources: object | None,
) -> _RuntimeContext:
    """Resolve the process-scoped heavy resources exactly once.

    Imported lazily so that constructing a :class:`SolverRuntime` and importing
    the public API stay free of the heavy engine.
    """
    from zesolver.catalog_resources import (
        Blind4DRuntimeError,
        CatalogResourceResolutionError,
        NearBatchRuntime,
        NearCatalogMode,
        SolverCatalogResources,
        resolve_blind4d_runtime,
        resolve_catalog_resources,
    )

    resources = injected_resources
    resources_error: str | None = None
    if resources is None:
        try:
            resources = resolve_catalog_resources(
                catalog_library=resources_path,
                enable_environment_discovery=(resources_path is None),
            )
        except CatalogResourceResolutionError as exc:
            # Expected operational failure: an explicitly requested catalog
            # library could not be resolved.  Degrade to "no resources" instead
            # of letting an internal catalog exception escape the public v1
            # boundary.  The diagnostic detail is preserved (non-stable, for
            # logs/debug only) and surfaced on the SolveResult.
            resources_error = str(exc)
            resources = SolverCatalogResources(
                library_path=resources_path,
                library_status=None,
                near=None,
                blind4d_indexes=(),
                blind4d_runtime_paths=(),
                blind4d_manifest_path=None,
                legacy_index_root=None,
                source="none",
                warnings=("catalog_resource_resolution_failed",),
                catalog_library_id=None,
                catalog_manifest_fingerprint=None,
                coverage=None,
                all_sky_blind4d=False,
                catalog_library=None,
            )

    near_shared = None
    if resources.near_available:
        near_shared = NearBatchRuntime(
            resources,
            mode=NearCatalogMode.AUTO,
            legacy_index_root=resources.legacy_index_root,
            legacy_cache_size=128,
        )

    blind_selection = None
    blind_selection_error = None
    if resources.blind4d_available:
        try:
            blind_selection = resolve_blind4d_runtime(resources, mode="auto")
        except Blind4DRuntimeError as exc:
            blind_selection_error = exc

    return _RuntimeContext(
        resources=resources,
        near_shared=near_shared,
        blind_selection=blind_selection,
        blind_selection_error=blind_selection_error,
        resources_error=resources_error,
    )


# ---------------------------------------------------------------------------
# Engine adapters (expected failures -> EngineSolveResult, bugs -> raise)
# ---------------------------------------------------------------------------


def near_solve_engine(
    internal_req,
    resources,
    configuration,
    shared_near,
    cancel_check: Callable[[], bool] | None,
):
    from astropy.io import fits
    from astropy.wcs import WCS

    from zeblindsolver.metadata_solver import NearSolveConfig

    from zesolver.catalog_resources import NearCatalogMode, NearCatalogRuntimeError
    from zesolver.core.models import EngineSolveResult, SolveStatus as InternalStatus
    from zesolver.zeblindsolver import (
        BlindSolverRuntimeError,
        InvalidInputError,
        near_solve,
    )

    if shared_near is None:
        return EngineSolveResult(
            status=InternalStatus.CATALOG_UNAVAILABLE,
            backend="NEAR",
            error="near_catalog_provider_unavailable",
        )
    try:
        runtime = shared_near.acquire()
    except NearCatalogRuntimeError as exc:
        return EngineSolveResult(
            status=InternalStatus.CATALOG_UNAVAILABLE,
            backend="NEAR",
            error=f"{exc.code}: {exc}",
        )
    if runtime.provider is None:
        error = runtime.error_message or runtime.error_code or "near_catalog_provider_unavailable"
        return EngineSolveResult(
            status=InternalStatus.CATALOG_UNAVAILABLE, backend="NEAR", error=str(error)
        )

    values = configuration.legacy_solve_config_values
    index_root = (
        runtime.legacy_index_root
        if runtime.effective_mode is NearCatalogMode.LEGACY_INDEX
        else None
    )
    target = internal_req.output_path or internal_req.input_path
    if target != internal_req.input_path:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(internal_req.input_path, target)

    near_cfg = NearSolveConfig(
        family=(resources.near.families[0] if resources.near and resources.near.families else "d50"),
        max_tile_candidates=int(values.get("near_max_tile_candidates", 48) or 48),
        tile_cache_size=int(values.get("near_tile_cache_size", 128) or 128),
        detect_backend=str(values.get("near_detect_backend") or "auto"),
        detect_device=(
            int(values["near_detect_device"])
            if values.get("near_detect_device") is not None
            else None
        ),
        detect_k_sigma=float(values.get("near_detect_k_sigma", 4.5) or 4.5),
        detect_min_area=int(values.get("near_detect_min_area", 8) or 8),
        detect_max_labels=int(values.get("near_detect_max_labels", 1200) or 1200),
        detect_gpu_slots=int(values.get("near_detect_gpu_slots", 1) or 1),
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
    try:
        result = near_solve(
            str(target),
            str(index_root) if index_root is not None else None,
            catalog_provider=runtime.provider,
            config=near_cfg,
            skip_if_valid=False,
            fallback_to_blind=False,
            cancel_check=cancel_check,
        )
    except InvalidInputError:
        raise
    except BlindSolverRuntimeError as exc:
        if "cancelled" in str(exc).lower():
            return EngineSolveResult(status=InternalStatus.CANCELLED, backend="NEAR", error="cancelled")
        raise

    if str(result.get("message") or "").strip().lower() == "cancelled":
        return EngineSolveResult(status=InternalStatus.CANCELLED, backend="NEAR", error="cancelled")

    wcs_obj = None
    if result.get("success"):
        with fits.open(target, memmap=False) as hdul:
            wcs_obj = WCS(hdul[0].header, naxis=2, relax=True)
    stats = result.get("stats") if isinstance(result, dict) else {}
    stats = stats if isinstance(stats, dict) else {}
    strict = stats.get("strict_acceptance") if isinstance(stats.get("strict_acceptance"), dict) else {}
    return EngineSolveResult(
        status=InternalStatus.SOLVED if result.get("success") else InternalStatus.UNSOLVED,
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


def blind_solve_engine(
    internal_req,
    resources,
    configuration,
    cancel_check: Callable[[], bool] | None,
    prep_cache: dict,
    blind_selection=None,
):
    from zesolver.catalog_resources import Blind4DRuntimeError, resolve_blind4d_runtime
    from zesolver.core.blind_models import BlindSolveRequest
    from zesolver.core.blind_result_adapter import engine_result_from_blind_result
    from zesolver.core.models import EngineSolveResult, SolveStatus as InternalStatus
    from zesolver.solver_config import build_blind_config_inputs, build_blind_solve_config
    from zesolver.zeblindsolver import (
        BlindSolverRuntimeError,
        InvalidInputError,
        blind_solve,
    )
    from zeblindsolver.index_manifest_4d import IndexManifestError

    if blind_selection is None:
        try:
            blind_selection = resolve_blind4d_runtime(resources, mode="auto")
        except Blind4DRuntimeError as exc:
            return EngineSolveResult(
                status=InternalStatus.CATALOG_UNAVAILABLE,
                backend="BLIND4D",
                error=f"{exc.code}: {exc}",
            )
    if not blind_selection.available or blind_selection.loaded_manifest is None:
        error = (
            blind_selection.error_message
            or blind_selection.error_code
            or "BLIND4D_RUNTIME_RESOURCE_UNAVAILABLE"
        )
        return EngineSolveResult(
            status=InternalStatus.CATALOG_UNAVAILABLE, backend="BLIND4D", error=str(error)
        )

    loaded_manifest = blind_selection.loaded_manifest
    blind_request = BlindSolveRequest.from_solve_request(internal_req, configuration=configuration)
    try:
        inputs = build_blind_config_inputs(
            blind_request, resources=resources, configuration=configuration, loaded_manifest=loaded_manifest
        )
        blind_cfg = build_blind_solve_config(
            inputs,
            ra_hint=blind_request.ra_hint_deg,
            dec_hint=blind_request.dec_hint_deg,
            loaded_manifest=loaded_manifest,
        )
    except IndexManifestError as exc:
        return EngineSolveResult(
            status=InternalStatus.CATALOG_UNAVAILABLE,
            backend="BLIND4D",
            error=f"blind_4d_manifest: {exc}",
        )

    index_root = loaded_manifest.manifest_path.parent
    try:
        with tempfile.TemporaryDirectory(prefix="zesolver-api-blind4d-") as tmp:
            temp_path = Path(tmp) / internal_req.input_path.name
            shutil.copyfile(internal_req.input_path, temp_path)
            result = blind_solve(
                str(temp_path),
                str(index_root),
                config=blind_cfg,
                log=logging.info,
                skip_if_valid=False,
                cancel_check=cancel_check,
                prep_cache=prep_cache,
            )
            engine = engine_result_from_blind_result(result, solved_path=temp_path)
            if engine.raw and isinstance(engine.raw, dict):
                raw = dict(engine.raw)
                raw.update(blind_selection.telemetry(include_paths=False))
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
    except InvalidInputError as exc:
        return EngineSolveResult(status=InternalStatus.INVALID_INPUT, backend="BLIND4D", error=str(exc))
    except BlindSolverRuntimeError as exc:
        if "cancelled" in str(exc).lower():
            return EngineSolveResult(status=InternalStatus.CANCELLED, backend="BLIND4D", error="cancelled")
        return EngineSolveResult(status=InternalStatus.FAILED, backend="BLIND4D", error=str(exc))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_solve(
    request: SolveRequest,
    *,
    resources,
    near_shared,
    blind_selection,
    resources_error: str | None = None,
    gpu_policy: GpuPolicy,
    network_policy: NetworkPolicy,
    resources_path: Path | None,
    near_solver: Callable = None,
    blind_solver: Callable = None,
    cancellation=None,
    progress=None,
    prep_cache: dict | None = None,
) -> SolveResult:
    from zesolver.core.models import SolveRequest as InternalSolveRequest
    from zesolver.core.models import SolveStatus as InternalStatus
    from zesolver.core.preflight import run_preflight

    near_solver = near_solve_engine if near_solver is None else near_solver
    blind_solver = blind_solve_engine if blind_solver is None else blind_solver

    start = time.perf_counter()
    options = request.options
    hints = request.hints
    write_policy = options.write_policy
    output_path = options.output_path
    overwrite = options.overwrite_existing_wcs
    timeout_s = options.timeout_s
    prep_cache = {} if prep_cache is None else prep_cache

    def elapsed() -> float:
        return round(time.perf_counter() - start, 6)

    _emit(progress, ProgressPhase.PREPARING, "preparing solve")

    if cancellation is not None and cancellation.is_cancelled():
        return _result(
            request, SolveStatus.CANCELLED, failure_code=None,
            diagnostic_code=None, message="cancelled", elapsed_s=elapsed(),
        )

    deadline = start + float(timeout_s) if timeout_s is not None else None

    # Pre-solve WRITE_COPY checks (determinable before solving).
    if write_policy is WritePolicy.WRITE_COPY:
        precheck = _precheck_write_copy(request, output_path, elapsed)
        if precheck is not None:
            return precheck

    # GPU required check (cheap, no CuPy import).
    if gpu_policy is GpuPolicy.REQUIRED and not _gpu_available():
        return _result(
            request, SolveStatus.FAILED, FailureCode.BACKEND_UNAVAILABLE,
            diagnostic_code="gpu_required_unavailable",
            message="gpu_policy=REQUIRED but no GPU is available", elapsed_s=elapsed(),
        )

    # Existing WCS handling.
    existing = _existing_wcs_state(request.input_path)
    if existing == "valid" and not overwrite:
        return _result(
            request, SolveStatus.SKIPPED_EXISTING_WCS, failure_code=None,
            diagnostic_code=None, message="existing valid WCS, overwrite not requested",
            elapsed_s=elapsed(),
        )
    if existing == "invalid" and not overwrite:
        return _result(
            request, SolveStatus.FAILED, FailureCode.EXISTING_WCS_INVALID,
            diagnostic_code="existing_wcs_invalid", message="existing WCS is invalid and overwrite is disabled",
            elapsed_s=elapsed(),
        )

    internal_req = InternalSolveRequest(
        input_path=request.input_path,
        output_path=(output_path if write_policy is WritePolicy.WRITE_COPY else None),
        overwrite_wcs=True,
    )
    preflight = run_preflight(internal_req, catalog_resources=None, require_catalog=False)
    if not preflight.ok:
        return _result(
            request, SolveStatus.FAILED, FailureCode.INVALID_INPUT,
            diagnostic_code=preflight.error, message=preflight.error, elapsed_s=elapsed(),
        )

    configuration = _build_configuration(request, resources_path, gpu_policy, network_policy, cancellation)
    cancel_check = _make_cancel_check(cancellation, deadline)

    near_available = bool(resources.near_available)
    blind_available = bool(resources.blind4d_available)

    attempts: list[tuple[str, Callable]] = []
    backend_policy = options.backend_policy
    if backend_policy is BackendPolicy.NEAR_ONLY:
        if not near_available:
            return _result(
                request, SolveStatus.FAILED, FailureCode.BACKEND_UNAVAILABLE,
                diagnostic_code="near_backend_unavailable",
                message="NEAR backend requested but no near catalog is available", elapsed_s=elapsed(),
            )
        attempts = [("NEAR", near_solver)]
    elif backend_policy is BackendPolicy.BLIND_ONLY:
        if not blind_available:
            return _result(
                request, SolveStatus.FAILED, FailureCode.BACKEND_UNAVAILABLE,
                diagnostic_code="blind_backend_unavailable",
                message="BLIND backend requested but no blind index is available", elapsed_s=elapsed(),
            )
        attempts = [("BLIND4D", blind_solver)]
    else:
        if near_available:
            attempts.append(("NEAR", near_solver))
        if blind_available:
            attempts.append(("BLIND4D", blind_solver))
        if not attempts:
            if resources_error is not None:
                return _result(
                    request, SolveStatus.FAILED, FailureCode.MISSING_RESOURCE,
                    f"catalog_resources_invalid: {resources_error}",
                    f"no solver catalog resources are available: {resources_error}",
                    elapsed_s=elapsed(),
                )
            return _result(
                request, SolveStatus.FAILED, FailureCode.MISSING_RESOURCE,
                diagnostic_code="catalog_resources_absent",
                message="no solver catalog resources are available", elapsed_s=elapsed(),
            )

    _emit(progress, ProgressPhase.SOLVING, "solving")

    last_engine = None
    for backend, solver_fn in attempts:
        if cancellation is not None and cancellation.is_cancelled():
            return _result(request, SolveStatus.CANCELLED, None, None, "cancelled", elapsed_s=elapsed())
        if deadline is not None and time.monotonic() > deadline:
            return _result(request, SolveStatus.FAILED, FailureCode.TIMEOUT, "timeout", "solve timed out", elapsed_s=elapsed())
        try:
            if backend == "NEAR":
                engine = solver_fn(internal_req, resources, configuration, near_shared, cancel_check)
            else:
                engine = solver_fn(internal_req, resources, configuration, cancel_check, prep_cache, blind_selection)
        except ZeSolverApiError:
            raise
        except Exception as exc:
            raise ZeSolverApiError(f"unexpected solver error in {backend} backend") from exc

        last_engine = engine
        if engine.solved:
            return _finalize_success(request, engine, write_policy, output_path, elapsed, progress)

        if cancellation is not None and cancellation.is_cancelled():
            return _result(request, SolveStatus.CANCELLED, None, None, "cancelled", elapsed_s=elapsed())
        if deadline is not None and time.monotonic() > deadline:
            return _result(request, SolveStatus.FAILED, FailureCode.TIMEOUT, "timeout", "solve timed out", elapsed_s=elapsed())
        if engine.status is InternalStatus.CANCELLED:
            return _result(request, SolveStatus.CANCELLED, None, None, "cancelled", elapsed_s=elapsed())

    _emit(progress, ProgressPhase.FINALIZING, "finalizing")
    return _exhausted_result(request, last_engine, elapsed)


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------


def _result(
    request: SolveRequest,
    status: SolveStatus,
    failure_code: FailureCode | None,
    diagnostic_code: str | None,
    message: str | None,
    *,
    elapsed_s: float | None = None,
    warnings: tuple[str, ...] = (),
    output_path: Path | None = None,
    backend_used: str | None = None,
    wcs_header: CanonicalWcsHeader | None = None,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    pixel_scale_arcsec: float | None = None,
    orientation_deg: float | None = None,
) -> SolveResult:
    return SolveResult(
        status=status,
        input_path=request.input_path,
        output_path=output_path,
        wcs_header=wcs_header,
        backend_used=backend_used,
        failure_code=failure_code,
        diagnostic_code=diagnostic_code,
        message=message,
        warnings=warnings,
        elapsed_s=elapsed_s,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        pixel_scale_arcsec=pixel_scale_arcsec,
        orientation_deg=orientation_deg,
    )


def _finalize_success(
    request: SolveRequest,
    engine,
    write_policy: WritePolicy,
    output_path: Path | None,
    elapsed: Callable[[], float],
    progress,
) -> SolveResult:
    from astropy.io import fits
    from astropy.wcs import WCS

    from zesolver.core.result_adapter import orientation_deg_from_wcs, pixel_scale_arcsec_from_wcs
    from zesolver.core.wcs_io import write_wcs_safely
    from zesolver.zeblindsolver import has_valid_wcs

    _emit(progress, ProgressPhase.WRITING, "writing WCS")
    target = output_path if write_policy is WritePolicy.WRITE_COPY else request.input_path
    wcs_obj = engine.wcs
    if not engine.wcs_written:
        written = write_wcs_safely(
            input_path=request.input_path,
            output_path=output_path,
            wcs=wcs_obj,
            overwrite_wcs=True,
            header_updates={"SOLVED": 1, "SOLVER": "ZeSolver"},
        )
        if not written.ok:
            code = FailureCode.WCS_INVALID if _is_wcs_error(written.error) else FailureCode.WRITE_FAILED
            return _result(
                request, SolveStatus.FAILED, code, written.error,
                written.error or "wcs write failed", elapsed_s=elapsed(),
                output_path=written.path,
            )
        target = written.path or target

    try:
        with fits.open(target, memmap=False) as hdul:
            header = hdul[0].header.copy()
    except OSError as exc:
        return _result(
            request, SolveStatus.FAILED, FailureCode.WRITE_FAILED,
            f"wcs_read_back_failed: {exc}", f"unable to read written WCS: {exc}",
            elapsed_s=elapsed(), output_path=target,
        )

    if not has_valid_wcs(header):
        return _result(
            request, SolveStatus.FAILED, FailureCode.WCS_INVALID,
            "written_wcs_invalid", "written WCS failed validation",
            elapsed_s=elapsed(), output_path=target,
        )

    canonical = CanonicalWcsHeader.from_fits_header(header)
    ra_deg = engine.center_ra_deg
    dec_deg = engine.center_dec_deg
    scale = (
        engine.pixel_scale_arcsec
        if engine.pixel_scale_arcsec is not None
        else pixel_scale_arcsec_from_wcs(wcs_obj)
    )
    orientation = (
        engine.orientation_deg
        if engine.orientation_deg is not None
        else orientation_deg_from_wcs(wcs_obj)
    )
    if ra_deg is None or dec_deg is None:
        try:
            w = WCS(header, relax=True)
            ra_deg = float(w.wcs.crval[0])
            dec_deg = float(w.wcs.crval[1])
        except Exception:
            pass

    _emit(progress, ProgressPhase.FINALIZING, "finalizing")
    return _result(
        request, SolveStatus.SOLVED, None, None, None,
        elapsed_s=elapsed(), warnings=tuple(engine.warnings or ()),
        output_path=target, backend_used=engine.backend,
        wcs_header=canonical, ra_deg=ra_deg, dec_deg=dec_deg,
        pixel_scale_arcsec=scale, orientation_deg=orientation,
    )


def _exhausted_result(request: SolveRequest, last_engine, elapsed) -> SolveResult:
    from zesolver.core.models import SolveStatus as InternalStatus

    if last_engine is None:
        return _result(
            request, SolveStatus.FAILED, FailureCode.BACKEND_UNAVAILABLE,
            "no_backend_attempted", "no solver backend was attempted", elapsed_s=elapsed(),
        )
    status = last_engine.status
    if status is InternalStatus.UNSOLVED:
        return _result(
            request, SolveStatus.FAILED, FailureCode.NO_SOLUTION,
            last_engine.error, last_engine.error or "no solution produced",
            elapsed_s=elapsed(), warnings=tuple(last_engine.warnings or ()),
            backend_used=last_engine.backend,
        )
    if status is InternalStatus.CATALOG_UNAVAILABLE:
        return _result(
            request, SolveStatus.FAILED, FailureCode.MISSING_RESOURCE,
            last_engine.error, last_engine.error or "catalog unavailable",
            elapsed_s=elapsed(), backend_used=last_engine.backend,
        )
    if status is InternalStatus.INVALID_INPUT:
        return _result(
            request, SolveStatus.FAILED, FailureCode.INVALID_INPUT,
            last_engine.error, last_engine.error, elapsed_s=elapsed(),
        )
    return _result(
        request, SolveStatus.FAILED, FailureCode.BACKEND_UNAVAILABLE,
        last_engine.error, last_engine.error or "backend unavailable",
        elapsed_s=elapsed(), warnings=tuple(last_engine.warnings or ()),
        backend_used=last_engine.backend,
    )


def _precheck_write_copy(request, output_path, elapsed):
    if output_path is None:  # pragma: no cover - validated at construction
        return _result(request, SolveStatus.FAILED, FailureCode.WRITE_FAILED, "output_path_required", "WRITE_COPY requires output_path", elapsed_s=elapsed())
    if output_path.exists():
        return _result(request, SolveStatus.FAILED, FailureCode.WRITE_FAILED, "output_path_exists", f"output_path already exists: {output_path}", elapsed_s=elapsed())
    parent = output_path.parent
    if not parent.exists():
        return _result(request, SolveStatus.FAILED, FailureCode.WRITE_FAILED, "output_parent_missing", f"output parent does not exist: {parent}", elapsed_s=elapsed())
    if not os.access(parent, os.W_OK):
        return _result(request, SolveStatus.FAILED, FailureCode.WRITE_FAILED, "output_parent_not_writable", f"output parent not writable: {parent}", elapsed_s=elapsed())
    return None


def _is_wcs_error(error: str | None) -> bool:
    text = str(error or "").lower()
    return "wcs" in text or "written_wcs_invalid" in text


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _emit(progress, phase: ProgressPhase, message: str) -> None:
    if progress is None:
        return
    try:
        progress(ProgressEvent(phase=phase, message=message))
    except Exception:
        pass


def _existing_wcs_state(input_path: Path) -> str:
    from astropy.io import fits

    from zesolver.zeblindsolver import has_valid_wcs

    try:
        with fits.open(input_path, memmap=False) as hdul:
            header = hdul[0].header
            if has_valid_wcs(header):
                return "valid"
            if _has_wcs_keywords(header):
                return "invalid"
            return "none"
    except Exception:
        return "none"


def _has_wcs_keywords(header) -> bool:
    keys = ("CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2")
    return any(k in header for k in keys)


def _make_cancel_check(cancellation, deadline: float | None) -> Callable[[], bool] | None:
    if cancellation is None and deadline is None:
        return None

    def _check() -> bool:
        if cancellation is not None and cancellation.is_cancelled():
            return True
        if deadline is not None and time.monotonic() > deadline:
            return True
        return False

    return _check


def _build_configuration(
    request: SolveRequest,
    resources_path: Path | None,
    gpu_policy: GpuPolicy,
    network_policy: NetworkPolicy,
    cancellation,
):
    from zesolver.settings import ProductSettings, RuntimeOptions, build_solver_configuration

    hints: SolveHints = request.hints
    # Source of truth for network behavior in API v1: the public API is
    # local-only.  ``web_fallback`` is forced off here and MUST NOT be read
    # from any persisted GUI preference (``use_web_fallback`` /
    # ``astrometry_fallback_after_blind``).  ``network_policy`` can only ever
    # be ``NetworkPolicy.DISABLED`` in API 1.0 (see models.NetworkPolicy).
    product_settings = ProductSettings(
        catalog_library_path=resources_path,
        gpu_mode=_gpu_mode_text(gpu_policy),
        web_fallback=False,
        blind_enabled=True,
        blind_only=False,
        near_catalog_mode="auto",
        blind4d_catalog_mode="auto",
        interface_mode="expert",
        overwrite_wcs=True,
        fov_deg=(hints.fov_deg if hints.fov_deg is not None else 1.5),
        hint_ra_deg=hints.ra_deg,
        hint_dec_deg=hints.dec_deg,
        hint_radius_deg=hints.radius_deg,
        hint_focal_mm=hints.focal_length_mm,
        hint_pixel_um=hints.pixel_size_um,
        hint_resolution_arcsec=hints.pixel_scale_arcsec,
    )
    runtime_options = RuntimeOptions(
        cancel_token=(lambda: cancellation.is_cancelled()) if cancellation is not None else None,
    )
    return build_solver_configuration(
        product_settings=product_settings,
        runtime_options=runtime_options,
    )


def _gpu_mode_text(gpu_policy: GpuPolicy) -> str:
    if gpu_policy is GpuPolicy.DISABLED:
        return "cpu"
    if gpu_policy is GpuPolicy.REQUIRED:
        return "cuda"
    return "auto"


def _gpu_available() -> bool:
    try:
        from zesolver.gpu_support.models import EffectiveBackend
        from zesolver.gpu_support.probe import probe_gpu_capability

        report = probe_gpu_capability(run_self_test=False)
        return report.effective_backend is EffectiveBackend.CUDA
    except Exception:
        return False
