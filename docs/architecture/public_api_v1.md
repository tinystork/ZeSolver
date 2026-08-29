# Public API v1 (`zesolver.api.v1`)

Status: implemented (ZS-PUBLIC-API)

This document describes the stable, public, engine-free API surface of ZeSolver
intended for external consumers such as ZeMosaic.  It lives **only** under
`zesolver.api.v1`; the historical `zesolver` package `__init__` is *not* the
public contract.

## Import boundary

```python
from zesolver.api.v1 import (
    API_VERSION, get_api_info, probe,
    create_solver_runtime, SolverRuntime, SolverSession,
    SolveRequest, SolveHints, SolveOptions, SolveResult,
    BackendPolicy, GpuPolicy, NetworkPolicy, WritePolicy,
    SolveStatus, FailureCode, CanonicalWcsHeader,
    CancellationToken, ProgressPhase, ProgressEvent,
    CapabilityState, CapabilityAvailability, CapabilityUnavailableReason,
    ZeSolverApiError, SolverClosedError, InvalidRequestError,
)
```

Importing `zesolver.api.v1` is **lightweight**: it does not import Qt/PySide6,
CuPy, `zesolver.gui_pipeline`, the heavy solver engine, or the catalog resource
resolver.  This is enforced by two mechanisms:

1. `zesolver/__init__.py` keeps its historical re-exports lazy (PEP 562
   `__getattr__`), so `import zesolver` no longer pulls in the engine.
2. `zesolver.api.v1._adapters` (the private coupling to the engine) performs all
   heavy imports lazily inside its functions, never at module import time.

Heavy engine, Astropy, and GPU imports therefore happen only when a runtime is
first resolved or a solve is actually run.

## Versioning

- `API_VERSION = "1.2"` is the single source of truth.
- `API_MAJOR` / `API_MINOR` are derived from `API_VERSION` (never independent
  constants).

## Public surface

The exact public symbols are enumerated in `zesolver.api.v1.__all__`:

- Version/metadata: `API_VERSION`, `API_MAJOR`, `API_MINOR`, `ApiInfo`,
  `get_api_info`, `RuntimeProbe`, `probe`.
- Capabilities: `CapabilityAvailability`, `CapabilityUnavailableReason`,
  `CapabilityState`.
- Policies: `BackendPolicy`, `GpuPolicy`, `NetworkPolicy`, `WritePolicy`.
- Solve models: `SolveHints`, `SolveOptions`, `SolveRequest`,
  `CanonicalWcsHeader`, `SolveStatus`, `FailureCode`, `SolveResult`.
- Progress/cancellation: `ProgressPhase`, `ProgressEvent`, `CancellationToken`.
- Errors: `ZeSolverApiError`, `SolverClosedError`, `InvalidRequestError`.
- Lifecycle: `create_solver_runtime`, `SolverRuntime`, `SolverSession`.

Everything else is private and must not be relied upon.

## Enum value form (stable wire values)

All public `str` enums expose **lowercase `snake_case` string values** as their
stable wire form for public interop (serialization, logging, comparison by
consumers).  Enum **member names** remain uppercase (for example
`CapabilityAvailability.AVAILABLE`, `GpuPolicy.AUTO`), but the `.value` of each
member is the lowercase contract form:

- `CapabilityAvailability`: `"available"` / `"unavailable"` / `"not_checked"`.
- `CapabilityUnavailableReason`: `"missing_resource"`, `"backend_unavailable"`,
  `"policy_disabled"`, `"gpu_unavailable"`, `"network_unavailable"`,
  `"license_or_auth_required"`, `"unsupported_platform"`, `"unknown"`.
- `GpuPolicy`: `"auto"` / `"disabled"` / `"required"`.
- `NetworkPolicy`: `"disabled"` only (API 1.0 is local-only; `"allowed"` is deferred to a future 1.1+).
- `BackendPolicy`: `"auto"` / `"near_only"` / `"blind_only"`.
- `WritePolicy`: `"overwrite_input"` / `"write_copy"`.
- `SolveStatus`: `"solved"` / `"skipped_existing_wcs"` / `"failed"` /
  `"cancelled"`.
- `FailureCode`: lowercase `snake_case` values (for example
  `"invalid_input"`, `"no_solution"`, `"backend_unavailable"`).
- `ProgressPhase`: `"preparing"` / `"solving"` / `"writing"` / `"finalizing"`.

## Metadata and probing

- `get_api_info()` returns **static** metadata only (`api_version`,
  `product_version`, `supported_capabilities`).  No catalog scan, GPU init, or
  I/O.
- `probe(check_catalogs=False, check_gpu=False, timeout_s=2.0, resources_path=None)`
  separates the *supported* capability from its negotiated availability:
  `AVAILABLE` / `UNAVAILABLE` / `NOT_CHECKED`.  The default probe performs no
  catalog scan and no GPU/CuPy import.

## Policies

- `GpuPolicy` is **runtime-scoped** (`AUTO` / `DISABLED` / `REQUIRED`); it is
  never a per-solve option.
- `NetworkPolicy` defaults to `DISABLED` everywhere in v1; the API performs no
  network access.
- `BackendPolicy` is non-ambiguous: `AUTO` / `NEAR_ONLY` / `BLIND_ONLY`.
- `WritePolicy` exposes only `OVERWRITE_INPUT` and `WRITE_COPY`.  There is no
  `NO_WRITE` in v1 and no temp-copy simulation of one.

## WCS transport

The primary public WCS transport is `CanonicalWcsHeader`, an ordered tuple of
FITS header **card strings** (`format="fits-header-cards-v1"`).  It preserves
card order and repeatable cards (`COMMENT`/`HISTORY`) and can round-trip to an
Astropy `Header` or `WCS`.  Public dataclasses never carry a live Astropy `WCS`
object.

## Runtime and session lifecycle

- **One `SolverRuntime` per process** owns process-scoped, shareable heavy
  resources (resolved catalog resources, the shared Near catalog provider, the
  resolved Blind-4D manifest selection).  These are resolved once and never
  rebuilt per image.
- **One `SolverSession` per concurrent worker/thread** owns the mutable,
  non-thread-safe solve context (a per-session Blind prep cache).  A session is
  not safe for concurrent `solve()` calls and detects that misuse.
- Sessions are not shared across processes.
- `SolverRuntime.close()` is idempotent and invalidates all its sessions;
  `SolverSession.close()` is idempotent.

```python
runtime = create_solver_runtime(resources_path="/path/to/library")
session = runtime.create_session()
result = session.solve(SolveRequest("image.fits", hints=SolveHints(ra_deg=..., dec_deg=...)))
session.close()
runtime.close()
```

## Error model

- **Contract misuse** (bad request, closed runtime/session, wrong types) raises
  public API exceptions: `InvalidRequestError`, `SolverClosedError`, or their
  base `ZeSolverApiError`.
- **Expected operational failures** (no solution, missing catalog, backend
  unavailable, timeout, WCS rejected, write impossible, cancellation) are
  returned as `SolveResult` objects — never raised.
- **Unexpected engine bugs** are re-raised wrapped in `ZeSolverApiError`
  (preserving `__cause__`), never silently converted into a failed
  `SolveResult`.

## Explicitly out of public scope

The following are intentionally *not* part of the public v1 surface:

- `SolverPipeline`, `ProductSettings`, `RuntimeOptions`, `SolverCatalogResources`
- solver ports/profiles and `TerminalReasonCode`
- any batch API
- `NO_WRITE` write policy and temp-copy `NO_WRITE` simulation
- live Astropy `WCS` objects in public dataclasses
