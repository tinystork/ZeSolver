# Changelog

## [1.2.2] - 2026-09-27

### Fixed

- **Catalogue star quota is now orientation-generic.** The strict ASTAP-ISO
  path sized its local catalogue window as a square of side
  `max(fov_x, fov_y)` (degrees) but derived the number of catalogue stars to
  request from `Nimg * height/width` (pixels). That is only correct for
  portrait frames: for landscape fields it under-provisioned the matcher by
  ≈2.15x, so the magnitude truncation removed the stars that actually fall
  inside the frame and no similarity transform could be estimated. The quota
  now follows the surface law `Nimg * window_area / footprint_area`, which is
  algebraically identical to the historical formula for every portrait frame
  (Seestar S50 results are bit-identical) and restores landscape solving
  (e.g. ASI294MC 4144x2822 @ ~1800 mm, FOV < 1°). No acceptance gate, quad
  tolerance or magnitude threshold was changed.
- The catalogue `oversize` factor is applied exactly once. The first
  implementation of the surface law folded `oversize` into the window and then
  applied it a second time, inflating the quota by up to 4x on star-poor
  portrait frames.
- Handled non-2D auxiliary HDUs during WCS probing (previously validated on
  the beta branch, included in this release).

### Added

- **Deterministic pointing/scale hint resolution** with an explicit, tested
  precedence: explicit user override > valid FITS acquisition metadata >
  instrument preset > generic fallback. Presets configured in the settings are
  now actually propagated to the near solver (previously they were dropped
  before `NearSolveConfig`), and a FITS without `RA`/`DEC` can now be solved
  from a preset hint. Per-field sources are reported in solve statistics and
  the GUI indicator reflects real consumption instead of configuration intent.

### Changed

- Near hint telemetry is stored per solve run instead of module state, so
  concurrent solves in the same process (parallel batch workers) can no longer
  cross-contaminate each other's reported hint sources.

### Notes

- Public API version is unchanged (`zesolver.api.v1` == 1.2).
- Qualified hint-offset domain for near solving is >= 1.00x FOV (8/8 directions)
  on both a portrait wide-field (Seestar S50) and a landscape narrow-field
  (ASI294MC) geometry, up from ~0.10x FOV on the landscape geometry before this
  release.

## [1.2.1] - 2026-08-30

### Fixed

- Made the default public API runtime use the same persisted catalogue
  discovery as `readiness()`, so embedded consumers can solve with a configured
  ZeSolver without importing private settings or supplying catalogue paths.
- Isolated settings-persistence and public resource-boundary tests from the
  real user settings file so test runs cannot overwrite catalogue configuration.

## [1.2.0] - 2026-08-29

### Added

- Public API v1 `readiness()` and `open_configuration()` (readiness /
  configuration surface) exposing operational status and catalog-configuration
  launch through the stable `zesolver.api.v1` contract.
- Public API v1.1 -> v1.2: `open_configuration()` returns an opaque
  `ConfigurationSession` handle (observable lifecycle via `is_running()` /
  `wait()`).
- Provider interop metadata (`zesolver/zesoftware_interop.json`, schema
  `zesoftware.interop.v1`) so the installed ZeSolver distribution is
  verifiable by the ZeAlfie compatibility gate when a consumer such as
  ZeMosaic declares `zesolver.api.v1` (API 1.2; capabilities:
  `near_solve`, `blind_solve`, `wcs_write`, `cancel`, `gpu`).
- Wheel-install witness test: the built wheel installs standalone and the
  public API still imports (ZS-INTEROP-PROVIDER-CLOSURE).

### Fixed

- Canonicalized `.fit`, `.fits`, and `.fts` GUI batch engine selection as one
  FITS family so mixed FITS extensions stay on the modern Pipeline in AUTO.
- Added a reproducible public `main` projection manifest and builder so the
  user-facing branch can be generated from `test` without merging tests,
  internal tools, reports, or development-only documentation.
- Promoted guided GPU provisioning for safe source-managed virtual
  environments, while keeping system Python, frozen builds and embedded hosts
  diagnostic-only.
- Added visible GPU installation progress in the startup wizard, including pip
  output, pip check, a fresh CUDA self-test subprocess, and a clear restart
  required state instead of leaving the user with a silent install.
- Fixed a GPU provisioning wizard crash after successful pip installation by
  separating the custom result signal from native `QThread.finished`, delaying
  worker cleanup until the Qt thread has really stopped, and continuously
  draining pip output.
- Added a guided optional GPU diagnostic/provisioning layer for ZeNear CUDA
  acceleration and stopped repeating the missing-CuPy fallback once per image
  in CPU-only batches.
- Fixed macOS CI portability failures around deterministic worker caps,
  spawn-based legacy executor shutdown, WCS-writer cancellation safety,
  Darwin thread-sampling telemetry, and deterministic Blind 4D ring sampling.
- Tightened macOS compatibility checks for catalog storage paths, Finder
  opening, spawn-based cancellation, Qt offscreen startup, and CPU-only
  operation when CuPy/CUDA is absent.
- Fixed a false Blind 4D partial-coverage warning shown before a full
  CatalogLibrary had been resolved.
- Made startup wizard CatalogLibrary activation transactional: existing
  libraries, official installs, and local packages now persist product modes
  (`near_catalog_mode=auto`, `blind4d_catalog_mode=auto`) before any settings
  read can validate a stale external Blind 4D manifest.
- Prevented the startup wizard from marking itself complete after a failed
  activation, avoiding contradictory saves from a stale wizard settings object.
- Restored the CatalogLibrary Blind 4D manifest-view CLI used by validation
  tests.

## [1.0.0] - 2026-04-23

### Added

- First Release Candidate Acceptance of ZeSolver (Near + ZeBlind pipeline, GUI/CLI integration).
- Formal semantic versioning baseline with release tag `v1.0.0`.


### Added

- Introduced the standalone `zeblindsolver` module/CLI for ASTAP-based blind solving
  (header sanitation, multi-database fallback, WCS tagging, CLI return codes).
- Wired `zeblindsolver` into `zesolver.py` as an automatic fallback with GUI/CLI
  run-info reporting and configuration flags (`--blind-db`, `--auto-blind-profile`,
  `--no-blind`, etc.).
- Added documentation/tests covering the blind solver workflow and resiliency.
- Added optional RA/Dec/radius and optical hints (focal length, pixel size,
  resolution bounds) to the GUI, CLI, and blind solver config so phases can
  pre-filter manifest tiles and report which hint set succeeded.
- Unified the `downsample` parameter across GUI/CLI and the blind pipeline: the
  factor now rescales the image pyramid, star detector kernel, and quad-vote
  bucket caps automatically.
- Implemented universal raster import (RAW/TIFF/JPG/PNG) for the blind solver,
  including float32 luminance conversion and `.wcs.json` sidecars when the input
  is not a FITS container.
- Converted the blind pipeline into multi-phase passes (hinted, scale-only,
  blind fallback) with early-exit ratios, per-phase logging, and stats surfaced
  through `WcsSolution`.
- Added a Seestar S50 instrument preset so the GUI FOV calculator pre-fills its
  optics fields for that scope/camera combo and immediately refreshes the solver
  hints.
- Shared the persistent settings dataclass/load/save helpers between the CLI
  entry point and the package, so tests can redirect the settings file path
  without touching the GUI stack.
- Added a configurable in-process tile cache for `_load_tile_positions`, exposed via
  `ZE_TILE_CACHE_SIZE` / `--tile-cache-size`, with hit/miss stats logged at DEBUG level.
- Observed quad hashes are now deduplicated per level and `tally_candidates` accepts
  `(hashes, counts)` tuples to weight votes; this drops redundant bucket lookups without
  changing solve order or scores.
- `zebuildindex` gained `--quad-storage {npz,npz_uncompressed,npy}`, `--tile-compression`,
  and `--workers` flags so quad tables can be written as mmap-friendly `.npy` folders or
  uncompressed `.npz` archives. `QuadIndex.load` auto-detects the format and logs load
  timings for observability.
- The GUI “Construire l’index” action mirrors those quad-storage/tile-compression options,
  so `.npy` or uncompressed `.npz` tables can be produced without dropping to the CLI.
- Documented the new builder/solver knobs in `README.md` and AGENTS.md, and added unit
  tests for the tile cache, weighted tallies, and the storage variants.

### Fixed

- The GUI/CLI batch runner now actually invokes the metadata-based near solver
  before falling back to the blind pipeline; the helper previously ignored the
  loaded FITS metadata, so only the manual “Near solve” tester would ever run it.
