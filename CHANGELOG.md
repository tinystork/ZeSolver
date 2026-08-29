# Changelog

## [Unreleased]

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
