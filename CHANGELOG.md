# Changelog

## [Unreleased]

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

### Fixed

- The GUI/CLI batch runner now actually invokes the metadata-based near solver
  before falling back to the blind pipeline; the helper previously ignored the
  loaded FITS metadata, so only the manual “Near solve” tester would ever run it.

