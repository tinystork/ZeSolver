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

