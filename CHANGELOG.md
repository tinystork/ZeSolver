# Changelog

## [Unreleased]

### Added

- Introduced the standalone `zeblindsolver` module/CLI for ASTAP-based blind solving
  (header sanitation, multi-database fallback, WCS tagging, CLI return codes).
- Wired `zeblindsolver` into `zesolver.py` as an automatic fallback with GUI/CLI
  run-info reporting and configuration flags (`--blind-db`, `--auto-blind-profile`,
  `--no-blind`, etc.).
- Added documentation/tests covering the blind solver workflow and resiliency.

