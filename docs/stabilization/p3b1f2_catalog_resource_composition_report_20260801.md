# P3B-1F2 - Catalog resource composition

## Context

Windows validation found that relaunching the startup wizard, selecting a valid
ASTAP root, and finishing the wizard removed the persisted ZeSolver Catalog
Library path. A stale legacy Blind 4D manifest then stayed in settings and could
block the run before ZeNear, even though the ASTAP root itself was valid.

## Root Cause

The startup wizard ASTAP handler called `_clear_catalog_library_selection()`.
That made `catalog_library_path` and `db_root` mutually exclusive in a flow where
they must be independent.

The runtime catalog resolver also treated legacy ASTAP and legacy Blind 4D
manifest input as one legacy bundle. In auto mode, an invalid legacy Blind 4D
manifest could therefore prevent a valid ASTAP Near resource from being used.

## Changes

- ASTAP selection from the startup wizard no longer clears
  `catalog_library_path`.
- `catalog_library_path` and `db_root` are preserved independently.
- `resolve_catalog_resources()` now supports `prefer_legacy_near=True` so an
  explicitly selected ASTAP root can provide ZeNear while a valid CatalogLibrary
  continues to provide ZeBlind 4D.
- `resolve_catalog_resources()` now supports
  `strict_legacy_blind4d_manifest=False`; in auto mode, invalid legacy Blind 4D
  manifests become structured warnings and Blind is unavailable instead of
  blocking ZeNear.
- Strict legacy Blind manifest errors are retained for explicit
  `external-manifest` and Blind-only paths.
- Runtime Blind 4D resolution no longer reuses a stale external manifest path in
  auto mode after the catalog resolver has ignored it.

## Expected Runtime Matrix

- CatalogLibrary only: ZeNear from the library, ZeBlind 4D from the library.
- ASTAP only: ZeNear only.
- CatalogLibrary + ASTAP selected: ZeNear from ASTAP, ZeBlind 4D from the
  CatalogLibrary.
- ASTAP + stale legacy manifest in auto mode: warning, Blind disabled, ZeNear
  remains available.
- Explicit external manifest invalid: blocking error.
- Blind-only invalid manifest: blocking error.
- No Near or Blind resource: preflight remains blocking.

## Persistence Rules

- The wizard does not erase resources it is not replacing.
- ASTAP selection persists only `db_root`, `near_catalog_mode=astap-native`, and
  `blind4d_catalog_mode=auto`.
- `catalog_library_path` remains unchanged when selecting ASTAP.
- Cancelling the wizard still leaves existing settings untouched.
- Stale legacy Blind manifest paths are ignored in auto/library mode and remain
  strict only when explicitly requested.

## Tests

Targeted tests added:

- `tests/test_p3b1f2_catalog_resource_composition.py`
- additional regressions in `tests/test_startup_wizard.py`

Executed:

- `tests/test_p3b1f2_catalog_resource_composition.py`: `7 passed`
- `tests/test_startup_wizard.py`: `27 passed`
- catalog resource/runtime lot:
  `tests/test_catalog_resource_resolution.py`,
  `tests/test_blind4d_runtime_source_policy.py`,
  `tests/test_catalog_library_near_integration.py`,
  `tests/test_solver_pipeline_near_provider.py`,
  `tests/test_s6b3_astap_only_routing.py`: `17 passed`
- `python -m compileall` on modified runtime/test files: OK
- `git diff --check`: OK

## Windows Validation

Manual validation still required on the Windows machine that reproduced the
issue:

1. Relaunch the startup wizard from the menu.
2. Select the valid ASTAP root.
3. Finish the wizard.
4. Confirm `catalog_library_path` is still present.
5. Confirm `db_root` is the selected ASTAP root.
6. Start a solve and confirm ZeNear is attempted.
7. Confirm stale legacy Blind 4D manifests no longer block auto mode.
8. Return to the existing ZeSolver CatalogLibrary without downloading.

## Limits

No ZeNear, ZeBlind, scientific thresholds, catalog formats, or solver algorithms
were changed. The fix is limited to GUI persistence and catalog resource
composition/preflight policy.
