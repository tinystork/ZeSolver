# S6B-2 - Simplified Capability And Unresolved Output

## 1. Etat Git initial

Initial checkpoint on branch `test`: clean worktree before S6B-2 changes.
`git diff --check` passed and `git diff` was empty.

## 2. Changements S6B-1 preexistants

S6B-1 is present in the baseline: the standard solver tab no longer exposes the
external Blind 4D manifest field, and the advanced selector offers Auto active
library vs external manifest.

## 3. Architecture actuelle des ressources

Catalog resources are resolved by `zesolver.catalog_resources`.  A
`SolverCatalogResources` reports Near availability, Blind 4D runtime paths,
library status, source, warnings and coverage.

## 4. Modele de capacites ajoute

Added `zesolver.simplified_capability` with `SimplifiedSolveCapability`:
`FULL_LOCAL`, `NEAR_ONLY`, `UNAVAILABLE`.

## 5. Matrice FULL_LOCAL / NEAR_ONLY / UNAVAILABLE

- Library with Near + Blind 4D -> `FULL_LOCAL`.
- Library with Near only -> `NEAR_ONLY`.
- Library with Blind only -> `UNAVAILABLE` in simplified flows.
- ASTAP/legacy Near only -> `NEAR_ONLY`.
- No Near source -> `UNAVAILABLE`.

## 6. Bibliotheque complete

Easy/Wizard keeps Blind enabled for the run and logs the effective local chain
as `ZeNear -> ZeBlind 4D`.

## 7. Bibliotheque Near-only

Easy/Wizard disables Blind only for the effective run.  The persisted setting is
not rewritten.

## 8. ASTAP seul

Legacy ASTAP Near resources map to `NEAR_ONLY`; Blind 4D is unavailable.

## 9. Sans source

The run is blocked before worker creation with `NO_LOCAL_CATALOG_SOURCE_AVAILABLE`.

## 10. Easy/Wizard/Expert

The simplified policy is applied only when `ProductSettings.interface_mode` is
`easy` or `wizard`.  Expert and CLI keep historical explicit overrides.

## 11. Preflight

Unavailable simplified capability returns a structured GUI summary before
creating batch workers.

## 12. Routage Blind effectif

`BatchSolveRequest.blind_enabled` controls whether phase Blind is submitted.
Near-only runs keep unresolved files terminal after Near.

## 13. Classification des echecs

`SolveResult` now carries `terminal_reason_code`.

## 14. Codes de raisons

Implemented codes include `NEAR_UNRESOLVED_BLIND_UNAVAILABLE`,
`ALL_ENABLED_SOLVERS_EXHAUSTED`, `INPUT_UNREADABLE`, `INPUT_MISSING`,
`RUNTIME_ERROR`, `WRITE_ERROR`, `PERMISSION_ERROR`,
`SKIPPED_EXISTING_WCS`, and `CANCELLED`.

## 15. Definition d'un unresolved scientifique

Only `UNSOLVED` results, or legacy failed results with an eligible scientific
reason, are considered terminal unresolved.  Technical errors are excluded.

## 16. Option GUI

Added Easy checkbox: move unresolved images to `unresolved_by_zesolver`.
Default is off.

## 17. Option CLI

Added `--move-unresolved` / `--no-move-unresolved`.  Existing CLI invocations
keep the default: no move.

## 18. Persistance

`move_unresolved_files` is persisted in `PersistentSettings` and exposed in
`ProductSettings`.

## 19. Moment du rangement

Movement is terminal and batch-owned, after Near, Blind and optional web
fallback decisions.

## 20. Completed / Cancelled / Failed

Completed normal runs may move eligible scientific unresolved files.  Cancelled
runs do not move anything.  Technical per-file failures are not eligible.

## 21. Exclusion du scanner

`unresolved_by_zesolver` is ignored at any recursive depth by `_iter_image_files`.

## 22. Arborescence relative

Relative input structure is preserved below `unresolved_by_zesolver`.

## 23. Collisions

Existing destinations are never overwritten; suffixes such as `__2` are used.

## 24. Sidecars

Only `<image>.wcs.json` and `<image>.meta.json` move with the image. Generic
same-stem JSON files stay in place.

## 25. Manifest JSON

The run writes `unresolved_manifest_YYYYMMDD_HHMMSS.json` atomically under the
unresolved directory when there are move records.

## 26. Telemetrie

Batch telemetry now includes `simplified_capability` and `unresolved_output`.
Run telemetry sidecars include these blocks.

## 27. Logs

The run logs the effective local chain at start and a compact unresolved sorting
summary at terminal time.

## 28. Test bibliotheque complete

Covered by `tests/test_s6b2_simplified_capabilities.py`.

## 29. Test ASTAP seul

Covered by `tests/test_s6b2_simplified_capabilities.py`.

## 30. Test aucune source

Covered by `tests/test_s6b2_simplified_capabilities.py`.

## 31. Test Stop

Cancelled batch behavior is covered by `tests/test_s6b2_unresolved_output.py`.

## 32. Test relance

Scanner exclusion is covered so moved files are not picked up by later scans.

## 33. Compatibilite Expert

Expert mode remains explicit; tests verify `interface_mode=expert` is preserved.

## 34. Compatibilite CLI

CLI gains only an opt-in flag; historical options remain unchanged.

## 35. Non-regression scientifique

No solver thresholds, matching policies, WCS tolerances or detection paths were
changed.  Sorting happens only after terminal scientific failure.

## 36. Fichiers modifies

Core models, batch runner, GUI request/settings adapters, telemetry, settings,
`zesolver.py`, new capability/output modules, tests, and this report.

## 37. Barrieres executees

Executed barriers:

```text
tools/check_core_boundaries.py: OK
S6B-1/S6B-2/catalog/batch targeted tests: 40 passed
tools/run_regression_suite.py --hermetic: 708 passed, 1 skipped, 9 deselected
QT_QPA_PLATFORM=offscreen pytest -q: 708 passed, 10 skipped
compileall: OK
git diff --check: OK
```

No FITS, backup, unresolved output directory, unresolved manifest, benchmark
artifact, or run telemetry sidecar is present in the Git status.  The artifact
grep only matches the expected source file `zesolver/resource_telemetry.py`.

## 38. Etat Git final

Final checkpoint:

```text
## test...origin/test
modified source/docs/tests only; no generated unresolved output artifacts
```

## 39. Gate final

```text
S6B2_SIMPLIFIED_CAPABILITY_POLICY_CONFIRMED
S6B2_FULL_LIBRARY_CHAIN_CONFIRMED
S6B2_ASTAP_NEAR_ONLY_FALLBACK_CONFIRMED
S6B2_TERMINAL_UNRESOLVED_CLASSIFICATION_CONFIRMED
S6B2_UNRESOLVED_OUTPUT_CONTRACT_CONFIRMED
S6B2_RESCAN_EXCLUSION_CONFIRMED
READY_FOR_ZEMOSAIC_ZSSS_INTEGRATION_PREPARATION
```
