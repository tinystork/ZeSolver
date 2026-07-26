# S6B-1 - Blind 4D External Manifest GUI Simplification

Date: 2026-07-26

## 1. Etat Git initial

Initial inspection on branch `test` after S6A-2 final commit:

```text
## test...origin/test
git diff --check: OK
git diff: empty
```

No commit and no push were made during this mission.

## 2. Contrat runtime Blind 4D actuel

`resolve_blind4d_runtime()` supports three internal modes:

```text
auto
library-view
external-manifest
```

With a valid active `CatalogLibrary`, `auto` resolves to the library-owned
Blind 4D view and does not let a stale external manifest path override it.
`external-manifest` remains an explicit override and requires a valid manifest.

## 3. Modes internes identifies

The internal mode names were preserved.  The GUI no longer exposes
`library-view` as a separate user choice because it is the effective runtime
source selected by Auto when a library is active.

## 4. Emplacement GUI initial

Before S6B-1, the Solver tab expert block directly exposed:

```text
External Blind 4D manifest
[path] [Browse] [Verify]
Not verified
```

The Settings compatibility group also exposed the legacy source mode and path.

## 5. Nouveaux emplacements

The standard Solver tab now keeps only the Blind backend profile selector.  The
external manifest controls live in the advanced compatibility/diagnostic group.

## 6. Elements retires du GUI standard

Removed from the standard Solver tab:

```text
External Blind 4D manifest label
path field
Browse button
Verify button
verification status
```

## 7. Selecteur avance ajoute

The advanced group now presents:

```text
Blind 4D index source
Auto - active library
External manifest
```

French labels are also present:

```text
Source des index Blind 4D
Auto - bibliotheque active
Manifeste externe
```

## 8. Mapping GUI vers reglages internes

```text
Auto - active library -> blind4d_catalog_mode = auto
External manifest     -> blind4d_catalog_mode = external-manifest
```

Historical persisted `library-view` profiles are represented as the GUI Auto
choice, without creating a fourth mode.

## 9. Comportement Auto

Auto uses the active ZeSolver library.  A saved external manifest path may
remain stored, but it is not consumed while `blind4d_catalog_mode=auto`.

## 10. Comportement external-manifest

External mode shows the path, Browse, Verify and status widgets immediately.
The manifest path is validated and transmitted only in this explicit mode.

## 11. Visibilite conditionnelle

Visibility is centralized in `_update_blind4d_source_visibility()`.  It controls
the manifest label, row, edit field, Browse button, Verify button and status
label together.

## 12. Persistance du chemin

The last external manifest path remains persisted even after returning to Auto.
Returning to External later restores the same path for diagnostic use.

## 13. Migration des profils historiques

Existing `library-view` settings map to the GUI Auto choice.  Existing
`external-manifest` settings open the advanced selector in External mode and
show the saved path.

## 14. Verification du manifeste

The existing strict manifest verification remains wired to the advanced Verify
button.  The status is reset when the path changes or when External mode is
selected.

## 15. Comportement d'un chemin invalide

Auto ignores stale external paths when a library is active.  External mode keeps
the existing blocking validation behavior for invalid or absent manifests.

## 16. Source reellement utilisee en Auto

Runtime tests confirm:

```text
requested=auto
used=library-view
external_fallback_used=false
```

with the library index selected even when a stale external path is supplied.

## 17. Source reellement utilisee en Externe

Runtime tests confirm:

```text
requested=external-manifest
used=external-manifest
manifest=<explicit path>
```

## 18. Compatibilite CLI/headless

CLI/headless support for `--blind4d-catalog-mode`, `library-view` and
`external-manifest` was not removed.  The GUI simplification does not alter the
public CLI mode set.

## 19. Traductions FR/EN

Added translated labels for:

```text
Source des index Blind 4D / Blind 4D index source
Auto - bibliotheque active / Auto - active library
Manifeste externe / External manifest
```

## 20. Tests GUI

Added `tests/test_s6b1_blind4d_source_gui.py`.

It verifies that the standard Solver tab no longer owns the external manifest
field/buttons/status, that the advanced selector has the two intended choices,
that controls hide in Auto, appear in External, update immediately, reset status
on path changes, preserve the path, and translate FR/EN.

## 21. Tests settings

Added `tests/test_s6b1_blind4d_source_settings.py`.

It verifies Auto with a stale external path uses the active library and External
mode remains an explicit manifest override.

## 22. Tests runtime

Existing runtime policy tests and the new S6B-1 settings tests cover:

```text
Auto + active library
Auto + stale external path
External + explicit manifest
historical library-view compatibility
```

## 23. Test manuel

A controlled GUI offscreen test opened the real Qt window, checked the standard
Solver tab, opened the advanced compatibility group, switched Auto/External,
checked immediate visibility, path persistence and language changes.

No separate astrophotography corpus run was required for this presentation and
settings-routing change.

## 24. Non-regression Blind

No Blind scientific thresholds, matching policy, family selection, WCS writing
or 4D algorithm code was changed.  Product-switch tests verify the selected
runtime source and index paths remain unchanged for the same selected source.

## 25. Fichiers modifies

```text
zesolver.py
zesolver/core/blind_port.py
docs/architecture/advanced_catalog_compatibility_surface.md
docs/architecture/gui_catalog_library_selection.md
tests/test_s6b1_blind4d_source_gui.py
tests/test_s6b1_blind4d_source_settings.py
docs/stabilization/s6b1_blind4d_external_manifest_gui_report_20260726.md
```

## 26. Barrieres executees

```text
.venv/bin/python tools/check_core_boundaries.py
-> OK

.venv/bin/python -m pytest -q \
  tests/test_product_settings.py \
  tests/test_configuration_assembly.py \
  tests/test_catalog_library_blind4d_product_switch.py \
  tests/test_catalog_resource_resolution.py \
  tests/test_catalog_blind4d_manifest_view.py \
  tests/test_blind4d_runtime_source_policy.py \
  tests/test_gui_catalog_library_solve_config.py \
  tests/test_gui_catalog_rollback_visibility.py \
  tests/test_gui_development_surface_reorganized.py \
  tests/test_gui_catalog_resource_type_validation.py \
  tests/test_gui_catalog_path_confusion.py \
  tests/test_gui_catalog_compatibility_surface.py \
  tests/test_s6b1_blind4d_source_gui.py \
  tests/test_s6b1_blind4d_source_settings.py \
  tests/test_batch_pipeline_scheduling.py
-> 58 passed

.venv/bin/python tools/run_regression_suite.py --hermetic
-> PASS, 689 passed, 1 skipped, 9 deselected

QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
-> 689 passed, 10 skipped

.venv/bin/python -m compileall -q \
  zeblindsolver zewcs290 zesolver tools tests \
  zesolver.py zewcscleaner.py zeindexcheck.py
-> OK

git diff --check
-> OK
```

## 27. Etat Git final

Final working tree state:

```text
## test...origin/test
 M docs/architecture/advanced_catalog_compatibility_surface.md
 M docs/architecture/gui_catalog_library_selection.md
 M zesolver.py
 M zesolver/core/blind_port.py
?? docs/stabilization/s6b1_blind4d_external_manifest_gui_report_20260726.md
?? tests/test_s6b1_blind4d_source_gui.py
?? tests/test_s6b1_blind4d_source_settings.py
```

No FITS, backup, benchmark or telemetry artifact is present in the S6B-1 diff.

## 28. Gate final

```text
S6B1_STANDARD_GUI_SIMPLIFIED
S6B1_EXTERNAL_MANIFEST_MOVED_TO_ADVANCED
S6B1_ACTIVE_LIBRARY_DEFAULT_CONFIRMED
S6B1_EXTERNAL_OVERRIDE_PRESERVED
S6B1_BACKWARD_COMPATIBILITY_CONFIRMED
READY_FOR_NEXT_GUI_SIMPLIFICATION
```
