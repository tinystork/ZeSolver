# RC-1A - Blind 4D Coverage Warning Alignment

Date: 2026-08-01
Branch: `test`

## 1. Initial Git State

Initial checks:

```text
git status --short
  clean
git branch -vv
  * test 57caf7f [origin/test] agent mis à jour
git rev-parse HEAD
  57caf7fc41e99033e599e29810a5eb5e96be097d
git rev-parse origin/test
  57caf7fc41e99033e599e29810a5eb5e96be097d
git diff --check
  clean
```

The local branch matched `origin/test` at the start of the mission.

## 2. Reproduction

The false warning came from the GUI solve request path before catalogue
resources were prepared:

```text
catalog_resources=None
-> blind4d_all_sky=bool(getattr(None, "all_sky_blind4d", False))
-> blind4d_all_sky=False
-> Engine selection warning blind4d_coverage_partial_not_all_sky
```

This made an unknown coverage state look like a known partial coverage state.

## 3. Root Cause

`EngineSelectionRequest.blind4d_all_sky` used `False` as its default. The GUI
adapter also coerced missing `SolverCatalogResources` to `False`. The engine
selector then emitted the partial-coverage warning from that default value.

The coverage was actually resolved later during GUI pipeline preparation, where
the real library reports:

```text
source=library
status=READY_FULL
blind4d_index_count=47
coverage=1476/1476
blind4d_all_sky=True
warnings=-
```

## 4. Chronology

Corrected chronology:

1. Engine selection receives `blind4d_all_sky=None` when resources are unknown.
2. Engine selection keeps AUTO -> PIPELINE unchanged and emits no catalogue
   coverage warning.
3. GUI pipeline preparation resolves `SolverCatalogResources`.
4. Catalogue preflight uses the resolved resources as the source of truth.
5. Runtime Blind 4D telemetry reports the effective source and coverage.

## 5. Coverage Semantics

The transported coverage state is now tri-state:

```text
True  = full coverage known
False = partial coverage known
None  = coverage unknown or not resolved yet
```

The engine selector no longer emits the coverage warning. The warning is emitted
only by resolved catalogue resources when Blind 4D coverage is positively known
as partial:

```text
blind4d_coverage_partial_not_all_sky
```

## 6. Architecture

Changes:

- `EngineSelectionRequest.blind4d_all_sky` is now `bool | None = None`.
- `GuiSettingsState` and `GuiSolveRequest` preserve the same tri-state value.
- `build_gui_solve_request_from_legacy_config()` returns `None` when resources
  are absent instead of converting absence to `False`.
- Engine selection does not generate catalogue coverage warnings.
- `resolve_catalog_resources()` emits the partial warning only after real
  resources are resolved.
- GUI pipeline preparation logs one coverage line after shared catalogue
  resources are available.
- Strict external 4D manifests can carry optional top-level coverage metadata;
  without it, a manifest listing only some tiles remains partial.

## 7. Files Modified

- `zesolver/engine_selection.py`
- `zesolver/gui_pipeline/requests.py`
- `zesolver/gui_pipeline/settings_adapter.py`
- `zesolver/gui_pipeline/pipeline_runner.py`
- `zesolver/catalog_resources.py`
- `zeblindsolver/index_manifest_4d.py`
- `tests/test_engine_selection.py`
- `tests/test_gui_settings_adapter.py`
- `tests/test_solver_pipeline_preflight.py`
- `tests/test_solver_pipeline_routing.py`
- `tests/test_catalog_library_blind4d_integration.py`
- `tests/test_catalog_library_blind4d_product_switch.py`
- `tests/test_catalog_library_pipeline_integration.py`
- `tests/test_solver_pipeline_result_adapter.py`
- `tests/test_s5e_settings_catalog_persistence.py`
- `tests/solver_pipeline_fixtures.py`
- `CHANGELOG.md`

## 8. Tests Added Or Updated

Covered cases:

- unknown coverage at selection time does not warn;
- READY_FULL resources do not warn;
- partial library resources warn once;
- Blind disabled does not warn from selection;
- Near-only/no Blind resources are not represented as partial by selection;
- strict external manifest without full coverage warns partial;
- strict external manifest with full coverage does not warn;
- pipeline result and telemetry deduplicate the partial warning.

## 9. Validation Results

Targeted tests:

```text
.venv/bin/python -m pytest -q \
  tests/test_engine_selection.py \
  tests/test_gui_settings_adapter.py \
  tests/test_gui_pipeline_runner.py \
  tests/test_solver_pipeline_preflight.py \
  tests/test_solver_pipeline_routing.py \
  tests/test_catalog_library_blind4d_integration.py \
  tests/test_catalog_library_pipeline_integration.py \
  tests/test_solver_pipeline_result_adapter.py \
  tests/test_catalog_library_blind4d_product_switch.py \
  tests/test_s5e_settings_catalog_persistence.py

56 passed
```

The same target list run with system `python` failed before executing the
mission code because that interpreter lacks `astroalign`. The project `.venv`
was used for validation.

Compileall:

```text
.venv/bin/python -m compileall zesolver zeblindsolver tools
OK
```

Expanded targeted lot:

```text
.venv/bin/python -m pytest -q \
  tests/test_engine_selection.py \
  tests/test_gui_settings_adapter.py \
  tests/test_configuration_assembly.py \
  tests/test_gui_pipeline_runner.py \
  tests/test_solver_pipeline_preflight.py \
  tests/test_solver_pipeline_routing.py \
  tests/test_catalog_library_status.py \
  tests/test_catalog_library_blind4d_integration.py \
  tests/test_catalog_library_pipeline_integration.py \
  tests/test_solver_pipeline_result_adapter.py \
  tests/test_catalog_library_blind4d_product_switch.py \
  tests/test_s5e_settings_catalog_persistence.py

68 passed
```

Raw global suite:

```text
.venv/bin/python -m pytest -q
806 passed, 36 skipped, 1 failed, 17 warnings
```

The single failure was:

```text
tests/test_zn310b_gui_fallback_dataset.py::test_zn310b_all_generated_copies_have_no_old_wcs
/home/tristan/near_bench_cmp30/zn310b_gui_fallback4d_20260717_021840/wrong_hints/ZN310B_BADHINT_003.fit
```

The same test was replayed in a detached worktree at `origin/test`
(`57caf7f`) and failed identically, demonstrating a pre-existing external
fixture state rather than an RC-1A regression.

Global suite with only that pre-existing fixture test deselected:

```text
.venv/bin/python -m pytest -q \
  --deselect tests/test_zn310b_gui_fallback_dataset.py::test_zn310b_all_generated_copies_have_no_old_wcs

806 passed, 36 skipped, 1 deselected, 17 warnings
```

## 10. Linux READY_FULL Validation

Library:

```text
/home/tristan/ZeSolverCatalog/new
```

Lightweight verification:

```text
unknown_request_blind4d_all_sky=None
engine_selected=pipeline
engine_warnings=-
catalog_source=library
catalog_status=READY_FULL
blind4d_index_count=47
coverage=1476/1476
blind4d_all_sky=True
warnings=-
```

GUI pipeline runner on a temporary FITS copy:

```text
Engine warnings=-
Blind4D coverage: source=library-view status=full indexes=47 covered=1476 total=1476 all_sky=true warning=-
result_status=SOLVED
result_backend=NEAR
summary_warnings=-
```

Blind-only runtime validation on a temporary FITS copy:

```text
status=SOLVED
backend=BLIND4D
warnings=-
blind4d_catalog_mode_effective=library-view
blind4d_index_count=47
blind4d_covered_tiles=1476
blind4d_total_tiles=1476
blind4d_all_sky=True
blind4d_external_fallback_used=False
```

## 11. Partial Negative Validation

Automated fixtures validate that a strict manifest or library with real partial
coverage still emits:

```text
blind4d_coverage_partial_not_all_sky
```

The warning is deduplicated in final pipeline result warnings and telemetry.

## 12. Windows Validation

No Windows runtime is available from this Linux workspace. The change is
path-syntax independent and only changes typed state propagation, resource
warnings, and Qt-independent logging. Windows should repeat the same READY_FULL
check with the installed complete CatalogLibrary before Release Candidate
acceptance.

## 13. Residual Risks

- External strict manifests without explicit full coverage metadata are treated
  as partial, which is conservative.
- Dynamic GUI visual validation on Windows remains pending outside this Linux
  workspace.

## 14. Final Commit

Final commit: this report is included in the final RC-1A commit. Read the
exact immutable hash with `git rev-parse HEAD` after the commit.

Verdict:

```text
RC1A_BLIND4D_COVERAGE_WARNING_ALIGNED
READY_FOR_RELEASE_CANDIDATE_ACCEPTANCE
```
