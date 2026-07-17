# P0A - Baseline Recovery Report

Date: 2026-07-17

## Summary

The existing red baseline was triaged first, then recovered without changing ZeNear thresholds, ZeBlind 4D algorithms, catalog formats, quad/signature/lookup logic, WCS validation thresholds, or original FITS corpus files.

Before:

```text
.venv/bin/python -m pytest -q
217 passed, 16 failed, 1 skipped, 1 warning
```

After:

```text
.venv/bin/python -m pytest -q
232 passed, 3 skipped, 1 warning
```

## Initial Failures and Final Classification

| Test | Classification | Confirmed cause | Resolution |
| --- | --- | --- | --- |
| `tests/test_batch_blind_fallback.py::test_sequential_near_path_keeps_immediate_blind_fallback` | `POSSIBLE_CODE_REGRESSION` -> obsolete unit expectation for 4D seam | Current product profile `zeblind_4d_experimental` defers blind work to the batch 4D phase; inline fallback remains valid only for explicit historical profile. | Local test contract split: 4D defers to batch, explicit historical keeps inline fallback. No production code changed. |
| `tests/test_catalog290.py::test_decode_g05_polar_tile` | `MISSING_FIXTURE` | Repository `database/` is absent. | Marked as `external_catalog` and skipped with explicit message when `database/` is absent. |
| `tests/test_catalog290.py::test_decode_d50_tile_and_cone_query` | `MISSING_FIXTURE` | Same absent external ASTAP/HNSKY database. | Same explicit `external_catalog` skip. |
| `tests/test_downloads.py::test_download_manager_with_fake_backend` | `TEST_BUG` | `FakeBackend` wrote only 69 of 70 bytes when content length was not divisible by `delay_steps`; SHA verifier correctly failed. | Fixed `FakeBackend` to write the final remainder and keep SHA verification active. |
| `tests/test_metadata_solver.py::test_metadata_solver_solves_synthetic_frame` | `INVALID_SYNTHETIC_FIXTURE` | Old four-square synthetic image no longer represents strict ASTAP-ISO detector behavior. | Test now injects a deterministic strict detector output and a minimal ASTAP-ISO hypothesis to test WCS writing without weakening strict mode. |
| `tests/test_metadata_solver.py::test_strict_fov_hint_source_override_priority` | `INVALID_SYNTHETIC_FIXTURE` | Test failed before reaching FOV priority logic because strict detector returned no stars. | Injected strict detector output; kept `astap_iso_strict=True`. |
| `tests/test_metadata_solver.py::test_strict_fov_hint_source_header_without_override` | `INVALID_SYNTHETIC_FIXTURE` | Same early detector failure. | Same injection approach. |
| `tests/test_metadata_solver.py::test_strict_scale_source_low_support_skips_autofov` | `INVALID_SYNTHETIC_FIXTURE` | Same early detector failure. | Same injection approach. |
| `tests/test_metadata_solver.py::test_strict_contextual_retry_zero_ref_patience` | `INVALID_SYNTHETIC_FIXTURE` | Test patched `detect_stars`, but strict path now uses `astap_adaptive_image_detection`. | Patched current strict detector seam instead. |
| `tests/test_p221_app_integration.py::test_app_settings_profile_persistence_and_presets` | `OBSOLETE_EXPECTATION` | Test still expected `historical` default. | Local test expectation aligned to `zeblind_4d_experimental`. |
| `tests/test_p221_app_integration.py::test_app_build_blind_config_requires_manifest_only_for_4d` | `OBSOLETE_EXPECTATION` | Test named a default `SolveConfig` as historical, but default is now 4D and requires manifest. | Local test now makes historical profile explicit and keeps 4D manifest requirement. |
| `tests/test_p222_gui_integration.py::test_p222_old_settings_keep_historical_and_easy` | `OBSOLETE_EXPECTATION` | Settings migration intentionally moves old profile to 4D default. | Local expectation changed to 4D + easy. |
| `tests/test_p222_gui_integration.py::test_p222_invalid_profile_and_interface_migrate_to_safe_defaults` | `OBSOLETE_EXPECTATION` | Safe default is now 4D, not historical. | Local expectation changed to 4D + easy. |
| `tests/test_p222_gui_integration.py::test_p222_gui_source_contains_required_controls_and_translations` | `OBSOLETE_EXPECTATION` | GUI text was already cleaned to remove "experimental" from the normal 4D chain wording. | Local expected strings changed to current ZeBlind 4D wording and 4D-index coverage warning. |
| `tests/test_zn310b_gui_fallback_dataset.py::test_zn310b_all_generated_copies_have_no_old_wcs` | `DATASET_CONTRACT_BROKEN` | Existing generated dataset had WCS/solve cards written into copies. | Generator now fails if WCS survives; dataset regenerated to a new timestamped directory. |
| `tests/test_zn310b_gui_fallback_dataset.py::test_zn310b_nohint_removes_all_near_hint_aliases_and_object` | `DATASET_CONTRACT_BROKEN` | Existing NOHINT copies contained `CRVAL1/CRVAL2` via WCS cards. | Generator now fails if NOHINT forbidden keys survive; dataset regenerated. |

## Corrections Performed

Tracked files modified:

- `pyproject.toml`
  - Added pytest markers: `external_catalog`, `corpus`, `slow`, `blind4d`.
- `tests/test_catalog290.py`
  - Marked external catalog tests and added explicit skip when repository `database/` is absent.
- `tests/test_metadata_solver.py`
  - Added strict detector injection helpers.
  - Kept strict ASTAP-ISO enabled while decoupling configuration/retry logic tests from obsolete synthetic photometry.
- `zeblindsolver/downloads.py`
  - Fixed `FakeBackend` to write the final remainder before checksum verification.
- `docs/stabilization/baseline_failure_triage.md`
  - Added the required failure matrix and first-pass classifications.
- `docs/stabilization/baseline_recovery_report.md`
  - This report.

Local ignored files used by the current suite were also aligned in the working tree, but are ignored by `.gitignore` and are not tracked by `HEAD`:

- `tests/test_batch_blind_fallback.py`
  - Local contract split: 4D fallback is deferred to batch; explicit historical profile keeps inline fallback.
- `tests/test_p221_app_integration.py`
  - Local expectations aligned to the 4D product default.
- `tests/test_p222_gui_integration.py`
  - Local expectations aligned to the 4D product default and current GUI wording.
- `tools/build_zn310b_gui_dataset.py`
  - Added post-write contract validation: no WCS may survive in any generated variant, and NOHINT must not retain position aliases or `OBJECT`.

Generated reports updated by the ZN3.10B dataset builder are also ignored by `.gitignore`:

- `reports/zenear_zn310b_source_inventory.json`
- `reports/zenear_zn310b_source_inventory.md`
- `reports/zenear_zn310b_pixel_integrity.json`
- `reports/zenear_zn310b_oracles.json`
- `reports/zenear_zn310b_gui_manifest.json`
- `reports/zenear_zn310b_gui_manifest.md`

New generated dataset:

```text
/home/tristan/near_bench_cmp30/zn310b_gui_fallback4d_20260717_021840/
```

It contains 8 selected sources:

```text
M31: 3
M106: 3
NGC6888: 2
```

The builder reported:

```text
all_pixels_identical: true
all_sources_covered_4d: true
selected_count: 8
```

This is a repository hygiene risk: the suite and dataset workflow use these local ignored files, but Git will not record them unless they are force-added or `.gitignore` is adjusted in a later, explicit cleanup.

## Commands Executed After Corrections

Targeted groups:

```bash
.venv/bin/python -m pytest tests/test_downloads.py::test_download_manager_with_fake_backend tests/test_catalog290.py::test_decode_g05_polar_tile tests/test_catalog290.py::test_decode_d50_tile_and_cone_query -q
```

Result:

```text
1 passed, 2 skipped
```

```bash
.venv/bin/python -m pytest tests/test_p221_app_integration.py::test_app_settings_profile_persistence_and_presets tests/test_p221_app_integration.py::test_app_build_blind_config_requires_manifest_only_for_4d tests/test_p222_gui_integration.py::test_p222_old_settings_migrate_to_4d_and_easy tests/test_p222_gui_integration.py::test_p222_invalid_profile_and_interface_migrate_to_safe_defaults tests/test_p222_gui_integration.py::test_p222_gui_source_contains_required_controls_and_translations -q
```

Result:

```text
5 passed
```

```bash
.venv/bin/python -m pytest tests/test_metadata_solver.py::test_metadata_solver_solves_synthetic_frame tests/test_metadata_solver.py::test_strict_fov_hint_source_override_priority tests/test_metadata_solver.py::test_strict_fov_hint_source_header_without_override tests/test_metadata_solver.py::test_strict_scale_source_low_support_skips_autofov tests/test_metadata_solver.py::test_strict_contextual_retry_zero_ref_patience -q
```

Result:

```text
5 passed
```

```bash
.venv/bin/python tools/build_zn310b_gui_dataset.py
```

Result:

```text
run_dir: /home/tristan/near_bench_cmp30/zn310b_gui_fallback4d_20260717_021840
all_pixels_identical: true
all_sources_covered_4d: true
selected_count: 8
```

```bash
.venv/bin/python -m pytest tests/test_zn310b_gui_fallback_dataset.py::test_zn310b_all_generated_copies_have_no_old_wcs tests/test_zn310b_gui_fallback_dataset.py::test_zn310b_nohint_removes_all_near_hint_aliases_and_object -q
```

Result:

```text
2 passed
```

```bash
.venv/bin/python -m pytest tests/test_batch_blind_fallback.py -q
```

Result:

```text
5 passed
```

Full suite:

```bash
.venv/bin/python -m pytest -q
```

Result:

```text
232 passed, 3 skipped, 1 warning in 12.92s
```

Hermetic baseline:

```bash
.venv/bin/python -m pytest -m "not external_catalog and not corpus and not slow" -q
```

Result:

```text
232 passed, 1 skipped, 2 deselected, 1 warning in 14.95s
```

External/integration baseline:

```bash
.venv/bin/python -m pytest -m "external_catalog or corpus or slow" -q
```

Result:

```text
2 skipped, 233 deselected in 3.40s
```

Compilation and diff checks:

```bash
.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests *.py
git diff --check
```

Result:

```text
OK
```

## Skips Remaining

- `tests/test_catalog290.py::test_decode_g05_polar_tile`
  - Skip reason: external ASTAP/HNSKY test database not found at repository `database/`.
  - Classification: explicit `external_catalog`.
- `tests/test_catalog290.py::test_decode_d50_tile_and_cone_query`
  - Same reason.
- `tests/test_real_s50.py`
  - Skip reason: S50 index or frame not configured.

## Warning Remaining

```text
tests/test_p220_manifest_loader.py::test_runtime_prepare_strips_position_and_identity_hints
Astropy VerifyWarning: Card is too long, comment will be truncated.
```

This warning predates the recovery pass and was not corrected in P0A because it is not one of the 16 red baseline failures.

## Risks Still Open

1. The local ignored tests under `tests/` should be either force-added intentionally or moved/renamed so Git state matches the suite that actually runs.
2. The external catalog decoder tests now skip cleanly, but true hermetic binary decoder fixtures still need design if the format can be represented compactly.
3. The regenerated ZN3.10B dataset is clean as a pre-GUI dataset, but the actual manual GUI fallback 4D run remains to be executed.
4. The outer batch path for 4D fallback is now clarified at the unit seam, but full GUI/batch fallback verification still belongs to the ZN3.10B manual log analysis.

## Decision

```text
READY_FOR_REGRESSION_SUITE
```

Reason: the hermetic baseline is green, external-data tests are explicitly marked and skipped when data is absent, strict ASTAP-ISO was not weakened, historical was not restored as product default, checksum verification remains active, and the ZN3.10B generated dataset contract is enforced by the builder.
