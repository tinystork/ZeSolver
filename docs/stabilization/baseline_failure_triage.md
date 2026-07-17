# P0A - Baseline Failure Triage

Date: 2026-07-17

Scope: first-pass classification of the 16 failures reported by `docs/stabilization/initial_state.md`.

No production code, tests, solver thresholds, FITS corpus, catalog, index, GUI behavior, or generated dataset was modified before this matrix was completed.

## References Read

- `AGENT.md`, full file
- `docs/stabilization/initial_state.md`

## Initial State

- Branch: `test`
- Commit referenced by initial state: `f5a76cd2ec8998d552332a3556325ebba5c116cf`
- Current worktree before this triage pass: `docs/stabilization/initial_state.md` already untracked from the previous initial-state step.
- Test command from initial state: `.venv/bin/python -m pytest -q`
- Initial suite result: `217 passed, 16 failed, 1 skipped, 1 warning`
- Compile command from initial state: `.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests *.py`
- Compile result from initial state: OK

## Isolated Reproduction

Each failing test was reproduced individually with `-vv`. Logs were captured outside the repository under:

```text
/tmp/zesolver_p0a_triage_logs/
```

Commands executed:

```bash
.venv/bin/python -m pytest tests/test_batch_blind_fallback.py::test_sequential_near_path_keeps_immediate_blind_fallback -vv
.venv/bin/python -m pytest tests/test_catalog290.py::test_decode_g05_polar_tile -vv
.venv/bin/python -m pytest tests/test_catalog290.py::test_decode_d50_tile_and_cone_query -vv
.venv/bin/python -m pytest tests/test_downloads.py::test_download_manager_with_fake_backend -vv
.venv/bin/python -m pytest tests/test_metadata_solver.py::test_metadata_solver_solves_synthetic_frame -vv
.venv/bin/python -m pytest tests/test_metadata_solver.py::test_strict_fov_hint_source_override_priority -vv
.venv/bin/python -m pytest tests/test_metadata_solver.py::test_strict_fov_hint_source_header_without_override -vv
.venv/bin/python -m pytest tests/test_metadata_solver.py::test_strict_scale_source_low_support_skips_autofov -vv
.venv/bin/python -m pytest tests/test_metadata_solver.py::test_strict_contextual_retry_zero_ref_patience -vv
.venv/bin/python -m pytest tests/test_p221_app_integration.py::test_app_settings_profile_persistence_and_presets -vv
.venv/bin/python -m pytest tests/test_p221_app_integration.py::test_app_build_blind_config_requires_manifest_only_for_4d -vv
.venv/bin/python -m pytest tests/test_p222_gui_integration.py::test_p222_old_settings_keep_historical_and_easy -vv
.venv/bin/python -m pytest tests/test_p222_gui_integration.py::test_p222_invalid_profile_and_interface_migrate_to_safe_defaults -vv
.venv/bin/python -m pytest tests/test_p222_gui_integration.py::test_p222_gui_source_contains_required_controls_and_translations -vv
.venv/bin/python -m pytest tests/test_zn310b_gui_fallback_dataset.py::test_zn310b_all_generated_copies_have_no_old_wcs -vv
.venv/bin/python -m pytest tests/test_zn310b_gui_fallback_dataset.py::test_zn310b_nohint_removes_all_near_hint_aliases_and_object -vv
```

All 16 tests failed individually with exit code `1`. This rules out suite-order dependence for the first triage pass.

## Classification Matrix

Allowed classifications:

```text
ENVIRONMENT
MISSING_FIXTURE
OBSOLETE_EXPECTATION
INVALID_SYNTHETIC_FIXTURE
DATASET_CONTRACT_BROKEN
POSSIBLE_CODE_REGRESSION
TEST_BUG
UNRESOLVED
```

| Test | Symptom exact | Expected contract | Observed behavior | Classification | Probable cause | Proposed action | Product risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `tests/test_batch_blind_fallback.py::test_sequential_near_path_keeps_immediate_blind_fallback` | `assert [False, False] == [False, False, True]` | Sequential Near failure with `allow_blind_fallback=True` should make two Near attempts with `fallback_to_blind=False`, then one immediate blind fallback call through `near_solve(... fallback_to_blind=True)`. | `_run_index_near_solver()` makes two calls with `fallback_to_blind=False`, then defers blind fallback when `blind_backend_profile == zeblind_4d_experimental`. No third call occurs in this unit seam. | `POSSIBLE_CODE_REGRESSION` | The test predates the 4D product path and assumes inline/historical fallback. Current code may intentionally defer 4D fallback to the batch blind phase, but the end-to-end guarantee still needs proof across sequential/parallel and blind enabled/disabled modes. | Do not change yet. Add/adjust targeted tests only after mapping current sequence: Near success, Near fail + Blind success, Near fail + Blind fail, blind disabled, deferred fallback, immediate fallback, sequential, parallel. | High: a real bug here could drop failed Near frames before ZeBlind 4D. |
| `tests/test_catalog290.py::test_decode_g05_polar_tile` | `FileNotFoundError: /home/tristan/.openclaw/workspace/projects/ZeSolver/database` | Unit decoder test should load a G05 tile and verify RA/Dec bounds. | Test hardcodes repository `database/`, which is absent in this checkout. | `MISSING_FIXTURE` | Test depends on a user-installed ASTAP database outside versioned test fixtures. | Make decoder unit tests hermetic with minimal fixtures under `tests/data/` if feasible. Otherwise mark real database tests as `external_catalog` with explicit path/env skip. | Medium: non-hermetic catalog tests hide real decode regressions in environments without local databases. |
| `tests/test_catalog290.py::test_decode_d50_tile_and_cone_query` | `FileNotFoundError: /home/tristan/.openclaw/workspace/projects/ZeSolver/database` | Unit decoder test should load a D50 tile, verify `bp_rp`, and run a cone query. | Same absent hardcoded `database/` path. | `MISSING_FIXTURE` | Same as G05: test is integration-like but unmarked. | Same as G05. Prefer minimal deterministic D50 fixture or explicit `external_catalog` marker. | Medium: D50 is core to Near and 4D derivation, so decode coverage must remain, but not through implicit local paths. |
| `tests/test_downloads.py::test_download_manager_with_fake_backend` | Item status is `failed`, error `SHA256 mismatch`; expected `done`/`verified`. `bytes_done=69`, `bytes_total=70`. | Fake backend should write exactly `content = b"abc123\n" * 10`, then SHA verification should pass. | Independent emulation shows `len(content)=70`, default `delay_steps=3`, `step=23`, loop writes only 69 bytes and drops final `b"\n"`. Written SHA is `835a77...`, expected SHA is `7bcd42...`. | `TEST_BUG` | `FakeBackend.fetch()` test helper truncates the remainder when content length is not divisible by `_steps`. The SHA verifier is correctly catching this. | Fix `FakeBackend` to write the full payload, including the final remainder, and keep SHA verification active. | Low/medium: production downloader verification is doing the right thing; test helper bug currently makes the suite red. |
| `tests/test_metadata_solver.py::test_metadata_solver_solves_synthetic_frame` | `AssertionError: no stars detected in the frame` | Synthetic four-star FITS should solve end-to-end through strict Near and write WCS. | Strict ASTAP-ISO path starts, selects one tile, loads 4 catalog stars, then returns no image stars detected. | `INVALID_SYNTHETIC_FIXTURE` | The old synthetic image is a 200x200 float frame with four 3x3 square plateaus at 5000 ADU. After ZN3.8, strict ASTAP-ISO uses native ADU and stricter ASTAP-like detection; this artificial photometry no longer represents detector input. | Replace with a detector-realistic fixture or split test: mock detector/catalog for solve orchestration, and test strict detector separately with a validated small image. Do not weaken strict mode. | Medium: false confidence in synthetic end-to-end tests; not evidence of corpus regression. |
| `tests/test_metadata_solver.py::test_strict_fov_hint_source_override_priority` | `AssertionError: expected debug records`, records list is empty. | Test wants to verify FOV override priority and no auto-FOV retries. | The patched `_astap_iso_hypothesis` is never reached because strict detection exits earlier with no usable stars/debug record. | `INVALID_SYNTHETIC_FIXTURE` | Test couples FOV-priority logic to the obsolete synthetic detector fixture. | Decouple priority/config logic from photometric detection by injecting/mocking stars or the correct internal boundary. | Medium: priority logic needs protection, but this test currently fails before exercising it. |
| `tests/test_metadata_solver.py::test_strict_fov_hint_source_header_without_override` | `IndexError: list index out of range` on `records[-1]`. | Test wants to verify FITS `FOVDEG` header priority when no override is provided. | No debug record emitted for the same early detector failure path. | `INVALID_SYNTHETIC_FIXTURE` | Same fixture/early-exit issue. | Same decoupling as above. | Medium. |
| `tests/test_metadata_solver.py::test_strict_scale_source_low_support_skips_autofov` | `IndexError: list index out of range` on `records[-1]`. | Test wants to verify scale-derived FOV source and low-support auto-FOV skip. | No debug record emitted. | `INVALID_SYNTHETIC_FIXTURE` | Same fixture/early-exit issue. | Mock the strict hypothesis boundary or provide detector-realistic stars. | Medium. |
| `tests/test_metadata_solver.py::test_strict_contextual_retry_zero_ref_patience` | `IndexError: list index out of range` on `records[-1]`. | Test wants to verify contextual retry patience when zero reference support persists. | Even with `detect_stars` monkeypatched, strict path logs `used=astap_adaptive`; the patched boundary does not produce a debug record in this setup. | `INVALID_SYNTHETIC_FIXTURE` | The monkeypatch targets the wrong seam for current strict ASTAP-ISO routing, or the test still fails before the intended diagnostic path. | Re-isolate retry policy with a current seam that bypasses detector photometry, or add a small current fixture that reaches `_astap_iso_hypothesis`. | Medium/high for retry-policy coverage, but not a proven solver regression. |
| `tests/test_p221_app_integration.py::test_app_settings_profile_persistence_and_presets` | `assert 'zeblind_4d_experimental' == 'historical'` | Empty persistent settings should default to `historical`. | Current `PersistentSettings`/migration default is `zeblind_4d_experimental`. | `OBSOLETE_EXPECTATION` | Product direction has changed to local chain `ZeNear -> ZeBlind 4D -> optional Astrometry.net`; historical backend is legacy/diagnostic, not product default. | Update expected default to `zeblind_4d_experimental`; keep roundtrip coverage for explicit profile and presets. | Medium: tests should prevent accidental historical reactivation, not demand it. |
| `tests/test_p221_app_integration.py::test_app_build_blind_config_requires_manifest_only_for_4d` | Unexpected `IndexManifestError: blind_4d_manifest_required` for `historical = zs.SolveConfig(...)`. | A `historical` config should not require a 4D manifest. | The constructed `SolveConfig()` no longer means historical; its default profile is 4D, so manifest is required. | `OBSOLETE_EXPECTATION` | Test variable name and expectation are stale. Historical branch still exists if explicitly selected; default is now 4D. | Change the test to pass `blind_backend_profile=HISTORICAL_PROFILE` for the historical branch, and keep the default/4D manifest-required assertion separately. | Medium: preserving explicit historical diagnostic path is okay, but product default must remain 4D. |
| `tests/test_p222_gui_integration.py::test_p222_old_settings_keep_historical_and_easy` | `assert 'zeblind_4d_experimental' == 'historical'` | Legacy v1 settings should preserve historical profile. | Migration converts historical/missing profile to `zeblind_4d_experimental`, keeps `interface_mode='easy'`. | `OBSOLETE_EXPECTATION` | ZN3.10B/current settings migration intentionally avoids historical as normal runtime default. | Update migration contract: old settings migrate safely to 4D default, with historical only as explicit diagnostic/development path if supported elsewhere. | Medium: stale migration tests could force unsafe legacy default. |
| `tests/test_p222_gui_integration.py::test_p222_invalid_profile_and_interface_migrate_to_safe_defaults` | `assert 'zeblind_4d_experimental' == 'historical'` | Invalid profile should migrate to historical/easy. | Invalid profile migrates to `zeblind_4d_experimental`; invalid interface migrates to `easy`. | `OBSOLETE_EXPECTATION` | "Safe default" changed from historical to 4D product profile. | Update expectation to 4D + easy; keep invalid profile coverage. | Medium. |
| `tests/test_p222_gui_integration.py::test_p222_gui_source_contains_required_controls_and_translations` | Missing strings: `Utiliser ZeBlind 4D expérimental`, `Use experimental ZeBlind 4D`, `Couverture limitée aux index installés`, `Coverage is limited to installed indexes`. | GUI source should contain older experimental labels and generic installed-index coverage text. | Current source no longer contains "experimental" user-facing text and now uses 4D-specific coverage wording, e.g. `Coverage is limited to installed 4D indexes`. | `OBSOLETE_EXPECTATION` | ZN3.10B intentionally cleaned GUI text to present ZeBlind 4D as the local chain, not an experimental default, while still warning about limited 4D index coverage. | Update required strings to current architecture text; retain assertions that historical is absent from simple/default wording and that 4D coverage is explicitly limited. | Low/medium: stale text tests are now enforcing old UX. |
| `tests/test_zn310b_gui_fallback_dataset.py::test_zn310b_all_generated_copies_have_no_old_wcs` | First failing path: `/home/tristan/near_bench_cmp30/zn310b_gui_fallback4d_20260716_153931/control_clean/ZN310B_CONTROL_008.fit`; `WCS(...).has_celestial` is true. Manual header scan shows all variants currently contain WCS cards and solve trace cards. | All generated copies under `control_clean`, `no_hints`, `wrong_hints`, and `gui_mixed` should be clean inputs with no old celestial WCS. | Current files contain `WCSAXES`, `CRPIX*`, `CTYPE*`, `CRVAL*`, `CD*`, `SOLVED`, `SOLVER`, `SOLVMODE`, etc. | `DATASET_CONTRACT_BROKEN` | The current generated dataset artifacts are no longer clean. They may have been modified by a GUI/manual solve after generation, or the report points to a non-pristine run directory. The builder contains `strip_solution_cards()`, so the artifact state must be regenerated/validated. | Correct or harden `tools/build_zn310b_gui_dataset.py` if needed, add mandatory post-generation contract validation, then regenerate a new timestamped dataset. Do not patch source FITS or hand-edit existing artifacts. | High: fallback validation dataset cannot be trusted while already-solved WCS cards remain. |
| `tests/test_zn310b_gui_fallback_dataset.py::test_zn310b_nohint_removes_all_near_hint_aliases_and_object` | First failing path: `no_hints/ZN310B_NOHINT_001.fit`, key `CRVAL1` still present. Header also contains WCS and solve trace cards. | NOHINT variant must remove all Near position aliases and `OBJECT`; file name must be neutral. | Current NOHINT files contain `CRVAL1`, `CRVAL2`, WCS cards, and solve trace cards. `OBJECT` and common RA/DEC object hints are absent, but the WCS itself is now a strong hint. | `DATASET_CONTRACT_BROKEN` | Same as above: current no-hint artifacts have WCS written into them, invalidating both "no old WCS" and "no hints" contracts. | Regenerate clean no-hint copies and make the generator fail explicitly if forbidden keys survive. | High: a NOHINT fallback test with CRVAL cards is not a real no-hint test. |

## Group Notes

### Environment

System Python without `.venv` still fails because `astroalign` is not installed. This is classified separately as `ENVIRONMENT`, not as one of the 16 `.venv` failures. `astroalign` is declared in `pyproject.toml`, and the command of record for this mission remains:

```bash
.venv/bin/python -m pytest
```

### Near -> Blind Sequence

Current observed unit seam for the failing sequential test:

```text
scan
  -> _run_index_near_solver(... allow_blind_fallback=True)
  -> near_solve(... fallback_to_blind=False)
  -> near rescue near_solve(... fallback_to_blind=False)
  -> if blind profile is zeblind_4d_experimental: defer blind fallback to batch blind phase
  -> return None
```

This explains `[False, False]`. It does not yet prove that the outer batch phase always calls ZeBlind 4D exactly once after a failed Near solve. That still needs targeted tests before changing the old expectation.

### ZN3.10B Dataset Header Scan

Command used:

```bash
python - <<'PY'
import json
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
root=Path('/home/tristan/.openclaw/workspace/projects/ZeSolver')
man=json.loads((root/'reports/zenear_zn310b_gui_manifest.json').read_text())
run=Path(man['run_dir'])
keys=['WCSAXES','CRPIX1','CRPIX2','CTYPE1','CTYPE2','CRVAL1','CRVAL2','CD1_1','CD1_2','CD2_1','CD2_2','CDELT1','CDELT2','PC1_1','PC1_2','PC2_1','PC2_2','RA','DEC','OBJCTRA','OBJCTDEC','OBJRA','OBJDEC','OBJECT','SOLVED','SOLVER','SOLVMODE']
for sub in ('control_clean','no_hints','wrong_hints','gui_mixed'):
    print('\\n##', sub)
    for path in sorted((run/sub).glob('*.fit'))[:10]:
        h=fits.getheader(path)
        present=[k for k in keys if k in h]
        print(path.name, 'has_celestial=', bool(WCS(h).has_celestial), 'present=', present)
PY
```

Result summary:

- `control_clean`: all inspected files have celestial WCS and solve trace cards.
- `no_hints`: all inspected files have celestial WCS and solve trace cards; `CRVAL1/CRVAL2` remain.
- `wrong_hints`: all inspected files have celestial WCS, solve trace cards, plus BADHINT RA/DEC aliases.
- `gui_mixed`: mixed copies also have celestial WCS and solve trace cards.

## Files Implicated by Triage

No file has been corrected yet. Files implicated by the failure analysis:

- `tests/test_batch_blind_fallback.py`
- `zesolver.py`
- `tests/test_catalog290.py`
- `zewcs290/catalog290.py`
- `tests/test_downloads.py`
- `zeblindsolver/downloads.py`
- `tests/test_metadata_solver.py`
- `zeblindsolver/metadata_solver.py`
- `tests/test_p221_app_integration.py`
- `tests/test_p222_gui_integration.py`
- `zesolver/settings_store.py`
- `tests/test_zn310b_gui_fallback_dataset.py`
- `tools/build_zn310b_gui_dataset.py`
- `reports/zenear_zn310b_gui_manifest.json`
- `/home/tristan/near_bench_cmp30/zn310b_gui_fallback4d_20260716_153931/`

## Proposed Correction Order

1. Fix `FakeBackend` remainder handling and keep SHA verification intact.
2. Mark/split external catalog tests, or add minimal hermetic fixtures if the binary format can be represented compactly.
3. Update obsolete product-profile tests to assert the 4D default and explicit historical diagnostic branch.
4. Decouple Near metadata/config logic tests from obsolete synthetic photometry.
5. Regenerate ZN3.10B dataset through the builder into a new timestamped directory and add post-generation forbidden-key validation.
6. Only after the outer fallback sequence is mapped, update or fix the Near -> Blind fallback test.

## Current Readiness

Decision after first-pass triage:

```text
NOT_READY_FOR_REGRESSION_SUITE
```

Reason: each failure now has an initial classification, but no correction has been applied yet, and two areas remain product-critical before the baseline can be trusted:

- Near failure must be proven to reach ZeBlind 4D exactly once through the real batch path.
- ZN3.10B fallback dataset must be regenerated or validated as clean before it can serve as a regression fixture.

