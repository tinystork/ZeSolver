# ZeSolver Stabilization Phase 1 - Initial State

Date: 2026-07-16 18:49-19:00 Europe/Paris

## Scope

This report freezes the observed repository state before starting the stabilization and catalog architecture work. No solver refactor, threshold change, catalog/index rebuild, or FITS corpus modification was performed during this initial-state pass.

## Repository

- Repository: `/home/tristan/.openclaw/workspace/projects/ZeSolver`
- Branch: `test`
- Upstream tracking: `origin/test`
- Commit: `f5a76cd2ec8998d552332a3556325ebba5c116cf`
- Worktree before this report: clean (`git status --short --untracked-files=all` returned no entries)

## Platform

- OS: Debian Linux on `TINYDEBIAN`
- Kernel: `6.12.95+deb13-amd64`
- System Python: `Python 3.13.5`
- Project virtualenv Python: `Python 3.13.5`

## Files and Areas Inspected

Read or inspected during this pass:

- `AGENT.md` in full
- `README.md`
- `CHANGELOG.md`
- `pyproject.toml`
- ZeNear entry points:
  - `zeblindsolver/metadata_solver.py`
  - `NearSolveConfig`
  - `solve_near`
- ZeBlind 4D entry points:
  - `zeblindsolver/zeblindsolver.py`
  - `solve_blind`
  - `zeblindsolver/index_manifest_4d.py`
  - `zeblindsolver/quad_index_4d.py`
  - `zeblindsolver/quad_index_builder.py`
- Catalog readers:
  - `zewcs290/catalog290.py`
  - `zeblindsolver/astap_db_reader.py`
- Tests under `tests/`
- Benchmark and diagnostic tools under `tools/`
- Current `reports/` inventory, especially `zenear_zn31` through `zenear_zn310b` and `zeblind_p20` through `zeblind_p29`
- Existing local corpus/index locations:
  - `/home/tristan/near_bench_cmp30/`
  - `/home/tristan/near_bench_cmp30/zn310b_gui_fallback4d_20260716_153931/`
  - `/home/tristan/zesolver_index/`
  - `config/zeblind_4d_experimental_manifest.json`

Note: `rg` is not installed in this environment, so source searches used `grep`/`find`.

## Main Dependencies Observed

From `.venv`:

```text
numpy==2.5.1
astropy==8.0.1
scipy==1.18.0
scikit-image==0.26.0
astroalign==2.6.2
rich==15.0.0
pytest==9.1.1
PySide6==6.11.1
sep==1.4.1
cupy-cuda12x==14.1.1
```

System Python is missing at least `astroalign`; a plain `pytest -q` outside `.venv` aborts during collection.

## Commands Executed

```bash
pwd
find /home/tristan/.openclaw/workspace -maxdepth 4 -type d -name ZeSolver
date
uname -a
python --version
sed -n '1,260p' AGENT.md
sed -n '261,620p' AGENT.md
sed -n '621,980p' AGENT.md
sed -n '1,220p' README.md
sed -n '1,220p' CHANGELOG.md
sed -n '1,240p' pyproject.toml
git status --short --branch
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
find . -maxdepth 2 -type f
find tests -maxdepth 3 -type f -name 'test*.py'
find zeblindsolver zewcs290 zesolver -maxdepth 4 -type f -name '*.py'
pytest -q
.venv/bin/python --version
.venv/bin/python -m pip show numpy astropy scipy scikit-image astroalign rich pytest PySide6
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests *.py
git status --short --untracked-files=all
grep -RInE 'def solve_near|def solve_blind|class CatalogDB|class Loaded4DManifest|class QuadIndex|class NearSolveConfig|class BlindSolveConfig|strict_acceptance_mode|SEED_SCALE' zeblindsolver zewcs290 zesolver.py zesolver tools tests
find reports -maxdepth 2 \( -name '*zn3*.json' -o -name '*zn3*.md' -o -name '*p2*.json' -o -name '*p2*.md' \)
find /home/tristan/near_bench_cmp30 -maxdepth 2 -type d
find /home/tristan/zesolver_index -maxdepth 3 -type f
```

## Test Results

### System Python

Command:

```bash
pytest -q
```

Result:

```text
Exit code: 3
No tests ran.
Collection aborted because importing zesolver.py raised:
SystemExit: astroalign is required. Install the project dependencies first.
Root cause: ModuleNotFoundError: No module named 'astroalign'
```

This is an environment issue for the system Python, not yet classified as a code regression.

### Project `.venv`

Command:

```bash
.venv/bin/python -m pytest -q
```

Result:

```text
16 failed, 217 passed, 1 skipped, 1 warning in 28.06s
```

Skipped:

```text
tests/test_real_s50.py: S50 index or frame not configured
```

Warning:

```text
tests/test_p220_manifest_loader.py::test_runtime_prepare_strips_position_and_identity_hints
Astropy VerifyWarning: Card is too long, comment will be truncated.
```

Failures observed:

```text
tests/test_batch_blind_fallback.py::test_sequential_near_path_keeps_immediate_blind_fallback
tests/test_catalog290.py::test_decode_g05_polar_tile
tests/test_catalog290.py::test_decode_d50_tile_and_cone_query
tests/test_downloads.py::test_download_manager_with_fake_backend
tests/test_metadata_solver.py::test_metadata_solver_solves_synthetic_frame
tests/test_metadata_solver.py::test_strict_fov_hint_source_override_priority
tests/test_metadata_solver.py::test_strict_fov_hint_source_header_without_override
tests/test_metadata_solver.py::test_strict_scale_source_low_support_skips_autofov
tests/test_metadata_solver.py::test_strict_contextual_retry_zero_ref_patience
tests/test_p221_app_integration.py::test_app_settings_profile_persistence_and_presets
tests/test_p221_app_integration.py::test_app_build_blind_config_requires_manifest_only_for_4d
tests/test_p222_gui_integration.py::test_p222_old_settings_keep_historical_and_easy
tests/test_p222_gui_integration.py::test_p222_invalid_profile_and_interface_migrate_to_safe_defaults
tests/test_p222_gui_integration.py::test_p222_gui_source_contains_required_controls_and_translations
tests/test_zn310b_gui_fallback_dataset.py::test_zn310b_all_generated_copies_have_no_old_wcs
tests/test_zn310b_gui_fallback_dataset.py::test_zn310b_nohint_removes_all_near_hint_aliases_and_object
```

Initial classification:

- Missing local `database/` directory causes the two `test_catalog290.py` failures.
- Several `test_metadata_solver.py` synthetic tests fail because strict ASTAP-ISO now reports no stars/debug records for their synthetic fixtures.
- `test_p221`/`test_p222` failures reflect an expectation mismatch between old historical defaults and the current 4D-default integration direction.
- `test_zn310b_gui_fallback_dataset.py` reports that at least one generated GUI dataset copy still has WCS cards, and NOHINT copies still contain `CRVAL1`. This conflicts with the intended ZN3.10B dataset contract and must be understood before relying on that dataset as a regression fixture.
- `test_downloads.py::test_download_manager_with_fake_backend` reports a SHA256 mismatch in the fake backend test.
- `test_batch_blind_fallback.py::test_sequential_near_path_keeps_immediate_blind_fallback` reports a changed call sequence: observed `[False, False]`, expected `[False, False, True]`.

Because the full `.venv` suite is red, subsequent stabilization steps should not assume the whole existing test suite is a clean baseline.

## Compilation

Command:

```bash
.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests *.py
```

Result:

```text
OK
```

Scope: project modules, package entry points, tools and tests. Vendored/external upstream trees (`ASTAP-main`, `astrometry-main`, `astrometry-net-main`) were not part of this compile command.

## Corpus and Reports Found

Repository reports include extensive historical ZeNear and ZeBlind 4D outputs:

- `reports/zenear_zn31_*` through `reports/zenear_zn310b_*`
- `reports/zeblind_p20_*` through `reports/zeblind_p29_*`
- Many forensic `r47i_*`, `r8_*`, `r10-*`, and M106/Near parity reports
- `benchmark_report.json`
- `benchmark_report.csv`

Key recent reports present:

- `reports/zenear_zn38_summary.md`
- `reports/zenear_zn39_summary.md`
- `reports/zenear_zn310b_summary.md`
- `reports/zenear_zn310b_gui_manifest.json`
- `reports/zenear_zn310b_gui_result.json`
- `reports/zeblind_p29_4d_bounded_multi_index_union_validation.json`

Local corpus/index material found:

- `/home/tristan/near_bench_cmp30/thread4/`
- `/home/tristan/near_bench_cmp30/zn310b_gui_fallback4d_20260716_153931/`
- `/home/tristan/zesolver_index/manifest.json`
- `/home/tristan/zesolver_index/hash_tables/quads_S.npz`
- `/home/tristan/zesolver_index/hash_tables/quads_M.npz`
- `/home/tristan/zesolver_index/hash_tables/quads_L.npz`
- many `/home/tristan/zesolver_index/tiles/d50_*.npz`

The repository-level `database/` directory referenced by `tests/test_catalog290.py` is absent.

## Main Risk Areas

1. The existing full test suite is not green in the current `.venv`.
2. The ZN3.10B dataset contract appears internally inconsistent in current tests: WCS remains in at least one control copy and NOHINT still has `CRVAL1`.
3. Historical-default tests conflict with the current product direction where ZeBlind 4D is the normal local fallback and historical blind is diagnostic-only.
4. Synthetic Near tests may no longer match strict ASTAP-ISO ADU detector behavior after ZN3.8.
5. Corpus and index data live outside the repository; future regression tests must reference them by manifest/checksum and skip clearly when unavailable.
6. `README.md` and `pyproject.toml` still describe release/version state as `1.0.0`, while `AGENT.md` recommends avoiding `1.0.0` until P0-P4 are satisfied.
7. Large vendored upstream trees and many historical diagnostic tools increase discovery noise and packaging risk.
8. `rg` is unavailable locally; automation scripts should not assume it exists.

## Preexisting Local Modifications

Before creating this report, `git status --short --untracked-files=all` returned no tracked or untracked entries. The only intended modification from this initial-state pass is this new report under `docs/stabilization/`.

## Gate for Next Step

Per the requested working order, do not proceed to corpus manifest generation or new characterization tests until the current red baseline is either:

1. accepted as the true initial state and explicitly documented as such; or
2. split into environment/data-missing failures versus real regressions; or
3. repaired by small, separately justified fixes before building the new regression suite.

