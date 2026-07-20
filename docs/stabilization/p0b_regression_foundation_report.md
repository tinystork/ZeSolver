# P0B - Reproducible Baseline & Regression Suite Report

Date: 2026-07-17

## Initial Git State

Starting point included P0A changes in the working tree:

- `.gitignore`, `pyproject.toml`, `tests/test_catalog290.py`, `tests/test_metadata_solver.py`, `zeblindsolver/downloads.py` modified or needing tracking;
- P0A docs untracked;
- several tests/tools/reports needed by the green local suite ignored by `.gitignore`.

No FITS corpus files, ASTAP databases, or 4D NPZ indexes were added.

## Reproducibility Work

`.gitignore` now:

- follows `tests/*.py` and `tests/corpus/**`;
- follows source tools required by the tests, especially `tools/diagnose_*.py` and `tools/build_zn310b_gui_dataset.py`;
- keeps `reports/**` ignored by default;
- adds narrow exceptions for selected report oracle files already consumed by existing tests.

The clean checkout was verified from a temporary detached Git commit built with a temporary index:

```text
c765ca6d16b8d0414d7182df24bc9a53d4c14241
```

Clean result:

```text
248 passed, 1 skipped, 6 deselected, 1 warning
compileall OK
runner status PASS
```

Editable install still fails because packaging lacks explicit package discovery. This is recorded as a P4 packaging gap.

## Corpus Manifest

Created:

```text
tests/corpus/README.md
tests/corpus/manifest.json
tests/corpus/oracles/zenear_reference.json
tests/corpus/oracles/zeblind4d_reference.json
tests/corpus/oracles/pipeline_reference.json
```

The manifest uses only explicit environment variables for data roots. It does not encode personal absolute paths.

Oracles available:

- ZeNear ZN3.9 positive corpus summary: 142/142 WCS confirmed, 0 wrong field.
- ZeBlind 4D P29 bounded multi-index reference metadata.
- ZN3.10B pipeline dataset contract metadata.

Oracles missing:

- mapped source FITS path for the P29 Blind4D reference case;
- completed manual GUI fallback result for ZN3.10B;
- normalized compact replacements for every selected historical report consumed by legacy ZN tests.

## Tests Added

```text
tests/corpus_loader.py
tests/test_regression_manifest.py
tests/test_regression_near.py
tests/test_regression_blind4d.py
tests/test_regression_pipeline.py
tests/test_regression_fits_integrity.py
tools/run_regression_suite.py
```

Coverage added:

- manifest schema and uniqueness;
- explicit skip for absent corpus roots;
- SHA mismatch failure for corrupt data;
- WCS validation independent of solver `success`;
- center, scale, matrix, parity, inliers and RMS checks when oracle fields exist;
- FITS pixel hash preservation after WCS header writes and WCS cleanup;
- Near success path does not invoke blind fallback;
- 4D product path defers failed Near frames to batch Blind;
- explicit historical diagnostic path preserves inline fallback.

## Commands Executed

Local targeted regression tests:

```bash
.venv/bin/python -m pytest tests/test_regression_manifest.py tests/test_regression_near.py tests/test_regression_blind4d.py tests/test_regression_pipeline.py tests/test_regression_fits_integrity.py -q
```

Result:

```text
16 passed, 4 skipped
```

Full suite:

```bash
.venv/bin/python -m pytest -q
```

Result:

```text
248 passed, 7 skipped, 1 warning
```

Hermetic:

```bash
.venv/bin/python -m pytest -m "not external_catalog and not corpus and not slow" -q
.venv/bin/python tools/run_regression_suite.py --hermetic
```

Result:

```text
248 passed, 1 skipped, 6 deselected, 1 warning
runner PASS
```

Corpus without configured data:

```bash
.venv/bin/python -m pytest -m "external_catalog or corpus or slow" -q
.venv/bin/python tools/run_regression_suite.py --corpus
```

Result:

```text
6 skipped, 249 deselected
runner PASS
```

Clean checkout:

```bash
/tmp/zesolver-clean-baseline-p0b/.venv-clean/bin/python tools/run_regression_suite.py --hermetic
```

Result:

```text
248 passed, 1 skipped, 6 deselected, 1 warning
compileall OK
PASS
```

## Skips

Expected skips:

- external ASTAP/HNSKY `database/` absent;
- S50 real frame/index not configured;
- `ZESOLVER_CORPUS_ROOT` unset;
- `ZESOLVER_ZN310B_ROOT` unset;
- `ZESOLVER_BLIND4D_MANIFEST` unset;
- P29 Blind4D source FITS path not mapped yet.

Unexpected skips: none identified.

## Risks Remaining

- The actual external corpus regression run has not been executed in this P0B pass.
- The GUI fallback 4D run remains manual.
- Selected historical report oracles are still under `reports/`; future cleanup should normalize them into `tests/corpus/oracles/`.
- Editable packaging is broken and must be handled in P4.

## Decision

```text
READY_FOR_CATALOG_ARCHITECTURE
```

Reason: the hermetic baseline is reproducible from a clean Git snapshot, the corpus manifest and initial oracles are versioned, missing data produce explicit skips, corrupt data fail, WCS validation no longer trusts `success=True`, FITS pixel integrity is tested on copies, and Near/Blind4D/pipeline contracts have initial characterization coverage without refactoring solver internals.
