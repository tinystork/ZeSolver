# P0B - Regression Suite Foundation

## Commands

Hermetic baseline:

```bash
.venv/bin/python tools/run_regression_suite.py --hermetic
```

Equivalent direct command:

```bash
.venv/bin/python -m pytest -m "not external_catalog and not corpus and not slow" -q
```

Corpus/integration baseline:

```bash
.venv/bin/python tools/run_regression_suite.py --corpus
```

Full local suite:

```bash
.venv/bin/python -m pytest -q
```

## External Data

Corpus tests use explicit variables only:

- `ZESOLVER_CORPUS_ROOT`
- `ZESOLVER_ZN310B_ROOT`
- `ZESOLVER_ASTAP_ROOT`
- `ZESOLVER_BLIND4D_MANIFEST`
- `ZESOLVER_LEGACY_INDEX_ROOT`

If data are absent, tests skip with the missing variable or missing path named. A SHA mismatch is a failure, not a skip.

## Manifest

Versioned manifest:

```text
tests/corpus/manifest.json
```

Initial coverage:

- `near_m106_232102`: ADU sentinel recovered by strict ZeNear.
- `near_m106_233459`: M106 positive control.
- `pipeline_zn310b_control_001`: expected Near-only path.
- `pipeline_zn310b_nohint_004`: expected Near failure then ZeBlind 4D.
- `pipeline_zn310b_badhint_007`: expected wrong-hint rejection then ZeBlind 4D.
- `blind4d_p29_232329`: P29 blind4d reference, not yet source-path mapped.

## Oracles

Small structured oracles:

```text
tests/corpus/oracles/zenear_reference.json
tests/corpus/oracles/zeblind4d_reference.json
tests/corpus/oracles/pipeline_reference.json
```

Selected legacy report files are also tracked because existing ZN tests read them directly.

## Tests Added

- `tests/corpus_loader.py`
  - manifest validation;
  - external path resolution;
  - SHA verification;
  - WCS validation helpers.
- `tests/test_regression_manifest.py`
  - schema, uniqueness, provenance, missing data, corrupt data.
- `tests/test_regression_near.py`
  - ZeNear oracle summary and WCS validation independent of `success=True`.
- `tests/test_regression_blind4d.py`
  - ZeBlind 4D oracle contract and manifest path behavior.
- `tests/test_regression_pipeline.py`
  - Near success, 4D deferred fallback, explicit historical inline fallback.
- `tests/test_regression_fits_integrity.py`
  - WCS header write and WCS cleanup do not alter pixel bytes.
- `tools/run_regression_suite.py`
  - single runner for hermetic/corpus/full validation.

## Current Results

Local full suite:

```text
248 passed, 7 skipped, 1 warning
```

Hermetic runner:

```text
248 passed, 1 skipped, 6 deselected, 1 warning
compileall OK
PASS
```

Corpus runner without external data:

```text
6 skipped, 249 deselected
compileall OK
PASS
```

Clean worktree hermetic runner:

```text
248 passed, 1 skipped, 6 deselected, 1 warning
compileall OK
PASS
```

## Known Gaps

- Actual corpus execution of Near/Blind4D is gated by external data variables.
- P29 Blind4D source FITS path is not yet mapped in the manifest.
- ZN3.10B GUI fallback remains a manual GUI run; the automated tests cover the same underlying routing contracts, not GUI clicks.
- Packaging is not installable via editable pip yet due missing package discovery.
