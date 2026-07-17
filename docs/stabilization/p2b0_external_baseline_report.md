# P2B-0 External Baseline Report

Date: 2026-07-17

Decision: `NOT_READY_FOR_CORE_EXTRACTION`

P2B-0 froze the P0-P2A milestone locally, configured known external data, and
ran the external corpus gate. P2B-1 must not start because the configured Near
corpus root contains files that do not match the versioned corpus manifest.

## Milestone Commit and Tag

Local commit:

```text
46208a9 stabilization: freeze P0-P2A functional baseline
```

Local annotated tag:

```text
zesolver-p2a-stable-baseline
```

The tag records:

- Date: 2026-07-17
- Hermetic baseline before tag: `320 passed, 1 skipped, 6 deselected, 1 warning`
- Full pytest before tag: `320 passed, 7 skipped, 1 warning`
- CatalogLibrary: P1C integrated at configuration boundary
- Active profiles: `zenear-v1`, `zeblind4d-v1`, `pipeline-v1`
- 4D coverage: six D50 indexes, partial only, `all_sky=false`
- External corpus: not yet executed at freeze

No remote push was performed.

## Variables Used

Values were exported inline for commands and were not written to versioned
configuration files.

```text
ZESOLVER_CORPUS_ROOT=<home>
ZESOLVER_ZN310B_ROOT=<home>/near_bench_cmp30/zn310b_gui_fallback4d_20260717_021840
ZESOLVER_ASTAP_ROOT=/opt/astap
ZESOLVER_BLIND4D_MANIFEST=<repo>/config/zeblind_4d_experimental_manifest.json
ZESOLVER_LEGACY_INDEX_ROOT=<home>/zesolver_index
```

## Data Present

- ASTAP/HNSKY `d50`: `1476/1476` `.1476` tiles under `/opt/astap`.
- ZeBlind 4D strict manifest: present.
- ZeBlind 4D NPZ indexes: 6 present, all SHA256 OK.
- Legacy index root: present with `manifest.json`, hash tables and tile NPZs.
- ZN3.10B generated dataset: present.

## Data Absent or Incomplete

- ASTAP/HNSKY `g05`: absent under `/opt/astap`; external `g05` test skips
  explicitly.
- P29 Blind 4D FITS source path: still unmapped in `tests/corpus/manifest.json`;
  skip remains explicit.
- General Near corpus root is present but contaminated/incompatible with the
  manifest SHA values.

## Integrity Audit

Catalogue audit command:

```bash
.venv/bin/python tools/audit_catalog_library.py \
  --astap-root /opt/astap \
  --blind4d-manifest config/zeblind_4d_experimental_manifest.json \
  --legacy-index-root "$ZESOLVER_LEGACY_INDEX_ROOT"
```

Result:

```text
astap.status=READY_PARTIAL
astap.d50.status=FULL
astap.d50.tile_count=1476
blind4d.status=READY_PARTIAL
blind4d.enabled_index_count=6
blind4d.all_sky=false
blind4d.coverage_fraction_by_tile_count=0.0040650406504065045
legacy_index.status=PRESENT
catalog_library.status=READY_PARTIAL
```

Near corpus integrity checks found:

```text
near_m106_232102 expected SHA256:
3edf2580268376dc51409591fd9f91452c8f1e3c2817d3fe29b77ae2d7980821

configured file actual SHA256:
c0f02d39226e5396efa7f41f969e0fd3ca5272fa884696af8fe8378f1b93a2c1
```

The configured file also already contains a celestial WCS and ZeSolver solve
cards:

```text
has_wcs=True
SOLVED=1
SOLVER=ZeSolver
```

A copy with the expected SHA exists in an old working-report directory, but the
configured corpus root does not point to that clean file. The gate must not
silently substitute it.

## Harness Corrections

Two non-algorithmic test harness fixes were made during P2B-0:

- `tests/test_regression_blind4d.py` now checks real
  `Loaded4DManifest.enabled_index_paths` and `tile_keys` instead of a stale
  `.tiles` attribute.
- `tests/test_catalog290.py` now honors `ZESOLVER_ASTAP_ROOT` and skips only the
  absent family (`g05`) when `d50` is configured.

Targeted harness validation:

```text
.venv/bin/python -m pytest tests/test_catalog290.py tests/test_regression_blind4d.py -q
s....s
4 passed, 2 skipped
```

## Hermetic Results

Before the milestone commit:

```text
tools/run_regression_suite.py --hermetic
320 passed, 1 skipped, 6 deselected, 1 warning
runner status PASS
compileall OK
```

After the P2B-0 harness corrections:

```text
tools/run_regression_suite.py --hermetic
320 passed, 1 skipped, 6 deselected, 1 warning
runner status PASS
compileall OK
```

`git diff --check` produced no output.

## Corpus Results

Command:

```bash
.venv/bin/python tools/run_regression_suite.py --corpus
```

with the external variables listed above.

Result after harness corrections:

```text
1 failed, 3 passed, 2 skipped, 321 deselected
compileall OK
runner status FAIL
```

Passed:

- `tests/test_catalog290.py::test_decode_d50_tile_and_cone_query`
- `tests/test_regression_blind4d.py::test_blind4d_manifest_loads_when_configured`
- `tests/test_regression_pipeline.py::test_pipeline_zn310b_cases_resolve_paths_or_skip_explicitly`

Skipped:

- `tests/test_catalog290.py::test_decode_g05_polar_tile`
  - Reason: `g05` family absent under `/opt/astap`.
- `tests/test_regression_blind4d.py::test_blind4d_cases_are_mapped_before_execution`
  - Reason: P29 source FITS path not mapped yet.

Failed:

- `tests/test_regression_near.py::test_zenear_corpus_files_resolve_or_skip_explicitly`
  - Reason: `near_m106_232102` SHA mismatch in configured corpus root.

## ZN3.10B Results

Automated pipeline validation was run on a temporary copy of `gui_mixed/`.

```text
Done in 174.9s — 8 solved, 0 skipped, 0 failed
```

Parsed result:

```text
CONTROL_NEAR_CORRECT=3
NOHINT_4D_CORRECT=3
BADHINT_4D_CORRECT=2
historical_blind_called=0
astrometry_web_called=0
verdict=PASS
```

See `docs/stabilization/zn310b_gui_validation_result.md`.

## Risks Open

- The configured Near corpus root is not a trustworthy baseline source because
  at least one manifest case has a SHA mismatch and residual WCS cards.
- The corpus manifest currently references one P29 Blind 4D case without a
  mapped FITS path.
- The external ASTAP root contains `d50` but not `g05`; this is acceptable only
  because the skip names the absent family precisely.
- Manual GUI validation remains unrun; automated pipeline validation passed on a
  temporary copy.

## Barrier Decision

```text
NOT_READY_FOR_CORE_EXTRACTION
```

Reason: although the P0-P2A milestone is frozen and ZN3.10B automated pipeline
validation passed, the configured external corpus gate fails on a real data
integrity mismatch. P2B-1 must wait until the Near corpus root is restored,
remapped, or the manifest is updated through an explicit baseline decision.
