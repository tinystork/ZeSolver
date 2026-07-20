# P2B-2A Blind Adapter Report

Phase: P2B-2A - Production Blind Adapter

## Summary

`SolverPipeline` now has a production Blind 4D port:

```text
zesolver/core/blind_models.py
zesolver/core/blind_port.py
zesolver/core/blind_result_adapter.py
```

The port calls the existing ZeBlind 4D backend through
`zesolver.zeblindsolver.blind_solve()` and does not modify `solve_blind()`,
`solve_near()`, solver thresholds, profiles, catalogs, indexes, FITS formats, or
manifest formats.

## Architecture

Files inspected:

```text
zesolver.py
zesolver/zeblindsolver.py
zeblindsolver/zeblindsolver.py
zeblindsolver/index_manifest_4d.py
zeblindsolver/profiles.py
zesolver/core/pipeline.py
zesolver/core/wcs_io.py
```

Mapping document:

```text
docs/architecture/blind_production_path_map.md
```

Decision:

```text
ProductionBlindSolverPort is a narrow 4D adapter.
ImageSolver remains available as rollback.
The historical backend remains diagnostic/legacy and is not removed.
```

## WCS Write Policy

The existing Blind wrapper writes WCS cards into the FITS path it receives.
To preserve the P2B-1 pipeline boundary, the new port:

1. copies the input FITS to a temporary directory;
2. runs the real Blind 4D wrapper on that temporary copy;
3. reads the WCS and header updates from the temporary copy;
4. returns `EngineSolveResult(wcs_written=False)`;
5. lets `zesolver/core/wcs_io.py` write the final output.

This keeps final output writes isolated and protects source pixels.

## Hint Policy

The port accepts resolved hints through `BlindSolveRequest`.

For `zeblind4d-v1`, config parity is still owned by the existing
`build_blind_solve_config()` path. The underlying `zeblind_4d_experimental`
profile clears RA/Dec/radius hints by design, matching the already validated 4D
contract. Adaptive neighbor hints remain in `ImageSolver` for now.

## P29 Mapping Attempt

Searched:

```text
/home/tristan/near_bench100_input/
reports/
docs/
tests/
config/
```

Representative candidate audit:

| Candidate | SHA256 | Clean |
| --------- | ------ | ----- |
| `/home/tristan/near_bench100_input/073_Light_mosaic_M 106_20.0s_IRCUT_20250518-232329.fit` | `79252659eb18110ae8fa1c1263de049a346ebc331c63a1515ff10787c7407a4a` | no, contains WCS + `SOLVED` + `SOLVER` |
| `/home/tristan/near_bench100_input/031_Light_mosaic_M 106_20.0s_IRCUT_20250518-232329.fit` | `249880281d9f2653c09a0b763d62e62454f56d203aba992e68a7ba255f3a2b8f` | no, contains WCS + `SOLVED` + `SOLVER` |
| `reports/eq_ircut_cleanbench_20260518_230249/data/Light_mosaic_M 106_20.0s_IRCUT_20250518-232329.fit` | `fb96e2d559a712f1ea7d7ae004797a235c7d7c20eccdc8d70be289d75de832e5` | no, contains WCS + `SOLVED` + `SOLVER` |
| `reports/r47i_s7_testzenear_full_product_clean_20260624/input/Light_mosaic_M 106_20.0s_IRCUT_20250518-232329.fit` | `4c5fd197d46ee92d33b517df641ab8d06d7b4d06b606593d778c13611ba4690b` | no, contains WCS + `SOLVED` + `SOLVER` |
| `reports/r47i_s8_p29_all30_internal45_then_image2xy_fallback_20260704/candidates/Light_mosaic_M 106_20.0s_IRCUT_20250518-232329.fit` | `a333461e1126db355e1ed16a6f69ef9c6a4e94c39d8134c0944f997fa0fd5217` | no, contains WCS + `SOLVED` + `SOLVER` |

No clean, reliable P29 fixture was mapped during P2B-2A. The existing explicit
skip remains correct and does not block this phase because ZN3.10B and the
configured corpus pass.

## Validation

Initial baseline before modification:

```text
Corpus: 4 passed, 2 skipped, 341 deselected, runner PASS
Hermetic: 340 passed, 1 skipped, 6 deselected, 1 warning, runner PASS
Full pytest: 344 passed, 3 skipped, 1 warning
compileall: OK
git diff --check: OK
```

P2B-2A targeted:

```text
12 passed in 141.57s
```

Corpus:

```text
5 passed, 2 skipped, 352 deselected in 134.49s
runner PASS
```

Expected corpus skips:

```text
external ASTAP/HNSKY family 'g05' not found under /opt/astap
blind4d source FITS paths not mapped yet: blind4d_p29_232329
```

Hermetic:

```text
351 passed, 1 skipped, 7 deselected, 1 warning
runner PASS
```

Full pytest:

```text
356 passed, 3 skipped, 13 warnings
```

Compile and diff:

```text
compileall: OK
git diff --check: OK
```

## Behavioral Changes

Intended additive behavior:

```text
SolverPipeline default Blind port now uses ProductionBlindSolverPort.
Blind 4D WCS output is normalized through EngineSolveResult and wcs_io.py.
CatalogLibrary/legacy resolved Blind 4D manifest paths are honored by the port.
```

No known astrometric regression was introduced. `ImageSolver` and the legacy
Blind path remain available.

## Decision

```text
READY_FOR_BATCH_EXTRACTION
```
