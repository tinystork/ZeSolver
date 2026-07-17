# P2B-2A ZN3.10B Blind Parity

Phase: P2B-2A - Production Blind Adapter

## Inputs

External variables used:

| Variable | Logical value |
| -------- | ------------- |
| `ZESOLVER_ZN310B_ROOT` | regenerated ZN3.10B GUI fallback dataset |
| `ZESOLVER_ASTAP_ROOT` | local ASTAP catalog root |
| `ZESOLVER_BLIND4D_MANIFEST` | ZeBlind 4D experimental manifest |
| `ZESOLVER_LEGACY_INDEX_ROOT` | local ZeSolver Near/legacy index root |

The test uses `reports/zenear_zn310b_gui_manifest.json`, which contains eight
GUI cases:

```text
CONTROL: 3
NOHINT: 3
BADHINT: 2
```

## Pipeline Route

The validation runs through `SolverPipeline` with:

```text
Near port: ExistingNearSolverPort
Blind port: ProductionBlindSolverPort
Profiles: zenear-v1, zeblind4d-v1, pipeline-v1
Astrometry.net fallback: disabled
```

`ProductionBlindSolverPort` invokes the real ZeBlind 4D wrapper on a temporary
copy, reads the produced WCS, and returns it to `wcs_io.py` for final output.
The source FITS files are not modified.

## Result

Command:

```bash
.venv/bin/python -m pytest tests/test_solver_pipeline_zn310b_production.py -q
```

Result:

```text
1 passed in 142.01s
```

The full P2B-2A targeted run also included this test:

```text
12 passed in 141.57s
```

Observed route contract:

| Group | Expected | Observed |
| ----- | -------- | -------- |
| CONTROL | 3/3 solved by Near; Blind not called | PASS |
| NOHINT | 3/3 solved by Blind 4D after Near failure | PASS |
| BADHINT | 2/2 solved by Blind 4D after wrong Near route rejection | PASS |

Additional checks:

```text
historical_blind_called=0 by construction of SolverPipeline production route
astrometry_web_called=0 by ProductSettings(web_fallback=False)
pixels_unchanged=true for all eight source inputs
WCS output present for all eight results
Blind calls observed: 5
```

Decision:

```text
ZN310B_PIPELINE_PRODUCTION_BLIND_PARITY_PASS
```
