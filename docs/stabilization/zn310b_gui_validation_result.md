# ZN3.10B GUI/Pipeline Validation Result

Date: 2026-07-17

## Scope

P2B-0 ran the ZN3.10B hybrid fallback dataset through the real CLI pipeline on
a temporary copy of `gui_mixed/`. The original ZN3.10B dataset was not modified.

This validates the automated pipeline behavior. It does not validate a manual
Qt GUI session.

## Data

Logical data roots:

- ZN3.10B dataset: `<home>/near_bench_cmp30/zn310b_gui_fallback4d_20260717_021840`
- ASTAP/HNSKY source: `/opt/astap`
- Blind 4D manifest: `config/zeblind_4d_experimental_manifest.json`
- Legacy index root: `<home>/zesolver_index`

Temporary run directory:

```text
/tmp/zesolver-zn310b-p2b0-8LB4he
```

## Command

The run copied `gui_mixed/` to a temporary input directory, then executed:

```bash
.venv/bin/python zesolver.py \
  --headless \
  --db-root /opt/astap \
  --input-dir "$TMP/input" \
  --family d50 \
  --workers 1 \
  --formats .fit \
  --overwrite \
  --blind-profile zeblind_4d_experimental \
  --blind-4d-manifest config/zeblind_4d_experimental_manifest.json \
  --blind-index "$ZESOLVER_LEGACY_INDEX_ROOT" \
  --log-level INFO
```

## Result

CLI summary:

```text
Done in 174.9s — 8 solved, 0 skipped, 0 failed
```

Parsed `ZN310B_EVENT` verdict:

```json
{
  "astrometry_web_called": 0,
  "counts": {
    "BADHINT_4D_CORRECT": 2,
    "CONTROL_NEAR_CORRECT": 3,
    "NOHINT_4D_CORRECT": 3
  },
  "historical_blind_called": 0,
  "verdict": "PASS"
}
```

Observed backend behavior:

- CONTROL: Near solved `3/3`; Blind 4D was not called.
- NOHINT: Near failed cleanly due missing metadata; Blind 4D solved `3/3`.
- BADHINT: Near rejected/failover path was exercised; Blind 4D solved `2/2`.
- Historical Blind was not called.
- Astrometry.net web fallback was not called.
- 4D partial coverage was logged through the six configured D50 indexes.

Blind 4D selected:

- BADHINT: `d50_2644`
- NOHINT: `d50_2823`

## Status

```text
pipeline automatisé validé
GUI manuelle non validée
```

Manual GUI validation remains a future task. The P2B-0 global gate is still
blocked by the external Near corpus integrity failure recorded in
`p2b0_external_baseline_report.md`.
