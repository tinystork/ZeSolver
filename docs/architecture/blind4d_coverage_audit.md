# Catalog Library Audit

Generated: `2026-07-17T01:10:07.734577+00:00`

## P1A Verdict

The current ZeBlind 4D coverage is **not all-sky**.

It is a verified partial D50 coverage set:

- six enabled indexes;
- six present NPZ files;
- SHA256 verified for all six;
- six D50 source tiles covered out of the `hnsky_1476` layout's 1476 D50 tile slots;
- approximate tile-count coverage: `0.004065` (`0.4065 %`);
- declination span covered by the selected tiles: `+36.0 deg` to `+51.42857143 deg`;
- no corrupt or missing enabled index was detected.

This is suitable for bounded validation on the installed M31/M106/NGC6888-style regions that match these tiles. It must not be presented as complete blind coverage.

## Summary

- ASTAP status: `NOT_CONFIGURED`
- Blind 4D status: `READY_PARTIAL`
- Legacy index status: `NOT_CONFIGURED`

## ASTAP Families

| Family | Status | Format | Tiles | Layout Tiles | Dec Range | Size |
| --- | --- | --- | ---: | ---: | --- | ---: |

## Blind 4D Indexes

| ID | Status | Tiles | Stars | Quads | SHA | Source |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `d50_2823_S_q40000` | `PRESENT` | 1 | 2000 | 40000 | ok | `tile_npz:x_deg/y_deg/ra_deg/dec_deg` |
| `d50_2822_S_q40000` | `PRESENT` | 1 | 2000 | 40000 | ok | `tile_npz:x_deg/y_deg/ra_deg/dec_deg` |
| `d50_2644_S_q40000` | `PRESENT` | 1 | 2000 | 40000 | ok | `tile_npz:x_deg/y_deg/ra_deg/dec_deg` |
| `d50_2645_S_q40000` | `PRESENT` | 1 | 2000 | 40000 | ok | `tile_npz:x_deg/y_deg/ra_deg/dec_deg` |
| `d50_2602_S_q40000` | `PRESENT` | 1 | 2000 | 40000 | ok | `tile_npz:x_deg/y_deg/ra_deg/dec_deg` |
| `d50_2702_S_q40000` | `PRESENT` | 1 | 2000 | 40000 | ok | `tile_npz:x_deg/y_deg/ra_deg/dec_deg` |

## Blind 4D Coverage

- Status: `PARTIAL`
- Enabled indexes: `6`
- Present indexes: `6`
- Enabled tiles: `6`
- Declination range: `36.0` to `51.42857143`
- All sky: `False`
- Family coverage:
  - `d50`: `6 / 1476` layout tiles, fraction `0.0040650406504065045`.
- Covered tiles:
  - `d50_2602`
  - `d50_2644`
  - `d50_2645`
  - `d50_2702`
  - `d50_2822`
  - `d50_2823`

## Missing Coverage

Everything outside the six listed D50 tiles is outside the current 4D index set. In product terms:

- no southern sky 4D coverage is installed;
- no polar all-sky 4D coverage is installed;
- no D05/D20/D80/V50/G05 4D family is installed;
- no complete D50 all-sky 4D index family is installed;
- `supported_scale_range_arcsec` is not declared per index, so scale support should not be advertised beyond validated runtime behavior.

The correct product status is `READY_PARTIAL`, not `READY_FULL`.

## Issues

- None
