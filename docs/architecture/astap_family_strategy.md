# ASTAP Family Strategy

P1A studies catalogue families as a product architecture question. It does not change any solver defaults or catalogue readers.

## Supported Families in Code

`zewcs290.catalog290.FAMILY_SPECS` currently supports:

| Family | Format | Announced density / role | Magnitude band | Color | Layout | Current runtime support |
| ------ | ------ | ------------------------ | -------------- | ----- | ------ | ----------------------- |
| `d05` | `.1476`, 5-byte records | Gaia DR3 up to 500 stars/deg^2 | Gaia BP | No | `hnsky_1476` | Supported by `CatalogDB`; not covered by current 4D manifest. |
| `d20` | `.1476`, 5-byte records | Gaia DR3 up to 2000 stars/deg^2 | Gaia BP | No | `hnsky_1476` | Supported by `CatalogDB`; not covered by current 4D manifest. |
| `d50` | `.1476`, 5-byte records | Gaia DR3 up to 5000 stars/deg^2 | Gaia BP | No | `hnsky_1476` | Main validated family for current Near and 4D work. |
| `d80` | `.1476`, 5-byte records | Gaia DR3 up to 8000 stars/deg^2 | Gaia BP | No | `hnsky_1476` | Supported by `CatalogDB`; not covered by current 4D manifest. |
| `v50` | `.1476`, 6-byte records | Gaia Johnson-V plus color | Johnson-V | Yes | `hnsky_1476` | Supported by `CatalogDB`; color is decoded but not a current product differentiator. |
| `g05` | `.290`, 5-byte records | Gaia DR3 wide-field | Gaia BP | No | `hnsky_290` | Supported by `CatalogDB`; hermetic tests skip external real database when absent. |

## Evidence from Current Validation

The validated P0B/P1A evidence is strongest for D50:

- P0B regression foundation uses Near, Blind 4D and pipeline contracts derived from the ZN/P29 reports.
- ZN3.9 qualified Near strict on `142/142` positive images with `generic_fallback_called = 0`.
- The bundled 4D manifest contains only D50 tiles.
- The audited 4D manifest has six D50 indexes: `d50_2602`, `d50_2644`, `d50_2645`, `d50_2702`, `d50_2822`, `d50_2823`.
- Each current 4D index uses `catalog_source = tile_npz:x_deg/y_deg/ra_deg/dec_deg`, originally derived from D50 historical tile NPZs.

This supports D50 as the current primary operational family. It does not prove D50 is sufficient for every field, focal length, sky density or future product mode.

## Family Roles

### D50

Recommended role: primary installed source for the current ZeNear strict and ZeBlind 4D product track.

Reasons:

- already validated by the strongest campaigns;
- density is high enough for the current Seestar-style fields used in regression;
- file format is supported by the existing reader and builders;
- all current 4D indexes derive from D50.

Risks:

- not proven all-sky in 4D;
- not proven sufficient for every very wide field;
- not proven optimal for very dense fields if star caps discard useful structure.

### D05 / D20

Recommended role: optional sparse-family fallback or future lower-cost install tier.

They may be useful for:

- very large fields where dense D50 may be unnecessary;
- low disk footprint installations;
- diagnostic comparisons when D50 produces too many candidates.

They should not be exposed as normal GUI choices unless a concrete user workflow needs them.

### D80

Recommended role: optional dense-family candidate for future small-field or crowded-field studies.

Potential value:

- denser source catalogue may help very small fields or sparse detections where D50 loses useful stars.

Risks:

- higher disk and index cost;
- may require new index construction/validation;
- no current P0B/P1A regression oracle proves it improves product outcomes.

### V50

Recommended role: optional specialist source when Johnson-V/color data becomes useful.

Potential value:

- color-aware filtering or diagnostics in a future solver.

Risks:

- current Near/Blind paths do not use color as a validated product signal;
- 6-byte record handling exists but broad regression coverage is not established.

### G05

Recommended role: optional wide-field support family.

Potential value:

- `.290` layout is intended for wide-field use;
- can reduce star density for large field-of-view solving.

Risks:

- separate layout and format;
- current corpus does not prove wide-field behavior;
- no 4D index family currently derives from G05.

## Product Exposure Model

Distinguish these three concepts:

| Concept | Meaning | User-facing? |
| ------- | ------- | ------------ |
| Physically installed family | Files present on disk under the catalogue library. | Visible only as status/details. |
| Engine-required family | A solver profile can request a family or accept the library default. | Mostly hidden. |
| Product catalogue library | One logical local astronomy library with capabilities and coverage. | Yes. |

The product should present a single `ZeSolver Catalog Library`, not a normal workflow full of D05/D20/D50 choices. Expert/dev panels can show installed families and profile bindings.

## Recommendation

Decision: `D50_PRIMARY_WITH_FALLBACK`.

D50 should remain the primary source family for the validated current chain. The future `CatalogLibrary` should, however, support multiple managed families because:

- the code already supports six families;
- G05 and D80 serve different field-density regimes;
- the current 4D coverage is partial and D50-only;
- no evidence yet proves `D50_ONLY` is safe for all product use.

The library manifest should therefore support multiple source families from day one, while default product profiles can bind to D50 until additional families have their own regression evidence.

## Required Future Evidence Before D50-Only

Do not choose `D50_ONLY` unless future tests show:

- D50 Near succeeds on representative wide-field, small-field, sparse and dense fields;
- D50-derived 4D indexes cover intended blind-solving regions;
- D50 disk footprint is acceptable for installation;
- D05/G05/D80/V50 do not add meaningful robustness for target users;
- WCS validation, not `success=True`, confirms the result.
