# ZeBlind Astrometry 4D Experimental Backend

## Summary

`astrometry_ab_code_4d_v1` is an experimental ZeBlind backend that indexes quads in an Astrometry.net-like continuous 4D code space. The historical product backend remains `opposite_edge_ratio_8bit_v1`.

The 4D backend exists because the historical 3D opposite-edge hash can miss useful hypotheses that are present in Astrometry-style AB/C/D code space. P2.3 and P2.4 showed that the runtime 4D route works when its source-list contract is explicit.

## Default State

The backend is OFF by default.

Default product behavior remains:

```text
quad_hash_schema="opposite_edge_ratio_8bit_v1"
blind_astrometry_4d_source_policy="standard_runtime"
blind_star_quality_filter=True
```

`diagnostic_unfiltered` is not a product default. It is limited to the experimental 4D route.

## Activation

Use the 4D route only with an explicit configuration:

```text
quad_hash_schema="astrometry_ab_code_4d_v1"
blind_astrometry_4d_index_paths=[
  "reports/p21_astrometry_ab_code_4d_v1_d50_2823_S_stars2000_q40000.npz",
  "reports/p25_astrometry_ab_code_4d_v1_d50_2822_S_stars2000_q40000.npz",
]
blind_astrometry_4d_source_policy="diagnostic_unfiltered"
blind_astrometry_4d_validation_catalog_policy="union_candidate_tiles"
blind_astrometry_4d_accept_policy="best_within_budget"
blind_astrometry_4d_max_accepts=64
```

CLI equivalent:

```bash
python -m zeblindsolver.zeblindsolver IMAGE.fit \
  --index-root reports/s3_focused_index_20260701_p16_multicase_v3/index \
  --quad-hash-schema astrometry_ab_code_4d_v1 \
  --blind-astrometry-4d-index-paths reports/p21_astrometry_ab_code_4d_v1_d50_2823_S_stars2000_q40000.npz,reports/p25_astrometry_ab_code_4d_v1_d50_2822_S_stars2000_q40000.npz \
  --blind-astrometry-4d-source-policy diagnostic_unfiltered \
  --blind-astrometry-4d-validation-catalog-policy union_candidate_tiles \
  --blind-astrometry-4d-accept-policy best_within_budget \
  --blind-astrometry-4d-max-accepts 64
```

If the schema is left at `opposite_edge_ratio_8bit_v1`, ZeBlind uses the historical backend even if 4D index fields are present. If the schema is set to `astrometry_ab_code_4d_v1` but the explicit 4D index list is empty, missing, incompatible, or yields an empty union catalogue, the 4D route fails explicitly instead of falling back silently to the historical backend.

The legacy single-index knobs `blind_astrometry_4d_index_enabled` and `blind_astrometry_4d_index_path` remain understood for targeted single-tile experiments, but the recommended P2.11 path is `blind_astrometry_4d_index_paths` with a short explicit list.

## Multi-Index Validation

`blind_astrometry_4d_validation_catalog_policy="union_candidate_tiles"` validates each candidate WCS against the deduplicated catalogue union of the explicitly listed 4D indexes. This is useful for fields near tile borders: P2.9/P2.10 showed that `232329` and `232431` were geometrically valid but failed mono-tile validation because useful catalogue support was split between `d50_2823` and `d50_2822`.

The union policy keeps the existing thresholds:

```text
quality_inliers=40
quality_rms=1.2
blind_astrometry_4d_match_radius_px=3.0
```

It is only applied inside the experimental 4D backend.

## Accept Policy

`first_accept` stops at the first candidate that passes the product thresholds.

`best_within_budget` continues within bounded budgets and chooses the best accepted candidate by:

1. accepted product threshold;
2. lower RMS;
3. higher inliers;
4. better geometric coverage;
5. lower candidate rank.

P2.10 showed `best_within_budget` kept `10/10` success and improved RMS on `9/10` M106 mini-corpus cases. P2.12 expanded the bounded M106 corpus to 30 local images: `best_within_budget` solved `28/30`, improved RMS on `27/30`, and had no RMS degradations versus `first_accept`. It remains the recommended experimental policy. Keep `blind_astrometry_4d_max_accepts` bounded.

## Indexes

Current targeted 4D indexes:

```text
reports/p21_astrometry_ab_code_4d_v1_d50_2823_S_stars2000_q40000.npz
reports/p25_astrometry_ab_code_4d_v1_d50_2822_S_stars2000_q40000.npz
```

`d50_2822` is a targeted extension built with the same catalog-side parameters as `d50_2823`: level `S`, `catalog_ring_coverage`, max catalog stars per tile from the reference index, and max catalog quads per tile from the reference index. It is not a full index rebuild.

## Limits

- Experimental 4D is not enabled by GUI or ZeNear.
- `astrometry_like_candidate` remains a separate source-list candidate for diagnostics only.
- No global rescue, fallback, or reranking behavior is added by this backend.
- The current 4D indexes are targeted tile slices, not broad coverage.
- No runtime WCS oracle is used. Astrometry.net WCS may be used only by offline diagnostic reports.
- The runtime does not discover all-sky indexes and does not build missing indexes.
- P2.12 leaves two unresolved M106 cases on the bounded `d50_2823+d50_2822` list:
  - `233828`: best reject `37` inliers / RMS `0.942`, below `quality_inliers=40`;
  - `234013`: best reject `28` inliers / RMS `0.910`, below `quality_inliers=40`.
- Mini-corpus testing should stay bounded until source-list, index coverage, and the remaining low-inlier cases are understood.

## Runtime Cost

P2.12 measured the 30-image bounded M106 corpus through `solve_blind` with:

```text
blind_astrometry_4d_accept_policy="best_within_budget"
blind_astrometry_4d_max_accepts=64
blind_astrometry_4d_max_hypotheses=2000
blind_global_hard_budget_s=45.0
```

Observed runtime:

```text
best_within_budget successes: 28/30
median total: 19.220 s
p95 total: 23.617 s
median validation: 7.621 s
p95 validation: 11.812 s
median tested hypotheses: 187
max_wall_s hits: 0
```

`first_accept` is faster (`12.130 s` median total on P2.12) but often selects a higher-RMS accepted candidate. Use it only when speed matters more than candidate quality.

## Minimal Probe

Run the hardened runtime replay through the main solver:

```bash
python tools/diagnose_p211_4d_experimental_runtime_hardening.py
```

It validates:

- loading the explicit `d50_2823 + d50_2822` 4D index list;
- solving the 10-case bounded M106 mini-corpus via `solve_blind`;
- using `union_candidate_tiles` without lowering thresholds;
- using `best_within_budget` with bounded accepted candidates;
- keeping a bad explicit list (`d50_2822` alone) from producing acceptances on the control set.

Run the bounded 30-image M106 validation:

```bash
python tools/diagnose_p212_4d_m106_30_bounded_validation.py \
  --control-case-limit 10 \
  --bad-tile-control-limit 10
```

It validates:

- the 30 local M106 images with both `first_accept` and `best_within_budget`;
- the explicit `d50_2823 + d50_2822` union-catalogue policy;
- bad-list controls, reversed-order controls, partial-list controls, and strict missing/incompatible index errors;
- runtime budgets and per-policy cost.

For the older product-slice probe:

```bash
python tools/diagnose_p25_4d_experimental_product_slice.py --rebuild-2822
```

It validates:

- loading the `d50_2823` 4D index;
- solving `232350 / d50_2823`;
- solving `232102 / d50_2823`;
- keeping the historical backend selected when the 4D schema is not set;
- building only the targeted `d50_2822` 4D index;
- testing a bounded set of M106 neighbors against `d50_2822`.
