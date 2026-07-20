# Solver Profiles

P2A introduces immutable profile identifiers. Future threshold changes must add a
new profile id instead of modifying v1.

## `zenear-v1`

Captures the current Near baseline:

- strict ASTAP-ISO enabled;
- quality inliers `60`;
- quality RMS `1.0`;
- pixel tolerance `3.0`;
- RANSAC trials `1200`;
- image/catalog star caps `800` / `2000`;
- search radius scale `1.8`, attempts `3`;
- tile cap `48`;
- detection baseline `k_sigma=4.5`, `min_area=8`, `max_labels=1200`;
- immediate blind fallback remains the pipeline default.

## `zeblind4d-v1`

Captures the current product 4D profile:

- backend profile `zeblind_4d_experimental`;
- base app blind values `500` stars, `8000` quads, `10` candidates;
- validation `40` inliers, RMS `1.2`, pixel tolerance `2.5`;
- 4D schema `astrometry_ab_code_4d_v1`;
- 4D quad sources `120`, max quads `2500`;
- max hypotheses `2000`, max accepts `64`;
- 4D route budget `45.0s`;
- validation catalogue policy `union_candidate_tiles`;
- accept policy `best_within_budget`;
- source policy `diagnostic_unfiltered`;
- image strategy `log_spaced`;
- code tolerance `0.015`;
- max hits `2000`, per-image-quad cap `8`;
- `index_scale_overlap_prefilter` remains disabled by default.

## `pipeline-v1`

Captures the current orchestration:

- order: Near, Blind 4D, optional Astrometry.net;
- no deferred blind fallback by default;
- blind skip-if-valid is true;
- historical blind is diagnostic-only;
- cancellation remains cooperative.

Profile ids are logged through the P2A assembly metadata.
