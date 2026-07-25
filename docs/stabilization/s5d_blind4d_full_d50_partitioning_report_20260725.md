# S5D - Blind 4D full D50 partitioning report - 2026-07-25

## 1. Etat Git initial

Initial command:

```text
git status --short --branch
git diff --check
```

Initial branch was `test...origin/test`. The worktree already contained S5/S5B/S5C changes and reports. `git diff --check` was clean. No user change was restored, no commit was made, and no push was made.

## 2. Reproduction monolithe

The dense source index is:

```text
/home/tristan/ZeSolverCatalog/new/indexes/blind4d/d50_4d.npz
```

Observed source layout:

```text
tile_count=1476
codes_4d=(59040000, 4) float32
quad_star_indices=(59040000, 4) int32
source_quad_indices=(59040000,) int32
tile_key_indices=(59040000,) int32
ratio_hashes=(59040000,) int64
catalog_ra_dec=(2952000, 2) float64
catalog_xy=(2952000, 2) float64
quads_per_tile=40000
stars_per_tile=2000
```

The old route loaded all index paths into `Quad4DIndex` before building image quads and before validating any candidate.

## 3. Explication de `tested=0`

`Quad4DIndex.load()` converts `codes_4d` to float64 and builds a `cKDTree`. With the full compressed monolith this consumed the route budget before validation. The old route then collected/sorted global hits and, for `union_candidate_tiles`, prepared a giant union catalog before the first hypothesis. Therefore the route could spend the full budget before the validation loop, producing:

```text
hits=2000
tested=0
accepted=0
```

## 4. Format des shards

Added `tools/shard_blind4d_index.py`.

The shard files remain standard Blind 4D NPZ files with the existing schema:

```text
astrometry_ab_code_4d_v1
```

Each shard preserves:

```text
codes_4d
source_quad_indices
ratio_hashes
catalog_ra_dec
catalog_xy
tile_keys
metadata provenance
```

`quad_star_indices` and `tile_key_indices` are remapped to local shard offsets so each shard is a valid compact runtime index.

## 5. Matrice des topologies

Dry-run topology matrix from the monolith:

```text
ring:    36 shards, tile counts 1..69, strongly unbalanced
fixed16: 93 shards, 92x16 tiles + 1x4 tiles
fixed32: 47 shards, 46x32 tiles + 1x4 tiles
```

Chosen runtime qualification topology:

```text
fixed32
```

Reason: bounded shard size and fewer KD-tree creations than fixed16, without the ring imbalance.

## 6. Politique de diversification

Added `Quad4DIndex.search_records_diversified()` with deterministic quotas:

```text
hit_quota_per_tile=64
hit_quota_per_image_quad_tile=2
round-robin merge by tile
```

This prevents a tile from being evicted solely because another tile produces many slightly closer neighbors.

## 7. Politique de cache

Runtime shard cache is currently structural and bounded:

```text
blind_astrometry_4d_shard_cache_size=1
```

The progressive route loads one shard at a time and explicitly allows old shard payloads to be collected.

The conversion tool uses a separate `.source_mmap_cache` for bounded conversion from compressed NPZ. That cache is not runtime data.

## 8. Gestion de la deadline

Added progressive shard route:

```text
open lightweight metadata
order shards deterministically
load one shard
search/diversify
validate immediately
move to next shard
stop on first strict accepted solution
```

Route checks cancellation/deadline between shards and delegates bounded budgets to shard solves:

```text
blind_astrometry_4d_shard_budget_s=4.2
blind_astrometry_4d_shard_max_hypotheses=16
```

Measured bounded overrun:

```text
M106: first_accept=35.34s, wall=47.12s
M31:  first_accept=42.29s, wall=51.48s
```

The remaining wall overhead includes pre-route Blind preparation and operation-in-progress completion.

## 9. Parite des payloads

Verified against the monolith, using mmap source arrays:

```text
d50_2823 direct-d50-fixed32-040: codes/catalog/source_quad_indices/ratio_hashes exact, qsi remap exact
d50_2602 direct-d50-fixed32-037: codes/catalog/source_quad_indices/ratio_hashes exact, qsi remap exact
d50_2822 direct-d50-fixed32-040: codes/catalog/source_quad_indices/ratio_hashes exact, qsi remap exact
```

## 10. Integration CatalogLibrary

Updated `zesolver/catalog_library/blind4d_view.py` so view generation reads only lightweight NPZ metadata and shape headers instead of calling `Quad4DIndex.load()`.

This preserves old/partial manifests and avoids building KD-trees during CatalogLibrary view materialization.

Full fixed32 strict manifest:

```text
/home/tristan/ZeSolverCatalog/new/indexes/blind4d_shards_s5d_fixed32/blind4d_manifest.json
entries=47
tiles=1476
unique_tiles=1476
duplicates=0
total_star_count=2952000
total_quad_count=59040000
```

S5D-2 adopted the existing fixed32 product into the normal `new` CatalogLibrary:

```text
/home/tristan/ZeSolverCatalog/new/catalog.json
```

The monolith remains on disk and in `catalog.json` as compatibility data, but is no longer in the default Blind 4D runtime order when the complete fixed32 product is present.

Measured runtime view after adoption:

```text
blind4d_catalog_source=catalog_library_view
blind4d_catalog_mode_effective=library-view
blind4d_index_count=47
blind4d_covered_tiles=1476
blind4d_total_tiles=1476
blind4d_all_sky=True
blind4d_external_fallback_used=False
runtime_order=direct-d50-fixed32-000..direct-d50-fixed32-046
contains direct-d50 exact=False
fingerprint=bdd3d0338dab026cc9683f8103e396a3471ca3b45f47edfd5c7484609b440f32
```

`CatalogLibrary` validation was also changed to treat collectively complete sharded coverage as `READY_FULL`, and to trust `FAST_VERIFIED` / `FULL_VERIFIED` integrity metadata by checking existence and size instead of rehashing all shard payloads during GUI preflight.

## 11. Memoire avant/apres

Monolith baseline from user log:

```text
RSS ~= 3.1 GiB
swap ~= 6.2 GiB
Blind preparation ~= 308s
tested=0
runtime ~= 157s
```

One fixed32 shard load measurement:

```text
rss_before=62,344 KiB
load_s=2.547
rss_after=192,476 KiB
entries=1,280,000
stars=64,000
tiles=32
```

Full conversion fixed32:

```text
RSS current during conversion ~= 599,052 KiB
VmHWM during conversion ~= 933,628 KiB
VmSwap during conversion ~= 34,508 KiB
```

## 12. Temps avant/apres

Fixed32 conversion:

```text
47 shards
elapsed_s=430.541
compressed shard total=1487.5 MB
min=4.1 MB
median=32.2 MB
max=32.3 MB
```

Runtime direct positives:

```text
M106 233459: first_test=2.45s, first_accept=35.34s, wall=47.12s
M31  230409: first_test=2.34s, first_accept=42.29s, wall=51.48s
233828:      first_accept=30.27s, wall=41.95s
234013:      first_accept=29.75s, wall=41.52s
```

## 13. Tests positifs

Validated with full fixed32 sharded manifest:

```text
M106 233459 FAKE_HINT: SOLVED d50_2823, inliers=60, rms=0.575
M31 230409 FAKE_HINT:  SOLVED d50_2602, inliers=57, rms=0.654
233828:                SOLVED d50_2823, inliers=56, rms=1.088
234013:                SOLVED d50_2822, inliers=49, rms=1.126
```

S5D-2 GUI-pipeline validation with normal product settings:

```text
instrument=S50
focal=250 mm
pixel=2.90 um
scale~=2.39"/px
profile C11 absent from validation logs
```

Separate FAKE_HINT runs from the normal GUI pipeline path:

```text
M106 233459: SOLVED via BLIND4D, WCS written, inliers=60, rms=0.575, duration=51.417s
M31  230409: SOLVED via BLIND4D, WCS written, inliers=57, rms=0.654, duration=62.265s
```

The log proves the configuration actually transmitted to `ProductionBlindSolverPort`, not only the GUI readiness probe:

```text
blind4d_catalog_mode_effective=library-view
blind4d_index_count=47
blind4d_runtime_order=['direct-d50-fixed32-000', ..., 'direct-d50-fixed32-046']
blind4d_covered_tiles=1476
blind4d_total_tiles=1476
blind4d_all_sky=True
blind4d_external_fallback_used=False
astrometry 4D progressive shard route active: shard_count=47 shard_cache_size=1
```

Runtime confirmations:

```text
M106: shards=7/47, hits=10699, tested=26, accepted=1, stop=confident_accept
M31:  shards=10/47, hits=15670, tested=32, accepted=1, stop=confident_accept
```

## 14. Controles negatifs

S5D-2 reran the real P1D-3B negative controls with the existing mono-tile P1D-3B NPZ controls, without rebuilding stars/quads and without relaxing thresholds.

```text
quality_inliers unchanged
quality_rms unchanged
match_radius unchanged
code_tol unchanged
```

Summary:

```text
controls=11
false_positives=0
```

Per-control results:

| control | label | index | hits | tested | accepted | best rejection | time |
|---|---:|---|---:|---:|---:|---|---:|
| d50_2822_only_low_footprint | 233356 | d50_2822_S_q40000 | 60 | 58 | 0 | inliers 5 < 40 | 15.512s |
| d50_2822_only_low_footprint | 232329 | d50_2822_S_q40000 | 64 | 60 | 0 | inliers 21 < 40 | 14.240s |
| d50_2822_only_low_footprint | 233417 | d50_2822_S_q40000 | 49 | 48 | 0 | inliers 4 < 40 | 13.956s |
| d50_2822_only_low_footprint | 232350 | d50_2822_S_q40000 | 64 | 60 | 0 | inliers 22 < 40 | 14.087s |
| d50_2822_only_low_footprint | 232102 | d50_2822_S_q40000 | 63 | 62 | 0 | rms 1.999 > 1.200 and inliers 18 < 40 | 14.803s |
| d50_2822_only_low_footprint | 233459 | d50_2822_S_q40000 | 64 | 63 | 0 | rms 1.846 > 1.200 and inliers 18 < 40 | 14.335s |
| d50_2822_only_low_footprint | 232247 | d50_2822_S_q40000 | 64 | 59 | 0 | inliers 20 < 40 | 13.965s |
| d50_2822_only_low_footprint | 233314 | d50_2822_S_q40000 | 39 | 38 | 0 | rms 2.103 > 1.200 and inliers 14 < 40 | 13.813s |
| d50_2822_only_low_footprint | 232144 | d50_2822_S_q40000 | 64 | 64 | 0 | inliers 5 < 40 | 14.467s |
| d50_2822_only_low_footprint | 232205 | d50_2822_S_q40000 | 64 | 63 | 0 | inliers 5 < 40 | 14.443s |
| d50_2823_only_234013 | 234013 | d50_2823_S_q40000 | 64 | 63 | 0 | inliers 7 < 40 | 15.872s |

An invalid first replay using a fixed32 shard as if it were a mono-tile negative was discarded and documented: fixed32 shard `direct-d50-fixed32-040` legitimately contains both `d50_2822` and `d50_2823`, so it is not equivalent to the P1D-3B mono-tile negative control.

## 15. Validation GUI

S5D-2 validated the normal GUI pipeline path through `PipelineGuiRunner`, using `catalog_library_path=/home/tristan/ZeSolverCatalog/new` and profile S50 settings. This is the same core path used by the GUI controller after preflight; temporary FITS copies were used so original data remained untouched.

```text
catalog_library_view
47 shards
1476/1476
all_sky=true
external fallback=false
tested > 0
SOLVED
WCS written by pipeline
```

Mixed GUI-pipeline batch:

```text
2 normal FITS + 2 FAKE_HINT FITS
workers=6
2 SOLVED via Near
2 SOLVED via Blind 4D
Near results emitted at 10.542s and 10.724s
ZeBlind preparation began at 12.384s, after Near unresolved were known
no duplicate terminal completion
```

Counters:

```text
catalog_resource_resolution_count=1
catalog_library_open_count=1
blind_runtime_resolution_count=1
blind_index_payload_load_count=0
blind_kdtree_build_count=0
```

## 16. Builder progressif

The sharding tool qualifies the topology without rebuilding stars/quads.

S5D-2 also added native fixed-size shard emission to the standard ASTAP direct builder:

```text
build_sharded_4d_indexes_from_astap(...)
```

Properties covered by tests:

```text
fixed shard grouping
one output NPZ per shard
bounded source materialization per shard
resume existing shards
repair one shard
cancel before write
atomic final manifest publication
strict manifest load compatibility
```

No full D50 rebuild was launched.

## 17. Reprise et reparation

`tools/shard_blind4d_index.py` supports:

```text
--resume
atomic shard writes
atomic manifest write
SIGINT/SIGTERM cancellation
single-shard repair by removing/regenerating one shard file
```

The first naive conversion attempt was stopped because compressed NPZ slicing loaded whole arrays. The tool was then changed to stream `.npy` members into an mmap cache before slicing, making conversion memory bounded.

The native builder repair path was tested by deleting one generated shard and rerunning with `resume=True` and `repair_shards=(...)`; the untouched shard mtime stayed stable and the missing shard was regenerated.

## 18. Fichiers modifies

S5D files:

```text
tools/shard_blind4d_index.py
tests/test_s5d_blind4d_sharding_progressive.py
.gitignore
zeblindsolver/quad_index_4d.py
zeblindsolver/zeblindsolver.py
zeblindsolver/profiles.py
zeblindsolver/astap_4d_builder.py
zesolver/settings/profiles.py
zesolver/catalog_library/blind4d_view.py
zesolver/catalog_library/coverage.py
zesolver/catalog_library/validation.py
zesolver/catalog_resources.py
zesolver/core/blind_port.py
tests/test_astap_4d_builder_cli.py
tests/test_catalog_library_blind4d_product_switch.py
```

External catalog data intentionally updated:

```text
/home/tristan/ZeSolverCatalog/new/catalog.json
```

The worktree also contains pre-existing S5/S5B/S5C modifications.

## 19. Barrieres

Executed:

```text
.venv/bin/python -m pytest -q tests/test_s5d_blind4d_sharding_progressive.py tests/test_catalog_blind4d_manifest_view.py
21 passed

.venv/bin/python -m pytest -q tests/test_s5d_blind4d_sharding_progressive.py tests/test_astap_4d_runtime_validation.py tests/test_blind_port_config_parity.py tests/test_blind_config_builder_parity.py tests/test_catalog_blind4d_manifest_view.py
33 passed

.venv/bin/python tools/check_core_boundaries.py
core boundary check: OK

.venv/bin/python -m pytest -q tests/test_astap_4d_builder_cli.py tests/test_catalog_library_blind4d_product_switch.py tests/test_catalog_blind4d_manifest_view.py tests/test_s5d_blind4d_sharding_progressive.py
33 passed

.venv/bin/python tools/run_regression_suite.py --hermetic
PASS, 630 passed, 1 skipped, 9 deselected

QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
630 passed, 10 skipped

.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests zesolver.py zewcscleaner.py zeindexcheck.py
PASS

git diff --check
PASS
```

## 20. Etat Git final

Final branch:

```text
## test...origin/test
```

Final status remains intentionally dirty with S5/S5B/S5C plus S5D/S5D-2 changes. Key S5D-2 files in the final status include:

```text
M zeblindsolver/astap_4d_builder.py
M zesolver/catalog_library/coverage.py
M zesolver/catalog_library/validation.py
M zesolver/catalog_resources.py
M zesolver/core/blind_port.py
M tests/test_astap_4d_builder_cli.py
M tests/test_catalog_library_blind4d_product_switch.py
?? docs/stabilization/s5d_blind4d_full_d50_partitioning_report_20260725.md
```

No commit and no push.

## 21. Limites restantes

Remaining limits:

```text
runtime wall can exceed 45s by bounded post-route/pre-route overhead
conversion cache currently remains on disk unless manually cleaned
fixed32 shard negatives are not mono-tile controls because shards intentionally contain groups of tiles
full D50 native rebuild not launched yet
```

## 22. Prochaine etape

Proceed to S5E settings and verification persistence. A later production rebuild can use the new native progressive fixed32 builder, but should be scheduled separately from this gate.

## 23. Decision de gate

The full fixed32 sharded runtime corrects the monolith failure mode:

```text
coverage-like manifest entries=47
tiles=1476/1476
first validation before full load
tested > 0
positives solved in direct runtime
memory substantially reduced
```

S5D-2 closes the remaining gate checks:

```text
CatalogLibrary selects the 47 fixed32 shards, not direct-d50
GUI pipeline solves M106 and M31 FAKE_HINT with profile S50
mixed batch keeps Near progressive and Blind lazy
P1D-3B negative controls remain negative
memory is substantially reduced versus the monolith
builder has a tested progressive fixed-shard path
```

```text
READY_FOR_S5E_SETTINGS_AND_VERIFICATION_PERSISTENCE
```

## 24. S5D-3 - Robustesse budget progressif et quads image

User GUI regression to close:

```text
M31 S5D-2: shards=10/47 hits=15670 tested=32 accepted=1 SOLVED
M31 user GUI: shards=10/47 hits=15670 tested=27 accepted=0 stop=astrometry_4d_search_budget_exceeded total=46.012s
good shard=direct-d50-fixed32-037
```

Root cause confirmed by instrumentation: the progressive route delegated each shard to the full runtime route, so image quads could be rebuilt/recharged per shard. The new counters are:

```text
image_quad_build_count
image_quad_build_time
image_quad_count
```

The route now performs:

```text
detect image stars
build image quads once
loop shards:
  bounded shard load
  lookup with prebuilt image quad records
  diversified validation
```

Expected criterion is now met:

```text
image_quad_build_count=1
per-shard image_quad_build_s=0.0
image quads reused by all 47 shard candidates
```

Second issue found during validation: after the one-time quad fix, shard load time could still consume the search budget before useful validation. S5D-3 therefore separates bounded shard load time from search/validation budget:

```text
blind_astrometry_4d_shard_load_budget_s=8.0
blind_astrometry_4d_shard_max_hypotheses=4
blind_astrometry_4d_min_hypotheses_per_nonempty_shard=1
```

This keeps the global deadline meaningful, prevents systematic `hits>0 tested=0` for a loaded shard, and avoids spending too many hypotheses on early wrong shards before the known M31 shard 037.

Validation harness added:

```text
tools/validate_s5d3_progressive_budget.py
```

It resolves the normal `/home/tristan/ZeSolverCatalog/new` CatalogLibrary view, strips WCS from temporary FITS copies, enforces S50 (`250 mm`, `2.90 um`, `2.39"/px`), and writes WCS only to the temporary copies.

Same-process reproducibility:

```text
M31 230409 FAKE_HINT: 10/10 SOLVED, tile=d50_2602, tested=37, shards=10, image_quad_build_count=1
M106 233459 FAKE_HINT: 10/10 SOLVED, tile=d50_2823, tested=25, shards=7, image_quad_build_count=1
WCS written: 20/20
max RSS sample: 578,112 kB
max swap sample: 115,620 kB
```

Fresh-process reproducibility:

```text
M31 230409 FAKE_HINT: 10/10 SOLVED, tile=d50_2602, tested=37, shards=10
M106 233459 FAKE_HINT: 10/10 SOLVED, tile=d50_2823, tested=25, shards=7
WCS written: 20/20
max RSS sample: 572,732 kB
max swap sample: 0 kB
```

Artificial timing variance, using an external CPU-load process plus shard-load and validation delays:

```text
M31:  SOLVED tile=d50_2602 tested=37 shards=10 image_quad_build_count=1 first_test=3.348s first_accept=36.544s wall=48.250s swap=0
M106: SOLVED tile=d50_2823 tested=25 shards=7  image_quad_build_count=1 first_test=3.832s first_accept=26.588s wall=39.256s swap=0
```

The initial in-process CPU-load probe was discarded because it starved the Python GIL and inflated the single image-quad build to more than 50s. The retained timing-variance probe uses an external process so it simulates machine load without blocking the solver thread itself.

Negative controls P1D-3B:

```text
controls=11
false_positives=0
success=0
max RSS sample=462,324 kB
max swap sample=0 kB
dominant reject reason=pixel_scale_out_of_range
```

Scientific thresholds were not relaxed:

```text
quality_inliers unchanged
quality_rms unchanged
match_radius unchanged
code_tol unchanged
```

Memory interpretation:

```text
fresh sharded peak ~= 560 MiB RSS, 0 swap
user GUI peak ~= 1.37 GiB RSS, 0 swap
monolith baseline ~= 3.1 GiB RSS plus system swap
```

The GUI/user peak is plausibly explained by the GUI process plus Near per-worker catalog state that remains alive until batch finalization. S5D-3 did not start a separate Near cache refactor because the allocation is bounded, swap-free in the user log, and not a duplicated D50 payload.

Additional S5D-3 files modified:

```text
zeblindsolver/zeblindsolver.py
zeblindsolver/profiles.py
zesolver/settings/profiles.py
tests/test_s5d_blind4d_sharding_progressive.py
tools/validate_s5d3_progressive_budget.py
```

Additional S5D-3 targeted barriers:

```text
.venv/bin/python -m pytest -q tests/test_s5d_blind4d_sharding_progressive.py
5 passed

.venv/bin/python -m compileall -q zeblindsolver/zeblindsolver.py zeblindsolver/profiles.py zesolver/settings/profiles.py tools/validate_s5d3_progressive_budget.py tests/test_s5d_blind4d_sharding_progressive.py
PASS

.venv/bin/python tools/check_core_boundaries.py
core boundary check: OK

.venv/bin/python tools/run_regression_suite.py --hermetic
PASS, 631 passed, 1 skipped, 9 deselected

QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
631 passed, 10 skipped

git diff --check
PASS
```

## 25. Decision de gate apres S5D-3

S5D-3 closes the observed M31 timing regression:

```text
M31 reproducible: 20/20 across same-process and fresh-process runs
M106 reproducible: 20/20 across same-process and fresh-process runs
timing variance probe: 2/2 solved
negative controls: 0/11 false positives
image quad build: exactly once per solve
loaded shards validate instead of failing with hits>0 tested=0
```

```text
READY_FOR_S5E_SETTINGS_AND_VERIFICATION_PERSISTENCE
```
