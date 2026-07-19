# P1D-4A — Vue stricte Blind 4D générée par CatalogLibrary

Decision: `READY_FOR_P1D4B_BLIND4D_PRODUCT_SWITCH`

P1D-4A adds deterministic generation and optional atomic materialization of
the strict Blind 4D manifest view from `CatalogLibrary`.  It does not switch
the product runtime, replace the product strict manifest, replace any NPZ,
rebuild indexes, change ZeNear/ZeBlind algorithms, change thresholds, change
GUI behavior or start P1D-4B.

## 1. Objectif

Allow:

```text
catalog.json -> CatalogLibrary -> CatalogIndex Blind 4D
-> CatalogBlind4DManifestView -> load_4d_index_manifest()
```

The current strict manifest remains an oracle and compatibility input, but is
no longer the only object that can describe the Blind 4D runtime contract.

## 2. Etat Git initial et HEAD

Initial state:

```text
git status --short --branch
## test...origin/test

git rev-parse HEAD
f8e92b3b6e80de16f04c9f55b5841449c7701446

git diff --check
OK
```

`AGENT.md` stated P1D-2A, P1D-2B, P1D-3A and P1D-3B were complete, P1D-4 was
next, and P3B remained paused.

## 3. Audit du manifeste strict

`load_4d_index_manifest()` consumes:

- top-level `schema = zeblind.astrometry_4d_index_manifest.v1`;
- `manifest_version = 1`;
- ordered `indexes`;
- entry `id`, enabled flag, `path`, optional `filename`;
- SHA256 integrity;
- `quad_schema = astrometry_ab_code_4d_v1`;
- `index_version`;
- `tile_keys`;
- `star_count`, `quad_count`;
- `sampler_tag`;
- optional `code_tol_recommended`;
- optional `catalog_source`;
- NPZ metadata loaded through `Quad4DIndex.load()`.

It rejects duplicate ids, duplicate paths, duplicate tiles, missing files,
checksum mismatch, schema/version mismatch, tile mismatch, count mismatch and
sampler mismatch.

Fields like `description`, `generated_at`, `priority`, `file_size_bytes`,
`generation_metadata`, `scale_note`, `source_index_root` and scale-range notes
are informative or historical for the strict loader.

## 4. Modele de vue

Added:

```python
CatalogBlind4DManifestView
build_blind4d_manifest_view(catalog_library)
```

The view exposes:

```text
payload, schema, version, entries, coverage, fingerprint, warnings, errors,
source_library_id, source_manifest_fingerprint, telemetry
```

Construction is read-only and in-memory.

## 5. Autorite des donnees

The view is built from `CatalogIndex(engine="blind4d")` objects and their
declared NPZ files.  It does not read the external strict manifest while
generating normal output.

The NPZ file is loaded to validate and emit fields the strict loader verifies
anyway: schema/version, tile keys, star count, quad count and sampler metadata.

## 6. Ordre runtime

Added additive manifest field:

```json
{
  "runtime_order": {
    "blind4d": [
      "d50_2823_S_q40000",
      "d50_2822_S_q40000",
      "d50_2644_S_q40000",
      "d50_2645_S_q40000",
      "d50_2602_S_q40000",
      "d50_2702_S_q40000"
    ]
  }
}
```

The adoption planner now preserves the strict manifest order in
`runtime_order.blind4d` and keeps `derived_indexes` in that same order.

Old manifests remain readable.  Missing order produces
`BLIND4D_VIEW_RUNTIME_ORDER_MISSING` during view generation rather than a
silent alphabetical or insertion-order guess.

## 7. Champs runtime

Each generated entry contains:

```text
id, enabled, path, filename, quad_schema, index_version, level, tile_keys,
star_count, quad_count, sampler_tag, code_tol_recommended, catalog_source,
sha256, priority, file_size_bytes
```

Checksums come from `CatalogIndex.derived_files` or `integrity.files` and are
verified against the declared file before the entry is emitted.

## 8. Couverture

The view keeps Blind 4D coverage separate from source ASTAP coverage.  The
external validation kept:

```text
coverage = PARTIAL
covered_tiles = 6
total_tiles = 1476
all_sky = false
```

No source all-sky state is promoted into Blind 4D all-sky coverage.

## 9. Empreinte

The view fingerprint covers:

- schema/version;
- ordered entries;
- index ids;
- path policy/value from `catalog.json`;
- SHA256;
- tile keys;
- counts;
- sampler;
- `code_tol_recommended`;
- coverage.

It excludes timestamps, temporary materialization paths, resolved-path
diagnostics and accidental dictionary order.  Tests verify deterministic
repeat builds and changes for order and `code_tol_recommended`.

## 10. Materialisation

`view.materialize(path)` is explicit and optional.  It:

```text
serializes in memory
-> writes a temp file in target directory
-> flushes and fsyncs
-> validates temp via load_4d_index_manifest()
-> os.replace()
-> fsyncs directory when supported
-> reloads final strict manifest
```

Overwrite is refused unless `overwrite=True`.  Controlled failures clean the
temp file.

## 11. Compatibilite avec le manifeste existant

External validation generated a temporary manual direct manifest and a
CatalogLibrary-owned view:

```text
manual: /tmp/p1d4a_library_view_runtime_likp9245/direct_manifest.json
view:   /tmp/p1d4a_library_view_runtime_likp9245/library_view_manifest.json
```

Comparison:

```text
order_match: true
paths_match: true
checksums_match: true
tile_keys_match: true
view_errors: []
view_warnings: []
view_fingerprint: 05a52c168a73c2a67fe44fbc09b87ed3be7b7b11c5fe74b35ef926b320450a19
```

Classification: `SEMANTICALLY_EQUIVALENT` for runtime fields.  Only diagnostic
metadata/description formatting differ.

## 12. Integration adoption

Updated `CatalogLibraryAdoptionPlan.reference_existing()` so the preview
captures:

- strict index order;
- checksums;
- families and source tiles;
- schema/version;
- `code_tol_recommended`;
- coverage status.

The atomic writer was not changed except that it serializes the additive field
already present in `manifest_preview`.  External validation confirmed a second
identical commit returns `NO_CHANGE`.

## 13. Erreurs stables

Implemented stable view errors:

```text
BLIND4D_VIEW_NO_INDEXES
BLIND4D_VIEW_INDEX_MISSING
BLIND4D_VIEW_INDEX_CORRUPT
BLIND4D_VIEW_CHECKSUM_MISMATCH
BLIND4D_VIEW_SCHEMA_UNSUPPORTED
BLIND4D_VIEW_RUNTIME_ORDER_MISSING
BLIND4D_VIEW_RUNTIME_ORDER_DUPLICATE
BLIND4D_VIEW_TILE_DUPLICATE
BLIND4D_VIEW_COVERAGE_INCONSISTENT
BLIND4D_VIEW_PATH_INVALID
BLIND4D_VIEW_MATERIALIZATION_FAILED
```

## 14. Fichiers modifies

Code:

- `zesolver/catalog_library/blind4d_view.py`
- `zesolver/catalog_library/__init__.py`
- `zesolver/catalog_library/adoption.py`
- `zesolver/catalog_library/manifest.py`
- `zesolver/catalog_library/models.py`
- `tools/generate_blind4d_manifest_view.py`

Docs/schema:

- `docs/architecture/catalog_manifest_schema.json`
- `docs/architecture/catalog_manifest_example.json`
- `docs/architecture/catalog_provenance_and_adoption.md`
- `docs/architecture/library_owned_blind4d_manifest.md`
- `docs/stabilization/p1d4a_library_owned_blind4d_manifest_report.md`

Tests:

- `tests/test_catalog_blind4d_manifest_view.py`
- `tests/test_catalog_blind4d_manifest_view_cli.py`
- `tests/test_catalog_library_adoption_plan.py`
- `tests/test_catalog_library_manifest.py`

## 15. Tests cibles

Command:

```bash
.venv/bin/python -m pytest \
 tests/test_catalog_library_manifest.py \
 tests/test_catalog_library_adoption_plan.py \
 tests/test_catalog_library_atomic_writer.py \
 tests/test_catalog_library_blind4d_integration.py \
 tests/test_catalog_blind4d_manifest_view.py \
 tests/test_catalog_blind4d_manifest_view_cli.py \
 tests/test_catalog_resource_resolution.py \
 tests/test_astap_4d_runtime_validation.py \
 tests/test_catalog_library_adopted_runtime.py \
 -q
```

Result:

```text
53 passed
```

## 16. Barrieres generales

```text
.venv/bin/python tools/check_core_boundaries.py
core boundary check: OK

.venv/bin/python tools/run_regression_suite.py --hermetic
PASS, 540 passed, 1 skipped, 9 deselected, 58 warnings

.venv/bin/python -m pytest -q
540 passed, 10 skipped, 58 warnings

.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests zesolver.py zewcscleaner.py zeindexcheck.py
OK

git diff --check
OK
```

Warnings are known categories from previous runs:

- `datetime.utcnow()` deprecation in `zeblindsolver/db_convert.py`;
- multiprocessing fork deprecation warnings in legacy/process tests;
- Astropy FITS card truncation warning in `test_p220_manifest_loader`.

## 17. Validation runtime

Validation used:

```text
direct build source: /tmp/p1d3b_direct_runtime_m9e0xbxr
work dir: /tmp/p1d4a_library_view_runtime_likp9245
ASTAP: /opt/astap
legacy index: /home/tristan/zesolver_index
product strict manifest: config/zeblind_4d_experimental_manifest.json
product indexes: indexes/astrometry_4d
```

Flow:

1. Reused P1D-3B direct indexes after A/B scientific fingerprint recheck.
2. Wrote a temporary manual direct strict manifest.
3. Adopted that direct manifest into a temporary `CatalogLibrary`.
4. Generated and materialized the strict view from `CatalogLibrary`.
5. Loaded both manifests with `load_4d_index_manifest()`.
6. Ran M106 all30 under `first_accept` and `best_within_budget`, comparing
   manual direct manifest vs library-generated view.

Result:

```text
60/60 SAME_SUCCESS_EQUIVALENT_WCS
0 DIRECT_LOSS
0 INVALID_DIRECT_SOLUTION
233828 preserved
234013 preserved
```

Runtime summary:

| policy/mode | success | median total | p95 total | max total |
|---|---:|---:|---:|---:|
| `first_accept:manual` | 30 | 3.593s | 8.473s | 10.639s |
| `first_accept:view` | 30 | 3.572s | 8.236s | 8.619s |
| `best_within_budget:manual` | 30 | 7.967s | 28.839s | 32.853s |
| `best_within_budget:view` | 30 | 7.981s | 28.658s | 32.931s |

The separate P1D-3B negative controls remain the behavioral oracle for the
same direct NPZ paths and reported `0/12` false positives.  P1D-4A's view
validation used identical index paths/checksums/order to the direct manifest,
so no new negative-control surface was introduced.

## 18. Integrite

Before/after snapshots were identical for:

- `/opt/astap`;
- `/home/tristan/zesolver_index`;
- `indexes/astrometry_4d`;
- `config/zeblind_4d_experimental_manifest.json`;
- reused direct index build `direct_build_a`.

All new manifests, catalog previews and FITS copies were under
`/tmp/p1d4a_library_view_runtime_likp9245`.

## 19. Warnings

- FAST adoption does not cryptographically hash ASTAP source shards.
- P1D-4A does not yet switch product runtime to the generated view.
- The current architecture example remains illustrative; adoption-generated
  manifests are the authoritative shape for per-NPZ runtime entries.

## 20. Limites

- Product runtime still points at the external strict manifest until P1D-4B.
- Product NPZ and strict manifest were not replaced.
- No GUI or settings surface was changed.

## 21. Etat Git final

Final `git status --short --branch` after implementation and validation shows
only P1D-4A modifications and new files.  No commit and no push were made.

## 22. Prochaine etape unique

P1D-4B: switch the product Blind 4D runtime to request/use the
CatalogLibrary-owned strict view, with the existing strict manifest retained as
explicit rollback/diagnostic input.

## 23. Decision de gate

All P1D-4A gate criteria are satisfied:

- strict view generated from `CatalogLibrary`;
- no silent external strict-manifest read needed;
- runtime order explicitly owned by the library;
- checksums, families, tiles and runtime parameters complete;
- coverage remains `PARTIAL 6/1476`;
- view fingerprint deterministic;
- materialization explicit and atomic;
- strict loader accepts the view;
- adoption captures required fields;
- runtime with generated view reproduces P1D-3B direct results;
- external resources unchanged;
- tests and barriers green.

```text
READY_FOR_P1D4B_BLIND4D_PRODUCT_SWITCH
```
