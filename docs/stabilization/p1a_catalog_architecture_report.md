# P1A Catalog Architecture Report

Decision: `READY_TO_IMPLEMENT_CATALOG_LIBRARY`

P1A was completed as an architecture and validation mission. No solver pipeline, threshold, catalogue format, index format, GUI behavior, catalogue file or index file was modified.

## Initial State

Read first, in order:

- `AGENT.md`
- `docs/stabilization/initial_state.md`
- `docs/stabilization/baseline_recovery_report.md`
- `docs/stabilization/repository_reproducibility.md`
- `docs/stabilization/regression_suite.md`
- `docs/stabilization/p0b_regression_foundation_report.md`
- `tests/corpus/README.md`
- `tests/corpus/manifest.json`
- `tests/corpus/oracles/zenear_reference.json`
- `tests/corpus/oracles/zeblind4d_reference.json`
- `tests/corpus/oracles/pipeline_reference.json`

Initial P1A validation before modifications:

```text
.venv/bin/python tools/run_regression_suite.py --hermetic
253 passed, 1 skipped, 6 deselected, 1 warning
runner status PASS

.venv/bin/python -m pytest -q
248 passed, 7 skipped, 1 warning

.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests
OK
```

`git status --short` already contained many P0A/P0B changes and untracked regression files before P1A. P1A did not revert or rewrite them.

## Commands Executed

Key inspection and validation commands:

```bash
grep -RInE "CatalogDB|db_root|index_root|blind_4d|4d|manifest|\\.1476|\\.290|famil(y|ies)|ASTAP|HNSKY" \
  zewcs290 zeblindsolver zesolver.py zesolver tools config tests \
  --include='*.py' --include='*.json' --include='*.md'

.venv/bin/python tools/audit_catalog_library.py \
  --blind4d-manifest config/zeblind_4d_experimental_manifest.json \
  --output-json docs/architecture/blind4d_coverage_audit.json \
  --output-md docs/architecture/blind4d_coverage_audit.md

.venv/bin/python -m pytest tests/test_catalog_library_audit.py -q
.venv/bin/python tools/run_regression_suite.py --hermetic
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests
git diff --check
```

## Consumers Inventory

Created:

- `docs/architecture/catalog_consumers_inventory.md`

Main findings:

- `zewcs290.catalog290.CatalogDB` is the central ASTAP/HNSKY source reader.
- Supported source families are `d05`, `d20`, `d50`, `d80`, `v50`, `g05`.
- `.1476` and `.290` layouts are embedded in `zewcs290.data.layouts.json`.
- `zeblindsolver.db_convert` builds the historical derived index from ASTAP/HNSKY shards.
- Historical index manifests still store absolute `db_root` provenance.
- `zeblindsolver.quad_index_4d.build_experimental_4d_index` derives 4D indexes from historical `tile_npz` data, not directly from raw ASTAP shards.
- `zeblindsolver.index_manifest_4d` strictly validates the runtime 4D manifest and NPZ indexes.
- `zesolver.py` and `PersistentSettings` still expose raw `db_root`, `index_root`, `blind_4d_manifest_path`, and family choices.
- `zesolver.blind4d_runtime` uses `ZEBLIND_4D_MANIFEST`; P0B test conventions used `ZESOLVER_BLIND4D_MANIFEST`. This should be rationalized by the future library/settings boundary.

## Data Flow

Created:

- `docs/architecture/catalog_data_flow.md`

Current flow:

```text
ASTAP/HNSKY shards
  -> CatalogDB
    -> ZeNear direct queries
    -> db_convert
      -> historical tile NPZ + quad hashes
        -> historical backend
        -> 4D index builder
          -> 4D NPZ indexes
            -> 4D manifest
              -> ZeBlind 4D runtime
```

The future architecture should unify management, provenance, validation and coverage. It should not force Near and Blind 4D to use the same runtime binary format.

## Local Installations and Families

P1A found no external data environment variables set in this shell:

```text
ZESOLVER_ASTAP_ROOT=<unset>
ZESOLVER_BLIND4D_MANIFEST=<unset>
ZESOLVER_LEGACY_INDEX_ROOT=<unset>
ZESOLVER_CORPUS_ROOT=<unset>
ZESOLVER_ZN310B_ROOT=<unset>
ZEBLIND_4D_MANIFEST=<unset>
```

The repository does contain the bundled 4D manifest and six NPZ indexes:

```text
config/zeblind_4d_experimental_manifest.json
indexes/astrometry_4d/d50_2602_S_q40000.npz
indexes/astrometry_4d/d50_2644_S_q40000.npz
indexes/astrometry_4d/d50_2645_S_q40000.npz
indexes/astrometry_4d/d50_2702_S_q40000.npz
indexes/astrometry_4d/d50_2822_S_q40000.npz
indexes/astrometry_4d/d50_2823_S_q40000.npz
```

Local report directories also contain historical index roots, but they are generated campaign artifacts and not portable product catalogues.

## 4D Coverage Audit

Created:

- `tools/audit_catalog_library.py`
- `tests/test_catalog_library_audit.py`
- `docs/architecture/blind4d_coverage_audit.json`
- `docs/architecture/blind4d_coverage_audit.md`

The read-only audit verifies:

- manifest schema: `zeblind.astrometry_4d_index_manifest.v1`;
- enabled index count: `6`;
- present index count: `6`;
- SHA256: OK for all six;
- source family: D50 only;
- covered tiles: `d50_2602`, `d50_2644`, `d50_2645`, `d50_2702`, `d50_2822`, `d50_2823`;
- layout tile-count coverage: `6 / 1476`, approximately `0.4065 %`;
- declination range: `+36.0 deg` to `+51.42857143 deg`;
- all-sky coverage: `false`.

Conclusion:

```text
Current ZeBlind 4D coverage is READY_PARTIAL, not READY_FULL.
```

Everything outside those six D50 tiles is not covered by the installed 4D index set.

## ASTAP Family Strategy

Created:

- `docs/architecture/astap_family_strategy.md`

Recommendation:

```text
D50_PRIMARY_WITH_FALLBACK
```

D50 is the validated primary family for the current Near and 4D chain. However, P1A does not prove that D50 alone is sufficient for every field regime. The future library should manage multiple families while product profiles default to the validated D50 path until additional evidence exists.

## CatalogLibrary Specification

Created:

- `docs/architecture/catalog_library.md`

Main decisions:

- `CatalogLibrary` is a management facade, not a solver.
- It owns source/index manifests, validation, coverage and capabilities.
- Near may keep using `CatalogDB`.
- Blind 4D may keep using strict 4D manifests and NPZ indexes.
- GUI should see library status and coverage, not internal index filenames.
- Initial P1B implementation should be adapter-based and reversible.

Proposed statuses:

```text
READY_FULL
READY_PARTIAL
NEAR_ONLY
BLIND4D_ONLY
SOURCE_ONLY
INDEX_BUILD_REQUIRED
INCOMPATIBLE
CORRUPT
MISSING
```

Expected current product status with bundled six 4D indexes:

```text
READY_PARTIAL
```

## Manifest Schema

Created:

- `docs/architecture/catalog_manifest_schema.json`
- `docs/architecture/catalog_manifest_example.json`

The schema describes:

- library metadata;
- source ASTAP/HNSKY families;
- derived indexes;
- coverage;
- checksums;
- provenance;
- status.

The example intentionally models the current six-index 4D set as partial coverage and does not claim all-sky support.

## Migration Strategy

Created:

- `docs/architecture/catalog_migration_matrix.md`

Migration modes:

```text
REFERENCE_EXISTING
IMPORT_COPY
FRESH_INSTALL
```

Preferred first mode:

```text
REFERENCE_EXISTING
```

The future migration should detect existing `db_root`, `index_root`, and 4D manifest settings, then create a common manifest by reference without moving or deleting user data.

## Product UI Boundary

Defined in `docs/architecture/catalog_library.md`.

Normal GUI should show:

- library location;
- status;
- size;
- coverage;
- Near availability;
- Blind 4D availability;
- missing/corrupt data;
- Verify/Install/Complete actions.

Normal GUI should not show:

- direct 4D manifest path;
- NPZ/NPY filenames;
- quad schemas;
- buckets or internal limits;
- individual index paths;
- construction parameters.

## Tests Added

Created:

- `tests/test_catalog_library_audit.py`

Coverage:

- ASTAP filename scan without decoding large source files;
- 4D manifest/index audit without rebuilding indexes;
- SHA mismatch classification as corruption;
- CLI JSON/Markdown output;
- JSON parse sanity for schema/example.

Isolated result:

```text
.venv/bin/python -m pytest tests/test_catalog_library_audit.py -q
5 passed
```

## Final Validation

After P1A changes:

```text
.venv/bin/python tools/run_regression_suite.py --hermetic
253 passed, 1 skipped, 6 deselected, 1 warning
runner status PASS

.venv/bin/python -m pytest -q
253 passed, 7 skipped, 1 warning

.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests
OK

git diff --check
OK
```

Expected skips are external-data related:

- external ASTAP/HNSKY database not found under repository `database/`;
- S50 index/frame not configured;
- `ZESOLVER_BLIND4D_MANIFEST` unset;
- P29 source FITS path not mapped;
- `ZESOLVER_CORPUS_ROOT` unset;
- `ZESOLVER_ZN310B_ROOT` unset.

## Files Created or Modified by P1A

Created:

- `docs/architecture/catalog_consumers_inventory.md`
- `docs/architecture/catalog_data_flow.md`
- `docs/architecture/blind4d_coverage_audit.json`
- `docs/architecture/blind4d_coverage_audit.md`
- `docs/architecture/astap_family_strategy.md`
- `docs/architecture/catalog_library.md`
- `docs/architecture/catalog_manifest_schema.json`
- `docs/architecture/catalog_manifest_example.json`
- `docs/architecture/catalog_migration_matrix.md`
- `docs/stabilization/p1a_catalog_architecture_report.md`
- `tools/audit_catalog_library.py`
- `tests/test_catalog_library_audit.py`

Modified:

- `.gitignore` to track `tools/audit_catalog_library.py` via a targeted exception.

## Risks and Open Questions

- Full ASTAP source roots were not configured in the shell, so P1A measured the bundled 4D set but not a complete local ASTAP installation.
- Current 4D indexes preserve source provenance as absolute report paths; this is useful forensic metadata but not portable library provenance.
- `ZEBLIND_4D_MANIFEST` and `ZESOLVER_BLIND4D_MANIFEST` naming should be resolved during settings/library integration.
- `supported_scale_range_arcsec` remains undeclared in current 4D manifest entries.
- Future checksum strategy for full ASTAP source shards must balance integrity and manifest size.
- D50 is validated as primary, but D50-only remains unproven.

## Final Decision

```text
READY_TO_IMPLEMENT_CATALOG_LIBRARY
```

The next phase can implement a read-only `CatalogLibrary` manifest/types layer and discovery adapters. It should remain non-destructive, keep existing solver paths intact, and use P0B/P1A regression tests before and after every integration step.
