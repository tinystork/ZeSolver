# P1D-2A Catalog Provenance Plan Report

Decision: `READY_FOR_P1D2B_ATOMIC_CATALOG_ADOPTION`

P1D-2A defines and implements the additive provenance contract and a
non-destructive adoption planner for existing CatalogLibrary installations.  It
does not change ZeNear, ZeBlind, solver thresholds, runtime routing, GUI
behavior, ASTAP shards, strict 4D manifests, NPZ indexes or historical indexes.

## 1. Objectif

Implement enough manifest/provenance/adoption structure for `CatalogLibrary` to
describe ASTAP/HNSKY sources, Blind 4D derived indexes, source/index links,
integrity, deterministic rebuild parameters, compatibility resources and repair
prescriptions.  The planner produces an in-memory `catalog.json` preview only;
atomic writing is reserved for P1D-2B.

## 2. Etat Git initial

Initial project checks:

```text
git status --short --branch
## test...origin/test [ahead 1]

git diff --check
OK
```

## 3. Audit du schema existant

Existing public models lived in `zesolver/catalog_library/models.py`:

- `CatalogLibrary`, `CatalogManifest`, `CatalogSource`, `CatalogIndex`;
- `CatalogStatus`, `CatalogDataStatus`, `CoverageStatus`, `IssueSeverity`;
- Near/Blind runtime descriptors and validation reports.

Existing required manifest fields were:

```text
schema_version, library_id, sources, derived_indexes, coverage, integrity,
provenance, status
```

The current supported schema version was `1`.  The loader rejected unknown
schema versions, rejected environment-variable placeholders in persisted paths,
allowed explicit absolute external references, and rejected relative managed
paths escaping `library_root`.  SHA256 validation was already conditional: only
files with a declared SHA were hashed.

## 4. Retrocompatibilite

The manifest remains `schema_version = 1`.  New fields are optional and parsed
additively.  Old manifests still load and produce the same Near and Blind 4D
runtime descriptors.

No existing field meaning was changed.  Partial Blind 4D coverage remains
partial.  Historical indexes are not promoted into product indexes.

## 5. Modele de provenance ASTAP

`CatalogSource` now supports optional:

```text
families, mode, path_policy, shards, layout_fingerprint, reader_version,
fingerprint_policy
```

Shard entries record relative source-root paths, family, tile code, size, optional
SHA256, informative mtime and verification status.

## 6. Modele de provenance Blind 4D

`CatalogIndex` now supports optional:

```text
category, families, derived_files, source_file_refs, build_parameters,
parameter_status, provenance_fingerprint, reconstruction_status
```

Known runtime parameters are retained.  Missing reconstruction parameters are
not invented; current historical tile-NPZ-derived indexes are marked
`PARTIAL` for deterministic rebuild metadata.

## 7. Canonicalisation

`canonical_provenance_fingerprint()` hashes canonical JSON:

- recursive sorted keys;
- compact separators;
- ASCII output;
- generated timestamps and mtimes excluded;
- stable scientific/provenance fields only.

Managed relative paths do not become machine-absolute in provenance
fingerprints.  File SHA256 remains separate integrity metadata.

## 8. Politique FAST/FULL

`fast` inventories files, sizes, families, tile codes and layout consistency
without source SHA256.  It records `FAST_VERIFIED`.

`full` hashes source shards and records `FULL_VERIFIED` for hashed shards.  Full
hashing is explicit and supports a progress callback.

Status vocabulary now includes:

```text
UNVERIFIED, FAST_VERIFIED, FULL_VERIFIED, MISMATCH, MISSING, CORRUPT
```

## 9. Compatibilite historique

Legacy historical indexes are represented as `compatibility_resources` with
`category = compatibility`.  They may reference `manifest.json`, `tiles/` and
`hash_tables/`, but they are not ASTAP sources, not normal Blind 4D resources,
and not proof of Blind 4D coverage.

## 10. API du planificateur

Added:

```python
CatalogLibraryAdoptionPlan.reference_existing(
    library_root=...,
    astap_roots=...,
    blind4d_manifest=...,
    legacy_index_root=...,
    fingerprint_policy="fast",
)
```

The result exposes:

```text
status, sources, indexes, compatibility_resources, coverage, warnings, errors,
repair_actions, manifest_preview, telemetry
```

## 11. Preview du manifeste

`manifest_preview` is deterministic for identical inputs, serializable, ordered
canonically and validable by `CatalogLibrary` when written only to a temporary
test directory.  It may include informative `created_at`, but fingerprints do
not depend on generated timestamps.

## 12. Actions de reparation

Implemented descriptive actions:

```text
VERIFY_SOURCE_SHA256
LOCATE_MISSING_SOURCE
LOCATE_MISSING_INDEX
REBUILD_BLIND4D_INDEX
REGENERATE_STRICT_4D_MANIFEST
REMOVE_STALE_COMPATIBILITY_REFERENCE
RECOMPUTE_COVERAGE
```

`REBUILD_BLIND4D_INDEX` is explicitly `P1D-3_OR_LATER` and never automatic in
P1D-2A.

## 13. Gestion des chemins

Managed relative paths remain confined to `library_root`.  POSIX and Windows
separators are normalized for managed relative paths.  Traversal such as
`..\outside` is rejected.  External references must be absolute and may live
outside the library root.

## 14. Fichiers modifies

Code:

- `zesolver/catalog_library/adoption.py`
- `zesolver/catalog_library/__init__.py`
- `zesolver/catalog_library/manifest.py`
- `zesolver/catalog_library/models.py`
- `zesolver/catalog_library/validation.py`

Docs:

- `docs/architecture/catalog_provenance_and_adoption.md`
- `docs/architecture/catalog_library_implementation.md`
- `docs/architecture/catalog_manifest_schema.json`
- `docs/architecture/catalog_manifest_example.json`
- `docs/stabilization/p1d2a_catalog_provenance_plan_report.md`

Tests:

- `tests/test_catalog_library_adoption_plan.py`
- `tests/test_catalog_library_fingerprints.py`
- `tests/test_catalog_library_paths.py`
- `tests/test_catalog_library_provenance.py`
- `tests/test_catalog_library_repair_actions.py`

## 15. Tests cibles

Command:

```text
.venv/bin/python -m pytest tests/test_catalog_library_manifest.py
tests/test_catalog_library_validation.py tests/test_catalog_library_status.py
tests/test_catalog_library_adapters.py tests/test_catalog_resource_resolution.py
tests/test_catalog_library_provenance.py tests/test_catalog_library_fingerprints.py
tests/test_catalog_library_adoption_plan.py tests/test_catalog_library_repair_actions.py
tests/test_catalog_library_paths.py -q
```

Result:

```text
51 passed
```

## 16. Barrieres generales

```text
.venv/bin/python tools/check_core_boundaries.py
core boundary check: OK

.venv/bin/python tools/run_regression_suite.py --hermetic
465 passed, 1 skipped, 9 deselected, 56 warnings
runner status PASS

.venv/bin/python -m pytest -q
465 passed, 10 skipped, 56 warnings

.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests
zesolver.py zewcscleaner.py zeindexcheck.py
OK

git diff --check
OK
```

Known warning categories only:

- `multiprocessing.popen_fork.DeprecationWarning`
- `Astropy VerifyWarning`

`zedatabase.py` is absent from this checkout and was not included.

## 17. Validation externe

Read-only FAST adoption plan used:

```text
ASTAP:        /opt/astap
Blind 4D:     config/zeblind_4d_experimental_manifest.json
Legacy index: /home/tristan/zesolver_index
```

Result:

```text
status: READY_PARTIAL
source astap-d50: 1476 shards, coverage FULL, FAST_VERIFIED
source_hashed_count: 0
indexes: 6 Blind 4D NPZ entries
index_hashed_count: 6 via strict 4D manifest validation
Blind 4D coverage: PARTIAL, all_sky false, 6 / 1476 D50 tiles
covered tiles: d50_2602, d50_2644, d50_2645, d50_2702, d50_2822, d50_2823
legacy-index: compatibility, FAST_VERIFIED
errors: none
warnings: INDEX_PARAMETERS_INCOMPLETE x6, BLIND4D_COVERAGE_PARTIAL
repair actions: VERIFY_SOURCE_SHA256, REBUILD_BLIND4D_INDEX x6,
RECOMPUTE_COVERAGE
files_written: 0
builder_called: false
```

No FULL source audit was run; source SHA256 was intentionally not computed in
FAST mode.

## 18. Absence d'ecriture et de builder

Automated tests assert no source mtimes change during adoption.  Telemetry
reports `files_written = 0` and `builder_called = false`.  No builder module is
called by `CatalogLibraryAdoptionPlan`.

## 19. Warnings

No new test warning category appeared.  External adoption warnings are expected:
current strict 4D indexes lack enough historical parameters for deterministic
rebuild and remain partial coverage.

## 20. Limites

- P1D-2A does not write `catalog.json`.
- P1D-2A does not synthesize a strict 4D runtime manifest from `catalog.json`.
- Current historical indexes have `PARTIAL` rebuild provenance.
- Exact direct ASTAP-to-4D rebuild remains P1D-3.
- No GUI surface was added.

## 21. Etat Git final

Final status:

```text
## test...origin/test
 M docs/architecture/catalog_library_implementation.md
 M docs/architecture/catalog_manifest_example.json
 M docs/architecture/catalog_manifest_schema.json
 M zesolver/catalog_library/__init__.py
 M zesolver/catalog_library/manifest.py
 M zesolver/catalog_library/models.py
 M zesolver/catalog_library/validation.py
?? docs/architecture/catalog_provenance_and_adoption.md
?? docs/stabilization/p1d2a_catalog_provenance_plan_report.md
?? tests/test_catalog_library_adoption_plan.py
?? tests/test_catalog_library_fingerprints.py
?? tests/test_catalog_library_paths.py
?? tests/test_catalog_library_provenance.py
?? tests/test_catalog_library_repair_actions.py
?? zesolver/catalog_library/adoption.py
```

No commit and no push were performed.

## 22. Une seule prochaine etape

P1D-2B: implement atomic manifest adoption writing for an explicit user-selected
library root, using this plan preview as input, with no implicit adoption.

## 23. Gate

P1D-2A satisfies the gate conditions:

- old manifests remain compatible;
- new provenance metadata is representable;
- ASTAP sources are inventoried without movement;
- Blind 4D indexes are linked to sources and tiles;
- partial coverage remains exact;
- fingerprints are deterministic and timestamp-independent;
- FAST and FULL are distinct;
- adoption does not modify files;
- repair actions are descriptive only;
- no builder is called;
- tests and barriers are green;
- external FAST validation produced a coherent plan;
- architecture boundaries are preserved.

Decision:

```text
READY_FOR_P1D2B_ATOMIC_CATALOG_ADOPTION
```
