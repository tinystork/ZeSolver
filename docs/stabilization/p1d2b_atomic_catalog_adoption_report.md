# P1D-2B Atomic Catalog Adoption Report

Decision: `READY_FOR_P1D3_DIRECT_ASTAP_TO_BLIND4D_BUILDER`

P1D-2B implements explicit atomic writing of a `CatalogLibrary` adoption plan.
It writes only the `manifest_preview` produced by
`CatalogLibraryAdoptionPlan.reference_existing(...)`.  It does not rediscover
resources at commit time, modify solvers, move catalogues, modify ASTAP shards,
modify Blind 4D NPZ indexes, modify the strict 4D manifest, modify the legacy
index, execute repair actions or call builders.

## 1. Objectif

Allow an explicit user action to create or replace:

```text
<library_root>/catalog.json
```

The adoption must be previewable, validated, atomic, idempotent, recoverable and
traceable.

## 2. Etat Git initial et HEAD

Initial state:

```text
git status --short --branch
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

Initial HEAD:

```text
de1bc8e064d813037750e11ab0bbb6ce9a3a673f
```

`git diff --check` was clean.  The branch had previously been reported
`ahead 1`; that commit was not modified, rewritten or overwritten.

## 3. Architecture du writer

Added:

```text
zesolver/catalog_library/atomic_adoption.py
```

Public API:

```python
CatalogLibraryAdoptionWriter.commit(
    plan,
    mode="create",
    expected_existing_sha256=None,
    create_backup=True,
)
```

The writer accepts only a prebuilt `CatalogLibraryAdoptionPlanResult` and writes
the plan preview with canonical JSON formatting.  It does not recalculate
informative fields at commit time.

## 4. Modes CREATE/REPLACE

`CREATE_NEW`:

- requires no existing different `catalog.json`;
- never replaces different content;
- returns `NO_CHANGE` if the exact canonical manifest is already present.

`REPLACE_EXISTING`:

- requires explicit mode;
- requires an existing `catalog.json`;
- requires `expected_existing_sha256`;
- creates or reuses a byte-identical backup before replacement.

## 5. Validation pre-commit

Before writing:

- plan errors are rejected;
- non-adoptable statuses are rejected;
- `manifest_preview` parses through current models;
- preview `library_root` must match the plan root;
- target is exactly `library_root/catalog.json`;
- target symlink is rejected;
- library root must already exist.

After locking:

- existing manifest SHA is rechecked;
- sources, source shards, strict 4D manifest and NPZ indexes are checked for
  disappearance or SHA mismatch where hashes are known;
- FAST mode reports its source mutation detection limit.

## 6. Verrou

The writer uses:

```text
<library_root>/.catalog-adoption.lock
```

It is created exclusively.  Existing lock raises:

```text
CATALOG_ADOPTION_LOCKED
```

The writer removes only the lock it created.  It does not auto-remove stale
locks.

## 7. Ecriture atomique

Sequence:

```text
preflight
→ lock
→ drift check
→ serialize in memory
→ write .catalog.json.<uuid>.tmp in library_root
→ flush + fsync
→ validate temp payload
→ backup when replacing
→ os.replace(temp, catalog.json)
→ fsync directory when supported
→ reopen CatalogLibrary
→ full validation
→ release lock
```

Temporary files are cleaned after controlled failures.

## 8. Sauvegarde

Replacement backup:

```text
catalog.json.backup.<previous_sha256>.json
```

Existing backups are reused only if their bytes match the expected previous
SHA.  Different backup content at that path is a conflict.

## 9. Rollback

If post-replacement validation fails, the writer restores the previous manifest
atomically from a copy of the backup, preserving the backup file, then validates
the restored manifest.  Create-mode post-validation failure removes the invalid
manifest.

## 10. Idempotence

If the canonical serialized preview already matches `catalog.json`, the result
is:

```text
NO_CHANGE
```

No file rewrite, no mtime change, no backup and `files_written = 0`.

## 11. Erreurs stables

Implemented stable errors:

```text
CATALOG_ADOPTION_PLAN_INVALID
CATALOG_ADOPTION_PLAN_HAS_ERRORS
CATALOG_ADOPTION_TARGET_EXISTS
CATALOG_ADOPTION_TARGET_MISSING
CATALOG_ADOPTION_CONFLICT
CATALOG_ADOPTION_LOCKED
CATALOG_ADOPTION_TARGET_SYMLINK
CATALOG_ADOPTION_TEMP_WRITE_FAILED
CATALOG_ADOPTION_VALIDATION_FAILED
CATALOG_ADOPTION_REPLACE_FAILED
CATALOG_ADOPTION_ROLLBACK_FAILED
CATALOG_ADOPTION_READ_ONLY
```

Failures raise `CatalogLibraryAdoptionError`; late transaction failures can
carry a structured result.

## 12. CLI

Added advanced tool:

```text
tools/adopt_catalog_library.py
```

Default mode is preview-only and read-only.  Writing requires `--write`.
Replacement additionally requires `--replace-existing` and
`--expected-existing-sha256`.  There is no interactive confirmation path and no
GUI integration.

## 13. Gestion des chemins

The writer requires an existing library root and writes only inside it.  It
rejects target symlinks and creates temp files in the same directory as the
target.  External resources outside the library are referenced and checked but
not modified.

P1D-2A path tests still cover spaces, non-ASCII-capable paths, Windows/POSIX
separators, traversal and external references.

## 14. Tests d'injection de panne

Covered:

- temp write failure;
- temp fsync failure;
- temp validation failure through post-write validation path;
- conflict before replacement;
- backup creation failure;
- `os.replace` failure;
- post-replacement validation failure;
- rollback success;
- rollback failure;
- existing lock;
- temp cleanup;
- second identical commit.

Assertions verify final manifest presence/absence, final validity, backup
presence, lock release and external-resource immutability for relevant cases.

## 15. Tests cibles

Command:

```text
.venv/bin/python -m pytest
tests/test_catalog_library_manifest.py
tests/test_catalog_library_validation.py
tests/test_catalog_library_adoption_plan.py
tests/test_catalog_library_atomic_writer.py
tests/test_catalog_library_adoption_conflicts.py
tests/test_catalog_library_adoption_rollback.py
tests/test_catalog_library_adoption_idempotence.py
tests/test_catalog_library_adoption_cli.py
tests/test_catalog_library_adopted_runtime.py
tests/test_catalog_library_paths.py
tests/test_catalog_library_fingerprints.py
tests/test_catalog_library_provenance.py
tests/test_catalog_library_repair_actions.py -q
```

Result:

```text
57 passed
```

Mission-indicative subset:

```text
48 passed
```

## 16. Barrieres generales

```text
.venv/bin/python tools/check_core_boundaries.py
core boundary check: OK

.venv/bin/python tools/run_regression_suite.py --hermetic
489 passed, 1 skipped, 9 deselected, 56 warnings
runner status PASS

.venv/bin/python -m pytest -q
489 passed, 10 skipped, 56 warnings

.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests
zesolver.py zewcscleaner.py zeindexcheck.py
OK

git diff --check
OK
```

`zedatabase.py` is absent from this checkout and was not included.

## 17. Validation externe

Read-only resources:

```text
/opt/astap
config/zeblind_4d_experimental_manifest.json
/home/tristan/zesolver_index
```

Temporary library root only:

```text
/tmp/p1d2b external library */ZeSolver Adopted Library
```

External validation:

```text
plan_status: READY_PARTIAL
create_status: CREATED
validate_status: READY_PARTIAL
near_provider: astap_native
near_legacy_index_root: null
blind4d_count: 6
coverage_status: PARTIAL
coverage_tiles: 6
coverage_total: 1476
coverage_fraction: 0.0040650406504065045
all_sky_blind4d: false
nochange_status: NO_CHANGE
replace_status: REPLACED
backup_exists: true
conflict_code: CATALOG_ADOPTION_CONFLICT
```

No FITS smoke solve was run; the runtime resource/provider re-read was completed
from the adopted temporary library.

## 18. Relecture runtime

The adopted manifest was reopened through `CatalogLibrary.open()`.  Product
resource resolution returned:

- Near source rooted at `/opt/astap`;
- ASTAP-native provider selected by `resolve_near_catalog_runtime(..., auto)`;
- no legacy index root in ASTAP-native mode;
- strict 4D manifest referenced correctly;
- six 4D NPZ paths;
- Blind 4D coverage still `PARTIAL`, `6 / 1476`.

## 19. Integrite des ressources externes

Before/after file counts and mtimes were unchanged for:

```text
/opt/astap
indexes/astrometry_4d
/home/tristan/zesolver_index
```

No files were added or modified in the adopted resources.

## 20. Warnings

No new test warning category appeared.  Broad-suite warnings remain the known
categories:

- `multiprocessing.popen_fork.DeprecationWarning`
- `Astropy VerifyWarning`

## 21. Limites

- No automatic startup adoption.
- No GUI integration.
- No strict 4D manifest synthesis from `catalog.json`.
- No repair action execution.
- FAST mode cannot cryptographically detect unhashed source shard changes.
- Direct ASTAP-to-Blind4D building remains P1D-3.

## 22. Etat Git final

Final status includes P1D-2A and P1D-2B changes, uncommitted:

```text
## test...origin/test
 M .gitignore
 M AGENT.md
 M docs/architecture/catalog_library_implementation.md
 M docs/architecture/catalog_manifest_example.json
 M docs/architecture/catalog_manifest_schema.json
 M zesolver/catalog_library/__init__.py
 M zesolver/catalog_library/coverage.py
 M zesolver/catalog_library/manifest.py
 M zesolver/catalog_library/models.py
 M zesolver/catalog_library/validation.py
 M zesolver/catalog_resources.py
?? docs/architecture/catalog_atomic_adoption.md
?? docs/architecture/catalog_provenance_and_adoption.md
?? docs/stabilization/p1d2a_catalog_provenance_plan_report.md
?? docs/stabilization/p1d2b_atomic_catalog_adoption_report.md
?? tests/test_catalog_library_adopted_runtime.py
?? tests/test_catalog_library_adoption_cli.py
?? tests/test_catalog_library_adoption_conflicts.py
?? tests/test_catalog_library_adoption_idempotence.py
?? tests/test_catalog_library_adoption_plan.py
?? tests/test_catalog_library_adoption_rollback.py
?? tests/test_catalog_library_atomic_writer.py
?? tests/test_catalog_library_fingerprints.py
?? tests/test_catalog_library_paths.py
?? tests/test_catalog_library_provenance.py
?? tests/test_catalog_library_repair_actions.py
?? tools/adopt_catalog_library.py
?? zesolver/catalog_library/adoption.py
?? zesolver/catalog_library/atomic_adoption.py
```

No commit and no push were performed.

## 23. Une seule prochaine etape

P1D-3: implement the direct ASTAP-to-Blind4D builder path, using the adopted
provenance and atomic library manifest as the input contract.

## 24. Decision de gate

The P1D-2B gate conditions are satisfied:

- no implicit adoption exists;
- preview mode is read-only;
- CREATE never replaces different content;
- REPLACE is explicit and SHA-guarded;
- writing is atomic;
- temp payload is validated before replacement;
- final manifest is reopened and validated;
- backups are byte-identical;
- rollback works and rollback failure is reported;
- second identical commit is `NO_CHANGE`;
- lock blocks concurrent writes;
- no external resource is modified;
- no builder is called;
- adopted manifest produces expected Near and Blind 4D resources;
- coverage remains `PARTIAL 6/1476`;
- tests and barriers are green;
- external validation is positive.

Decision:

```text
READY_FOR_P1D3_DIRECT_ASTAP_TO_BLIND4D_BUILDER
```
