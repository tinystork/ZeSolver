# Catalog Atomic Adoption

P1D-2B adds explicit atomic writing for a `CatalogLibrary` adoption plan.  The
writer accepts only an already-built `CatalogLibraryAdoptionPlan` and writes the
plan's `manifest_preview` to:

```text
<library_root>/catalog.json
```

It does not discover resources again, rebuild indexes, synthesize a Blind 4D
runtime manifest, move files, copy ASTAP catalogues, modify external resources,
call builders, import GUI modules, or change solver behavior.

## API

```python
from zesolver.catalog_library import CatalogLibraryAdoptionPlan, CatalogLibraryAdoptionWriter

plan = CatalogLibraryAdoptionPlan.reference_existing(...)

result = CatalogLibraryAdoptionWriter.commit(
    plan,
    mode="create",
    expected_existing_sha256=None,
    create_backup=True,
)
```

The writer serializes `plan.manifest_preview` with canonical project formatting:

- JSON object keys sorted;
- two-space indentation;
- UTF-8;
- final newline.

It does not update timestamps or recompute generated fields during commit.

## Modes

`CREATE_NEW`
: Requires no existing different `catalog.json`.  It never replaces content.
  If the exact canonical manifest is already present, the result is
  `NO_CHANGE`.

`REPLACE_EXISTING`
: Requires an existing `catalog.json`, an explicit
  `expected_existing_sha256`, and a byte-identical backup before replacement.
  If the exact canonical manifest is already present, the result is
  `NO_CHANGE` and no backup is created.

There is no implicit adoption at startup and no implicit replacement.

## Pre-Commit Validation

Before writing, the writer verifies:

- the plan has no blocking errors;
- the plan status can be adopted;
- the manifest preview parses through current `CatalogLibrary` models;
- the preview `library_root` matches the plan root;
- the target path is exactly `library_root/catalog.json`;
- the target is not a symlink;
- the library root already exists and is a directory;
- the mode is valid.

After acquiring the lock, it also checks resource drift:

- source root still exists;
- source shards still exist;
- source shard SHA256 still matches when the plan contains one;
- strict Blind 4D manifest still exists and matches its plan SHA256;
- Blind 4D NPZ files still exist and match declared SHA256;
- existing `catalog.json` SHA still matches the expected replacement SHA.

FAST adoption does not claim cryptographic mutation detection for unhashed
source shards.  Result telemetry includes this limitation.

## Lock

The writer uses:

```text
<library_root>/.catalog-adoption.lock
```

The lock is created with exclusive file creation.  If it already exists, the
writer raises `CATALOG_ADOPTION_LOCKED` and does not remove it.  Controlled
success and failure paths remove the lock they created.

The lock file contains only minimal diagnostic data:

```json
{"pid": ..., "purpose": "catalog_adoption"}
```

## Atomic Sequence

The commit sequence is:

```text
preflight validation
→ exclusive lock
→ drift check
→ serialize manifest in memory
→ write temp file in library_root
→ flush + fsync temp file
→ validate temp payload
→ backup existing manifest when replacing
→ os.replace(temp, catalog.json)
→ fsync directory when supported
→ reopen CatalogLibrary
→ full validation
→ release lock
```

Temporary files are named:

```text
.catalog.json.<uuid>.tmp
```

They are created in the target directory so the replacement stays on the same
filesystem.

## Backup

Replacement creates:

```text
catalog.json.backup.<previous_sha256>.json
```

If that backup already exists, it is reused only when its bytes match the
expected previous SHA.  A different existing backup path is treated as a
conflict.

## Rollback

If post-write validation fails after replacement, the writer restores the old
manifest atomically from a copy of the backup, then validates the restored
manifest.  The backup file is preserved.

For failed create validation, the writer removes the invalid `catalog.json` and
does not leave a usable partial manifest.

## Result

`CatalogLibraryAdoptionResult` exposes:

```text
status
mode
library_root
catalog_path
created
replaced
unchanged
backup_path
manifest_sha256
previous_sha256
lock_used
atomic_replace_used
post_write_validation
rollback_performed
files_written
warnings
errors
telemetry
```

Statuses:

```text
CREATED
REPLACED
NO_CHANGE
FAILED
ROLLED_BACK
```

## Stable Errors

Stable writer errors include:

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

Failures raise `CatalogLibraryAdoptionError`.  Late transaction errors may carry
a structured result on `error.result`.

## CLI

Advanced tool:

```text
tools/adopt_catalog_library.py
```

Default behavior is preview-only and read-only:

```bash
.venv/bin/python tools/adopt_catalog_library.py \
  --library-root /tmp/ZeSolverCatalog \
  --astap-root /opt/astap \
  --blind4d-manifest config/zeblind_4d_experimental_manifest.json \
  --legacy-index-root /home/tristan/zesolver_index \
  --report-json /tmp/adoption_preview.json
```

Writing requires:

```text
--write
```

Replacement additionally requires:

```text
--replace-existing
--expected-existing-sha256 <sha256>
```

There is no interactive confirmation path.

## Guarantees

- No implicit startup adoption.
- Preview mode is read-only.
- CREATE never replaces different content.
- REPLACE requires explicit mode and SHA conflict check.
- Temp file is validated before replacement.
- Final `catalog.json` is reopened and validated.
- Backup bytes are preserved.
- Rollback is tested.
- Second identical commit is `NO_CHANGE`.
- Concurrent writers are blocked by the lock.
- External ASTAP, Blind 4D and legacy resources are not modified.
- No builder is called.

## Non-Guarantees

In FAST mode, unhashed source shard byte changes are not cryptographically
detected.  FULL mode is required for source SHA256 coverage.
