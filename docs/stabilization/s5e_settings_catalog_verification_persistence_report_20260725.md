# S5E - Settings and CatalogLibrary verification persistence report - 2026-07-25

## 1. Etat Git initial

Initial commands:

```text
git status --short --branch
git diff --check
```

Initial state:

```text
## test...origin/test
git diff --check: clean
```

No commit, no push, no D50 rebuild, no threshold or WCS algorithm change.

## 2. Reproduction du retour C11

The audited GUI lifecycle showed two persistence gaps:

1. `wire_settings_tab_callbacks()` connected `presets_combo.currentIndexChanged`
   before finishing persistent preset restoration.
2. `closeEvent()` did not save an instrument snapshot unless the user clicked the
   general settings save button.

That made the following sequence possible:

```text
load settings with last_preset_id=seestar_s50
build combo with default index 0 = c11_0p63_asi533
connect currentIndexChanged too early
apply/repaint later through generic UI lifecycle
close without a targeted instrument save
next launch can present/persist C11 again
```

## 3. Cause exacte

There was no single source of truth for the instrument snapshot. `_read_settings_from_ui()`
read `presets_combo.currentData()` directly, while the GUI had no distinction
between "known preset selected" and "custom values edited after a preset".

CatalogLibrary verification had the opposite issue: path persistence existed, but
verified status lived only in memory (`_catalog_library_validated_resources`).
`_populate_settings_ui()` and `_apply_language()` reset a persisted path to
"unverified" after every new process.

The stale coverage warning came from `build_gui_solve_request_from_legacy_config()`,
which set `blind4d_all_sky=False` instead of using final `catalog_resources`.

## 4. Ordre de chargement avant/apres

Before:

```text
load settings
build widgets
connect preset signal
set saved combo index
apply preset through live signal path
later language refresh can reset CatalogLibrary status to unverified
```

After:

```text
load settings
build widgets
block preset/FOV signals during snapshot restore
apply known preset or custom snapshot atomically
connect preset and FOV callbacks only after initialization
restore CatalogLibrary status through FAST fingerprint/cache
language refresh reuses the same cache restore path
```

## 5. Modele de persistance instrument

The source of truth is now:

```text
_instrument_preset_id
last_fov_focal_mm
last_fov_pixel_um
last_fov_res_w
last_fov_res_h
last_fov_reducer
last_fov_binning
```

Known preset:

```text
last_preset_id=seestar_s50
canonical values restored from zeblindsolver.presets
```

Custom mode:

```text
last_preset_id=None
last_fov_* values preserved
```

Persistence is targeted on preset selection, validated FOV edit completion, and
clean close. It does not call full settings validation for unrelated fields.

## 6. Migrations

`SETTINGS_SCHEMA_VERSION` is now `12`.

Existing settings remain loadable. Instrument fields now use bounded parsing:
invalid or out-of-range fields log a warning and fall back locally without
discarding the rest of the settings file.

Covered cases:

```text
missing settings
legacy settings without preset_id
known preset_id
unknown/custom values
partially corrupt numeric instrument fields
out-of-range numeric instrument fields
```

## 7. Format du cache de verification

New persisted field:

```text
catalog_library_verification
```

Record fields:

```text
canonical_library_path
library_id
catalog_manifest_fingerprint
blind4d_view_fingerprint
verification_level
verification_status
verified_at
verification_schema_version
application_compatibility_version
lightweight_fingerprint
runtime_order
blind4d_index_count
blind4d_covered_tiles
blind4d_total_tiles
blind4d_all_sky
payload_hash_count
inspected_file_count
```

## 8. Fingerprint leger

The FAST fingerprint reads:

```text
catalog.json
referenced manifest paths
ordered Blind 4D runtime index list
relative/external path identities
declared sha/size/provenance metadata
file existence
file size
mtime_ns
declared coverage
runtime order
```

It does not hash shard payloads.

Production measurement on `/home/tristan/ZeSolverCatalog/new`:

```text
indexes=47
coverage=1476/1476
all_sky=True
payload_hash_count=0
inspected_file_count=95
cache_reused=True
restore_duration_s=0.384
```

## 9. Regles d'invalidation

The cache is invalidated when any of these differ:

```text
canonical_library_path
library_id
catalog_manifest_fingerprint
blind4d_view_fingerprint
verification_schema_version
application_compatibility_version
lightweight_fingerprint
verification_status
runtime order
file existence/size/mtime identity
coverage
```

This means edits to `catalog.json`, runtime order, referenced manifests, missing
shards, changed shard size, and changed coverage invalidate the cache.

Policy for moved libraries: current implementation treats a different canonical
path as invalidation, even if content is otherwise identical.

## 10. Comportement FAST/FULL

FAST:

```text
existence, sizes, mtime_ns, manifests, schema, runtime order, coverage
payload_hash_count=0
```

FULL:

```text
explicit GUI verification path
normal CatalogLibrary validation
successful result stores verification_level=FULL
```

The GUI messages distinguish:

```text
Vérifiée — cache valide
Vérification rapide réussie
Modification détectée — nouvelle vérification requise
Bibliothèque invalide
```

## 11. Warning de couverture

`build_gui_solve_request_from_legacy_config()` now propagates:

```text
blind4d_all_sky = catalog_resources.all_sky_blind4d
```

This removes the false `blind4d_coverage_partial_not_all_sky` warning for the
final fixed32 all-sky view while keeping it for genuinely partial fixtures.

## 12. Warning instrument

The GUI now emits a non-blocking warning when reliable FITS metadata imply an
image scale that diverges strongly from the active profile scale:

```text
FOCALLEN
XPIXSZ/YPIXSZ
XBINNING/YBINNING
existing WCS pixel scale
```

No filename guessing is used and the solve is not blocked.

## 13. Tests automatises

Added:

```text
tests/test_s5e_settings_catalog_persistence.py
```

Covered:

```text
S50 settings roundtrip
partial numeric corruption is local
FULL cache reuse
payload_hash_count=0
catalog.json/runtime order invalidation
shard size invalidation
coverage invalidation
fixed all-sky request suppresses partial coverage warning
```

Targeted commands run:

```text
.venv/bin/python -m compileall -q zesolver/settings_store.py zesolver/gui_settings_sections.py zesolver/catalog_library/verification_cache.py zesolver/gui_pipeline/settings_adapter.py zesolver.py tests/test_s5e_settings_catalog_persistence.py
.venv/bin/python -m pytest -q tests/test_s5e_settings_catalog_persistence.py tests/test_engine_selection.py
18 passed in 116.58s
.venv/bin/python -m pytest -q tests/test_settings_persistence.py tests/test_gui_settings_adapter.py tests/test_gui_catalog_library_control.py
12 passed in 92.24s
```

Note: some GUI-import tests print success before a slow teardown/GC period.

## 14. Validation GUI reelle

The production library FAST/cache restore path was validated against:

```text
/home/tristan/ZeSolverCatalog/new
```

Measured result:

```text
canonical_library_path=/home/tristan/ZeSolverCatalog/new
library_id=new
indexes=47
coverage=1476/1476
all_sky=True
payload_hash_count=0
inspected_file_count=95
cache_reused=True
message=Vérifiée — cache valide
duration_s=0.384
```

The full interactive open/select/FULL/close/reopen GUI sequence is represented
by the same code path but was not completed manually in this report draft.

## 15. Performance de demarrage

Production FAST/cache restore after Python import:

```text
restore_duration_s=0.384
payload_hash_count=0
inspected_file_count=95
```

No KD-tree or NPZ payload hash is performed by the FAST restore path.

## 16. Fichiers modifies

```text
AGENT.md
docs/stabilization/s5e_settings_catalog_verification_persistence_report_20260725.md
zesolver.py
zesolver/catalog_library/verification_cache.py
zesolver/gui_pipeline/settings_adapter.py
zesolver/gui_settings_sections.py
zesolver/settings_store.py
tests/test_s5e_settings_catalog_persistence.py
```

## 17. Barrieres

Final barrier run:

```text
.venv/bin/python tools/check_core_boundaries.py
core boundary check: OK

.venv/bin/python tools/run_regression_suite.py --hermetic
635 passed, 1 skipped, 9 deselected
status: PASS

QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
635 passed, 10 skipped

.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests zesolver.py zewcscleaner.py zeindexcheck.py
OK

git diff --check
OK
```

## 18. Etat Git final

Final `git status --short --branch`:

```text
## test...origin/test
 M AGENT.md
 M zesolver.py
 M zesolver/gui_pipeline/settings_adapter.py
 M zesolver/gui_settings_sections.py
 M zesolver/settings_store.py
?? docs/stabilization/s5e_settings_catalog_verification_persistence_report_20260725.md
?? tests/test_s5e_settings_catalog_persistence.py
?? zesolver/catalog_library/verification_cache.py
```

## 19. Limites restantes

- The cache path invalidates on canonical path change. This is conservative for
  moved libraries.
- FAST verifies file identity metadata and declared content identities; it does
  not prove payload bytes unchanged if an attacker preserves size and mtime.
  FULL remains the explicit expensive route.
- No scientific solve threshold was changed.

## 20. Mise a jour d'AGENT.md

`AGENT.md` was updated to mark:

```text
S5C terminé
S5D terminé
S5D-2 terminé
S5D-3 terminé
S5E mission active
```

Obsolete diagnostics about Blind loading before Near, monolith runtime as the
main path, `tested=0` fixed32, and GUI preflight freeze were removed.

## 21. Prochaine etape unique

If final barriers pass:

```text
P3B-1E — intégration de distribution officielle des Bibliothèques ZeSolver
```

## 22. Decision de gate

S5E passes the persistence/cache/warning criteria:

```text
READY_FOR_P3B1E_LIBRARY_DISTRIBUTION_INTEGRATION
```
