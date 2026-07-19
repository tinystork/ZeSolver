# GUI CatalogLibrary Selection

P1D-5A added the first normal GUI product surface for selecting an explicit
`CatalogLibrary`.

P1D-5B closes the ambiguity with historical catalog paths: the normal product
surface now presents only the ZeSolver library selector, while legacy fields and
tools live in an explicit compatibility/diagnostic surface.

## Control

The Settings tab now exposes:

```text
ZeSolver library
[ library root ] [ Browse... ] [ Verify ] [ Clear ]
```

The selected path is the library root containing `catalog.json`, not the
manifest file itself.  The field is initialized from
`PersistentSettings.catalog_library_path`, accepts paths with spaces and
non-ASCII characters, expands `~` during validation, and never creates,
adopts, writes or repairs a library.

`Clear` removes the selection from settings and logs that the library files are
not deleted.

## Validation

The `Verify` button uses the product library loader and resource resolver:

```text
CatalogLibrary.open()
-> CatalogLibrary.validate()
-> resolve_catalog_resources(catalog_library=...)
```

The GUI does not implement a second catalog discovery path.  Validation reports
stable, textual states:

```text
AUCUNE_BIBLIOTHEQUE
VALIDATION_EN_COURS
READY_FULL
READY_PARTIAL
NEAR_ONLY
BLIND4D_ONLY
INVALID
MISSING
```

The visible summary distinguishes source and derived-index coverage.  A full
ASTAP source with six Blind 4D indexes is shown as partial Blind 4D coverage,
not as a complete library:

```text
READY_PARTIAL
Near ASTAP: available, D50, 1476 tiles
Blind 4D: available, 6 indexes / 1476 tiles
Global Blind 4D coverage: no
```

Coverage warnings are informational.  Missing manifests, invalid schemas,
checksum mismatches and missing critical resources block use of the selected
library.

## Persistence

`_read_settings_from_ui()` treats the visible library field as the source of
truth.  A non-empty path is persisted as
`PersistentSettings.catalog_library_path`; an empty field is persisted as
`None`.

Historical fields are preserved during save:

```text
db_root
index_root
blind_4d_manifest_path
near_catalog_mode
blind4d_catalog_mode
```

Old settings files without `catalog_library_path` still migrate to `None`.

## Run Configuration

With a valid explicit library, the GUI can build a run config without requiring:

```text
db_root
index_root
blind_4d_manifest_path
```

The normal modes remain:

```text
near_catalog_mode = auto
blind4d_catalog_mode = auto
```

In that state the product selectors choose:

```text
Near -> astap-native
Blind 4D -> catalog_library_view
```

If an advanced setting forces rollback:

```text
near_catalog_mode = legacy-index
blind4d_catalog_mode = external-manifest
```

the GUI preserves that explicit choice and the preflight log shows it.

With no library selected, legacy validation rules are preserved.  With an
invalid library explicitly selected, the run is blocked and the GUI does not
fall back silently to legacy paths.

## Wizard

The simple startup wizard now checks an explicit library first:

```text
valid library -> ready
invalid library -> ask the user to verify or choose another root
no library -> legacy db/index compatibility checks
```

It never creates or adopts a library automatically.

## Preflight Telemetry

At run start the GUI logs resource selection without personal paths in
structured telemetry:

```text
Catalog resources: {...}
CatalogLibrary selected: <library id>
CatalogLibrary status: READY_PARTIAL
Near catalog preflight: {...}
ZeBlind 4D preflight: {...}
Catalog preflight timings: {...}
```

The timing payload separates:

```text
catalog_library_open_s
catalog_resource_resolution_s
near_runtime_resolution_s
blind4d_runtime_resolution_s
catalog_preflight_total_s
```

These timings are lightweight diagnostics only; no solver cache or algorithm
was changed in P1D-5A.

## Compatibility Surface

Historical fields remain available, but no longer in the normal path:

```text
Historical ASTAP source
Historical Near index
External Blind 4D manifest
Near rollback mode
Blind 4D rollback mode
```

They are grouped under:

```text
Legacy compatibility and diagnostics
```

French:

```text
Compatibilite historique et diagnostic
```

The group is hidden in Easy mode and closed by default in Expert mode.  Opening
it does not activate rollback.

The normal CatalogLibrary run ignores invalid historical sentinels while modes
remain `auto`.  Historical paths are consulted only for explicit rollback:

```text
near_catalog_mode = legacy-index
blind4d_catalog_mode = external-manifest
```

The GUI detects cross-field mistakes such as using a ZeSolver library as a
historical Near index and displays a targeted message instead of only a generic
legacy manifest error.
