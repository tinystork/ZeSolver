# Advanced Catalog Compatibility Surface

P1D-5B closes the normal CatalogLibrary GUI path by removing the ambiguous
historical catalog fields from the ordinary product surface.

The normal surface has one catalog concept:

```text
ZeSolver library
```

It is a directory containing `catalog.json`.  In `auto` modes, a valid library
drives:

```text
Near     -> astap-native
Blind 4D -> catalog_library_view
```

Historical ASTAP, Near index and external Blind 4D manifest fields remain
available only in an explicit advanced surface.

## Normal Surface

The Settings tab always exposes the ZeSolver library selector:

```text
ZeSolver library
[ root containing catalog.json ] [ Browse... ] [ Verify ] [ Clear ]
```

The visible status reports the library state and distinguishes source coverage
from derived Blind 4D coverage:

```text
READY_PARTIAL
Near ASTAP: D50, 1476 tiles
Blind 4D: 6 / 1476 tiles
Global Blind 4D coverage: no
```

The normal path does not show:

```text
Historical ASTAP source
Historical Near index
External Blind 4D manifest
Near legacy-index mode
Blind external-manifest mode
Index construction
Hash rebuilding
Legacy index validation tools
```

## Compatibility Surface

Expert mode can open the checkable group:

```text
Legacy compatibility and diagnostics
```

French title:

```text
Compatibilite historique et diagnostic
```

The group is hidden by default in Easy mode and unchecked by default in Expert
mode.  Opening it is informational; it does not activate rollback.

It contains:

```text
Historical ASTAP source
Historical Near index
External Blind 4D manifest
Near mode: auto / astap-native / legacy-index
Blind 4D mode: auto / library-view / external-manifest
Restore recommended automatic mode
```

The restore action sets:

```text
near_catalog_mode = auto
blind4d_catalog_mode = auto
```

It preserves all historical paths.

## Maintenance Tools

Index construction, hash rebuilding, old-index validation and catalogue
exploration are grouped under an advanced maintenance section.  They are not
part of the normal CatalogLibrary path and are never launched by selecting or
verifying a library.

## Typed Validation

The GUI uses typed validators rather than one generic directory check:

```text
CatalogLibrary root       -> catalog.json + CatalogLibrary.open/validate
Historical ASTAP source   -> *.1476 or *.290 family files
Historical Near index     -> manifest.json + legacy index validator
External Blind 4D manifest -> strict JSON manifest loader
```

Inactive legacy fields do not block a normal CatalogLibrary run.  They are
validated and blocking only when their corresponding rollback mode is explicitly
requested.

## Confusion Prevention

The GUI detects common cross-field mistakes before a run:

```text
CATALOG_LIBRARY_USED_AS_LEGACY_NEAR_INDEX
LEGACY_NEAR_INDEX_USED_AS_CATALOG_LIBRARY
ASTAP_SOURCE_USED_AS_CATALOG_LIBRARY
BLIND4D_MANIFEST_FILE_REQUIRED
```

For the user-reported case, entering a ZeSolver library in the historical Near
index field now reports:

```text
This folder is a ZeSolver library, not a historical Near index.
Use this folder in the ZeSolver library field.
```

French:

```text
Ce dossier est une Bibliotheque ZeSolver, pas un index Near historique.
Utilisez ce dossier dans le champ Bibliotheque ZeSolver.
```

The previous generic legacy error:

```text
LEGACY_NEAR_INDEX_INVALID: .../manifest.json
```

is no longer the only user-facing signal for that confusion.

## Rollback

Rollback remains explicit:

```text
Near     = legacy-index
Blind 4D = external-manifest
```

When active, the GUI shows a visible rollback status:

```text
ROLLBACK HISTORIQUE ACTIF
```

In normal `auto` modes, persisted historical paths are ignored when a valid
CatalogLibrary is selected, even if those historical paths are invalid test
sentinels.

## Qt Layout Warning

P1D-5B also fixes the existing Qt warning:

```text
QLayout::addChildLayout: layout QFormLayout "" already has a parent
```

The cause was duplicate insertion of form layouts during tab construction, not
the CatalogLibrary selector itself.  The fix removes the duplicate additions
instead of suppressing Qt warnings globally.

