# CatalogLibrary Architecture

`CatalogLibrary` is the proposed management facade for ZeSolver's local astronomical data. P1A defines the architecture only; it does not implement the component in the solver pipeline.

## Product Goal

From the user's point of view, ZeSolver should use one local astronomy library:

```text
ZeSolver Catalog Library
```

Internally, the library may contain:

- ASTAP/HNSKY source shards used directly by ZeNear;
- historical tile NPZs and quad hashes retained for compatibility/diagnostics;
- ZeBlind 4D NPZ indexes derived from the source catalogues;
- manifests and integrity metadata tying those layers together.

The goal is unified management, provenance, versioning, coverage and installation. It is not a requirement that every solver reads the same binary format at runtime.

## Non-Goals

`CatalogLibrary` must not:

- solve images;
- contain Near or Blind matching algorithms;
- rebuild indexes automatically during a resolution;
- download data silently;
- hide partial coverage;
- mutate user catalogue directories without explicit migration;
- expose NPZ/NPY, quad buckets, hash schemas or individual index filenames in the normal GUI.

## Responsibilities

`CatalogLibrary` should be able to:

- open a library directory;
- read `catalog.json`;
- validate schema version and required fields;
- identify installed ASTAP/HNSKY source families;
- identify derived historical and 4D indexes;
- verify compatibility between sources and derived indexes;
- verify checksums when present;
- report missing, corrupt or incompatible data;
- expose true sky coverage;
- expose Near readiness;
- expose Blind 4D readiness;
- provide paths/adapters needed by current ZeNear without changing the ASTAP reader;
- provide loaded or loadable 4D manifest/index entries needed by ZeBlind 4D;
- preserve provenance of source data and index-generation algorithms;
- support future installation/update planning;
- hide internal paths from normal product UI.

## Public API Proposal

Minimal conceptual API:

```python
library = CatalogLibrary.open(path)

status = library.status()
issues = library.validate()

if library.capabilities.near:
    near_source = library.near_source()

if library.capabilities.blind4d:
    blind4d = library.blind4d_indexes()

coverage = library.coverage()
```

### Classes

```python
@dataclass(frozen=True)
class CatalogLibrary:
    root: Path
    manifest: CatalogManifest

    @classmethod
    def open(cls, path: Path | str) -> "CatalogLibrary": ...
    def status(self) -> CatalogStatus: ...
    def capabilities(self) -> CatalogCapabilities: ...
    def coverage(self) -> CatalogCoverage: ...
    def validate(self) -> CatalogValidationReport: ...
    def near_source(self, *, profile: str = "zenear-v1") -> NearCatalogSource: ...
    def blind4d_indexes(self, *, profile: str = "zeblind4d-v1") -> Blind4DIndexSet: ...
```

```python
@dataclass(frozen=True)
class CatalogManifest:
    schema_version: int
    library_id: str
    sources: tuple[CatalogFamily, ...]
    derived_indexes: tuple[CatalogIndex, ...]
    coverage: CatalogCoverage
    provenance: dict[str, Any]
```

```python
@dataclass(frozen=True)
class CatalogCapabilities:
    near: bool
    blind4d: bool
    historical_blind: bool
    source_only: bool
    can_build_indexes: bool
```

```python
@dataclass(frozen=True)
class CatalogFamily:
    family: str
    format: str
    relative_path: str
    tile_count: int
    coverage: CatalogCoverage
    checksums: dict[str, str]
    status: CatalogDataStatus
```

```python
@dataclass(frozen=True)
class CatalogIndex:
    id: str
    engine: Literal["blind4d", "historical"]
    schema: str
    relative_path: str
    source_families: tuple[str, ...]
    source_tiles: tuple[str, ...]
    algorithm_version: str
    coverage: CatalogCoverage
    integrity: dict[str, str]
    status: CatalogDataStatus
```

```python
@dataclass(frozen=True)
class NearCatalogSource:
    db_root: Path
    families: tuple[str, ...]
    manifest_source_id: str
```

```python
@dataclass(frozen=True)
class Blind4DIndexSet:
    manifest_path: Path
    index_paths: tuple[Path, ...]
    index_ids: tuple[str, ...]
    tile_keys: tuple[str, ...]
    coverage: CatalogCoverage
```

```python
@dataclass(frozen=True)
class CatalogIssue:
    severity: Literal["info", "warning", "error"]
    code: str
    message: str
    path: Path | None = None
    subject: str | None = None
```

```python
@dataclass(frozen=True)
class CatalogValidationReport:
    status: CatalogStatus
    issues: tuple[CatalogIssue, ...]
```

### Exceptions

```python
class CatalogLibraryError(RuntimeError): ...
class CatalogManifestError(CatalogLibraryError): ...
class CatalogMissingError(CatalogLibraryError): ...
class CatalogIncompleteError(CatalogLibraryError): ...
class CatalogCorruptionError(CatalogLibraryError): ...
class CatalogCompatibilityError(CatalogLibraryError): ...
class CatalogVersionError(CatalogLibraryError): ...
```

Exceptions should be reserved for invalid operations or explicit validation calls. Product UI should usually receive `CatalogValidationReport` instead of a traceback.

## Status Model

| Status | Conditions | Allowed functionality | User message | Severity | Action |
| ------ | ---------- | --------------------- | ------------ | -------- | ------ |
| `READY_FULL` | Source catalogues and derived indexes cover the declared product sky/scale scope; integrity OK. | Near and Blind 4D. | Library ready. | OK | None. |
| `READY_PARTIAL` | Near and/or Blind 4D data are valid but coverage is partial. | Only within declared coverage; out-of-coverage blind fails explicitly. | Library ready with limited coverage. | Warning | Show coverage/completion option. |
| `NEAR_ONLY` | ASTAP/HNSKY source families valid; no valid Blind 4D indexes. | Near only. | Assisted solving available; blind solving unavailable. | Warning | Install/build Blind 4D indexes. |
| `BLIND4D_ONLY` | Valid 4D indexes exist but source catalogues unavailable. | Blind 4D only if validation does not need source shards. | Blind solving available; Near catalogue source missing. | Warning | Attach source catalogues. |
| `SOURCE_ONLY` | Source shards exist; no derived indexes. | Near and index-building tools only. | Catalogue source installed; blind index build required. | Info/Warning | Build or install indexes. |
| `INDEX_BUILD_REQUIRED` | Sources available and selected profile requires missing indexes. | Near; no Blind 4D for missing coverage. | Index build needed. | Warning | Schedule explicit build/import. |
| `INCOMPATIBLE` | Schema or algorithm version unsupported. | No affected engine. | Catalogue/index format incompatible with this ZeSolver version. | Error | Update ZeSolver or data. |
| `CORRUPT` | Checksum mismatch, unreadable manifest, invalid NPZ/source tile. | No affected engine. | Catalogue data appears corrupt. | Error | Reinstall/repair. |
| `MISSING` | Library root or required manifest absent. | None. | Catalogue library not found. | Error | Select/install library. |

`READY_PARTIAL` is the expected status of the current bundled 4D set.

## Manifest Handling

`CatalogLibrary.open(path)` should resolve:

```text
path/catalog.json
```

All data paths inside `catalog.json` should be relative to the library root unless explicitly declared as an external reference. External references are allowed for non-destructive adoption of current installations, but must be marked as such.

Validation phases:

1. parse manifest;
2. validate schema version;
3. validate source families and files;
4. validate derived index manifest/index references;
5. verify checksums when available;
6. compare derived index provenance to declared source families/tiles;
7. compute capability status and coverage.

## Physical Structure

Recommended target layout:

```text
ZeSolverCatalog/
├── catalog.json
├── sources/
│   └── astap/
│       ├── d50/
│       ├── d80/
│       ├── g05/
│       └── ...
├── indexes/
│   ├── blind4d/
│   │   ├── manifest.json
│   │   └── *.npz
│   └── legacy/
│       ├── manifest.json
│       ├── tiles/
│       └── hash_tables/
├── cache/
└── state/
```

### Directory Rules

- `catalog.json` is mandatory.
- `sources/astap/` is mandatory for `NEAR_ONLY`, `SOURCE_ONLY`, `READY_PARTIAL` with Near, and `READY_FULL`.
- `indexes/blind4d/` is optional but required for Blind 4D capability.
- `indexes/legacy/` is optional and should be treated as compatibility/build source, not a normal user-facing dependency.
- `cache/` is regenerable and can be deleted.
- `state/` is local, machine-specific and should not be shared blindly.
- Source shards and indexes are immutable once checksummed.
- Manifest writes should be atomic: write `catalog.json.tmp`, fsync where practical, then replace.
- A lock file should be used during explicit index construction/import.

### Platform Rules

- Manifest paths are POSIX-style relative paths.
- Absolute paths are allowed only under an explicit `external_reference` object.
- Windows/macOS/Linux path conversion happens at load time.
- Libraries on external drives should remain portable if they avoid absolute paths.

## Product UI Boundary

The normal GUI should show:

- selected library location;
- status (`READY_PARTIAL`, `NEAR_ONLY`, etc.);
- total size;
- installed source families in a detail view;
- Near available/not available;
- Blind 4D available/not available;
- sky coverage summary;
- missing/corrupt data;
- Verify button;
- future Install/Complete button.

The normal GUI should not show:

- direct 4D manifest path;
- individual NPZ/NPY filenames;
- quad hash formats;
- bucket/candidate/hash limits;
- builder parameters;
- experimental family selections;
- internal index source paths.

Expert/developer views may expose raw paths for diagnostics, but must clearly label them as internal.

## Existing Runtime Adapters

Initial implementation should be adapter-based:

```python
near_source = library.near_source()
solve_config = replace(
    solve_config,
    db_root=near_source.db_root,
    families=near_source.families,
)
```

```python
blind4d = library.blind4d_indexes()
solve_config = replace(
    solve_config,
    blind_4d_manifest_path=blind4d.manifest_path,
)
```

This keeps `solve_near()` and `solve_blind()` unchanged during early P1B phases.

## Progressive Implementation Plan

| Step | Scope | Files likely touched | Tests | Risk | Rollback | Exit criteria |
| ---- | ----- | -------------------- | ----- | ---- | -------- | ------------- |
| P1B-1 | Types and manifest reader | new `zesolver/catalog_library.py`, docs/tests | schema/example tests | Low | Remove new module | Parses example manifest and reports statuses. |
| P1B-2 | Read-only discovery of existing installs | new discovery helpers | temp-directory fixtures | Low | Disable discovery | Existing `db_root`/manifest can be represented without moving files. |
| P1B-3 | ZeNear adapter | config assembly only | Near regression hermetic/corpus | Medium | Revert adapter path | Near receives same `db_root`/families as before. |
| P1B-4 | ZeBlind 4D adapter | config assembly only | 4D manifest/pipeline tests | Medium | Revert adapter path | 4D receives same manifest/index paths as before. |
| P1B-5 | Coverage/integrity validation | library validation module | corrupt/missing fixtures | Low/Medium | Keep validation advisory | Partial vs full coverage surfaced correctly. |
| P1B-6 | Non-destructive settings migration | settings loader and GUI settings page | migration tests | Medium | Fall back to old settings | Old paths adopted by reference; no files moved. |
| P1B-7 | GUI integration | GUI settings/catalogue panel | GUI integration tests | Medium/High | Keep expert raw path panel | Normal UI uses library status/capabilities. |
| P1B-8 | Installation/update flow | installer/downloader/build tooling | offline fixture tests | High | Feature flag | Explicit install/update can create/update library manifests atomically. |

## Open Questions

- Should the canonical environment variable be `ZESOLVER_BLIND4D_MANIFEST`, `ZEBLIND_4D_MANIFEST`, or a library-root variable only?
- Should current bundled six 4D indexes become a built-in partial library, or remain profile resources until full library packaging exists?
- How much checksum detail is acceptable for ASTAP source shards without creating very large manifests?
- Which ASTAP families should be included in the first product install profile beyond D50?
