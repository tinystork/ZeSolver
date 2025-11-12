# Agent Guide for This Repository

Scope: This AGENTS.md lives at the repository root and applies to the entire tree.
Use it as the source of truth when you (the coding agent) interact with the codebase.

## Mission

Build and maintain a fast, fully-Python blind and metadata-assisted WCS solver for batches of astronomical images without invoking ASTAP or shipping any of its source code. You may read ASTAP/HNSKY catalogue shards to build your own Python index, but you must not run or embed ASTAP itself.

## Strict Rules (Read First)

- Do NOT call the ASTAP executable or any third-party solver binaries.
- Do NOT import, copy, or integrate ASTAP source code. The folder `astapsourcereadonly/` (if present) is for reading and inspiration only. You may use it to understand algorithms, but you must not paste or translate functions verbatim. The project must remain pure Python.
- DO use ASTAP/HNSKY databases (the star shards like `.1476`, `.290`) as input data only. Parsing and conversion are implemented here in Python.
- Keep indices separated from raw databases. The recommended layout is:
  - `database/` → ASTAP/HNSKY tiles (read-only data)
  - `index/` → Generated Python blind index (manifest + quads)

## Project Layout (Key Parts)

- `zewcs290/` – Catalogue reader for ASTAP/HNSKY tiles; builds a virtual view of the shards and serves cone/box queries.
- `zeblindsolver/` – End-to-end blind solver pipeline in Python.
  - `db_convert.py` – Converts ASTAP tiles to tangent-plane per-tile blobs and writes `manifest.json`, then builds quad hash tables.
  - `quad_index_builder.py` – Hashes quads for levels L/M/S and writes `hash_tables/quads_*.npz`.
  - `asterisms.py` – Quad sampling and hashing (three ratios + parity). Includes a “local_brightness” strategy for smaller-diameter quads.
  - `candidate_search.py` – Tally candidate tiles by hash votes.
  - `matcher.py` – Similarity transform estimation + RANSAC.
  - `wcs_fit.py` – Build TAN WCS from similarity and refine with SIP if needed.
  - `zeblindsolver.py` – CLI + orchestration of the blind solve.
- `zesolver.py` – Batch GUI/CLI wrapper for solving folders and managing index settings.
- `examples/` – Example FITS and `*_thn.jpg` thumbnails (JPGs are expected to fail/skip solving; they are for UI testing).
- `tests/` – Synthetic and integration tests.

## Index Build and Use

1) Build the index from the ASTAP database (separate folder from `database/`):
   - CLI: `zebuildindex --db-root database --index-root index --mag-cap 15.5 --max-stars 2000 --max-quads-per-tile 20000`
   - Output:
     - `index/manifest.json`
     - `index/tiles/*.npz`
     - `index/hash_tables/quads_L.*`, `quads_M.*`, `quads_S.*`
   - Storage knobs:
     - `--quad-storage {npz,npz_uncompressed,npy}` picks compressed `.npz`, store-only `.npz`, or mmap-friendly `quads_<level>/hashes.npy` directories.
     - `--tile-compression {compressed,uncompressed}` toggles NPZ compression for tiles (uncompressed = faster reads, larger files).
     - `--workers N` sets the number of processes used to hash quads (default: half the CPUs).

2) Run the blind solver:
   - CLI: `zeblindsolve examples/Light_*.fit --index-root index`
   - The solver writes a TAN WCS and validation keywords (`SOLVED`, `QUALITY`, `BLIND_VER`, etc.).

3) GUI (zesolver):
   - Settings → “Construire l’index” prompts for a dedicated index directory (prevents mixing with `database/`).
   - Settings are persisted in `~/.zesolver_settings.json` (`db_root`, `index_root`, and limits).

## Cross-Platform Paths

- Always write manifest tile paths with POSIX separators (`as_posix()`).
- When reading, normalize backslashes to forward slashes (Windows-origin manifests).

## Allowed vs Prohibited Sources

- Allowed:
  - Reading `.1476` / `.290` catalogue tiles (ASTAP/HNSKY) and deriving your own per-tile arrays and indices.
  - Using GPL data files under `database/` as input.
- Prohibited:
  - Including ASTAP source code or calling the ASTAP executable from this project.
  - Copying ASTAP functions verbatim (“read only for understanding”).

## Coding Conventions

- Stay in Python; keep code minimal, explicit, and testable.
- Prefer small, targeted changes; don't rename files or add frameworks unless necessary.
- Update docs and GUI strings when behavior changes (FR/EN supported in `GUI_TRANSLATIONS`).
- Normalize manifest paths; treat missing quad tables (`hash_tables/quads_<level>.npz` or `.npy` folders) as a clear, actionable error.
- `_load_tile_positions` uses an in-process LRU cache keyed by absolute path + `(mtime_ns, size)`. Use the existing helper functions, and expose size tuning via `ZE_TILE_CACHE_SIZE` / `--tile-cache-size` rather than new globals.

## Typical Agent Tasks

- Add/modify index build options in the GUI:
  - Ensure index destination is user-chosen and not inside `database/`.
  - Persist the chosen `index_root`.
- Improve blind matching robustness:
  - Use vote-based pair filtering, local-neighborhood sampling, and level-specific hash specs.
  - Ensure `CRPIX` and `CRVAL` are consistent with the plane center.
- Performance:
  - Tune `max_quads_per_tile` (e.g. 2000-5000 for faster builds) and level strategies (`local_brightness` for M/S).
  - Pick the right index layout for the machine: `--quad-storage npy` for mmap, `npz_uncompressed` when I/O dominates, `--tile-compression uncompressed` if disk allows faster solving.

## Troubleshooting

- “Missing quad tables (L/M/S)” → Rebuild the index; ensure each level has either `hash_tables/quads_<level>.npz` or a `hash_tables/quads_<level>/` directory with `.npy` payloads.
- “Tile file not found” with backslashes in path → Manifest from Windows; reader normalizes separators.
- Too many false candidates → increase quad selectivity, lower bucket caps, or favor smaller-diameter levels.

## Tests

- Keep/add tests under `tests/` following existing patterns (synthetic index fixture, candidate tally, WCS validation). Avoid adding heavy or external dependencies.

---

By following this guide, you ensure all future contributions remain compliant with the project’s constraints: pure Python, no ASTAP code execution or integration, and clean separation of database vs. index artifacts.

