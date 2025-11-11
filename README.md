# zewcs290

An offline WCS (World Coordinate System) solver designed specifically for Seestar / ZeSolver
workflows. The project ingests the ASTAP / HNSKY ``.1476`` (Dxx/V50) and ``.290`` (G05) Gaia
catalogue shards directly from disk and exposes:

- A modular Python API (`zewcs290`) for loading catalogue tiles, querying stars in spatial
  regions and feeding them into astrometric solvers.
- A binary inspection utility (`tools/inspect_290.py`) to reverse engineer catalogue metadata
  and sanity‑check local databases without touching the network.
- Stubs for the star detection and solving workflows required for a fully offline plate solver.

The project deliberately **does not** bundle any astronomical catalogues. You must point the code
to an existing ASTAP installation (e.g. D20/D50/V50/G05 databases).

## Repository layout

```
zewcs290/          # Python package (catalogue reader, tilings, solver stubs)
tools/             # Standalone utilities (inspect_290.py, future helpers)
database/          # User-provided ASTAP catalogues (not version controlled)
examples/          # FITS samples for solver development
tests/             # Pytest suite
pyproject.toml     # Build + dependency metadata
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Inspect a Gaia G05 catalogue directory
python tools/inspect_290.py --db ./database --family g05 --limit 3 --json report.json
```

Within Python:

```python
from zewcs290 import CatalogDB

db = CatalogDB(db_root="database", families=["d50", "g05"])
stars = db.query_cone(ra_deg=184.17, dec_deg=47.14, radius_deg=1.0, mag_limit=17.5)
print(stars[:3])
```

## Batch solver GUI/CLI

The `zesolver.py` entry point wraps the catalogue reader with a lightweight plate
solver, supporting both CLI and PySide6 GUI workflows:

```bash
# CLI mode (headless batch) – uses half of the available CPU threads by default
python zesolver.py --db-root ./database --input-dir ./examples --workers 4 --fov-deg 1.4

# GUI mode (requires PySide6, install via `pip install .[gui]`)
python zesolver.py --gui
```

The GUI lets you pick the ASTAP/HNSKY database folder, select an input directory
containing FITS/TIFF/PNG frames, adjust the FOV/downsample parameters, and launch
multi-threaded solving. Results are written back to the FITS headers (or stored
as JSON sidecars for raster formats).

## Blind solver (`zeblindsolver`)

`zeblindsolver` is a standalone ASTAP-based blind solver that sanitises FITS
headers, tries the local D80/D50/V50/D20/D05/G05/W08/H18 databases in cascade,
and writes back a validated WCS solution (including `SOLVED`, `SOLVER=ZeSolver`,
`SOLVMODE=BLIND`, `BLINDVER`, `USED_DB`, `RMSPX`, `INLIERS`). It exposes both a
Python API and a CLI:

```bash
# Blind solve a single FITS file in place with the default ASTAP sequence
zeblindsolver --input Light_M81.fit --db "D50;D20;W08" --profile S50 --timeout 90

# Copy the solved FITS somewhere else
zeblindsolver --input raw.fit --db "/data/D50;/data/G05" --write-to solved.fit
```

`zesolver.py` automatically triggers the blind fallback when a FITS header lacks
a trustworthy WCS (unless `--no-blind` is provided). Successful blind solves are
logged in both CLI and GUI modes (`run_info_blind_*` entries) and prevent GUI
freezes because ASTAP runs in worker threads.

In the GUI, the settings tab now includes a "Blind solver (Python)" group with
the main tunables and a "Fast mode (S-only, fallback M/L)" option that tries the
most selective level first for speed.

## Metadata-assisted near solver

The new `zeblindsolver/metadata_solver.py` path keeps the solving workflow
entirely in Python and only requires the manifest + tile projections produced by
`zebuildindex`.  No quad hash tables are needed.  Given a FITS file with an RA/Dec
hint and approximate optical metadata, the solver:

- Detects image stars via the shared `star_detect` module.
- Loads nearby catalogue tiles from `index/tiles/*.npz`, reprojects them onto the
  requested tangent plane, and matches stars via similarity RANSAC (with optional
  parity flips).
- Writes a TAN WCS (and SIP terms if needed) together with `SOLVED`, `SOLVER=ZeSolver`,
  `SOLVMODE=NEAR`, `QUALITY`, `NEAR_VER`, `INLIERS`, `RMSPX`, and `PIXSCAL` keywords.
- Falls back to the quad-based blind solver automatically when metadata is
  missing or inconsistent (if the quad tables are present).

Programmatic usage mirrors the blind helper:

```python
from pathlib import Path
from zesolver import NearSolveConfig, near_solve

config = NearSolveConfig(max_img_stars=400, max_cat_stars=1000)
result = near_solve("examples/Light_M31.fit", index_root="index", config=config)
print(result["message"])
```

The GUI settings tab exposes a dedicated **“Near solve (Python, no quads)”**
button next to the blind test action so you can validate a sample FITS file
against your local index without launching ASTAP.

## Status

Phase 1 focuses on catalogue ingest, inspection, and the assisted solver path. Blind solving,
triangle hashing, SIP fitting, and full CLI wiring will land in subsequent iterations.

## License

MIT. Catalogue data remains copyrighted by their respective owners (Gaia/ESA, ASTAP,
HNSKY). The code only reads those files in place.
