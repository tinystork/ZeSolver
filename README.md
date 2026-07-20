# ZeSolver

**ZeSolver** is an open-source, offline WCS (World Coordinate System) solver developed for
large Seestar imaging workflows.

It was created for the needs of **The Seestar Collective**, a small community of amateur
astronomers who may collect several thousand — sometimes more than 10,000 — short exposures
of a target before stacking and assembling them into mosaics.

ZeSolver is designed to solve these large batches locally and efficiently while preserving
the original image pixels.

> [!IMPORTANT]
> ZeSolver is an independent community project. It is not affiliated with, maintained by,
> or officially endorsed by ASTAP, HNSKY, Astrometry.net, ESA, or the Gaia project.

## Current status

ZeSolver is under active development and is not yet considered ready for a general public
release.

The solver core, batch pipeline, catalogue library, WCS safeguards, progress reporting,
cancellation, and the main Near/Blind solving paths are functional and covered by regression
tests. Packaging, first-install experience, documentation, simplified GUI work, and broader
Blind 4D catalogue coverage are still being finalized.

Blind 4D coverage may be partial depending on the installed indexes. ZeSolver reports this
explicitly and must not be presented as all-sky unless the installed library actually provides
all-sky coverage.

## How ZeSolver works

The local solving chain is:

```text
ZeNear
  -> ZeBlind 4D
  -> optional Astrometry.net web fallback
```

### ZeNear

**ZeNear** is the fast, metadata-assisted solver.

When approximate sky coordinates and optical information are available, it reads nearby stars
directly from locally installed ASTAP/HNSKY catalogue shards through ZeSolver's own Python
catalogue provider. It then detects image stars, matches them against catalogue stars, validates
the solution, and writes a TAN WCS when the result passes the configured quality checks.

The ASTAP executable is **not** invoked by the current ZeNear product path.

### ZeBlind 4D

**ZeBlind 4D** is the fully blind local solver.

It uses precomputed geometric quad indexes derived from locally installed catalogue data. Its
general blind-solving approach is conceptually inspired by the geometric indexing techniques
used by Astrometry.net, while the implementation and runtime integration in this repository are
part of ZeSolver.

ZeBlind 4D uses the same catalogue-library provenance as ZeNear whenever possible, so source
catalogues, derived indexes, generation parameters, versions, and coverage can be related and
validated together.

### Optional Astrometry.net fallback

An Astrometry.net web fallback can be configured as a last resort after the local Near and Blind
paths. When enabled, images may be uploaded to a third-party service; users are responsible for
their own API credentials, rights, and privacy choices.

## Acknowledgements

ZeSolver exists thanks to the work of the wider astronomical software and data community.

### ASTAP and HNSKY

Special thanks to **Han Kleijn**, developer of
[ASTAP](https://www.hnsky.org/astap.htm) and HNSKY.

ASTAP has been a major technical reference throughout the development of ZeSolver, particularly
for catalogue organization, practical plate-solving workflows, and the metadata-assisted
**ZeNear** path. ZeSolver reads the ASTAP/HNSKY `.1476` and `.290` catalogue formats through an
independent Python implementation.

ZeSolver does not include ASTAP source code, does not invoke the ASTAP executable in its current
local product chain, and does not claim to be an official ASTAP component.

### Astrometry.net

The **ZeBlind 4D** work was informed by the principles behind
[Astrometry.net](https://astrometry.net/), including geometric quad-based blind solving.

Credit is due to **Dustin Lang, David W. Hogg, Keir Mierle**, and the other Astrometry.net
contributors for their foundational work in blind astrometric calibration.

### Gaia and catalogue data

The stellar data used by ASTAP/HNSKY catalogue families originates from upstream astronomical
catalogues, including Gaia data produced by the European Space Agency and the Gaia collaboration.

All catalogue data remains subject to the rights, licences, acknowledgements, and distribution
conditions of its respective providers.

## Catalogue and derived-index policy

This repository deliberately does **not** bundle or redistribute the original ASTAP/HNSKY
catalogue shards.

Users currently provide their own local catalogue installation, such as D20, D50, V50, G05, or
other supported families.

ZeSolver can generate its own working assets from those catalogues, including:

- normalized tile snapshots;
- compressed or uncompressed `.npz` data;
- `.npy` memory-mapped data;
- quad tables;
- 4D geometric hash indexes;
- manifests, provenance, version, and coverage metadata.

These files are derived runtime assets for ZeSolver and are not the original ASTAP/HNSKY
downloads.

Precomputed derived assets are not included in this repository. Any future redistribution of
such assets must be handled separately, with appropriate permission, attribution, documentation,
and licence notices. The original ASTAP/HNSKY catalogue files will not be repackaged as part of
ZeSolver.

## Main features

- Offline batch WCS solving for FITS images.
- Metadata-assisted **ZeNear** solving.
- Local **ZeBlind 4D** fallback.
- Optional Astrometry.net web fallback.
- Direct ASTAP/HNSKY `.1476` and `.290` catalogue reading.
- Unified ZeSolver catalogue-library validation and provenance.
- FITS pixel-integrity safeguards.
- Existing-WCS detection and configurable overwrite policy.
- TAN WCS output, with validated metadata updates.
- JSON WCS sidecars for supported raster formats.
- Real-time batch progress and per-file status.
- Responsive cancellation and safe restart.
- CPU operation by default.
- Optional CUDA acceleration for Near star detection.
- GUI and command-line workflows.
- Index construction, validation, benchmarking, and diagnostic tools.

## Repository layout

```text
zesolver/           Product orchestration, catalogue library, settings, GUI pipeline, and core
zeblindsolver/      Blind solver, quad generation, index builders, and validation
zewcs290/           Native ASTAP/HNSKY catalogue readers and spatial queries
settings/           Stable product/runtime settings models
tools/              Benchmarks, audits, diagnostics, and maintenance utilities
tests/              Unit, integration, regression, GUI, corpus, and safeguard tests
docs/               Architecture, stabilization, validation, and development reports
packaging/          PyInstaller and release helpers
examples/           Development and example inputs where provided
pyproject.toml      Build and dependency metadata
```

Catalogue databases, generated indexes, large test corpora, and local runtime files are not meant
to be committed to the repository.

## Requirements

- Python 3.10 or newer.
- A local ASTAP/HNSKY catalogue installation for offline catalogue-backed solving.
- PySide6 for the desktop GUI.
- CUDA and CuPy only when optional GPU detection is desired.

ZeSolver must remain fully usable on CPU without CUDA.

## Quick start from source

```bash
git clone https://github.com/tinystork/ZeSolver.git
cd ZeSolver

python -m venv .venv
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python zesolver.py
```

The project is still being prepared for a clean public installation experience. Until packaging
and first-run documentation are finalized, source installations should be considered development
or beta installations.

## Catalogue library

The target product model is a single user-selected concept:

```text
ZeSolver Library
```

A valid library relates:

- the ASTAP/HNSKY source catalogue families used by ZeNear;
- the derived Blind 4D indexes;
- generation settings and format versions;
- file fingerprints and provenance;
- installed coverage and known limitations.

Legacy database-root, index-root, family, and external-manifest paths may remain available as
advanced compatibility or diagnostic overrides while the product surface is being simplified.

Coverage warnings must not be hidden. A partial Blind 4D library is valid for development or
limited use, but it is not equivalent to all-sky coverage.

## Building derived indexes

The `zebuildindex` command can generate ZeSolver-friendly tiles and quad indexes from a local
ASTAP/HNSKY catalogue installation.

Example:

```bash
python -m zeblindsolver.db_convert \
  --db-root database \
  --index-root index \
  --max-quads-per-tile 20000 \
  --quad-storage npy \
  --tile-compression uncompressed \
  --workers 8
```

On Windows PowerShell:

```powershell
python -m zeblindsolver.db_convert `
  --db-root .\database `
  --index-root .\index `
  --max-quads-per-tile 20000 `
  --quad-storage npy `
  --tile-compression uncompressed `
  --workers 8
```

Relevant storage options include:

- `--quad-storage npz` for compressed `.npz`;
- `--quad-storage npz_uncompressed` for store-only `.npz`;
- `--quad-storage npy` for memory-mapped `.npy` bundles;
- `--tile-compression compressed`;
- `--tile-compression uncompressed`.

When only the quad sampler or hash encoding has changed, a quads-only rebuild can reuse existing
tile data:

```bash
python -m zeblindsolver.db_convert \
  --db-root database \
  --index-root index \
  --mag-cap 15.5 \
  --max-stars 2000 \
  --max-quads-per-tile 20000 \
  --quad-storage npy \
  --tile-compression uncompressed \
  --workers 8 \
  --quads-only
```

Generated assets should be validated before use. ZeSolver includes index and library validation
tools that detect missing files, incompatible formats, stale metadata, provenance mismatches, and
partial coverage.

## Batch solver GUI and CLI

The `zesolver.py` entry point provides the main batch workflow.

GUI launch:

```bash
python zesolver.py
```

or, where supported:

```bash
python zesolver.py --gui
```

A headless CLI workflow can be launched with explicit paths and parameters. The exact advanced CLI
surface may evolve while the settings and packaging layers are being stabilized.

Example legacy-compatible invocation:

```bash
python zesolver.py \
  --db-root ./database \
  --input-dir ./examples \
  --workers 4 \
  --fov-deg 1.4
```

The batch pipeline can:

- scan FITS and supported raster files;
- detect existing WCS information;
- preserve or replace WCS according to the selected policy;
- route compatible inputs through the current pipeline;
- keep explicit legacy compatibility paths where necessary;
- report processed, total, and remaining files in real time;
- emit at most one terminal result per file;
- stop cleanly and restart after cancellation.

For FITS files, the primary status reflects the WCS state of the `PRIMARY` HDU. Pixel data must not
be modified by WCS solving or cleanup operations.

For supported raster formats, WCS information is stored in a sidecar file rather than modifying
the source raster.

## Optional GPU acceleration

ZeSolver works on CPU by default.

Optional CUDA acceleration currently targets Near star detection on supported Linux/NVIDIA
systems.

Install the optional runtime where appropriate:

```bash
python -m pip install -U \
  cupy-cuda12x \
  nvidia-cuda-runtime-cu12 \
  nvidia-cuda-nvrtc-cu12
```

In the GUI, select `Auto` or `CUDA` for the star-detection backend.

When CUDA or one of its runtime libraries is unavailable, ZeSolver should fall back safely to CPU
and record the reason in the log.

## Benchmarking and diagnostics

The repository includes utilities for controlled solver benchmarks, catalogue inspection, index
validation, regression gates, and FITS/WCS integrity checks.

Example benchmark:

```bash
python tools/benchmark_solver.py \
  --index-root index \
  --output-json bench.json \
  --output-csv bench.csv \
  examples/*.fit
```

Benchmark inputs may include files, directories, glob patterns, or list files. By default,
benchmarks should preserve pristine copies and avoid modifying original FITS files unless writing
is explicitly enabled.

Development and validation utilities include, depending on the current branch:

- catalogue and binary inspection;
- index and manifest validation;
- source and GUI inventories;
- WCS cleanup;
- pixel-integrity checks;
- core-boundary checks;
- hermetic regression suites;
- corpus and graphical validation instructions.

## macOS readiness preflight

Before a production-oriented macOS test, run:

```bash
.venv/bin/python -m zesolver.macos_preflight
```

This checks imports, multiprocessing behavior, and a real `zesolver.py --help` launch.

A successful preflight is useful, but it does not replace a real GUI test or a complete clean-machine
installation test.

## Packaging

PyInstaller helpers are available under:

```text
packaging/pyinstaller/
```

Typical development build:

```bash
.venv/bin/python -m pip install pyinstaller
.venv/bin/python packaging/pyinstaller/build.py --clean
```

Optional one-file build:

```bash
.venv/bin/python packaging/pyinstaller/build.py --clean --onefile
```

Packaging is still an active work area. A version number in `pyproject.toml` does not by itself mean
that a stable public release has been completed.

Before release, builds must be tested on clean Windows, macOS, and Linux environments, including
CPU-only systems.

## Release checklist

Before publishing a release:

- [ ] Project name, package name, and version are consistent.
- [ ] `pyproject.toml` contains a valid build-system configuration.
- [ ] Wheel and source distribution build successfully.
- [ ] Editable installation works.
- [ ] Installation from the built wheel works.
- [ ] CPU-only operation works without CUDA packages.
- [ ] GUI and CLI entry points work.
- [ ] Windows, macOS, and Linux launch tests pass.
- [ ] Paths containing spaces and non-ASCII characters are tested.
- [ ] Existing WCS and pixel-integrity safeguards pass.
- [ ] Near, Blind 4D, pipeline, and fallback regression tests pass.
- [ ] Installed Blind 4D coverage is documented accurately.
- [ ] Credits, licences, and third-party notices are complete.
- [ ] Catalogue and derived-index distribution terms are documented.
- [ ] First-run and clean-machine installation guides are verified.
- [ ] Known limitations and fallback behavior are documented.

## Development safeguards

The project follows several non-negotiable rules:

- never modify image pixels while writing or cleaning WCS metadata;
- never overwrite an existing WCS silently;
- preserve unrelated FITS HDUs and metadata;
- keep raster WCS in sidecar files;
- reject incompatible forced routes with a clear error;
- avoid silent fallbacks between solver implementations;
- report partial Blind 4D coverage honestly;
- keep the solver core independent from the GUI;
- ensure cancellation cannot leave a file advertised as solved with an incomplete header;
- ensure each file receives at most one terminal GUI result per run.

## Contributing

Contributions, testing, bug reports, and technical discussion are welcome.

Because ZeSolver is still evolving rapidly, contributors should first read the current project
mission and the relevant architecture or stabilization notes under `docs/`.

Changes to solver thresholds, catalogue formats, WCS acceptance rules, FITS-writing behavior, or
routing logic should be isolated, tested, and justified by reproducible evidence.

## Licence

ZeSolver source code is released under the **MIT Licence**, subject to the contents of the
repository's licence and notice files.

This MIT licence applies to ZeSolver's own source code. It does not grant rights to redistribute
third-party catalogue data, upstream software, trademarks, or other external assets.

ASTAP, HNSKY, Astrometry.net, Gaia, ESA, and all other third-party names and materials remain the
property of their respective owners and are governed by their own terms.

## Contact

Project repository:

https://github.com/tinystork/ZeSolver

Maintainer:

**Tristan Nauleau — Tinystork**
