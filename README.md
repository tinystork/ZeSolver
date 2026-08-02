# ZeSolver

**ZeSolver** is an open-source, offline WCS (World Coordinate System) solver designed for
large astronomical imaging batches, with a particular focus on Seestar workflows.

It was created for the needs of **The Seestar Collective**, a community of amateur astronomers
who may collect several thousand — sometimes more than 10,000 — short exposures of a target
before stacking them and assembling mosaics.

ZeSolver solves these images locally, writes validated astrometric metadata, and is designed to
preserve the original image pixels.

> [!IMPORTANT]
> ZeSolver is an independent community project. It is not affiliated with, maintained by, or
> officially endorsed by ASTAP, HNSKY, Astrometry.net, ESA, or the Gaia project.

## Current status

ZeSolver is currently in **Release Candidate Acceptance**.

The solver core, Near-to-Blind batch pipeline, catalogue library, first-run wizard, WCS safeguards,
progress reporting, cancellation, theme selection, catalogue download recovery, and optional GPU
diagnostics are functional.

The current state should be understood as a **source-based beta / release candidate**, not as a
completed stable packaged release:

* the Linux source workflow and a complete `READY_FULL` catalogue library have been validated;
* the final clean-profile Windows package acceptance gate is still pending;
* automated macOS compatibility checks pass on Apple Silicon, but no physical Mac runtime or
  clean-machine GUI installation has yet been validated;
* no signed or notarized macOS application is currently provided;
* packaged GPU support remains undefined and CPU operation is the reference fallback.

A version number in `pyproject.toml` does not, by itself, mean that all release gates or packaged
artifacts have been completed.

## Platform support

| Platform            | Current state                                                                                                                    |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Linux               | Source workflow and CPU runtime validated. Optional NVIDIA CUDA acceleration is available in supported source environments.      |
| Windows             | Source/runtime development is supported. Final acceptance of the exact user-facing package on a clean profile is still required. |
| macOS Apple Silicon | Automated CI and compatibility audit passed. Physical GUI/runtime validation is still pending.                                   |
| macOS Intel         | Not validated.                                                                                                                   |
| macOS GPU           | CPU only. CUDA/CuPy provisioning is not supported.                                                                               |
| Frozen/package GPU  | Diagnostic-only until a dedicated packaging flow is validated.                                                                   |

## Main features

* Offline batch WCS solving for FITS images.
* Fast metadata-assisted **ZeNear** solving.
* Local **ZeBlind 4D** fallback when Near cannot solve an image.
* Optional Astrometry.net web fallback as a final external route.
* Direct reading of ASTAP/HNSKY `.1476` and `.290` catalogue formats.
* Unified **ZeSolver Library** for catalogue data, Blind 4D indexes, provenance, and coverage.
* Official multi-component catalogue downloads with integrity checks and recovery controls.
* Installation from an official distribution, an existing library, or a local package.
* Library verification and repair tools.
* FITS pixel-integrity safeguards and configurable existing-WCS handling.
* TAN WCS output with validated metadata updates.
* JSON WCS sidecars for supported raster formats.
* Real-time batch progress, per-file status, cancellation, and safe restart.
* Optional sorting of scientifically unresolved files into `unresolved_by_zesolver`.
* System, light, and dark interface themes.
* CPU operation by default.
* Optional CUDA acceleration for ZeNear star detection.
* GUI and headless command-line workflows.

## How ZeSolver works

The normal solving chain is:

```text
ZeNear
  -> ZeBlind 4D
  -> optional Astrometry.net web fallback
```

### ZeNear

**ZeNear** is the fast, metadata-assisted solver.

When approximate sky coordinates and optical information are available, it reads nearby stars
directly from locally installed ASTAP/HNSKY catalogue shards through ZeSolver's own Python
catalogue provider. It detects image stars, matches them against catalogue stars, validates the
solution, and writes a TAN WCS only when the configured quality checks pass.

The ASTAP executable is **not invoked** by the current ZeNear product path.

### ZeBlind 4D

**ZeBlind 4D** is the fully blind local solver.

It uses precomputed geometric quad indexes derived from catalogue data. Its general blind-solving
approach is conceptually inspired by the geometric indexing principles developed by Astrometry.net,
while the implementation and runtime integration in this repository are part of ZeSolver.

A complete ZeSolver Library provides both the catalogue resources used by ZeNear and the derived
Blind 4D indexes used by ZeBlind. This keeps catalogue origin, generation parameters, versions,
fingerprints, and sky coverage related and verifiable.

### Optional Astrometry.net web fallback

An Astrometry.net web fallback can be configured as a final route after the local Near and Blind
solvers.

When this option is enabled, images may be uploaded to a third-party service. Users are responsible
for their own API credentials, upload rights, network access, and privacy choices.

## Requirements

* Python 3.10 or newer.
* PySide6 for the desktop GUI.
* A ZeSolver Library or a compatible ASTAP/HNSKY catalogue installation for catalogue-backed
  offline solving.
* Several gigabytes of free storage for a complete catalogue library. The startup wizard calculates
  and displays the required download, installation, cache, and safety-margin space.
* CUDA and CuPy only when optional NVIDIA acceleration is desired.

ZeSolver remains fully usable on CPU without CUDA, CuPy, or an NVIDIA GPU.

The source repository does not contain the large catalogue packages or precomputed indexes. They are
installed separately and are governed by their own data-distribution terms.

## Install from source

Clone the public repository:

```bash
git clone https://github.com/tinystork/ZeSolver.git
cd ZeSolver
```

### Linux and macOS

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[gui]"

python zesolver.py
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[gui]"

python zesolver.py
```

Running `python zesolver.py` without CLI input arguments opens the desktop interface.

> [!NOTE]
> CPU-only installation is the default and does not require CUDA or CuPy.

## First launch and catalogue setup

On a fresh profile, ZeSolver opens a startup wizard. The wizard can also be launched again later
from the **Interface** menu.

The available catalogue routes are:

### Install the official ZeSolver Library

This is the recommended route for normal users.

The wizard can download a prepared library made of several components. The distribution layer can:

* use multiple download sources;
* download independent components in parallel;
* pause, resume, cancel, retry, and resume interrupted transfers;
* reuse already verified cache files;
* verify component size and SHA-256 integrity;
* check storage space before installation;
* install through a staging directory;
* activate the final library only after successful validation.

An interrupted or failed installation must not be presented as a valid library.

### Use an existing ZeSolver Library

Select the library root containing `catalog.json`.

ZeSolver validates the library before activation. A successful activation sets the normal product
modes automatically:

```text
catalog_library_path=<selected library root>
near_catalog_mode=auto
blind4d_catalog_mode=auto
```

Inactive historical paths may remain stored for diagnostics, but they are not validated or used
while the corresponding product mode is `auto`.

### Use an existing ASTAP catalogue

This route provides **ZeNear only** when a complete ZeSolver Library is not available.

The expected product modes are:

```text
near_catalog_mode=astap-native
blind4d_catalog_mode=auto
```

ZeBlind 4D remains unavailable until compatible Blind 4D indexes are installed or generated.

### Install a local ZeSolver Library package

This advanced route installs and validates a prepared local package without downloading the
official distribution.

### Configure later

ZeSolver can be opened without completing catalogue setup, but offline catalogue-backed solving
will remain unavailable until a valid source is configured.

## ZeSolver Library

A **ZeSolver Library** is the normal catalogue configuration used by the product.

A library can relate:

* the ASTAP/HNSKY catalogue families used by ZeNear;
* the derived Blind 4D indexes used by ZeBlind;
* generation settings and format versions;
* file fingerprints and provenance;
* installed sky coverage and known limitations;
* the legal and attribution files required for redistribution.

The GUI includes a library manager for:

* installing a prepared local package;
* creating a library from an existing ASTAP catalogue installation;
* validating an existing library;
* repairing missing or invalid derived resources.

Legacy database roots, historical Near indexes, and external Blind 4D manifests remain available
only as explicit advanced compatibility or diagnostic overrides.

### Library states

A validated library may report states such as:

* `READY_FULL`: Near and Blind resources are available;
* `READY_PARTIAL`: usable resources exist, but Blind coverage is incomplete;
* `NEAR_ONLY`: Near catalogue resources are available without Blind indexes;
* `BLIND4D_ONLY`: Blind resources are available without a Near source;
* invalid or incomplete states that require repair.

Coverage warnings must not be hidden. A partial Blind 4D library is valid for limited use or
development, but it is not equivalent to all-sky coverage.

For the currently validated complete D50 library, the expected diagnostic state is:

```text
status=READY_FULL
near_catalog_mode_effective=astap-native
near_catalog_source=library
blind4d_catalog_mode_effective=library-view
blind4d_index_count=47
blind4d_covered_tiles=1476
blind4d_total_tiles=1476
blind4d_all_sky=true
blind4d_external_fallback_used=false
```

These values describe the currently validated full D50 distribution and should not be used to
misrepresent another library with different or partial coverage.

### Default storage locations

Installed libraries default to:

```text
~/ZeSolverCatalog/libraries
```

The download cache defaults to:

```text
Linux:   ~/.cache/ZeSolver/catalogs
macOS:   ~/Library/Caches/ZeSolver/catalogs
Windows: %LOCALAPPDATA%\ZeSolver\catalogs
```

The destination can be changed in the wizard. ZeSolver rejects unsafe destinations such as a
partial staging path, the download cache itself, the application directory, or a macOS `.app`
bundle.

## Normal GUI workflow

1. Start ZeSolver and complete or verify the catalogue setup.
2. Select the directory containing the images.
3. Scan the directory.
4. Review the WCS overwrite policy and solver options.
5. Start the batch.
6. Follow per-file status and overall progress.
7. Use **Stop** when needed; cancellation is designed to finish cleanly without publishing an
   incomplete WCS as solved.

The default product route is Near first, followed lazily by Blind only for unresolved images.

The interface provides Easy, Wizard, and Expert surfaces. Historical and diagnostic paths are kept
outside the normal simplified workflow.

## Image and WCS safety

ZeSolver follows these rules:

* image pixels must not be modified by WCS solving or WCS cleanup;
* an existing WCS must not be overwritten silently;
* unrelated FITS HDUs and metadata must be preserved;
* a file must not be reported as solved until its WCS write is complete;
* each file receives at most one terminal result per run;
* unsupported or incompatible forced routes must fail explicitly rather than silently switching
  implementations.

For FITS files, the main displayed WCS state refers to the `PRIMARY` HDU.

For supported raster formats, ZeSolver stores WCS information in a JSON sidecar instead of modifying
the source raster.

## Unresolved files

The GUI can optionally move scientifically unresolved images into:

```text
unresolved_by_zesolver/
```

This action occurs only after a batch has completed normally.

It is deliberately limited:

* terminal scientific non-solves are eligible;
* cancelled files are not moved;
* technical failures are not moved unless their terminal reason explicitly represents an eligible
  non-solve;
* files with a successfully written WCS are never moved;
* relative subdirectories are preserved;
* existing destination files are not overwritten;
* known ZeSolver sidecars are moved with the image;
* a timestamped JSON manifest records the move result.

This directory is intended to make later inspection, fallback processing, or dataset cleanup easier
without mixing technical errors with genuine unsolved images.

## Headless CLI

The main entry point also supports headless batch processing.

A normal library-backed invocation is:

```bash
python zesolver.py \
  --headless \
  --catalog-library "/path/to/ZeSolver-Library" \
  --input-dir "/path/to/images" \
  --workers 4
```

On Windows PowerShell:

```powershell
python zesolver.py `
  --headless `
  --catalog-library "C:\Path\To\ZeSolver-Library" `
  --input-dir "C:\Path\To\Images" `
  --workers 4
```

Useful discovery commands:

```bash
python zesolver.py --help
python -m zesolver.gpu_diagnostic --json --show-install-plan
```

Historical `--db-root`, legacy index, family, and external-manifest options remain available for
advanced compatibility and diagnostics. They are not the recommended starting point for a normal
installation.

## Optional GPU acceleration

ZeSolver works fully on CPU by default.

CUDA and CuPy are currently used only for optional **ZeNear star-detection acceleration**. They do
not replace the CPU solver and are not required for ZeBlind 4D.

Run the diagnostic without installing anything:

```bash
python -m zesolver.gpu_diagnostic --json --show-install-plan
```

In a safe source-managed virtual environment, the startup wizard may propose a guided CuPy
installation after explicit confirmation. The provisioner does not install:

* NVIDIA drivers;
* the system CUDA Toolkit;
* operating-system packages;
* more than one CuPy variant.

Frozen executables, embedded hosts, system Python installations, and unproven environments remain
diagnostic-only unless a future packaging flow explicitly supports mutation.

The declared `gpu` extra currently targets Linux source environments:

```bash
python -m pip install -e ".[gui,gpu]"
```

Manual installation for a locally validated CUDA 12 environment can use:

```bash
python -m pip install "cupy-cuda12x[ctk]"
```

Never install more than one of the following in the same environment:

```text
cupy
cupy-cuda12x
cupy-cuda13x
```

Environment controls:

```text
ZESOLVER_DISABLE_GPU_PROVISIONING=1
ZESOLVER_ALLOW_GPU_PROVISIONING=1
```

The advanced allow override does not make an unsafe system Python installation suitable for
modification.

When CUDA or a required runtime library is unavailable, ZeSolver falls back to CPU and records the
reason. A permanent missing-GPU condition is selected once per batch instead of producing the same
CuPy error for every image.

### Windows GPU note

The GPU policy contains a Windows source-environment candidate path, but final Windows validation is
still required before it becomes a public packaging promise. The current `pyproject.toml` GPU extra
is intentionally declared for Linux only.

### macOS GPU note

CUDA provisioning is not supported on macOS. ZeSolver uses the CPU backend.

## WCS cleanup

The GUI includes a non-blocking WCS cleanup workflow for FITS files.

A standalone helper is also available:

```bash
python zewcscleaner.py
```

Cleanup must preserve image pixels and only remove or update the requested WCS metadata.

## macOS experimental support

macOS support is currently **experimental**.

What has been validated:

* automated CI on Apple Silicon (`arm64`) with Python 3.11;
* imports of the main scientific and GUI dependencies;
* `spawn`-based multiprocessing behavior;
* Unicode and space-containing user, cache, temporary, and FITS paths;
* macOS catalogue-library and cache path selection;
* Qt offscreen widget startup;
* application icon availability;
* CPU operation when CuPy is absent;
* a real `zesolver.py --help` subprocess launch.

What has not yet been validated:

* a complete interactive run on a physical Mac;
* a clean-machine GUI installation by an end user;
* an Intel Mac;
* a public `.app` or DMG;
* code signing, notarization, and Gatekeeper distribution behavior.

Before a production-oriented macOS source test, run:

```bash
.venv/bin/python -m zesolver.macos_preflight --strict-gui
```

Without `--strict-gui`, a missing PySide6 installation or Qt offscreen failure may be reported as a
warning rather than a blocking failure.

A successful preflight is a compatibility check. It does **not** mean that the macOS runtime has
been validated on a physical machine.

## Advanced: building derived indexes

> [!WARNING]
> This is a low-level advanced workflow. Normal users should install a complete prepared ZeSolver
> Library or use the GUI library manager to create one from an existing ASTAP installation.

The `zebuildindex` entry point, or the underlying module, can generate ZeSolver-friendly tiles and
quad indexes from a local ASTAP/HNSKY catalogue installation.

Linux/macOS example:

```bash
python -m zeblindsolver.db_convert \
  --db-root "/path/to/astap/catalogues" \
  --index-root "/path/to/output/index" \
  --max-quads-per-tile 20000 \
  --quad-storage npy \
  --tile-compression uncompressed \
  --workers 8
```

Windows PowerShell example:

```powershell
python -m zeblindsolver.db_convert `
  --db-root "C:\Path\To\ASTAP" `
  --index-root "C:\Path\To\Output\Index" `
  --max-quads-per-tile 20000 `
  --quad-storage npy `
  --tile-compression uncompressed `
  --workers 8
```

Relevant storage options include:

* `--quad-storage npz` for compressed `.npz` files;
* `--quad-storage npz_uncompressed` for store-only `.npz` files;
* `--quad-storage npy` for memory-mapped `.npy` bundles;
* `--tile-compression compressed`;
* `--tile-compression uncompressed`.

When only quad generation needs to be repeated, a quads-only rebuild can reuse existing tile data:

```bash
python -m zeblindsolver.db_convert \
  --db-root "/path/to/astap/catalogues" \
  --index-root "/path/to/output/index" \
  --mag-cap 15.5 \
  --max-stars 2000 \
  --max-quads-per-tile 20000 \
  --quad-storage npy \
  --tile-compression uncompressed \
  --workers 8 \
  --quads-only
```

Generated assets must be validated before use or redistribution. A low-level index directory is not
automatically equivalent to a fully activated `READY_FULL` ZeSolver Library.

## Catalogue and derived-index distribution

The ZeSolver source repository does not contain the large original ASTAP/HNSKY catalogue packages
or precomputed ZeSolver indexes. These assets may be provided as separate optional downloads.

Written permission was received from **Han Kleijn** on 20 July 2026 to use ASTAP/HNSKY databases as
intermediate catalogue data and to redistribute the relevant catalogue data for non-commercial use,
subject to attribution and the applicable ESA/Gaia/DPAC conditions.

The project may therefore distribute:

* supported ASTAP/HNSKY catalogue packages;
* ZeSolver-derived normalized tiles;
* `.npz` or `.npy` data;
* quad tables and geometric hash indexes;
* associated manifests and provenance metadata.

Catalogue packages and catalogue-derived assets are distributed for **non-commercial use only**.
They must credit Han Kleijn, ASTAP, and HNSKY, and must include the applicable ESA/Gaia/DPAC
attribution and copyright text.

Every redistributed catalogue or derived-index package must include:

```text
NOTICE.md
legal/ASTAP_HNSKY_DATA_TERMS.md
legal/GAIA_DATA_TERMS.txt
```

These data conditions are separate from the GNU GPL licence governing ZeSolver's own source code
and executable builds.

Users generating indexes locally remain responsible for obtaining catalogue data legitimately and
preserving all required upstream notices when redistributing derived assets.

## Acknowledgements

ZeSolver exists thanks to the work of the wider astronomical software and data community.

### ASTAP and HNSKY

Special thanks to **Han Kleijn**, developer of [ASTAP](https://www.hnsky.org/astap.htm) and HNSKY.

ASTAP has been a major technical reference throughout the development of ZeSolver, particularly for
catalogue organization, practical plate-solving workflows, and the metadata-assisted ZeNear path.
ZeSolver reads ASTAP/HNSKY `.1476` and `.290` catalogue formats through an independent Python
implementation.

ZeSolver does not include ASTAP source code, does not invoke the ASTAP executable in its current
normal local solving chain, and does not claim to be an official ASTAP component.

### Astrometry.net

ZeBlind 4D was informed by the geometric blind-solving principles developed by
[Astrometry.net](https://astrometry.net/).

Credit is due to **Dustin Lang, David W. Hogg, Keir Mierle**, and the other Astrometry.net
contributors for their foundational work in blind astrometric calibration.

ZeSolver is an independent implementation and is not affiliated with or endorsed by Astrometry.net.

### Gaia and catalogue data

The stellar data used by ASTAP/HNSKY catalogue families originates from upstream astronomical
catalogues, including data produced by the European Space Agency Gaia mission and processed by the
Gaia Data Processing and Analysis Consortium.

All catalogue data remains subject to the rights, licences, acknowledgements, and distribution
conditions of its respective providers.

## Repository layout

The generated public `main` branch contains the runtime, public entry points, required resources,
licence notices, icons, and essential user documentation.

```text
zesolver/           Product orchestration, catalogue library, settings, GUI pipeline, and core
zeblindsolver/      Blind solver, quad generation, index builders, and validation
zewcs290/           Native ASTAP/HNSKY catalogue readers and spatial queries
config/             Small runtime manifests
icon/               Application icons
legal/              Third-party catalogue and data terms
docs/               Essential public user documentation
pyproject.toml      Build and dependency metadata
LICENSE             GNU GPL v3 licence text for ZeSolver source code
NOTICE.md           Third-party credits and distribution notices
```

Catalogue databases, generated indexes, large test corpora, local settings, logs, caches, and runtime
files must not be committed to the public repository tree.

## Packaging and releases

Official user-facing packages must be tested on clean environments before publication.

At minimum, release acceptance must verify:

* clean installation or extraction;
* first launch and startup wizard;
* official, existing-library, and local-package activation paths;
* close and restart persistence;
* `READY_FULL` library status;
* a successful Near solve;
* a successful Blind 4D fallback;
* interrupted-download recovery;
* system, light, and dark theme persistence;
* paths containing spaces and non-ASCII characters;
* clean Stop and cancellation behavior;
* absence of unexpected Python tracebacks or consoles;
* complete licence, credit, and third-party data notices.

PyInstaller helpers and internal packaging reports live on the development `test` branch. They may
not be present in the generated public `main` tree.

## Development and branch policy

The repository uses two branches with deliberately different roles:

* `test` is the development source of truth. It contains the complete codebase, tests, tools,
  reports, architecture notes, stabilization documents, and publication scripts.
* `main` is a generated public distribution tree containing the runtime and essential public files.

Contributions and pull requests should target `test`, not `main`.

Do not develop or apply direct fixes on `main`. Public `main` updates are generated from a validated
`test` commit through the repository's allowlisted publication workflow.

Changes to solver thresholds, catalogue formats, WCS acceptance rules, FITS-writing behavior, or
routing logic should be isolated, tested, and supported by reproducible evidence.

## Licence and third-party data

ZeSolver's own source code is released under the **GNU General Public License, version 3 or any
later version** (`GPL-3.0-or-later`).

See [`LICENSE`](LICENSE) for the complete licence text.

The GNU GPL applies to ZeSolver's source code and executable distributions. It does not replace or
override the separate conditions governing third-party catalogue data, catalogue-derived assets,
trademarks, or upstream projects.

Optional ASTAP/HNSKY catalogue packages and ZeSolver-derived catalogue indexes are distributed
separately for **non-commercial use only**, subject to attribution and the applicable ESA/Gaia/DPAC
notices.

See:

* [`NOTICE.md`](NOTICE.md)
* [`legal/ASTAP_HNSKY_DATA_TERMS.md`](legal/ASTAP_HNSKY_DATA_TERMS.md)
* [`legal/GAIA_DATA_TERMS.txt`](legal/GAIA_DATA_TERMS.txt)

ASTAP, HNSKY, Astrometry.net, Gaia, ESA, and all other third-party names and materials remain the
property of their respective owners. ZeSolver is an independent project and is not affiliated with
or endorsed by those projects or organizations.

## Contact

Project repository: https://github.com/tinystork/ZeSolver

Maintainer: **Tristan Nauleau — Tinystork**
