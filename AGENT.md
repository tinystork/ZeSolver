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
---

## Strategic Roadmap: From Monolithic to Tiled Architecture

**Diagnosis:** Analysis has shown that `zeblind`'s slow performance and low success rate are caused by two primary architectural issues:
1.  **Monolithic Index:** A single, large `.npz` index file is loaded entirely into memory, causing extremely long start-up times and high RAM usage.
2.  **Brittle Quad Sampling:** The method for generating quad hashes (based on the "brightest-N" stars) is not robust, leading to a low probability of matching image hashes with index hashes.

**Objective:** Refactor `zeblind` to implement a high-performance, tiled architecture inspired by the successful concepts of `astap` and `astrometry.net`, while remaining pure Python.

**Action Plan Summary (See `followup.md` for full details):**

### Phase 1: Transition to a Tiled/Bucketed Index
- **Action:** Replace the single `.npz` index with a "bucketed" structure of many small, independent files. Each file will contain quads for a specific range of hashes (e.g., based on the first 12 bits of the hash).
- **Implementation:**
  - Modify the index builder (`tools/build_blind_index.py`) to write to these bucket files.
  - Rewrite the query logic (`zesolver/blindindex.py`) to use **memory-mapping (`mmap`)** to open only the necessary bucket files on-demand.
- **Outcome:** Near-instantaneous load times and drastically reduced memory consumption.

### Phase 2: Implement Robust, Geometric Quad Sampling
- **Action:** Replace the "brightest-N" quad sampling logic with a geometrically stable algorithm.
- **Implementation:**
  - In both the indexer and the solver (`zeblindsolver/asterisms.py`), implement a sampling method that generates quads based on pairs of stars and their neighbors at multiple distance scales. This makes hash generation far more reliable and resilient to star magnitude variations.
- **Outcome:** A massive increase in the solver's success rate.

### Phase 3: (Recommended) Enhance Star Detection
- **Action:** Improve the quality of the initial star list.
- **Implementation:**
  - In `zeblindsolver/star_detect.py`, move from a global threshold to an **adaptive local threshold** to better handle varying sky backgrounds.
- **Outcome:** A cleaner input signal for the entire solving pipeline, further boosting reliability.

For a detailed technical guide on how to implement these changes, refer to the **`followup.md`** file.


## Current Focus (2026-04-11)

A focused audit was run on speed + success-rate versus ASTAP.

### Observed baseline

- Repository benchmark (`benchmark_report.json`): **6 / 111** successful solves.
- Most failures return `no valid solution`; failures are also costly in runtime.
- `downsample-2` is currently the only profile with consistent wins (5/22).

### Suspected root causes

1. Metadata-assisted near solve is brittle when headers are incomplete/noisy (RA/DEC/scale hints).
2. Quad/hash robustness issue remains (test path can hit `no quads were hashed for level L`).
3. Validation thresholds and solve acceptance criteria need harmonization (near vs blind).
4. Candidate + matching loops still spend too much CPU in failing branches.

### Implementation priority

- P0 Robustness first, then P1 quality gates, then P2 throughput optimization.
- Any optimization that reduces success rate is out of scope.
- Keep full Python compliance (no ASTAP executable/code integration).

### Working log convention

A repo-local `memory.md` file tracks discoveries, actions, and outcomes chronologically.
Update it as work progresses (diagnostics, patches, benchmark deltas, regressions).

## Current Focus (2026-04-12, throughput pivot)

Reliability has improved substantially on heavy runs (first 300+ images reported stable with ZeNear).
The primary bottleneck is now end-to-end throughput for the 10k+ images objective.

### Updated priority order

1. **P0 Throughput engineering (CPU first):**
   - optimize worker count, I/O concurrency, and tile cache behavior,
   - reduce expensive fail-path work,
   - benchmark images/min at scale (500 -> 2k -> 10k).
2. **P1 GPU re-activation (targeted stages):**
   - restore effective GPU usage where it pays off (star detection first),
   - keep robust automatic CPU fallback.
3. **P2 Hybrid pipeline (CPU + GPU simultaneously):**
   - overlap stages (GPU detect on N+1 while CPU matches/validates N),
   - pursue measurable throughput gains without sacrificing reproducibility.

### Guardrails for this phase

- Do not trade away solve stability for raw speed.
- Keep deterministic behavior auditable (seeded RANSAC paths).
- Measure before/after every tuning step on the same dataset slice.


## Audit précis ASTAP-main (2026-04-14)

### 1) État des lieux du code source ASTAP utilisé pour la résolution

Fichiers cœur (lus et tracés):

- `ASTAP-main/unit_astrometric_solving.pas`
  - `solve_image(...)` (ligne ~746): orchestration complète de la résolution.
  - `bin_and_find_stars(...)` (ligne ~473): binning/crop + fond + détection étoiles.
  - `read_stars(...)` (ligne ~238): lecture catalogue autour d’un centre RA/Dec.
- `ASTAP-main/unit_star_align.pas`
  - `find_quads(...)` (ligne ~505), `find_many_quads(...)` (ligne ~228): génération des quads.
  - `find_fit(...)` (ligne ~972), `find_fit_using_hash(...)` (ligne ~1083): matching quads.
  - `find_offset_and_rotation(...)` (ligne ~1651): choix brute-force/hash + fit affine LSQ.
- `ASTAP-main/command-line_version/unit_command_line_solving.pas`
  - version CLI avec la même logique de base (constantes/formules cohérentes avec les deux unités ci-dessus).

### 2) Méthode ASTAP de résolution (factuelle, avec constantes)

1. **Initialisation FOV / base de données**
   - `max_fov := 5.142857143` pour DB `.1476` (unit_astrometric_solving.pas:821).
   - `max_fov := 9.53` pour DB `.290` (ligne 824).
   - Auto-FOV:
     - départ à `9.5°` (ou `90°` pour wide DB), puis division par `1.5` à chaque boucle (`fov_org := fov_org / 1.5`, lignes 847, 857).
     - borne basse `fov_min := 0.38` (ou `12` en wide, ligne 848).

2. **Détection étoiles image**
   - Binning auto: `Result := max(Result, round(1.5 / arcsec_per_px))` (ligne 563).
   - `hfd_min := max(0.8, min_star_size_arcsec / (binning * arcsec_per_px))` (dans `solve_image`).
   - Pipeline concret: `get_background(...)` puis `find_stars(...)` (lignes 500-501, 545).

3. **Génération quads image**
   - Si `nrstars_image < 30` -> `find_many_quads(..., mode=6)` (unit_star_align.pas:525-527).
   - Si `< 60` -> `mode=5` (lignes 531-533).
   - Sinon `find_quads` standard.
   - `minimum_quads := 3 + nrstars_image div 140` (unit_astrometric_solving.pas:968).

4. **Balayage du ciel (spirale carrée)**
   - `STEP_SIZE := search_field` (ligne 983), donc pas = FOV courant.
   - `max_distance := round(radius / (fov2 + 0.00001))` (ligne 991) en DB standard.
   - Filtre géométrique: séparation angulaire `<= radius + step_size/2` (ligne 1055).

5. **Lecture catalogue à chaque centre testé**
   - `read_stars(...)` fait un split sur 1..4 zones via `find_areas(...)` (ligne 254).
   - Quotas par zone proportionnels `frac1..frac4` (ex. ligne 260).

6. **Matching quads et fit**
   - `find_offset_and_rotation(...)`:
     - si `nrquads < 180` -> `find_fit` brute force (unit_star_align.pas:1664-1666),
     - sinon `find_fit_using_hash` (ligne 1674).
   - `find_fit`: test de 5 ratios normalisés (indices 1..5) sous `quad_tolerance` (ligne 994).
   - Filtrage ratio médian:
     - `median_ratio := smedian(...)` (ligne 1034),
     - garde si `abs(median_ratio-ratio)<=quad_tolerance*median_ratio` (ligne 1046).
   - Fit affine LSQ sur centres de quads (`A_XYpositions`, `b_Xrefpositions`, `b_Yrefpositions`).
   - Garde-fou final: `xy_sqr_ratio` doit rester dans `[0.9, 1.1]` (lignes 1692-1693).

7. **Double passe précision**
   - boucle `match_nr` jusqu’à `match_nr >= 2` (unit_astrometric_solving.pas:1218),
   - 2e passe avec recadrage sur la solution 1 pour améliorer la précision.

8. **Écriture WCS + SIP**
   - WCS TAN via `CRPIX/CRVAL/CD/CROTA`.
   - SIP seulement si activé et assez de correspondances (`if len < 20 then` abandon SIP, lignes 625-627; appel ligne 1298).

### 3) Comparaison précise avec notre ZeSolver actuel

Référence code ZeSolver: `zeblindsolver/metadata_solver.py`.

1. **Entrée et préfiltrage spatial**
   - **ZeSolver**: échec immédiat si RA/DEC absents (`metadata RA/DEC missing`, lignes 1533-1534).
   - **ASTAP**: pas de garde équivalente dans `solve_image`; il travaille avec la position de départ courante.
   - **ZeSolver**: préfiltrage manifeste tuiles (`_select_tiles`, lignes 220, 1563) avec cap `max_tile_candidates=48` (ligne 71).
   - **ASTAP**: lecture DB directement au fil de la spirale, sans manifeste indexé intermédiaire.

2. **Détection étoiles**
   - **ZeSolver**: `detect_stars` NumPy/SciPy/CUDA (ou ASTAP extract optionnel), seuils `k_sigma/min_area/max_labels` (NearSolveConfig lignes 87-89).
   - **ASTAP**: `get_background + find_stars` avec logique HFD/SNR interne Pascal.

3. **Noyau quads**
   - **ZeSolver**: port ISO ASTAP (`_astap_iso_find_quads`, `_astap_iso_find_fit_using_hash`) avec sweep tolérances `0.007..0.060` (ligne 1092).
   - **ASTAP**: matching quads + LSQ via `find_fit` / `find_fit_using_hash`.
   - Point important: le comportement mode-6 de la source ASTAP actuelle est atypique (combinaisons dupliquées et 4e index non exploité dans ce bloc: `x4 := ... quad_indices[2]`, unit_star_align.pas:409). Le port Python reproduit volontairement ce comportement (`combos6`, lignes 756+).

4. **Chemins supplémentaires non-ASTAP**
   - **ZeSolver** ajoute des couches absentes d’ASTAP:
     - `_build_candidate_pairs` (ligne 1339),
     - hypothèse quad additionnelle `_find_quad_hypothesis` (ligne 584),
     - RANSAC global `estimate_similarity_RANSAC(...)` (lignes 2230, 2271).
   - **ASTAP** n’utilise pas ce mix pair-votes + RANSAC dans ce flux near.

5. **Gates d’acceptation solution**
   - **ZeSolver** applique 3 niveaux de rejet:
     1) `validate_solution` RMS/inliers (lignes 2403+),
     2) `validate_wcs_for_zemosaic` (ligne 2471),
     3) `_near_conformance_check` (ligne 1435, appelé après).
   - **ASTAP**: après fit géométrique valide, écrit WCS; il log des warnings d’échelle mais n’empile pas ces mêmes gates.

6. **Profil actuel des écarts qui comptent pour la parité**
   - Nous ne sommes pas en “ASTAP pur”: notre flux ajoute des branches de matching (pairs/RANSAC) et des critères de rejet supplémentaires.
   - Notre near est plus contraint en post-validation, ce qui peut transformer des hypothèses “acceptables ASTAP” en échecs ZeSolver.
   - Le préfiltrage par manifeste + fenêtre locale peut changer le set d’étoiles comparé au `read_stars` ASTAP en spirale.

### 4) Conclusion opérationnelle (non conceptuelle)

Pour comparer à iso-comportement ASTAP, il faut exécuter un mode diagnostic qui:

1. garde uniquement la chaîne `ASTAP-ISO quads -> find_fit_using_hash -> LSQ`,
2. désactive les branches `candidate_pairs/quad_hypothesis/RANSAC`,
3. désactive temporairement les gates additionnels non-ASTAP (`zemosaic` + `near_conformance`).

Sinon, on compare ASTAP à un pipeline différent, et le diagnostic reste mécaniquement biaisé.


## Current Focus (2026-04-21, post-parity strict)

### Mission ASTAP-gap: statut
- La mission "écart ZeNear vs ASTAP" est considérée **accomplie pour le scope actuel** (mode strict ASTAP-ISO).
- Validation side-by-side exécutée ASTAP CLI vs ZeNear strict sur lot NGC6888 co-résolu:
  - séparation centre médiane ~3.14",
  - écart scale médian ~0.0034%,
  - écart rotation médian ~0.291°.
- Test mixte opérationnel préparé pour ZeMosaic sur M106 (50% ASTAP / 50% ZeNear) dans:
  - `/home/tristan/zemosaic/example/testzenear/`.

### Criticité restante (ZeSolver)
- **Aucun P0 bloquant immédiat** identifié sur la parité stricte ASTAP pour ce scope.
- Backlog technique (P1):
  1. benchmark C (ASTAP CLI) multi-champs complet, pour clôture statistique large,
  2. stratégie non-strict en mode rescue (profils safe/balanced/aggressive).

### Nouveau cap prioritaire (P0 produit)
- Passer en mode **scale-up/throughput** pour lots massifs (1k -> 10k images):
  1. stabilité long run (aucun faux échec de fin de run),
  2. saturation CPU/GPU contrôlée et reproductible,
  3. instrumentation débit/latence/mémoire robuste pour tuning itératif.
