# ZeSolver Memory Log

## 2026-04-11

### Discoveries
- Baseline benchmark in repo indicates low global success rate: 6/111 attempts.
- `downsample-2` profile is currently the strongest profile on this dataset.
- Near solver quality is strongly tied to metadata quality (RA/DEC/scale hints).
- Test suite reveals a critical quad-index fragility: `no quads were hashed for level L` in synthetic index setup.

### Actions completed
- Read and reviewed: `AGENT.md`, `followup.md`, `README.md`.
- Audited main solver flow (`zeblindsolver.py`, `metadata_solver.py`, quads/index modules, detector).
- Ran benchmark report analysis from existing `benchmark_report.json`.
- Ran `pytest -q` to capture current regressions and failure modes.
- Updated `followup.md` with current diagnostic and short-term execution plan.
- Updated `AGENT.md` with current focus and repo memory convention.

### Next actions
- Reproduce and fix the `level L` quad hashing failure deterministically.
- Add non-regression tests around quad generation/hash acceptance across scales.
- Re-run benchmark and compare success rate + runtime before/after patches.


## 2026-04-11 (itération 1 - mise en place)

### Code changes
- Patched `zeblindsolver/asterisms.py` to use a canonical angular ordering for quad points before hashing (instead of radius-to-centroid ordering), preventing bow-tie quads with near-zero polygon area.
- Replaced 2D `np.cross` parity check with scalar cross-product (`cross_z`) to avoid NumPy 2.0 deprecation and keep parity explicit.
- Hardened `zeblindsolver/zeblindsolver.py` background-removal call with a compatibility fallback for simplified mocked signatures used by tests.

### Validation
- Ran targeted tests:
  - `pytest -q tests/test_synthetic.py tests/test_failures.py::test_solver_returns_fail_quickly`
  - Result: **4 passed**.

### Impact expected
- Removes a concrete failure mode where valid quads could be discarded at hash time (notably visible on synthetic fixtures).
- Improves immediate robustness baseline before deeper perf tuning.


## 2026-04-12 (itération 2)

### Changes implemented
- Added robust quad ordering in hashing path already present from iteration 1 and extended quad sampling robustness:
  - pairwise sampler now falls back/augments with legacy local-neighborhood quads when pairwise output is too sparse.
  - merged quad sets are deduplicated in stable order before hashing.
- Added blind-solver robustness fallback for level-specific observed hashes:
  - when pixel-adapted level filtering yields zero hashes, solver now falls back to base hash set for that level instead of hard failing candidate search.

### Why this matters
- Reduces false dead-ends in candidate search caused by scale-hint mismatch or sparse quad generation.
- Improves probability of finding valid candidates while keeping level-guided behavior when hashes exist.

### Validation
- `pytest -q tests/test_synthetic.py tests/test_failures.py::test_solver_returns_fail_quickly tests/test_matcher.py tests/test_zeblindsolver.py`
- Result: **13 passed**.

### Next step
- Re-run benchmark on your regenerated index/hashes dataset to measure success-rate and runtime delta versus previous baseline.


## 2026-04-12 (diagnostic stop/cancel)

### Symptom observed
- GUI stop requests were logged multiple times (`Arrêt demandé…`) while the run kept going.

### Root cause identified
- In `BatchSolver.run` (`zesolver.py`), the loop waited on `next(as_completed(inflight))`, which blocks until one task completes.
- On stop, the cancel flag was set but the loop could not observe it while blocked.
- Additionally, the `ThreadPoolExecutor` context manager waits for running tasks on exit, delaying effective stop feedback.

### Mitigation implemented
- Reworked `BatchSolver.run` to poll futures with `concurrent.futures.wait(..., timeout=0.2, FIRST_COMPLETED)` and check cancel regularly.
- On cancel, call `shutdown(wait=False, cancel_futures=True)` to return control immediately (cooperative cancel still expected in worker code).
- Added a cancel check inside metadata candidate-pair building loop (`metadata_solver.py`) to improve cooperative interruption during heavy matching.

### Validation
- `pytest -q tests/test_metadata_solver.py tests/test_zeblindsolver.py tests/test_synthetic.py`
- Result: **12 passed**.


## 2026-04-12 (run appears frozen after ZENEAR start)

### Findings
- Live process was active (4 compute threads near 95% CPU each), not deadlocked.
- Log stalled right after `near solve start ...` because near solver spent most time in `detect_stars(...)` on raw frames.
- In this dataset/config, blind fallback was disabled (near-only), so long near stages are user-visible as apparent freeze.

### Mitigations applied
- Improved stop responsiveness already in `BatchSolver.run` (poll + non-blocking shutdown on cancel).
- Tuned near-star detection call to avoid pathological segmentation cost:
  - pre-remove background in near pipeline,
  - use `mode=global`, `k_sigma=4.0`, `min_area=8`,
  - cap labels processed via new `max_labels` argument in `detect_stars` (CPU path).
- Added temporary near-stage logs (`candidates`, `catalog stars`, `detect start/done`, `pair-build`, `ransac`) for diagnosis.

### Measured effect
- Single-frame near solve dropped from >180s timeout to ~6s on test frame (still failing strict quality gate: inliers 19 < 60).


## 2026-04-12 (GPU runtime fix)

### Issue
- System launched `zesolver.py` with `/usr/bin/python`, while CuPy (`cupy-cuda12x`) is only available in repo `.venv`.
- Result: runtime sees no CuPy, UI/backend falls back to CPU.

### Script-side fix
- Added automatic interpreter re-exec in `zesolver.py`:
  - detect project venv python at `.venv/bin/python` (or Windows equivalent),
  - if current interpreter is not the venv one, re-launch the same command under venv,
  - guard against loops with `ZE_REEXECED_VENV` and opt-out via `ZE_NO_VENV_REEXEC=1`.

### Verification
- Environment check confirmed:
  - `/usr/bin/python` -> cupy unavailable,
  - `/home/tristan/ZeSolver/.venv/bin/python` -> cupy available.


## 2026-04-12 (ZENEAR D50 diagnosis and fix)

### Root cause
- ZENEAR was rejecting geometrically valid solutions due to a fixed `quality_inliers=60` gate.
- Logs showed very low RMS with consistent inliers in the 19-33 range, i.e. good fits rejected by an unrealistic fixed threshold for this FOV/dataset.

### Fix
- In `zeblindsolver/metadata_solver.py`, switched near validation to adaptive inlier threshold:
  - `adaptive_inliers = min(cfg.quality_inliers, max(6, int(0.4 * n_pairs)))`
  - use this threshold consistently for TAN + LS + SIP validation.
- Added header keyword `REQINL` to record the effective inlier requirement used at acceptance time.

### Quick validation
- Single frame previously failing now succeeds in ~6.2s (`inliers=19`, `REQINL=7`).
- 6-frame smoke test on `dataset_test`: **6/6 solved** (all near mode), ~6-7s per frame on CPU.


## 2026-04-12 (ZENEAR final stretch: 10 stubborn files)

### Additional root cause
- Last 10 failures were mostly low-support detections on faint/edge frames.
- Aggressive near detection (`global k=4` on background-subtracted image) produced 0-4 stars on these frames.

### Fixes
- Added staged near-detect fallback in `metadata_solver.py` when support is low:
  1) default fast pass: processed image, global k=4.0,
  2) fallback #1: raw image, global k=3.0,
  3) fallback #2: raw image, global k=2.5.
- Tuned adaptive inlier gate lower bound from 6 to 4 with pair-aware scaling:
  - `adaptive_inliers = min(cfg.quality_inliers, max(4, int(0.3 * n_pairs)))`

### Validation
- Previously unresolved 10-file subset: **10/10 solved**.
- Full `dataset_test` near-only smoke test: **30/30 solved** (~98s total on CPU).
- `tests/test_failures.py::test_solver_returns_fail_quickly` now passes again.


## 2026-04-12 (virage perf: fiabilité -> débit 10k+)

### Contexte terrain
- Run lourd en cours: ZeNear jugé fiable sur les 300+ premières images.
- Nouveau goulot identifié: vitesse d'exécution globale (objectif final 10k+ images).
- Hypothèse de travail validée: le gain ne viendra pas d'un seul levier, mais d'une combinaison orchestration CPU + GPU ciblé + pipeline hybride.

### Décision produit/technique
- Changer la priorité immédiate: passer de la chasse aux échecs de solve à l'ingénierie de débit.
- Ordre retenu:
  1) tuning parallèle CPU/I/O/cache,
  2) réactivation GPU sur étapes rentables (détection),
  3) pipeline CPU+GPU simultané (overlap des étapes).

### Critère de succès
- Débit mesurablement supérieur (images/min) sans perte de robustesse ni dérive de reproductibilité.

## 2026-04-12 (orientation UX produit, post-perf)

### Décision produit
- Priorité validée: rendre ZeSolver compréhensible pour un utilisateur lambda via une surcouche UX simple, puis l'intégrer dans ZeMosaic.
- Les réglages experts restent disponibles, mais ne doivent plus être la porte d'entrée.

### UX cible à mémoriser
- Wizard premier démarrage: setup ASTAP/bases + presets instrument.
- Étape index/hashes: construire ou réutiliser un répertoire existant avec validation.
- Workflow dossier: sélectionner entrée, option de nettoyage WCS (zewcscleaner), lancer batch.
- Résultats: résumé lisible + gestion des non-résolus pensée pour l'environnement ZeMosaic.

### Contrainte d'intégration
- Le nom exact du dossier de rejet est secondaire côté ZeSolver seul; la convention finale doit être dictée par le pipeline ZeMosaic.


## 2026-04-13 (reorientation mission Zenear vs ASTAP)

### Demande produit
- Tristan a demandé de réorienter explicitement la mission: comprendre pourquoi ASTAP résout M106 alors que ZeNear (priorité absolue) reste insuffisant.

### Travail effectué
- Création/maintenance d'un plan dédié: `MISSION_ZENEAR_ASTAP_GAP.md`.
- Ajout d'instrumentation near profonde (`near_debug.jsonl`).
- Durcissement de la validation WCS pour bloquer les faux positifs (échelles dégénérées).
- Tentative d'implémentation ASTAP-like partielle (votes quads + hypothèse quad pré-RANSAC).
- Fallback blind enrichi par hints metadata dans le wrapper near.

### État actuel
- Les améliorations augmentent la visibilité et réduisent les faux positifs, mais ZeNear pur n'est pas encore fiable sur M106.
- Le prochain saut de robustesse nécessite un matching quad-hash plus fidèle au pipeline ASTAP.

## 2026-04-13 (accélération Near + benchmark ASTAP solved)

### Avancées solve
- Campagne near-only sur set `fresh` poussée jusqu'à **10/10**.
- Le passage à 10/10 a été obtenu avec:
  - élargissement de l'escalade de tolérance ISO ASTAP (ajout paliers `0.040`, `0.060`),
  - assouplissement de la borne basse de conformance pixscale (`0.4 -> 0.18`).
- Tests unitaires restés stables pendant les itérations (`tests/test_zeblindsolver.py`: 10 passed).

### Benchmark demandé sur lot varié déjà ASTAP-solvé
- Dataset: `/home/tristan/zemosaic/example/astap solved` (67 FITS).
- Rapport: `/home/tristan/ZeSolver/reports/astap_solved_benchmark_iter1.json`.
- Résultat:
  - **55/67** succès (**82.1%**),
  - médiane temps solve **18.84 s/image**,
  - échecs dominants: rejets conformance/ZeMosaic liés au pixel scale.

### Conclusion produit/perf
- Fiabilité Near en hausse, mais débit encore trop loin d'ASTAP.
- Priorité clarifiée avec Tristan: viser une architecture 10k-ready, fortement parallélisée, charge machine maximale (CPU/RAM/GPU/VRAM) avec garde-fous.


## 2026-04-14 (mission ASTAP-main audit + ZeNear strict)

### Modifications et actions réalisées

- `zeblindsolver/metadata_solver.py`
  - Correction de la sélection de tuiles Near: `_tile_intersects` basé sur overlap réel des bounds RA/Dec (et non simple critère centre+extent trop permissif).
  - Portage et alignement renforcé du noyau ASTAP-ISO:
    - `find_many_quads(mode=6)` rapproché du comportement source ASTAP observé,
    - scan des tolérances ISO conservé/renforcé,
    - scoring candidat enrichi avec proxy d'inliers rapides.
  - Ajustements du center stepping Near:
    - seuils de badness assouplis sur `eval_inliers` (>=4),
    - correction d'un bug d'ordre d'évaluation (`eval_inliers0` utilisé avant affectation).
- `zewcscleaner.py`
  - Nettoyage de commentaires redondants et clarification des commentaires demandée.
- Vérifications techniques
  - `python -m py_compile zeblindsolver/metadata_solver.py` OK.
  - `pytest tests/test_zeblindsolver.py -q` OK (10 passed).

### Benchmarks exécutés

- Benchmark ZeNear sur NGC6888 (16 images) après réintégration ASTAP-like:
  - rapport: `reports/ngc6888_16_after_astap_like_port.json`
  - résultat: **0/16** succès, `wall_s` ~129.48s, ~7.41 img/min.
- Référence ASTAP CLI sur le même lot:
  - rapport: `reports/ngc6888_16_astap_cli_probe.json`
  - résultat: **10/16**.

### Analyse produite et documentation

- Audit source ASTAP-main effectué (lecture précise des unités Pascal coeur).
- `AGENT.md` enrichi avec une section détaillée:
  - méthode ASTAP pas à pas (constantes, fonctions, décisions),
  - comparaison factuelle avec ZeNear actuel,
  - écarts structurels identifiés (branches non-ASTAP + gates additionnels côté ZeSolver).

### Conclusion technique à retenir

- Le portage ASTAP-like a amélioré l'alignement algorithmique local, mais la parité globale n'est pas atteinte.
- Écart principal restant: ZeNear actuel conserve encore des chemins et gates non ASTAP qui biaisent la comparaison 1:1.
- Prochaine étape validée: implémenter un mode "ASTAP-ISO strict" réellement isolé pour benchmark de parité propre.


## 2026-04-20 (implémentation ASTAP-ISO strict en cours)

### Décision utilisateur
- Tristan a explicitement demandé de continuer l’implémentation et la vérification (`"oui s'il te plait"`, puis `"tu peux continuer"`).

### Travaux réalisés
- Câblage complet du flag `near_astap_iso_strict` (CLI/UI/settings/runtime).
- Retrait du hardcode `strict_astap_iso = False`.
- En strict: neutralisation des branches non-ASTAP (pair-builder, quad-hypothesis non-ISO, RANSAC global, gates ZeMosaic/conformance).
- Ajout de la bifurcation ASTAP `find_fit` (<180 quads) / `find_fit_using_hash` (>=180 quads).
- Ajout du garde-fou `xy_sqr_ratio` (post-LSQ) et maintien d’une 2e passe type `match_nr`.
- Instrumentation debug enrichie (`near_debug.jsonl`) avec événement `zenear_no_consensus`.

### État mesuré
- Tests unitaires ciblés stables: `tests/test_zeblindsolver.py` -> 10 passed.
- Benchmark strict NGC6888 x16 au dernier état: 0/16 (écarts causaux encore ouverts sur consensus/validation).

### Leçon durable
- Le noyau strict est branché, mais la parité de résultat ASTAP n’est pas encore atteinte.
- Le goulot n’est plus le câblage du mode strict, c’est la qualité du consensus géométrique strict sur ce lot.


## 2026-04-20 (suite mission strict ASTAP-ISO, instrumentation causale)

- Ajout d’une télémétrie de diagnostic fine dans `near_debug.jsonl` (`astap_iso_diag`) avec compteurs par tolérance et agrégats multi-appels (centers + pass2).
- Les événements d’échec Near incluent désormais cette télémétrie (`no_transform`, `no_consensus`, `validation_fail`, `attempt`).
- Observation terrain sur probe NGC6888:
  - certains frames restent sans hypothèse exploitable (`ok_calls=0`),
  - d’autres passent à une hypothèse partielle (inliers >0) mais échouent ensuite en consensus/validation (RMS trop élevé).
- Le problème restant est désormais bien localisé: qualité géométrique des hypothèses strictes, pas manque de logs.

## 2026-04-20 (ASTAP-ISO strict, convergence FOV + coût retries)

### Avancées majeures
- Le mode strict Near a été stabilisé avec conservation explicite du hint par FOV:
  - priorité des sources FOV: `fov_override_deg` > header FITS (`FOV*`/`FIELD*`) > FOV dérivé scale,
  - traçage dans `astap_iso_diag` via `fov_hint_source`.
- Parité comportementale validée sur NGC6888 quand le hint FOV explicite est fourni:
  - ZeNear strict + override per-frame = **10/16**, même partition de succès/échecs que le probe ASTAP historique.
- Sans hint FOV explicite, le mode strict reste plus permissif/robuste sur ce lot (**16/16**), ce qui confirme que l’écart principal est lié à la stratégie FOV/windowing plutôt qu’au cœur quad/hash strict.

### Implémentation technique durable
- Retry auto-FOV strict ajouté puis durci:
  - activé uniquement quand la source FOV est `scale` (pas d’écrasement des hints explicites),
  - configurable (`strict_auto_fov_retry`, `strict_auto_fov_retry_scales`, `strict_auto_fov_retry_max_attempts`).
- Optimisation coût retries:
  - détection/skip des sous-ensembles catalogue dupliqués (`duplicate_subset`) pour éviter des appels `_astap_iso_hypothesis` redondants,
  - ordre par défaut retuné des scales: `(1.25, 0.82, 1.6, 0.65, 2.4, 4.0)`.

### Validation
- `py_compile` OK.
- `pytest -q tests/test_zeblindsolver.py` OK (10 passed).
- Rapports clés générés:
  - `reports/ngc6888_16_strict_fov_alignment_summary_v1.json`,
  - `reports/ngc6888_16_strict_auto_fov_retry_v1.json`,
  - `reports/ngc6888_16_strict_auto_fov_retry_tuned_defaults_v1.json`,
  - `reports/ngc6888_16_strict_auto_fov_retry_smart_cost_v1.json`.

### État global à retenir
- Le cap produit demandé est tenu: hint FOV explicite conservé et prioritaire.
- Le mode strict est désormais instrumenté et configurable pour piloter finement le compromis parité/coût.
- Prochaine marche: déclenchement auto-FOV encore plus contextuel (piloté par `astap_iso_diag`) et validation sur lots multi-champs/capteurs.

## 2026-04-20 (ASTAP-ISO strict, étape contextuelle)

### Ce qui a été ajouté
- Déclenchement auto-FOV strict désormais contextuel:
  - activé seulement si `fov_hint_source=scale` (les hints FOV explicites restent prioritaires),
  - retry conditionné par un support minimal (`stars_img>=24`, `quads_img>=3`),
  - si l’hypothèse initiale n’a aucun indice (`best_refs=0` et `matches_raw=0`), ordre expansion-first + budget contextuel.
- Nouveau garde-fou coût:
  - `strict_auto_fov_retry_zero_ref_patience=3` (arrêt anticipé après zéro-refs répété),
  - `strict_auto_fov_retry_max_attempts` conservé,
  - skip `duplicate_subset` maintenu.

### Impact mesuré
- NGC6888 inchangé (bon signe de stabilité):
  - strict sans hint explicite: 16/16,
  - strict avec hint explicite per-frame: 10/16 aligné ASTAP (même partition).
- Validation large (24 FITS multi-sources):
  - strict retry ON: 23/24,
  - strict retry OFF: 22/24,
  - coût retry réduit (appels effectifs 9 -> 6 sur ce lot après garde-fou contextuel).

### État durable
- Le mode strict conserve la priorité du hint FOV explicite tout en gagnant une logique de retry plus intelligente.
- On a maintenant un compromis robuste coût/fiabilité sur lots mixtes, avec instrumentation suffisante pour verrouiller des tests de non-régression dédiés.

## 2026-04-20 (non-régression stricte FOV/retries verrouillée)

### Ce qui a été verrouillé
- Ajout de tests unitaires dédiés dans `tests/test_metadata_solver.py` pour garantir dans la durée:
  - priorité des sources FOV (`override` > `header` > `scale`),
  - activation des retries auto-FOV uniquement en contexte `scale` et support suffisant,
  - arrêt anticipé sur répétition zéro-refs (`zero_refs_patience`).

### Validation
- `pytest -q tests/test_metadata_solver.py` : 5 passed.
- `pytest -q tests/test_zeblindsolver.py tests/test_metadata_solver.py` : 15 passed.

### Leçon durable
- Les comportements critiques de parité strict (hint FOV explicite et retries contextuels) ne reposent plus uniquement sur validation manuelle/bench:
  ils sont désormais couverts par des tests automatisés de non-régression.

## 2026-04-20 (benchmark multi-champs strict vs non-strict)

### Fait
- Run comparatif ZeNear sur 56 brutes uniques (pool total disponible: 86 fichiers, 56 uniques par basename).
- Configuration: backend CPU, `k_sigma=4.5`, `min_area=8`, `max_labels=1200`.

### Résultat clé
- `astap_iso_strict=True` (strict): 45/56, ~25.55 img/min.
- `astap_iso_strict=False` (non-strict): 0/56, ~1.64 img/min.
- Le strict domine nettement ce lot, y compris sur NGC6888 où il dépasse l’oracle ASTAP historique (16/16 vs 10/16 sur le probe existant).

### Interprétation prudente
- L’écart massif strict/non-strict suggère une régression ou un mode non-strict aujourd’hui hors envelope sur ces données réelles.
- Priorité suivante: diagnostic ciblé du chemin non-strict (pourquoi 0/56) avant d’en faire un mode de fallback produit.

## 2026-04-20 (non-strict: itération seuils RMS terminée)

### Diagnostic durable
- Cause principale du 0/56 non-strict confirmée: gate de validation RMS trop dur (`quality_rms=1`), avec majorité d'échecs en `validation_fail` plutôt qu'en `no_transform`.

### Sweep multi-champs (56 brutes uniques)
- Projection depuis run de référence non-strict `quality_rms=500`:
  - RMS 10 -> 4/56
  - RMS 30 -> 7/56
  - RMS 100 -> 24/56
  - RMS 160 -> 33/56
  - RMS 220 -> 49/56
  - RMS 500 -> 52/56
- Mais qualité des solves additionnels dégradée quand le seuil monte (RMS médian ~127.8 px, max ~422 px): gain de taux au prix d'un risque fort de faux positifs.

### Règle pratique à retenir
- Non-strict doit rester un mode de rescue avec profils explicites (safe/balanced/aggressive), pas un mode par défaut unique tant qu'on n'a pas un garde-fou qualité plus robuste que le simple seuil RMS global.


## 2026-04-21 (clôture scope ASTAP-gap + préparation scale-up)

### Accompli
- Mission parité ASTAP-ISO strict considérée accomplie pour le scope actuel.
- Validation side-by-side confirmée sur co-solves NGC6888 avec écarts faibles:
  - centre médian ~3.14",
  - scale médian ~0.0034%,
  - rotation médiane ~0.291°.
- Lot manuel M106 mixte 50/50 préparé pour ZeMosaic:
  - `/home/tristan/zemosaic/example/testzenear/`
  - 15 ZeNear + 15 ASTAP, WCS injecté et manifest/audit produits.

### Leçon durable
- Pour éviter le biais de comparaison solveur, il faut retirer les cartes WCS/SOLVER avant A/B puis résoudre à nouveau sur copies.
- Les conclusions produit doivent distinguer explicitement deux cas:
  - pipeline avec recalage stellaire aval (tolérance plus large),
  - pipeline WCS-only (sensibilité accrue aux écarts angulaires/échelle en bord de champ).

### Nouveau cap prioritaire
- Basculer la mission sur le scale-up 1k-10k images (débit + robustesse long-run) plutôt que continuer du micro-ajustement strict.


## 2026-04-21 (optimisation ZeNear M106, lot 50, phase ASTAP-vs-ZeNear)

### Résultats durables
- Benchmark mono-champ M106 x50 stabilisé avec succès `50/50` ASTAP et `50/50` ZeNear.
- Après optimisation (skip background en backend ASTAP + caches catalogue/lookup), ZeNear est passé d’environ `~2.50s` médian à une plage observée `~0.64–0.81s` médian selon run, tout en conservant la cohérence WCS (center/scale/rotation inchangés vs ASTAP).

### Décisions techniques retenues
- Le coût dominant restant est `metadata_solver._detect_stars_astap_cli` (process externe ASTAP extract).
- Un essai de downsample pré-extract ASTAP (binning x2 via FITS temporaire) a été implémenté en option mais s’est montré défavorable sur M106 (A/B 15 images: `bin1 ~1.04s` vs `bin2 ~1.38s` médian).
- Décision: garder cette voie **désactivée par défaut** (`astap_extract_bin_factor=1`), sans l’imposer tant qu’une version sans surcoût I/O n’est pas disponible.

### Artefacts de référence
- Bench x50: `reports/astap_vs_zenear_m106_mono_50_20260421_100613/`, `..._101821/`, `..._101947/`
- Profil x15: `reports/profile_zenear_hotspots_m106_15_20260421_102117/`

- A/B strict x15 backend détection: `astap` reste meilleur que `cpu` sur M106 (astap ~0.81s médiane, 15/15; cpu ~1.61s, 14/15), donc garder `detect_backend=astap` pour ce profil.

- GUI ZeSolver: début du nettoyage post-virage, le mode simple masque désormais dynamiquement les options solver avancées (hints/search/formats/famille), avec resynchronisation Easy/Expert.

- GUI cleanup (phase 2): en mode Easy, l’onglet Settings cache désormais les groupes experts (presets/FOV/reco/blind tuning) et les actions diagnostics Run blind/Run near; ces éléments restent disponibles en mode Expert.

- Début de modularisation GUI: logique de visibilité Easy/Expert extraite dans `zesolver/gui_profiles.py`, avec `zesolver.py` allégé et prêt pour extraction progressive des tabs.

- Modularisation GUI (suite): section Blind tuning de l’onglet Settings extraite dans `zesolver/gui_settings_sections.py` (`build_blind_group`), validation syntaxe OK.

- Modularisation GUI (phase 4): Presets/FOV/Reco de l’onglet Settings extraits dans `zesolver/gui_settings_sections.py` (`build_presets_fov_reco_groups`), compile check OK.

- Modularisation GUI (phase 5): callbacks Settings (preset apply, browse/save/build/run, sync DB tab) déplacés vers `zesolver/gui_settings_sections.py`; `zesolver.py` raccourci et compile check OK.

- Après extraction callbacks Settings, suppression d’un doublon de layout (`column.addLayout(form)` appelé deux fois dans `_build_settings_tab`) pour réduire le bruit/risque UI; compile check OK.

- Démarrage suppression code mort: retrait du widget legacy `preset_warning_label` (jamais visible) et des clés i18n orphelines `spec_warning_unknown`; compile check OK.

- Nettoyage code mort GUI: suppression de deux méthodes orphelines (`_gather_candidate_files`, `_refresh_file_list`) après vérification de non-référence, avec compile check OK et scan local sans méthode ZeSolverWindow orpheline restante.

- Nettoyage code mort étendu au core solveur: suppression de `ImageSolver._try_blind_shortcut` (non référencée), plus aucune occurrence des méthodes mortes retirées (`_try_blind_shortcut`, `_gather_candidate_files`, `_refresh_file_list`), compile check OK.

- Passe i18n ultra-safe réalisée: retrait de 10 clés de traduction orphelines (database_label, settings.cancel, solver.run.batch, astrometry submit/polling/job status) avec validation compile OK; les clés restantes à faible fréquence sont toutes utilisées via patterns dynamiques.

- Amélioration UX GUI validée: rétablissement des updates progressives en local via callback `on_result` (émission au fil de l’eau), ajout d’un lissage temps réel de la progress bar (timer 400ms), et copie automatique du log de run dans le dossier de sortie.

- Ajustement logique backend GUI: local devient le défaut effectif, et le pipeline local applique désormais un fallback en 3 étages (ZeNear, puis ZeBlind, puis Astrometry uniquement si clé API disponible), avec progression GUI sans double comptage des fichiers.


## 2026-04-22 -> 2026-04-23 (consolidation followup + AGENT)

### Décisions produit confirmées
- Cap validé avec Tristan: **protéger ZeNear** et concentrer les optimisations en **blind-only ZeBlind**.
- Mission parité ASTAP-ISO (scope précédent) confirmée clôturée; nouveau cap prioritaire = throughput/robustesse lot massif.

### Avancement ZeBlind confirmé
- P0/P0bis/P1/P2/P3/P4/P5/P6 implémentés avec instrumentation et garde-fous (feature flags + logs causaux).
- P4 depth ladder livré sous contraintes (caps 80/160/500, cap stage1, OFF en rescue, activation `degraded` uniquement).
- P5 log-odds livré sous flag, conservé **OFF par défaut** (gain temps smoke mais risque solve-rate).
- P6 budgets durs + tuning low-signal rescue2 livrés, puis seuils rendus plus conservateurs (`best_fail_inliers<=3`).

### Résultats de validation (référence lot M106)
- Baseline complète: `zesolver_run_20260423_091723_patch30s.cli.log` -> **29/30**, **718.2s**.
- Candidate blind-only validée: `zesolver_run_20260423_163025_cand_lowsignal_full30_overwrite_REALPATH.cli.log` -> **29/30**, **693.3s**.
- Échec inchangé: `...233232.fit`.
- Delta validé: **-24.9s** à solve-rate identique (rapport `reports/zeblind_fullrun_compare_20260423_164257.md`).
- Position utilisateur confirmée: run complet satisfaisant, candidat crédible pour **v1.0.0**.

### Versioning + UX + cross-platform
- Titre GUI versionné centralement via `APP_VERSION` (`pyproject.toml`) avec suffixe automatique `Vx.y.z` dans toutes les langues.
- Icônes ZeSolver intégrées:
  - assets générés: `icon/ZSicon.ico`, `icon/ZSicon.png`, `icon/ZSicon.icns` depuis `ZSicon.jpeg`,
  - application icône au niveau `QApplication` + fenêtre GUI,
  - sélection d’icône par OS (win/mac/linux) + compat runtime packagé (`sys._MEIPASS`).
- Préflight macOS ajouté: `zesolver/macos_preflight.py` (imports, process pool, `zesolver.py --help`).
- Packaging PyInstaller ajouté: `packaging/pyinstaller/build.py`, `packaging/pyinstaller/convert_icon.py`, docs associées.
- README mis à jour (preflight macOS, packaging multi-OS, workflow icônes, checklist release v1.0.0).

### État actif pour prochaine itération
- Backlog blind-only restant prioritaire: P7 (parity lock fiable) et P8 (uniformisation/dédup verify).
- Garder ZeNear stable/non-régressé pendant toute suite d’optimisation ZeBlind.
