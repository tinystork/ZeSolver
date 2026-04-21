# Follow-up ZeSolver (virage perf) — 2026-04-12

## Contexte actuel

- Fiabilité Near en forte hausse, run lourd en cours jugé solide sur les **300+ premières images**.
- Le nouveau goulot est maintenant le **débit** (objectif produit: **10k+ images**).
- ASTAP reste une référence de robustesse mais est limité par son mode séquentiel dans notre usage.

## ✅ Ce qui est validé

- RA/Dec lock phase-aware (hinted uniquement, pool global en scale_only/blind).
- Reproductibilité Near renforcée:
  - seed RANSAC stable par FITS,
  - warm-start Near désactivé en parallèle.
- Rescue Near (incluant relâchement RMS) validé.
- P2/P3 clôturés:
  - `near_ransac_seed` exposé en CLI/UI/config,
  - audit terminé: plus d'appel `estimate_similarity_RANSAC` sans `random_state`.

## 🎯 Nouveau cap prioritaire (Q1)

Passer de “solver fiable” à “solver fiable **et** rapide à grande échelle”.

### Priorité 1 — parallélisation CPU + mode auto (court terme, ROI max)

1. Stabiliser un profil workers/I/O/cache pour run massif.
2. Ajouter un **mode auto** qui adapte la concurrence à la machine (CPU/RAM/IO/GPU).
3. Réduire le temps des branches d'échec (timeouts internes, retries inutiles).
4. Mesurer throughput réel en images/min sur paliers (500 / 2k / 10k).

### Priorité 2 — GPU utile (ciblé, pas dogmatique)

1. Réactiver proprement la voie GPU de détection d'étoiles.
2. Bench A/B CPU vs GPU sur mêmes lots.
3. Garder le fallback CPU robuste et automatique.

### Priorité 3 — pipeline hybride CPU+GPU (progressif)

Objectif: faire tourner CPU et GPU en parallèle sur des étapes différentes, avec fallback automatique.

- Étape 1: GPU détection étoiles (N+1), CPU matching/RANSAC/WCS (N)
- Étape 2: étude de portage de sous-blocs pair-build/RANSAC si gain mesuré
- Étape 3: scheduler hybride piloté par télémétrie (occupation CPU/GPU, latence I/O, VRAM)

Attendu: gain de débit net sans dégrader la reproductibilité.

## 🧭 Décisions d'architecture (update 12:27)

- **GPU confirmé fonctionnel**, mais charge faible observée (normal avec offload limité à la détection).
- Direction retenue: **pipeline hybride progressif** plutôt que simple "charger plus d'images en GPU".
- Principe cible:
  - GPU pour les briques massivement parallèles,
  - CPU pour orchestration/fallback et parties non portées,
  - chevauchement CPU/GPU (pipeline) pour éviter les temps morts.
- Contrainte non négociable: le mode **CPU-only** reste pleinement opérationnel.

## 🚀 Mise en place accélération (démarrée)

Implémentation initiale effectuée pour ouvrir un vrai cycle d'optimisation débit sans retoucher le code à chaque essai:

- Nouveaux réglages Near de détection exposés en **CLI + config + UI**:
  - `near_detect_k_sigma`
  - `near_detect_min_area`
  - `near_detect_max_labels`
- Nouvelles options CLI:
  - `--near-detect-k-sigma`
  - `--near-detect-min-area`
  - `--near-detect-max-labels`
- Propagation complète jusqu'au moteur Near (`metadata_solver.NearSolveConfig`) et logs de run.

Objectif: lancer des benchs contrôlés CPU/GPU et quantifier l'impact perf/qualité sans modifier le code entre runs.

## ✅ Suite follow-up en cours (CPU bottleneck)

Première brique implémentée pour attaquer le goulot CPU métier sans risque de régression fonctionnelle:

- **Instrumentation fine des timings Near** dans les logs:
  - `detect`, `pair-build`, `ransac`, `fit`, `write`, `total`
- **Micro-optimisation CPU sur la réduction catalogue** (`max_cat_stars`):
  - bascule vers `argpartition` sur gros volumes,
  - tri stable du sous-ensemble retenu pour garder un comportement déterministe.

But: identifier précisément le vrai coût dominant (pair-build vs ransac vs fit) avant portage hybride ciblé.

- **Run analysé (CPU6, 500)**: hotspot confirmé `detect` ≈ **95.5%** du temps Near moyen (`pair-build` ≈ 0.2%, `ransac` ≈ 2.2%).
- **Étape suivante implémentée**: optimisation CPU du détecteur (`star_detect.py`) avec extraction flux/centroïdes vectorisée (suppression de boucles masques par label).
- Validation smoke: sur image test, `near total` est passé d'environ **2.30s** à **1.41s** (détecteur dominant réduit).
- **Validation post-opt (CPU6, 500)**: 500/500 en 325.54s (92.15 img/min), near mean 3.73s, p95 5.2s.

## Plan d'exécution recommandé

### Palier A (immédiat)
- Terminer run lourd en cours et extraire métriques:
  - solved/failed/skipped,
  - temps total,
  - images/min,
  - distribution Near/Blind.

### Palier B (CPU tuning + auto)
- Sweep contrôlé des workers + I/O concurrency + cache tile.
- Implémenter un preset **auto**:
  - auto-workers (CPU/logical cores),
  - auto I/O concurrency (débit disque),
  - garde-fous mémoire (RAM/VRAM),
  - adaptation conservative sur petites machines.
- Choisir un preset “production 10k” stable et documenté.

### Palier C (GPU + hybride)
- Vérifier activation GPU effective (pas seulement config).
- Bench sur lot identique:
  - CPU-only,
  - GPU-detect,
  - pipeline hybride.
- Ajouter un mode d'ordonnancement adaptatif CPU/GPU (auto) selon la machine.
- Décision guidée par métriques (débit + stabilité + variabilité).

### ✅ Point 2 (mode auto smart) — implémentation initiale faite

- `workers=0` conserve le mode auto adaptatif (déjà exposé UI) avec profil moins conservatif.
- Ajout de garde-fous auto pour `io_concurrency=0`:
  - prise en compte RAM machine,
  - prise en compte backend near (`cpu`/`cuda`) pour éviter la saturation,
  - log explicite des ajustements auto (`I/O auto-guard adjusted ...`).
- Hint UI mis à jour: auto adapté machine (CPU/RAM), mode manuel = potentiellement plus rapide mais moins stable.

## Critères de sortie “10k-ready”

- Stabilité: pas de dérive run-to-run sur un lot fixe.
- Robustesse: taux d'échec maîtrisé et explicable.
- Débit: gain significatif vs profil actuel CPU-only.
- Opérationnel: logs exploitables et paramètres reproductibles.

## Notes

- Le gain GPU est réaliste, mais la VRAM faible peut limiter l'accélération brute.
- Le meilleur levier durable est un **pipeline hybride** + un **mode auto** qui s'adapte au matériel.
- Le mode hybride CPU+GPU est techniquement pertinent et **non délirant**: c'est une stratégie standard de pipeline haut débit.
- CPU-only doit rester un chemin nominal (robuste et maintenu).

## ✅ Point 3 (pipeline hybride) — étape 1 implémentée

- Ajout d'un garde-fou de pipeline hybride côté Near: `detect_gpu_slots` (défaut `1`).
- But: limiter la contention CPU↔GPU quand plusieurs workers lancent la détection en même temps.
- Comportement:
  - backend `cpu` => aucun impact,
  - backend `auto/cuda` + runtime CUDA OK => section détection GPU régulée via slots partagés,
  - runtime CUDA non prêt => fallback CPU inchangé (pas de sérialisation inutile).
- Exposition CLI: `--near-detect-gpu-slots` (et logging de config + usage réel).
- Logs Near enrichis: `near detect backend used ... gpu_slots=...` pour audit de charge.

Première intention: créer un vrai chevauchement progressif (détection GPU régulée, appariement/RANSAC CPU en parallèle) sans casser le chemin CPU-only.

### ✅ Validation hybride rapide (350 images, workers=6)

- `cpu_350_run_6_slots1_latest`: 247.0s, 85.02 img/min
- `auto_350_run_6_slots1_latest`: 230.7s, 91.03 img/min (**+7.07%** vs CPU)
- `auto_350_run_6_slots2_latest`: 226.7s, 92.63 img/min (**+1.76%** vs slots=1)

Lecture: la régulation `detect_gpu_slots` apporte un gain mesurable; `slots=2` est meilleur que `slots=1` sur ce profil machine/lot, avec backend CUDA effectivement utilisé.

## 🧩 Nouvelle mission UX (pré-intégration ZeMosaic) — 2026-04-12

Objectif produit validé: ZeSolver doit devenir consommable par un utilisateur non technique,
puis s'intégrer proprement dans l'environnement ZeMosaic (CLI d'abord, UI simple ensuite).

### Principes

- Le mode avancé actuel reste disponible (power users).
- Le mode par défaut doit être **Simple / guidé** (assistant first-run + workflow clair).
- Le dossier de rejet est secondaire côté ZeSolver brut: la logique finale de routage des non-résolus sera alignée sur ZeMosaic.

### Parcours UX cible (mode simple)

1. **Premier démarrage (wizard)**
   - Vérifier la présence d'un répertoire `database` (catalogues étoiles) et d'un index ZeSolver valide.
   - Proposer soit l'ajout/téléchargement de bases compatibles, soit l'usage d'un jeu recommandé (ex: équivalent D50).
   - Choisir un profil instrument (S50/S30/C11/Evolux 62, presets extensibles).

2. **Index/hashes**
   - proposer soit:
     - construire les hashes depuis la base,
     - soit pointer vers un répertoire de hashes existant.
   - afficher un check de validité avant lancement solve.

3. **Run dossier**
   - choisir dossier d'entrée,
   - option explicite: nettoyage WCS existants (via `zewcscleaner.py` côté CLI),
   - lancer solve batch.

4. **Sortie utilisateur**
   - résumé simple (résolus/non résolus/temps),
   - export d'un rapport,
   - non résolus routés selon convention d'intégration ZeMosaic.

### Décision d'implémentation

- Prioriser d'abord la **surcouche UX simple** (wizard + presets + checks),
- garder les réglages experts derrière un mode "Avancé",
- préparer l'API/CLI pour appel propre depuis ZeMosaic (contrat d'entrée/sortie stable).

### Clarification importante (source de vérité)

- **ZeSolver n'exige pas l'exécutable ASTAP** pour son fonctionnement normal (solveur Python + index local).
- Ce qui est requis: les **catalogues/bases** (répertoire `database`) et/ou un **index ZeSolver** valide (`manifest` + `tiles/hash_tables`).
- Donc en UX: on demande un chemin de bases/index, pas un binaire ASTAP.


## 🚧 Avancement mission UX — démarrage effectif

- [x] Ajout d’un toggle **Mode simple (recommandé)** dans l’onglet Solveur.
- [x] Ajout d’un assistant pré-lancement (mode simple):
  - vérifie `database`,
  - guide vers l’onglet Réglages si `index` manquant,
  - affiche une confirmation de run lisible (dossier, nb fichiers, backend, overwrite).
- [ ] Étape suivante: intégrer l’option de nettoyage WCS (`zewcscleaner.py`) dans ce flux simple.
- [x] Flux simple: option intégrée de nettoyage WCS pré-run (FITS uniquement, via `zewcscleaner.py`, backup `.bak` activé).
- [x] Menu "Interface" ajouté (Expert / Easy / Wizard), avec **Easy par défaut** au démarrage.

---

## 2026-04-13 — Cap prioritaire: 10k fichiers en charge maximale machine

### Constat opérationnel (nouveau)

- Campagne Near-only sur set `fresh`: progression validée jusqu'à **10/10**.
- Bench sur set varié déjà résolu ASTAP (`/home/tristan/zemosaic/example/astap solved`, 67 FITS):
  - **55/67 (82.1%)** acceptés,
  - temps solve médian ≈ **18.84 s/image** (beaucoup trop élevé vs ASTAP),
  - causes d'échecs dominantes: gates de conformance/ZeMosaic liées au pixel scale.

### Pourquoi c'est plus lent qu'ASTAP aujourd'hui

1. Pipeline Python hybride plus long (détection + hypothèses + stepping + gates + refit).
2. Coûts process externes quand `astap -extract` est utilisé pour la détection.
3. Requêtes catalogue/projections répétées par image et par centre.
4. Validation produit supplémentaire (conformance + ZeMosaic), absente du flux ASTAP natif.

### Nouveau plan exécution (objectif throughput massif)

#### Phase A — Débit brut (immédiat)

1. Passer en **workers persistants** (éviter les relances process image-par-image).
2. Introduire un scheduler **multi-queues** (I/O, detect, match, validate, write).
3. Limiter les passes redondantes (center stepping adaptatif, sortie anticipée quand consensus fort).
4. Mesurer paliers débit: 500 / 2k / 10k (img/min, p95 latence, taux d'échec).

#### Phase B — Saturation machine contrôlée

1. Mode `auto-max` qui pousse CPU/GPU au maximum avec garde-fous RAM/VRAM.
2. Backpressure dynamique (réduction concurrence si pression mémoire/IO).
3. Cache index/catalog global partagé (éviter query/reproject répétitifs).

#### Phase C — Parité ASTAP pragmatique

1. Réduire les divergences de workflow (chaîne solve plus directe, moins de refits).
2. Séparer explicitement les profils:
   - `fast` (throughput max),
   - `strict` (conformance forte / contrôle qualité).
3. Validation hors `fresh` sur lot varié déjà ASTAP-solvé pour surveiller faux positifs.

### KPI de sortie

- **Débit**: multiplier fortement img/min (objectif x3 puis x5+).
- **Robustesse**: stabilité run-to-run sur lot fixe.
- **Scalabilité**: capacité opérationnelle démontrée sur 10k sans OOM/livelock.
- **Observabilité**: timings par étape + causes d'échec agrégées.


## 2026-04-14 - Plan d'action ZeNear en mode "ASTAP-ISO strict"

### Objectif

Isoler un chemin Near qui reproduit le noyau ASTAP pour obtenir une comparaison 1:1 (robustesse + débit), sans branches heuristiques additionnelles ZeSolver.

### Cadrage (important)

Le mode strict doit exécuter uniquement la chaîne:

1. détection étoiles,
2. quads ASTAP-ISO,
3. `find_fit_using_hash` ASTAP-ISO,
4. fit affine LSQ,
5. génération WCS TAN,
6. validation minimale équivalente ASTAP.

Tout le reste doit être désactivable explicitement en mode strict.

### Travaux à implémenter (ordre imposé)

#### Phase 1 - Câblage d'un vrai mode strict (P0)

- Ajouter un flag de config Near explicite, ex: `near_astap_iso_strict=true` (CLI + UI + settings).
- Sous ce flag, désactiver:
  - `_build_candidate_pairs`,
  - `_find_quad_hypothesis`,
  - `estimate_similarity_RANSAC` global,
  - center stepping enrichi non ASTAP,
  - second-pass refine non strict,
  - gates additionnels (`validate_wcs_for_zemosaic`, `_near_conformance_check`).
- Conserver uniquement l'acceptation basée sur:
  - cohérence géométrique du fit,
  - seuils RMS/inliers minimaux alignés ASTAP (à documenter dans code).

#### Phase 2 - Parité fonctionnelle fine du noyau (P0)

- Vérifier la parité de `find_many_quads(mode=6)` par rapport à la source ASTAP lue (comportement effectif actuel, y compris ses particularités).
- Vérifier la parité du hash matching:
  - binning,
  - voisinage de bins,
  - filtrage au ratio médian.
- Vérifier la parité du fit affine LSQ et du garde-fou XY ratio.
- Ajouter des logs stricts par étape (compteurs exacts):
  - `stars_img`, `stars_cat`, `quads_img`, `quads_cat`, `matches_raw`, `matches_kept`, `refs_final`, `rms_px`.

#### Phase 3 - Protocole benchmark 1:1 (P0)

- Lot cible primaire: NGC6888 x16.
- Protocole:
  1. ASTAP CLI référence (déjà existante),
  2. ZeNear strict seul,
  3. ZeNear non-strict actuel.
- Sorties attendues:
  - `success/count`,
  - temps total et img/min,
  - raisons d'échec catégorisées.
- Critère de sortie phase:
  - réduction nette de l'écart vs ASTAP sur NGC6888,
  - aucun faux positif évident (RMS/échelle incohérents).

#### Phase 4 - Réintégration contrôlée des garde-fous produit (P1)

- Réactiver ensuite, un par un:
  1. gate conformance near,
  2. gate zemosaic.
- Mesurer l'impact incrémental de chaque gate (succès perdus vs faux positifs évités).
- Produire deux profils stables:
  - `strict-astap-iso` (debug/parité),
  - `prod-strict` (qualité produit complète).

### Livrables demandés

1. Diff minimal et lisible dans `metadata_solver.py` (et config/CLI/UI associés).
2. Rapport benchmark comparatif JSON + résumé markdown.
3. Tableau d'écarts ASTAP vs ZeNear strict, avec causes techniques vérifiables.

## 2026-04-20 - Plan d'implémentation ASTAP-ISO strict (version précise, exécutable)

> Cette section **remplace opérationnellement** le plan strict du 2026-04-14 pour l'ordre d'implémentation et les critères de parité.

### 0) Cible exacte de parité

En mode `near_astap_iso_strict=true`, le chemin Near doit reproduire la chaîne ASTAP suivante:

1. détection étoiles,
2. génération quads,
3. matching `find_fit` **ou** `find_fit_using_hash` selon le volume,
4. fit affine LSQ,
5. garde-fou ratio XY,
6. recalage 2e passe (match_nr 2),
7. écriture WCS TAN.

### 1) Câblage du flag strict de bout en bout (P0 bloquant)

#### Fichiers à modifier

- `zeblindsolver/metadata_solver.py`
- `zesolver.py`
- `zesolver/settings_store.py`
- (si présent) panneau UI Near dans `zesolver.py` (widgets + sérialisation)

#### Contrat attendu

- Nouveau booléen persistent: `near_astap_iso_strict`.
- Valeur par défaut: `False`.
- Propagation complète:
  - settings JSON -> `PersistentSettings` -> `SolveConfig` -> `NearSolveConfig` -> `solve_near(...)`.
- Log explicite en début de solve:
  - `"astap_iso_strict": true|false`.

### 2) Suppression stricte des branches non-ASTAP quand strict=true (P0)

Dans `solve_near` (`metadata_solver.py`), quand `cfg.astap_iso_strict` est vrai:

- **désactiver** `_build_candidate_pairs`;
- **désactiver** `_find_quad_hypothesis`;
- **désactiver** `estimate_similarity_RANSAC(...)` (one-shot + global);
- **désactiver** les heuristiques de secours non-ASTAP (sélection par quick-inliers/ratio custom hors logique ASTAP de base);
- garder un flux déterministe fondé sur le noyau quad/hash + LSQ.

### 3) Parité matching ASTAP complète: `find_fit` vs `find_fit_using_hash` (P0)

Point manquant à implémenter explicitement:

- reproduire la bifurcation ASTAP:
  - si `nrquads_ref < 180` -> chemin `find_fit` (bruteforce),
  - sinon -> `find_fit_using_hash`.

Actions:

- ajouter un helper Python ASTAP-iso pour `find_fit` (mêmes tests ratios 1..5 + filtre ratio médian),
- réutiliser le même downstream LSQ que le chemin hash,
- conserver `minimum_count` cohérent (`3 + nrstars_image // 140`).

### 4) Parité garde-fou géométrique ASTAP (P0)

Après LSQ (comme ASTAP):

- calculer `xy_sqr_ratio = (sx0^2 + sx1^2) / (sy0^2 + sy1^2)`;
- rejeter hors intervalle `[0.9, 1.1]`;
- en strict, ce rejet fait foi avant toute autre validation produit.

### 5) Parité 2e passe ASTAP (P0)

En strict, implémenter la logique "match_nr" (2 passes max):

- passe 1: solve initial autour du centre hint,
- recadrage sur centre solution,
- passe 2: resolve avec fenêtre mise à jour,
- conserver la meilleure solution de la passe 2 si amélioration.

Important:

- la 2e passe est **obligatoire en strict** (sauf absence de solution passe 1),
- ne pas injecter les rescues non-ASTAP dans cette boucle.

### 6) Validation en mode strict (P0)

Quand `near_astap_iso_strict=true`:

- **bypass** `validate_wcs_for_zemosaic(...)`;
- **bypass** `_near_conformance_check(...)`;
- garder uniquement:
  - validité WCS/TAN,
  - cohérence géométrique (LSQ + XY ratio),
  - seuils RMS/inliers strictement documentés pour le mode iso.

Quand `near_astap_iso_strict=false`:

- comportement actuel inchangé (gates produit actifs).

### 7) Instrumentation obligatoire pour A/B propre (P0)

Ajouter un bloc de debug structuré par image (JSONL) avec:

- `strict_mode`,
- `stars_img`, `stars_cat`,
- `quads_img`, `quads_cat`,
- `path_used` (`find_fit` | `find_fit_using_hash`),
- `matches_raw`, `matches_kept`,
- `lsq_xy_ratio`,
- `pass_index` (1 ou 2),
- `rms_px`, `inliers`,
- `final_accept_reason` / `final_reject_reason`.

### 8) Protocole benchmark de validation (P0)

#### Jeux de test minimum

1. `NGC6888 x16` (référence historique),
2. lot `fresh` strict-iso,
3. lot varié déjà ASTAP-solvé.

#### Matrice de run

- A: ASTAP référence (déjà produite),
- B: ZeNear strict-iso,
- C: ZeNear actuel (non-strict).

#### KPI obligatoires

- `success/count`,
- `img/min`, `wall_s`,
- distribution raisons d'échec,
- taux de divergence B vs A (fichiers résolus différemment).

### 9) Critères de sortie (Definition of Done)

Le chantier strict est considéré terminé si:

1. le flag `near_astap_iso_strict` est disponible et persisté CLI/UI/settings,
2. le chemin strict n'exécute plus de branches non-ASTAP,
3. la bifurcation `find_fit` (<180) / `find_fit_using_hash` (>=180) est active,
4. la 2e passe ASTAP est active en strict,
5. les gates ZeMosaic/conformance sont bypassées en strict,
6. un rapport A/B/C documente l'écart restant vs ASTAP avec causes vérifiables.

### 10) Plan de merge sécurisé

1. PR-1: flag + propagation config + logs (sans changement algo).
2. PR-2: noyau strict pur (bypass branches non-ASTAP + find_fit path + XY ratio + 2e passe).
3. PR-3: benchmark + rapport + ajustements mineurs de seuils.
4. PR-4: réintégration progressive des gates produit pour profil `prod-strict` (hors mode iso).



## 2026-04-20 - État d’avancement réel (ASTAP-ISO strict, après implémentation)

### Vérification de complétude

Le plan 2026-04-20 est bien présent et sert de source de vérité.
Il est désormais **partiellement exécuté** avec un état code mesurable.

### ✅ Implémenté dans le code

- Flag strict propagé de bout en bout: `near_astap_iso_strict`
  - CLI (`--near-astap-iso-strict`),
  - settings persistants,
  - UI (onglet fast solver),
  - runtime `NearSolveConfig -> solve_near(...)`.
- Suppression des branches non-ASTAP en strict:
  - bypass `_build_candidate_pairs`,
  - bypass `_find_quad_hypothesis`,
  - bypass `estimate_similarity_RANSAC` global,
  - bypass gates produit `validate_wcs_for_zemosaic` et `_near_conformance_check`.
- Parité noyau matching: bifurcation ASTAP active
  - `find_fit` si `<180` quads,
  - `find_fit_using_hash` sinon.
- Garde-fou géométrique ASTAP ajouté après LSQ (`xy_sqr_ratio`).
- Deuxième passe type ASTAP (logique `match_nr`) maintenue en strict (recentrage puis resolve).
- Instrumentation debug enrichie (`near_debug.jsonl`):
  - `zenear_no_transform`, `zenear_no_consensus`, `zenear_validation_fail`.

### 📊 État benchmark actuel (à froid)

- ZeNear strict ASTAP (NGC6888 x16): **0/16**
  - ex: `reports/ngc6888_16_zenear_strict_astap_iso_progress_v3.json`
- ZeNear non-strict (référence interne actuelle sur ce lot): **0/16**
  - ex: `reports/ngc6888_16_zenear_nonstrict_progress.json`
- Référence ASTAP historique mentionnée dans ce repo: **10/16**.

### 🧪 Vérifications de stabilité

- `python -m py_compile ...` OK
- `pytest -q tests/test_zeblindsolver.py` OK (10 passed)

### 🔜 Reste prioritaire

- Faire remonter le consensus géométrique strict (échecs dominants: `no geometric consensus` + `validation fail`).
- Ajouter/normaliser l’instrumentation stricte de parité demandée (counters par passe, `path_used`, etc.) pour diagnostic causal ASTAP vs ZeNear strict.
- Produire un comparatif A/B/C final propre une fois le noyau strict stabilisé.


### Update technique (13:55-14:10) — instrumentation causale no_transform/no_consensus

- Instrumentation ASTAP-ISO enrichie dans `near_debug.jsonl` via `astap_iso_diag`:
  - `initial`, `best_center`, `second_pass`,
  - compteurs agrégés `aggregate` (`calls`, `ok_calls`, `max_refs`, `max_quick_inliers`, `path_counts`),
  - par tolérance: `matches_raw`, `matches_kept`, `ok`, `quick_inliers`, `quick_med`, `scale_ratio`.
- Traces propagées dans les événements: `zenear_no_transform`, `zenear_no_consensus`, `zenear_validation_fail`, `zenear_attempt`.
- Ajustements de gating strict en cours d’exploration:
  - `iso_min_inliers` strict abaissé à `1` (depuis `2`) pour ne plus bloquer trop tôt.
- Ajustement consensus:
  - sweep de tolérance ajouté aussi sur `_build_matches_from_affine` (1.0/1.8/3.0/5.0) avant fallback transform.

#### Résultats rapides (probe x2)
- Frame A: reste `zenear_no_transform` (aucune hypothèse iso exploitable, `ok_calls=0`).
- Frame B: progression `no_transform` -> `no_consensus` -> `validation_fail` selon variantes; avec état actuel on atteint la validation (inliers non nuls) mais RMS encore très mauvais (solution géométriquement fausse).

#### Conclusion provisoire
- Le goulot principal n’est plus l’absence d’observabilité: on a maintenant la télémétrie nécessaire pour isoler où ça casse.
- Le noyau strict produit parfois une hypothèse partielle mais encore instable (inliers faibles ou géométrie erronée), d’où échec final.

## Update mission ASTAP-ISO strict (2026-04-20, soirée)

### ✅ Accompli

- Parité stricte consolidée côté Near sur lot NGC6888:
  - mode strict sans hint FOV explicite: **16/16**,
  - mode strict avec hint FOV explicite (override per-frame issu du probe ASTAP): **10/16**,
  - partition alignée ASTAP (mêmes 6 échecs), donc la fonction de hint FOV est bien préservée.
- `metadata_solver.py` enrichi pour la gestion FOV:
  - source FOV tracée (`fov_hint_source`: `override` > `header` > `scale`),
  - support hints header (`FOV`, `FOVDEG`, `FOVH`, `FOVY`, `FIELD`, `FIELDDEG`),
  - retry auto-FOV strict activé seulement si la source est `scale`.
- Retry auto-FOV strict rendu configurable:
  - `strict_auto_fov_retry` (bool),
  - `strict_auto_fov_retry_scales` (ordre/scales configurables),
  - `strict_auto_fov_retry_max_attempts` (0 = illimité).
- Optimisation coût retries:
  - skip des sous-ensembles catalogue dupliqués (`duplicate_subset`) pour éviter des appels ISO redondants,
  - ordre par défaut des scales retuné: `(1.25, 0.82, 1.6, 0.65, 2.4, 4.0)`.

### 🧪 Vérifications

- `python -m py_compile ...` OK.
- `pytest -q tests/test_zeblindsolver.py` OK (**10 passed**).
- Rapport de synthèse principal:
  - `reports/ngc6888_16_strict_auto_fov_retry_smart_cost_v1.json`.
- Rapports de tuning:
  - `reports/strict_auto_fov_retry_order_tune_singleframe_v1.json`,
  - `reports/strict_auto_fov_retry_order_tune_multisrc_v1.json`,
  - `reports/strict_auto_fov_retry_cost_multisrc_v2.json`.

### 🔜 Suite recommandée

- Ajouter un déclenchement encore plus contextuel du retry auto-FOV (basé sur signaux `astap_iso_diag` initiaux) sans toucher à la priorité des hints FOV explicites.
- Mesurer l'impact coût/robustesse sur un lot plus varié que NGC6888 (plusieurs champs/caméras), en conservant les mêmes garde-fous de parité strict.

## Update mission ASTAP-ISO strict (2026-04-20, nuit) — déclenchement contextuel + validation large

### ✅ Accompli

- Déclenchement auto-FOV strict rendu **contextuel** dans `metadata_solver.py`:
  - retry auto-FOV seulement si source FOV = `scale`,
  - garde-fou support image minimal (`stars_img >= 24`, `quads_img >= 3`),
  - pilotage contextuel des retries si hypothèse initiale sans indice (`best_refs=0` et `matches_raw=0`):
    - ordre d’essai orienté expansion d’abord,
    - budget contextuel par défaut (3 tentatives) sauf override explicite.
- Contrôle coût supplémentaire:
  - `strict_auto_fov_retry_zero_ref_patience` (défaut 3): arrêt anticipé si zéro refs répété,
  - `duplicate_subset` conservé pour éviter des appels ISO redondants,
  - `strict_auto_fov_retry_max_attempts` toujours disponible.

### 🧪 Résultats de validation

- Stabilité code:
  - `python -m py_compile ...` OK,
  - `pytest -q tests/test_zeblindsolver.py` OK (**10 passed**).
- Parité NGC6888 inchangée (attendu):
  - strict sans hint FOV explicite: **16/16**,
  - strict + hint FOV explicite (override per-frame): **10/16**,
  - confusion alignée ASTAP (10 succès communs, 6 échecs communs).
- Validation plus large (24 FITS, mix NGC6888 + M106 backup + fresh):
  - `strict_retry_on`: **23/24**, retries sur 2 frames, **6** appels retry effectifs,
  - `strict_retry_off`: **22/24**,
  - donc +1 frame résolue avec coût retry désormais borné.

### 📁 Rapports

- `reports/strict_contextual_retry_broad_validation_v2.json`
- `reports/ngc6888_16_strict_auto_fov_retry_smart_cost_v1.json`

### 🔜 Suite

- Ajouter des tests non-régression dédiés pour la logique contextuelle des retries (source FOV, patience zero-ref, max attempts).
- Lancer un A/B/C final ASTAP CLI vs ZeNear strict vs ZeNear non-strict sur un lot multi-champs stabilisé.

## Update mission ASTAP-ISO strict (2026-04-20, nuit tardive) — non-régression ciblée

### ✅ Accompli (tests)

- Ajout d'une batterie de non-régression dédiée dans `tests/test_metadata_solver.py` (5 tests sur ce fichier):
  1. priorité FOV `override` > `header`,
  2. source FOV `header` prise en compte sans override,
  3. source `scale` avec support faible => skip auto-FOV (`low_support`),
  4. retry contextuel borné avec patience zéro-ref (`zero_refs_patience`),
  5. test synthétique de résolution (existant, conservé).
- Ces tests valident explicitement la logique demandée: hint FOV explicite prioritaire, retries contextuels seulement quand pertinent.

### 🧪 Validation exécutée

- `pytest -q tests/test_metadata_solver.py` => **5 passed**.
- `python -m py_compile ...` + `pytest -q tests/test_zeblindsolver.py tests/test_metadata_solver.py` => **15 passed**.

### Effet opérationnel

- Le comportement "hint FOV d'abord" est désormais verrouillé par tests.
- Le déclenchement contextuel des retries auto-FOV est couvert par tests unitaires (plus seulement par bench manuel).

## Update mission A/B/C (2026-04-20, run multi-champs)

### Dataset exploité
- Brutes disponibles: 86 fichiers bruts, 56 uniques par nom (`astap solved` + `fresh` + `backuplightsastap` + `lights`).

### Exécution
- Comparatif lancé sur 56 brutes uniques:
  - A = ZeNear strict (`astap_iso_strict=True`)
  - B = ZeNear non-strict (`astap_iso_strict=False`)
- C (ASTAP CLI) non relancé ici, comparaison C faite via probe existant sur le sous-ensemble NGC6888 x16.

### Résultats
- Multi-champs x56:
  - ZeNear strict: **45/56** (25.55 img/min)
  - ZeNear non-strict: **0/56** (1.64 img/min)
  - Diff: 45 succès strict-only, 11 échecs communs.
- Sous-ensemble NGC6888 x16 avec oracle ASTAP existant:
  - ASTAP: 10/16
  - ZeNear strict: 16/16
  - ZeNear non-strict: 0/16
  - Confusion ASTAP vs strict: 10 succès communs, 6 strict-only, 0 ASTAP-only, 0 double-échec.

### Rapports
- `reports/ab_zenear_strict_vs_nonstrict_multifield_v1.json`
- `reports/abc_subset_ngc6888_16_from_multifield_v1.json`

## Update mission non-strict (2026-04-20, itération seuils RMS)

### ✅ Diagnostic finalisé

- Le 0/56 en non-strict est confirmé comme un **problème de gate `quality_rms`** (pas majoritairement un no-transform):
  - à `quality_rms=1`: projection/observation = 0 succès, 53 validation fails, 3 no-transform.
- Sweep multi-champs exécuté sur 56 brutes uniques (non-strict, CPU, `k_sigma=4.5`, `min_area=8`, `max_labels=1200`) avec run de référence `quality_rms=500` puis projection des seuils.

### 📊 Résultats sweep (56)

- `rms=10` -> 4/56
- `rms=30` -> 7/56
- `rms=100` -> 24/56
- `rms=160` -> 33/56
- `rms=220` -> 49/56
- `rms=500` -> 52/56 (reste 3 no-transform + 1 validation à inliers=4)

### ⚠️ Lecture qualité

- Les solutions supplémentaires apparaissent avec des RMS très élevés (médiane run `rms=500` ≈ 127.8 px, max ≈ 422 px), donc seuils élevés = forte hausse du risque de faux solve.
- Le non-strict reste aujourd’hui un mode de rescue à piloter prudemment (profil safe vs agressif), pas un remplacement du strict.

### 📁 Rapport

- `reports/nonstrict_quality_rms_sweep_multifield_v2.json`


## Update mission ASTAP-ISO strict (2026-04-21, matin) — clôture scope parité + bascule scale-up

### ✅ Accompli

- Vérification side-by-side ASTAP CLI vs ZeNear strict confirmée sur lot NGC6888 co-résolu.
- Re-run "dé-biaisé" effectué (suppression préalable des cartes WCS/SOLVER sur copies avant résolution):
  - séparation centre médiane ~3.14",
  - écart scale médian ~0.0034%,
  - écart rotation médian ~0.291°.
- Interprétation opérationnelle validée:
  - faible risque de dérive visuelle finale si recalage stellaire aval,
  - risque réel en WCS-only sur bords (effet rotation/échelle).
- Préparation d'un lot manuel M106 mixte 50/50 pour ZeMosaic:
  - 15 FITS taggés `SOLVER=ZeNear` (WCS ZeNear injecté),
  - 15 FITS taggés `SOLVER=ASTAP` (WCS ASTAP injecté),
  - dossier: `/home/tristan/zemosaic/example/testzenear/`.

### ✅ Décision statut mission

- Mission "gap ASTAP vs ZeNear strict" considérée **accomplie pour le scope courant**.
- Pas de P0 bloquant restant sur le noyau strict pour continuer produit.

### 🔜 Nouveau focus prioritaire: scale-up & vitesse (1k-10k)

- P0-A: throughput massif reproductible (CPU/GPU) avec KPI standardisés (`img/min`, p50/p95 latence frame, RAM/VRAM max).
- P0-B: robustesse long-run (zéro faux échec de fin de run, reprise propre, crash breadcrumbs exploitables).
- P0-C: tuning architecture lot massif (batching, caches index, I/O pipeline, workers adaptatifs selon mémoire dispo).

### P1 utile (non bloquant immédiat)

- Rejouer un benchmark C (ASTAP CLI) multi-champs complet pour refermer la boucle A/B/C à grande échelle.
- Finaliser le profil non-strict en mode rescue (safe/balanced/aggressive) avec garde-fous qualité explicites.


## Plan optimisation unitaire ZeNear vs ASTAP (2026-04-21, M106 x50)

### Constat mesuré
- ASTAP médian: ~0.59 s/image
- ZeNear strict médian: ~2.50 s/image
- Gap: ~x4.2

### Hotspots (profiling m106 x15)
1. `image_prep.remove_background` (~19.3 s cum / 15 images)
2. `_detect_stars_astap_cli` (~7.25 s cum / 15)
3. `catalog290.__init__` + `_parse_catalog_file` (~11.0 s cum / 15)

### Patch 1 (P0, ROI max): court-circuit background quand backend astap
- Fichier: `zeblindsolver/metadata_solver.py`
- Changement: ne pas exécuter `remove_background(...)` si `detect_backend == "astap"`.
- Impact attendu: -0.9 à -1.3 s/image.

### Patch 2 (P0): cache catalogue persistant intra-process + writeback NPZ
- Fichier: `zeblindsolver/metadata_solver.py`
- Changement:
  - éviter `CatalogDB(...)` reconstruit à répétition dans `_get_tile_raw_arrays` fallback,
  - réutiliser un cache catalogue partagé,
  - écrire les tuiles fallback en NPZ pour les runs suivants.
- Impact attendu: -0.2 à -0.6 s/image après warmup.

### Patch 3 (P1 rapide): réduire overhead ASTAP extract/parse
- Fichier: `zeblindsolver/metadata_solver.py`
- Changement:
  - parser CSV plus vectorisé,
  - réduire les allocations/listes Python dans `_detect_stars_astap_cli`,
  - option de nettoyage silencieux des warnings non bloquants.
- Impact attendu: -0.1 à -0.3 s/image.

### Critère de succès cible (itération courte)
- M106 x50 mono-champ: ZeNear strict médian <= 1.2 s/image (étape 1), puis <= 0.9 s/image (étape 2).


## Résultats après optimisation cache catalogue + lookup tuiles (2026-04-21 10:06)

- Patch appliqué dans `zeblindsolver/metadata_solver.py`:
  - cache partagé `CatalogDB` réutilisé aussi pour `db_for_stepping`,
  - cache lookup des tuiles ASTAP (`_get_raw_tile_lookup`) pour éviter reconstruction par image.
- Validation syntaxe: `py_compile` OK.
- Benchmark M106 x50 (mono-champ):
  - ASTAP: `50/50`, médiane `~0.517s`
  - ZeNear: `50/50`, médiane `~0.637s`
  - rapport: `/home/tristan/ZeSolver/reports/astap_vs_zenear_m106_mono_50_20260421_100613/results.json`
- Profiling M106 x15 (post-patch): médiane `~0.720s` (mean `~0.811s`, premier frame cold-start plus lent), hotspot dominant restant: `_detect_stars_astap_cli`.
- Objectif produit atteint pour cette étape: ZeNear `<= 0.9s` médian sur M106 x50.


## Évaluation downsample ASTAP extract (2026-04-21 10:20)

### Ce qui a été implémenté
- `NearSolveConfig` enrichi avec:
  - `astap_extract_bin_factor`
  - `astap_extract_bin_strict_only`
  - `astap_extract_bin_min_stars`
- Nouveau chemin optionnel de détection ASTAP:
  - binning moyen d’image avant `astap -extract`,
  - remontée des coordonnées en pleine résolution,
  - retry full-res si détection binned trop pauvre.
- Durcissement parse CSV ASTAP:
  - garde-fou sur CSV vide/tronqué,
  - suppression des warnings `genfromtxt` vides.

### Résultat mesuré (A/B rapide sur M106)
- A/B 15 images, strict ASTAP-ISO:
  - `bin_factor=1`: médiane ~`1.04s` (15/15)
  - `bin_factor=2`: médiane ~`1.38s` (15/15)
- Lecture: sur ce setup, le downsample pré-extract via FITS temporaire est **plus lent** (surcoût I/O + fallback full-res fréquent).

### Décision
- Garder la fonctionnalité binned **disponible mais désactivée par défaut** (`astap_extract_bin_factor=1`).
- Ne pas forcer le downsample ASTAP en production tant qu’on n’a pas une voie sans surcoût disque.

### Next best ROI
1. Réduire encore le coût `_detect_stars_astap_cli` sans I/O temporaire (chemin process + parse + lifecycle).
2. Étudier une voie détecteur interne équivalente ASTAP (CPU/GPU) en mode strict, pour supprimer le coût process externe.
3. Rebench x50 après chaque changement avec garde-fous succès + métriques WCS (center/scale/rot).

- A/B backend strict (x15): `detect_backend=astap` reste devant `cpu` sur ce lot (astap médiane ~0.81s, 15/15; cpu médiane ~1.61s, 14/15).


## Étape suivante validée: tuning charge ZeSolver pour ZeMosaic (2026-04-21 10:48)

- Décision: après optimisation unitaire, prochaine étape = ajuster la charge parallèle ZeSolver côté ZeMosaic.
- Point de fonctionnement observé (machine actuelle, lot ~100 images):
  - zone optimale: `3` workers (perf),
  - mode prudent: `2` workers (stabilité/mémoire).
- Action prévue: exposer un preset simple dans GUI (Safe=2 / Perf=3) et aligner l’auto pour éviter >4 workers sur ce profil matériel.


## GUI cleanup post-virage ZeNear (2026-04-21 10:48)

### Constat
- Le mode Easy masque déjà une partie des onglets avancés, mais le code GUI reste monolithique et conserve beaucoup de réglages historiques (blind/dev/bench/astrometry) qui alourdissent maintenance et UX.

### Recommandation (ordre d’exécution)
1. **Nettoyage UX sans risque**
   - Garder en Easy uniquement: chemins, workers, overwrite, blind on/off, lancement, logs.
   - Déporter tous les réglages experts vers Expert (hints détaillés, tuning blind, benchmark/dev).
2. **Nettoyage structurel code**
   - Extraire les tabs en modules séparés (`tabs/solver.py`, `tabs/database.py`, `tabs/expert_*.py`) pour réduire la complexité de `zesolver.py`.
3. **Retrait fonctionnel prudent**
   - Supprimer réellement des features seulement après audit d’usage (ne pas retirer blind fallback tant que pipeline ZeMosaic l’utilise).

### Cible
- GUI lisible en mode Easy, maintenance simplifiée, et dette technique réduite sans casser le pipeline de prod.

- Implémentation immédiate: masquage automatique des options avancées dans l’onglet Solveur quand le mode simple est coché (hints détaillés, search tuning, formats/famille).
- Le toggle `Mode simple` est désormais connecté dynamiquement, et la visibilité est resynchronisée lors du changement Easy/Expert.

- Nouvelle passe GUI: en mode Easy, l’onglet Settings masque désormais les blocs experts (`presets_group`, `fov_group`, `reco_group`, `blind_group`) et les boutons de diagnostic `Run blind/Run near`.

- Refactor code: extraction de la logique de visibilité Easy/Expert vers `zesolver/gui_profiles.py` (`apply_solver_simple_visibility`, `apply_settings_easy_visibility`) pour alléger `zesolver.py` et préparer la modularisation des tabs.

- Modularisation GUI (phase 3): extraction du bloc "Blind solver tuning" de `_build_settings_tab` vers `zesolver/gui_settings_sections.py` (`build_blind_group`) pour réduire le volume du fichier principal.

- Modularisation GUI (phase 4): extraction du bloc Presets/FOV/Reco de `_build_settings_tab` vers `zesolver/gui_settings_sections.py` (`build_presets_fov_reco_groups`), avec `zesolver.py` encore allégé.

- Modularisation GUI (phase 5): extraction des callbacks de l’onglet Settings (preset apply + connexions boutons/inputs) vers `zesolver/gui_settings_sections.py` (`apply_settings_preset`, `wire_settings_tab_callbacks`).

- Suppression de code mort/parasite: retrait d’un `column.addLayout(form)` dupliqué dans `_build_settings_tab` (double insertion du même layout).

- Nettoyage code mort (Settings): suppression du widget legacy `preset_warning_label` (toujours masqué, jamais affiché).

- Nettoyage i18n: retrait des clés de traduction orphelines `spec_warning_unknown` (aucune référence runtime).

- Nettoyage code mort (GUI core): suppression des méthodes orphelines `_gather_candidate_files` et `_refresh_file_list` (aucune référence dans ZeSolverWindow après migration scanner/refresh incrémental).

- Vérification statique locale post-nettoyage: plus de méthode `ZeSolverWindow` sans référence détectée dans `zesolver.py` + modules `zesolver/*.py`.

- Suppression code mort (solver core): retrait de `ImageSolver._try_blind_shortcut` (aucune référence runtime/tests).

- Hygiène post-suppression: vérification grep globale, plus aucune occurrence de `_try_blind_shortcut`, `_gather_candidate_files`, `_refresh_file_list` dans le repo.

- Nettoyage i18n (ultra-safe): suppression de 10 clés FR/EN réellement orphelines (`database_label`, `settings.cancel`, `solver.run.batch`, et 7 clés `astrometry.submit/job/polling` non référencées).

- Vérification post-nettoyage: compile OK et audit des clés restantes à faible fréquence, il ne reste que des clés utilisées via construction dynamique (`status_*`, `language_action_*`, `benchmark_dialog_filter_*`, `settings_*_option_*`).

- UX GUI (temps réel): le backend local envoie désormais les résultats au fil de l’eau au GUI (`BatchSolver.run(..., on_result=...)` + `SolveRunner` callback), ce qui réactive la progression progressive de la liste des fichiers traités.

- UX GUI (progress bar): ajout d’un timer de lissage (`_on_progress_tick`, 400 ms) pour afficher une progression intermédiaire pendant le traitement d’un fichier (progress bar en base 100 pas/fichier).

- Journalisation sortie: ajout d’une copie automatique du log runtime en fin de run dans le dossier de traitement (`zesolver_run_YYYYmmdd_HHMMSS.log`).

- Backend GUI: forçage du backend par défaut sur `local` (ZeNear flow) au chargement UI, sans réutiliser automatiquement la valeur persistée `astrometry`.

- Chaîne fallback locale implémentée dans `BatchSolver.run`: ZeNear → ZeBlind → Astrometry (dernier recours) si et seulement si une clé API Astrometry est présente.

- Émission GUI conservée sans doublons de comptage: en cas de fallback Astrometry, les échecs intermédiaires sont retenus puis remplacés par le résultat final (évite de dépasser le compteur de fichiers).

- Libellés UI mis à jour pour refléter la nouvelle logique (`Local (ZeNear → ZeBlind → Astrometry*)`).
