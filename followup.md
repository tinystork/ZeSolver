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
