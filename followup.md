# Follow-up ZeSolver — mission ZeBlind perf (2026-04-22)

## 1) Baseline de travail (actuelle)

Lot: `/home/tristan/zemosaic/example/testzenear/`

- Run A (scale 3.0): `zesolver_run_20260422_142645.log` -> **28/30**, **1000.1s**
- Run B (scale 1.5): `zesolver_run_20260422_145319.log` -> **28/30**, **918.9s**
- Run C (post P0/P1, scale 1.5): `zesolver_run_20260422_155824.log` -> **28/30**, **834.1s**
- Run D (tightening P2, scale 1.5): `zesolver_run_20260422_162330.log` -> **28/30**, **793.3s**
- Run E (Near 1-retry, scale 1.5): `zesolver_run_20260422_170404.log` -> **28/30**, **842.4s**
- Gain confirmé: **~ -8.1%** à résultat identique (mêmes 2 fichiers daubés)
- P0/P1 post-patch: nouveau run à **834.1s** (~**-9.2%** vs 918.9s), mêmes 2 échecs
- Bottleneck confirmé sur run D: Near ~**122.5s** vs Blind ~**551.6s** (5 succès blind + 2 échecs blind coûteux).
- Run E confirme le tradeoff: Near un peu plus court (~**115.8s**) mais coût des échecs blind en hausse (~**221.2s** cumulés), donc ZeBlind reste le point dur dominant.

## 2) Ce qui est déjà fait (sorti du backlog actif)

- Série d’optimisations blind-only patch1->patch6 validée sur lot propre.
- Meilleur run historique du jour: `patch6` autour de **785s** pour **30/30**.
- Essai `near_defer_blind_fallback` effectué puis rollback (pas retenu en défaut produit).
- Câblage runtime confirmé: caps S/M/L, vote percentile, search scale persistés et actifs en run.

---

## 3) Nouveau cap mission (backlog actif uniquement)

### P0 — Réduire le coût des échecs blind (priorité absolue)

Constat: les tentatives blind en échec coûtent nettement plus cher que les succès.

### TODO
- [x] Instrumenter un budget temps par tentative blind (succès/échec) avec ventilation par phase (`hinted`, `hinted_wide`, `scale_only`, `blind`). *(stats `attempt_elapsed_s`, `phase_perf`, `total_candidates_tried` ajoutées)*
- [x] Ajouter un **early-abort orienté échec** (quand support reste faible après N candidats / T secondes). *(v1: budget + garde-fous sur validations/inliers/candidats)*
- [x] Introduire un mode de sortie anticipée sur échec “certain” (faible support + validations faibles répétées + score decay). *(v1 branchée sur `scale_only`/`blind`)*
- [x] Produire un rapport A/B: median fail-attempt time avant/après. *(rapport signé: `reports/zeblind_ab_signed_20260422_200733.{json,md}`)*

### Critère de sortie P0
- [ ] **-25% à -35%** sur le temps médian des tentatives blind en échec, sans dégrader le taux de solve.

### Notes exécution P0 (2026-04-22)
- Gate rescue enrichi: `blind_rescue_attempt=2` peut désormais être court-circuité quand `fail_early_abort=true` sur la tentative précédente (skip explicite en logs).
- Calibration rapide sur 2 frames daubées: instrumentation et cache actifs, mais fail-fast non déclenché avec seuils par défaut (cas trop courts), ce qui confirme le besoin d’un A/B sur lot complet.
- `SolveConfig` enrichi avec paramètres fail-fast (`fail_attempt_budget_s`, `fail_attempt_min_validations`, `fail_attempt_max_best_inliers`, `fail_attempt_min_candidates`).
- `fail_stats` blind enrichis au retour d’échec avec raisons d’abort et métriques phase.
- Logs orchestrateur enrichis (`[ZEBLIND] fail metrics ...`) pour faciliter les comparaisons A/B.



---

### P0bis — ZeNear fail-early / handoff plus tôt vers blind

Constat: sur le run D, `near_rescue_attempt=2` a été lancé plusieurs fois sans produire de succès Near; mieux vaut limiter le coût Near et passer plus tôt à ZeBlind.

### TODO
- [x] Limiter ZeNear à **1 retry** par défaut (2e retry optionnel via `near_allow_second_rescue`).
- [x] Ajouter un skip explicite du 2e retry sur erreur persistante `could not estimate a similarity transform`.
- [x] Faire un A/B lot complet pour mesurer l’impact near->blind (temps total + solve rate). *(run E effectué)*

### Critère de sortie P0bis
- [ ] Baisse du temps cumulé Near sur frames difficiles, sans perte de solve rate.

### Notes exécution P0bis (2026-04-22)
- Patch appliqué dans `zesolver.py` (`_run_index_near_solver`): plan de rescue Near ramené à 1 tentative par défaut.
- Le 2e retry reste activable explicitement (`near_allow_second_rescue=true`) pour sécurité/rollback rapide.
- Exposition opératoire ajoutée: flags CLI `--near-allow-second-rescue` / `--no-near-allow-second-rescue` et persistance settings.
- Logs Near enrichis (`allow_second_rescue`) pour vérifier en run réel le mode actif.
- Validation run E: `near_rescue_attempt=2` absent (mode 1-retry actif), gain Near modeste mais non suffisant tant que les échecs blind restent coûteux.

### P1 — Optimiser la zone chaude `_collect_tile_matches`

Constat: trop de boucles Python (hashes -> buckets -> tiles -> paires).

### TODO
- [x] Profiling fin de `_collect_tile_matches` (temps + compteurs par niveau). *(métriques `collect_metrics` injectées dans stats blind)*
- [ ] Remplacer l’accumulation Python par stratégie vectorisée (`numpy`) quand possible.
- [x] Ajouter un cache local des correspondances utiles par `(level, tile_index, observed_hash_signature)`. *(cache local `collect_matches_cache` branché dans `_attempt_level`)*
- [ ] Évaluer une implémentation accélérée (Numba/Cython) derrière un flag expérimental.

### Critère de sortie P1
- [ ] Gain mesurable micro-benchmark + baisse du temps “candidate search” sur runs réels.

### Notes exécution P1 (2026-04-22)
- Ajout d’un cache local `collect_matches_cache` pour éviter les recomputes de `_collect_tile_matches` sur combinaisons répétées (niveau/tuile/hash-set observé).
- Ajout d’un compteur `collect_metrics` (`calls`, `cache_hits`, `cache_misses`, `compute_s`) remonté dans `solution.stats` (succès + échec).
- Logs échec enrichis côté orchestrateur avec résumé cache-hit (`[ZEBLIND] collect metrics ...`).
- Smoke test local sur frame difficile: cache actif observé (`calls=318`, `hits=229`, `hit_rate~72%`, `compute_s~1.02s`).
- P1 démarré sous flag expérimental `collect_matches_vectorized_experimental` (CLI: `--dev-collect-matches-vectorized` / `--no-dev-collect-matches-vectorized`), sans activation par défaut.
- Harness bench mis à jour avec `--collect-vectorized` pour comparer rapidement les deux modes sur le focus4 pipeline-like.
- Smoke bench initial (repeat=1) après wiring: pas de gain clair en mode vectorisé (variance élevée), décision = conserver flag OFF par défaut et poursuivre mesures x3 avant promotion.
- Bench x3 comparatif P1 exp: baseline `zeblind_focus4_pipeline_hinted_20260422_202628.json` vs vectorisé `..._204416.json` → gains mixtes/non robustes (ex: `233130` +1.6s), et `233211` passe de 1/3 à 0/3. Décision: garder le flag OFF par défaut.

---

### P2 — Budget adaptatif plus agressif en global mode

Objectif: couper plus tôt les candidats faibles au lieu d’explorer la queue inutilement.

### TODO
- [x] Rendre `candidate_limit` dynamique selon ratio top/runner-up et pente des scores. *(dominance top/runner-up en global mode)*
- [x] Renforcer l’arrêt sur streak faible support (par niveau, avec seuils distincts S/M/L). *(ajout d’un `medium_validation_streak` <=8 inliers)*
- [x] Ajouter un hard budget global par phase blind selon risque image. *(knobs `global_budget_fast_s`/`global_budget_slow_s`)*
- [x] Tracer explicitement la raison d’arrêt (`time_budget`, `weak_support_streak`, `score_floor_prune`). *(champ `phase_perf.early_abort_reason`)*

### Critère de sortie P2
- [ ] Réduction du temps global blind sur cas difficiles, sans hausse des faux échecs.

### Notes exécution P2 (2026-04-22)
- Harness benchmark ciblé ajouté: `tools/bench_zeblind_focus4.py` (run Zeblind-only sur 4 fichiers problématiques, JSON de sortie).
- Harness pipeline-like ajouté: `tools/bench_zeblind_focus4_pipeline.py` (copie temporaire des FITS + boucle de rescue type orchestrateur) pour des mesures plus proches du run réel.
- Premier artefact généré: `zeblind_focus4_hinted_20260422_173029.json` (base de comparaison rapide entre patches Zeblind).
- Artefact pipeline-like: `zeblind_focus4_pipeline_hinted_20260422_174323.json` (représentatif des rescues blind).
- Benchmark pipeline-like x3 exécuté (`zeblind_focus4_pipeline_hinted_20260422_174839/175230/175623`), agrégé dans `zeblind_focus4_pipeline_hinted_triplet_20260422_175639.json`: `233130`=0/3 (~76.6s), `233211`=0/3 (~72.4s), `233828`=3/3 (~66.1s), `232945`=3/3 (~11.3s).
- Validation rapide post-heuristique via `zeblind_focus4_pipeline_hinted_20260422_180823.json`: amélioration modeste sur `233130` (~74.0s vs ~76.6s médian historique), succès préservés sur `233828`/`232945`, `233211` toujours dur.
- Benchmark pipeline-like x3 post-heuristique (`181432/181820/182205`), agrégé dans `zeblind_focus4_pipeline_hinted_triplet_20260422_182223.json`: `233130`=0/3 (~73.4s), `233211`=0/3 (~72.3s), `233828`=3/3 (~64.7s), `232945`=3/3 (~11.1s).
- Benchmark pipeline-like x3 post-failfast-rescue2 (`182922/183307/183650`), agrégé dans `zeblind_focus4_pipeline_hinted_triplet_20260422_183709.json`: `233130`=0/3 (~73.7s), `233211`=0/3 (~69.6s), `233828`=3/3 (~63.5s), `232945`=3/3 (~10.8s).
- Tightening global-mode appliqué dans `_attempt_level`:
  - `candidate_limit` dynamique basé sur la dominance top/runner-up,
  - nouveau garde-fou `medium_validation_streak` (<=8 inliers) avec gate temporel,
  - logs explicites de sortie anticipée en queue faible.
- Analyse du run `155824`: fail-fast P0 n’était pas déclenché (`fail_early_abort=false` sur les 2 échecs), ce qui motive ce tightening P2.
- Budget global actif aussi en mode rescue lent (`fast_mode=false`) via nouveaux knobs `global_budget_fast_s`/`global_budget_slow_s` + raison d’abort tracée dans `phase_perf.early_abort_reason`.
- Heuristique hard-cases ajoutée côté orchestration blind: **soft-cap du rescue #2** (au lieu de full élargissement) quand tentative précédente coûteuse et profil peu prometteur (`no_validation` ou `stalled_high_churn`).
- Heuristique hard-cases renforcée: skip `rescue2` sur `no-validation stall` et sur `mid-support plateau` (fenêtre 6-8 inliers / 10-30 validations / tentative >60s).
- Hardening fail-fast sur `rescue2` uniquement: budget ramené à ~42s, seuil validations abaissé (<=12) et inliers plafond élargi (<=8) pour couper plus tôt les plateaux coûteux.
- Δwall vs triplet baseline: `233130` **-3.2s**, `233211` **-0.1s**, `233828` **-1.4s**, `232945` **-0.3s` (succès inchangés).
- Δwall vs baseline triplet (175639): `233130` **-2.9s**, `233211` **-2.8s**, `233828` **-2.6s**, `232945` **-0.5s** (succès inchangés).

---

### P3 — Filtre qualité image avant blind

Objectif: détecter tôt les frames “arbres/feuilles/pollution forte” et passer en mode dégradé rapide.

### TODO
- [x] Définir des métriques cheap pré-blind (densité d’étoiles robustes, texture parasite, ratio régions saturées).
- [x] Introduire un “profil blind dégradé” pour ces frames (budget borné + stratégie conservative).
- [x] Logger le routing qualité (`quality_profile=normal|degraded`) et son impact.

### Critère de sortie P3
- [ ] Moins de temps perdu sur images non solvables, sans casser les images solvables limites.

### Notes exécution P3 (2026-04-22)
- Routage qualité pré-blind ajouté dans `zesolver.py` (`_select_blind_quality_profile`) avec métriques cheap: `star_count`, `edge_star_fraction`, `texture_grad`, `sat_ratio`.
- Fallback robuste quand `peaks` indisponible: estimation étoiles via `_detect_stars` sur `frame_data` pour éviter un profil aveugle.
- Profil `degraded` branche maintenant des budgets blind plus bornés (`max_candidates/max_quads`, fail-fast plus court, plans rescue allégés).
- Logging explicite ajouté: `[ZEBLIND] quality profile for <file>: normal|degraded metrics={...}` + injection dans `run_info_blind_config`.
- Smoke ciblé validé:
  - `233130` → `quality_profile=degraded` (échec en ~90.4s, budget borné actif)
  - `233232` → `quality_profile=degraded` (échec en ~87.7s)
  - `233828` → `quality_profile=degraded` (succès en ~58.7s)
  - `233048` / `233211` → `quality_profile=normal` (routing non dégradé conservé)

---

## 4) Reprise ciblée d’idées Astrometry (à intégrer progressivement)

### Bloc A — KD-tree en espace des codes
- [ ] Prototype de retrieval candidats via structure type KD-tree / index voisinage code-space.

### Bloc B — Filtres géométriques précoces
- [ ] Ajouter des rejets rapides avant validation lourde (invariants de forme, échelle, cohérence locale).

### Bloc C — Verify / log-odds + seuils d’arrêt
- [ ] Introduire un score de crédibilité cumulatif pour arrêter tôt les mauvaises pistes.

### Bloc D — Budget strict `maxquads`
- [ ] Contrôle plus strict et adaptatif du budget quads selon niveau/qualité image.

---

## 5) Plan d’exécution recommandé

### Sprint 1 (court)
- [x] P0 instrumentation + early-abort échecs.
- [x] Baseline A/B sur le même lot (mêmes fichiers, mêmes settings). *(baseline `170404` vs candidate `190505_failfastr2`)*

### Sprint 2
- [~] P1 `_collect_tile_matches` (cache local livré, vectorisation restante).
- [ ] Mesure micro + run complet.

### Sprint 3
- P2 budget global adaptatif renforcé.
- P3 filtre qualité image pré-blind.

### Sprint 4
- Intégration progressive des briques Astrometry (A -> B -> C -> D), avec gate de non-régression à chaque étape.

---

## 6) Garde-fous de validation (obligatoires)

- [ ] Pas de régression solve rate sur lot de référence. *(Exception temporaire validée par Tristan: 3 frames obstruées peuvent rester non résolues tant que le tri stack reste acceptable.)*
- [ ] Pas de régression Near.
- [ ] Logs explicites exploitables pour chaque nouvelle heuristique.
- [ ] Comparatif avant/après signé (temps total, near median, blind success median, blind fail median).


## 7) Prochaine exécution immédiate

- Dernière mesure lot complet: `zesolver.log` (run ~19:16) -> **763.8s**, **27 solved / 3 failed**.
- Fichiers non résolus: `233232`, `233828`, `233130` (obstructions feuillage).
- Décision utilisateur (Tristan): échecs acceptables temporairement pour ces 3 frames; on poursuit l’optimisation perf.

- [x] Valider le patch ZeNear 1-retry sur lot complet (même dossier `testzenear`, mêmes réglages GUI).
- [x] Valider patch P2 budgets globaux (slow rescue) sur lot complet pour vérifier baisse du coût blind fail. *(bench pipeline-like x3)*
- [~] Cibler une heuristique spécifique pour `233130`/`233211` (échecs stables en pipeline-like). *(soft-cap rescue2 livré, tuning supplémentaire requis)*
- [x] Lancer un run lot complet post-patch (même dossier `testzenear`, mêmes réglages GUI) et mesurer l’impact global.
- [x] Produire le comparatif A/B signé (temps total, near median, blind success median, blind fail median) en notant l’exception temporaire sur les 3 frames obstruées. *(fait: `reports/zeblind_ab_signed_20260422_200733.md`)*
- [~] Démarrer P1 vectorisation ciblée de `_collect_tile_matches` (au moins la phase de vote/pairs), derrière un flag expérimental. *(flag dev ajouté, gain non confirmé sur premier bench, à retester x3)*

### Update run complet post-patch (2026-04-23)

- Run lot complet exécuté: `/home/tristan/zemosaic/example/testzenear/zesolver_run_20260423_091723_patch30s.log`
- Résultat: **29/30** en **718.2s** (1 échec: `233232`).
- Comparatif A/B signé généré:
  - `reports/zeblind_ab_signed_20260423_093823.json`
  - `reports/zeblind_ab_signed_20260423_093823.md`
- Delta vs baseline followup (`170404`, 842.4s):
  - Temps total: **-124.2s**
  - Solve count: **+1** (29/30 vs 28/30)
  - Blind fail médiane: **110.58s -> 26.96s** (**-75.6%**)

Statut mission:
- P0 (réduction coût des échecs blind sans régression solve rate): **atteint sur ce run**.
- Gate strict no-regression solve rate: **OK**.

## 8) Extension plan perf ZeBlind (Astrometry-inspired, blind-only)

Contrainte validée avec Tristan:
- **Ne plus toucher ZeNear** (protection stricte).
- Les optimisations ci-dessous s’appliquent **uniquement à ZeBlind**.

### P4 — Depth ladder (étoiles brillantes d’abord)

Référence Astrometry:
- `--depth`, `--objs` (man/readme)

TODO:
- [x] Ajouter un ladder progressif des étoiles côté ZeBlind. *(v1 sous flag, défaut: `80 -> 160 -> 500`)*
- [~] Ne monter de palier qu’en cas d’échec ou d’évidence de support insuffisant. *(v1: montée séquentielle, tuning adaptatif restant)*
- [x] Logger le palier utilisé et le coût par palier. *(stats `depth_ladder_rows` + message de stage)*

Critère de sortie:
- [ ] Réduction mesurable du temps médian blind sur succès, sans baisse du solve rate.

Notes exécution P4 (2026-04-23):
- Implémentation blind-only dans `zeblindsolver.py` (orchestration multi-stages via `max_stars`), avec métriques agrégées `depth_ladder_*`.
- Guardrail vitesse ajouté: stage 1 borné (`hard_max_candidates_tried=96`) pour limiter le coût des faux-départs.
- Rescue path: depth ladder désactivé sur les rescues (`depth_ladder_enabled=False`) pour éviter la multiplication des coûts.
- Orchestrateur: depth ladder activé uniquement sur `quality_profile=degraded` (désactivé sur profil normal pour protéger le débit).
- Smoke focus4 (pipeline-like, hinted, repeat=1):
  - OFF: `reports/zeblind_depthladder_off_20260423_focus4.json`
  - ON (v4): `reports/zeblind_depthladder_on_20260423_focus4_v4.json`
  - Comparatif: `reports/zeblind_depthladder_compare_20260423_141842.{json,md}`
- Résultat smoke: ON améliore le solve-rate (3/4 vs 2/4) mais reste plus coûteux en temps global; décision prudente maintenue: **OFF par défaut** tant que tuning vitesse non finalisé.

### P5 — Vérification log-odds avec arrêt anticipé

Référence Astrometry:
- `logodds_bail` + `logodds_stoplooking` (`solver/verify.c`, `solver/solver.c`)

TODO:
- [x] Introduire un score log-odds cumulatif par hypothèse. *(implémentation v1 côté ZeBlind sous flag)*
- [x] Ajouter un seuil de rejet anticipé (bail) sur trajectoire défavorable. *(v1 branchée sur validations échouées)*
- [x] Ajouter un seuil d’acceptation anticipée (stoplooking) quand la preuve est suffisante. *(v1: neutralise les aborts "weak support" quand score cumulé positif)*
- [x] Instrumenter les raisons d’arrêt (bail/stoplooking/fin naturelle). *(stats + logs enrichis)*

Critère de sortie:
- [ ] Baisse nette du coût de vérification sur cas difficiles, solve rate non régressif.

Notes exécution P5 (2026-04-23):
- `SolveConfig` étendu: `verify_logodds_enabled`, `verify_logodds_bail`, `verify_logodds_stoplooking`, `verify_logodds_min_validations`.
- Intégration dans le fail-fast ZeBlind (`_maybe_fail_early_abort`) + télémétrie (`verify_logodds_cum`, `verify_logodds_last`).
- Smoke A/B focus4 hinted (repeat=1):
  - ON: `reports/zeblind_logodds_on_20260423_focus4.json`
  - OFF(default): `reports/zeblind_default_after_patch_20260423_focus4.json`
  - Comparatif: `reports/zeblind_logodds_smoke_compare_20260423_103542.{json,md}`
- Résultat smoke: gain temps global ON, mais risque solve-rate (1/4 ON vs 2/4 OFF). Décision prudente: **flag OFF par défaut**.

### P6 — Budgets durs de recherche

Référence Astrometry:
- `maxquads`, `maxmatches` (`solver/solver.c`)

TODO:
- [x] Ajouter des caps explicites par tentative (quads générés / matches testés). *(v1: caps durs sur candidats/validations, désactivés par défaut)*
- [ ] Rendre ces caps adaptatifs selon profil qualité image (normal/degraded).
- [x] Exposer les caps en config + logs de hit budget.

Critère de sortie:
- [ ] Limitation des queues longues blind, pas de régression robuste sur lot de référence.

Notes exécution P6 (2026-04-23):
- Nouveaux knobs ZeBlind: `hard_max_candidates_tried`, `hard_max_validations` (CLI + config runtime).
- Raisons d’arrêt hard-budget remontées dans `fail_early_abort_reason` et `phase_perf`.
- Tuning rescue blind-only ajouté: skip `rescue2` sur profils low-signal (support trop faible après rescue1) avec garde-fous pour préserver `233828`.
- Benchmark focus4 pipeline-like (hinted):
  - Baseline r2: `reports/zeblind_speed_baseline_20260423_focus4_r2.json`
  - Candidate r2: `reports/zeblind_speed_lowsignal_20260423_focus4_r2.json`
  - Comparatif: `reports/zeblind_speed_compare_lowsignal_20260423_151900.{json,md}`
- Résultat r2: solve-rate inchangé (**4/8**) avec baisse nette du temps cumulé (candidate **-115.80s** sur focus4 r2).
- Ajustement post-r2: seuil low-signal rendu plus conservateur (`best_fail_inliers <= 3` au lieu de `<= 4/6`) pour éviter les faux négatifs rescue2.
- Validation lot complet M106 (même protocole que baseline, `--overwrite`, chemin dataset réel):
  - Baseline: `zemosaic/example/testzenear/zesolver_run_20260423_091723_patch30s.cli.log`
  - Candidate: `reports/zesolver_run_20260423_163025_cand_lowsignal_full30_overwrite_REALPATH.cli.log`
  - Compare: `reports/zeblind_fullrun_compare_20260423_164257.{json,md}`
  - Verdict fullrun: solve-rate conservé (**29/30**, même unique échec `233232`) avec gain temps **-24.9s** (693.3s vs 718.2s).

### P7 — Verrou parité (quand l’info est fiable)

Référence Astrometry:
- `--parity pos/neg` (man + `solver.h`)

TODO:
- [ ] Permettre un mode ZeBlind parity-locked (nominal uniquement ou mirror uniquement).
- [ ] Utiliser ce lock seulement quand l’information de parité est considérée fiable.
- [ ] Fallback automatique vers dual-parity si confiance insuffisante.

Critère de sortie:
- [ ] Gain CPU sur recherche candidats, solve rate préservé.

### P8 — Uniformisation / dédup en vérification

Référence Astrometry:
- `verify_uniformize`, `verify_dedup` (`solver/verify.c`)

TODO:
- [ ] Uniformiser spatialement les étoiles de vérification (downsample intelligent, non aléatoire).
- [ ] Dédupliquer les paires et correspondances redondantes avant scoring lourd.
- [ ] Mesurer impact direct sur coût de verify et sur robustesse.

Critère de sortie:
- [ ] Temps verify réduit à précision équivalente.

### Ordre d’exécution recommandé (blind-only)

1. P5 log-odds bail/stoplooking
2. P4 depth ladder
3. P6 budgets durs
4. P8 uniformisation/dédup verify
5. P7 verrou parité (garde-fous stricts)

