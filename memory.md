# ZeSolver Memory Log (accompli uniquement)

> Règle: ce fichier contient uniquement ce qui a été fait/validé.
> Le reste à faire est dans `followup.md`.

## 2026-04-11 → 2026-04-24 (synthèse consolidée)

### Robustesse solver / infra
- Audit global du flow solver (zeblind, metadata, index/quads, détection) et stabilisation progressive.
- Multiples correctifs de robustesse quad/hash (ordering canonique, fallbacks sampling, fallback hashes par niveau).
- Correctifs stop/cancel batch (poll futures + shutdown non bloquant) pour éviter les faux freeze.
- Fix runtime GPU/venv (re-exec dans `.venv`), suppression de causes CPU-only involontaires.

### ZeNear (phase fiabilisation)
- Passage progressif à une validation near adaptative (inliers requis non figés), avec gains nets de solve local.
- Staging de fallback détection near (passes successives) pour cas faibles supports.
- Smokes validés sur sous-ensembles puis jeux plus larges (améliorations de solve observées à l’époque de cette phase).

### ZeBlind (phase conformité Astrometry initiale)
- Mise en place/incrémentation du moteur verify log-odds (match/conflict/distractor) + télémétrie.
- Préfiltres d’échelle/hypothèse et instrumentation détaillée des causes d’échec.
- Mécaniques de pruning/tri des paires (uniformisation, dédup, scoring géométrique) avec gains de coût mais solve-rate encore limité sur lot fallback.
- Références fallback Astrometry conservées (`rejected_cases_wcs_reference.{json,md}`).

---

## 2026-04-25 (sprint portage Astrometry blind)

### Décisions structurantes
- Validation explicite: poursuivre le portage complet avant toute désactivation prématurée de briques amont.
- Checkpoint anti-boucle formalisé: **NO-GO** sur micro-tuning resolve-hit seul tant que l’amont ne produit pas de signal.

### Portage réalisé (code/méthode)
- Portage `try_permutations` dans le chemin principal quad-match (pas seulement rescue).
- Ajout bloc add_stars/inbox côté ZeBlind et conservation en défaut actif.
- Évolution lookup code-space:
  - near lookup (bucket-level puis entry-level),
  - filtre continu code-space,
  - backend rangesearch continu (approximation).
- Introduction objet canonique `code-hit` (entry/score/source) + score gating + geom gating amont.
- Ajout mode `code-hit only` (génération candidates centrée hits canonique, fallback conditionnel).

### Resolve-hit / consommation des hits
- Ajout resolve-hit local + refit, puis évaluation globale post-refit.
- Ajout instrumentation globale (`global_evals`, `global_better`, deltas) et variante hybrid-refit.
- Correctif de flux: un quality reject resolve-hit ne supprime plus l’hypothesis de base.

### Résultats observés (subset/cas bloquants)
- Les briques s’exécutent et sont instrumentées (hits/evals présents), mais:
  - `nummatches` reste à 0 sur les cas bloquants M106,
  - `global_better`/`hybrid_better` restent à 0 dans les A/B décisifs.
- Gains surtout côté coût/filtrage bruit, pas encore de percée solve.

### Rapports de référence de la soirée
- Complétude/gaps portage: `zeblind_astrometry_portage_completeness_20260425_v74.md`
- Reassessment gaps: `..._v77b.md`, `..._v78b.md`
- Go/No-Go formel: `zeblind_portage_go_nogo_20260425_v86b.md`
- A/B décisif resolve-hit OFF/ON: `zeblind_p1_subset2_20260425_v86_resolve_hit_ab_decisive.json`
- A/B mode code-hit only: `zeblind_p1_subset2_20260425_v87_code_hit_only_ab.json`

---

## État consolidé (fin 2026-04-25)
- Conformité structure/code Astrometry: partielle avancée.
- Parité comportementale sur cas bloquants: non atteinte.
- Verrou principal confirmé: qualité des correspondances amont (avant resolve), pas le tuning local du refit.


- 2026-04-25 (G2.65): début concret du bloc A5 (recherche continue plus native) avec backend `blind_astrometry_code_rangesearch_backend` (`linf`/`l2`) + rayon `blind_astrometry_code_rangesearch_l2_radius`, cache des codes log + tri 1D pour pruning local (approx rangesearch continu). A/B subset2 `v88`: backend `l2` bien actif (`code_rs_l2_activations>0`), coût runtime en baisse sur les 2 cas, mais `nummatches=0` et `global_better=0` inchangés.
- 2026-04-25 (G2.66): enrichissement de l’objet canonique `code-hit` (ajout `rank` et `metric`), puis consommation ordonnée en mode `code-hit only` (priorité rank/score). Smoke `v88b` valide l’exécution sans régression fonctionnelle; blocage solve toujours présent sur 233027.


- 2026-04-25 (G2.67): ajout du mode `blind_astrometry_hit_pipeline_first_*` (strict/fallback) pour forcer une consommation hits->candidats avant heuristiques legacy, avec métriques d’activation/fallback/skip. A/B subset2 `v89`: pipeline-first s’active massivement (activations >0) et le mode strict réduit parfois le coût, mais aucun signal solve (`nummatches=0`, `resolve_global_better=0`).


- 2026-04-25 (G2.68): extension du backend de recherche continue avec un troisième mode `grid3d` (binning 3D en log-space + voisinage par bins), en plus de `linf` et `l2`. Ajout des métriques `astrometry_code_rangesearch_grid_activations` et options `blind_astrometry_code_rangesearch_grid_*`. A/B subset2 `v90` (linf/l2/grid3d + hit-pipeline-first strict): backends bien activés selon mode mais aucun signal solve (`nummatches=0`, `resolve_global_better=0`).


- 2026-04-25 (stabilisation avant pause): interruption des runs Python longue durée restés actifs (3 processus `python3 -` CPU-bound) pour repartir proprement. Les backends expérimentaux code-space (`l2`, `grid3d`) sont désormais **désactivés par défaut** via `blind_astrometry_code_rangesearch_allow_experimental_backends=False`; toute demande `backend=l2/grid3d` est forcée en `linf` tant que ce flag n’est pas activé explicitement. Compile/tests rapides validés après changement.


- 2026-04-25 (G2.69): accomplissement du bloc B4 minimal avec ajout d’un mode dédié non-legacy `blind_astrometry_hit_pipeline_dedicated_enabled` (+ fallback/strict) qui force la consommation des `code-hit` avant parcours legacy. Nouvelles métriques `astrometry_hit_pipeline_dedicated_*`. A/B subset2 `v91`: chemin dédié bien activé (activations/skips/fallbacks cohérents) et réduction nette du runtime, mais solve-rate inchangé (`nummatches=0`, `resolve_global_better=0`).

- 2026-04-29 (état des lieux ZeBlind): analyse complète du run `zesolver.log` confirmant le pattern actuel: ZeNear robuste sur cas simples, ZeBlind en échec sur cas difficiles avec `best_fail_inliers=0` et longues recherches candidates, puis fallback Astrometry qui résout 8/8. Relecture croisée Astrometry source (`solver.c`, `verify2.c`) + rapports `v78b/v86b` => gaps encore bloquants priorisés: KD/rangesearch natif, pipeline strict hit->resolve->fit->verify, parité complète permutations/index-driven, alignement compteurs/arrêts, activation cohérente verify log-odds en run blind.

- 2026-04-29 (itération autonome ZeBlind, robustesse runtime): patch du contrôle de budget pour les probes difficiles. Les hard caps (`hard_max_candidates_tried`, `hard_max_validations`) sont maintenant consommés globalement à travers les stages du depth ladder (au lieu d’être réinitialisés implicitement), avec arrêt explicite quand le budget est épuisé. Le scale-ladder est aussi bypassé automatiquement quand un hard budget explicite est posé, pour éviter les reruns coûteux en diagnostic. Validation: compile OK + probe mono-image (`testzeblind`) qui s’arrête proprement sur budget épuisé au lieu de repartir en longues itérations.

- 2026-04-29 (probe mono-image post-patch): test `testzeblind` en mode borné (`depth_ladder=0`, `scale_ladder=0`, `hit_pipeline_dedicated=1`, hard cap=40) confirme un échec en amont avec `validations=0` avant épuisement budget. Conclusion pratique: le verrou reste côté génération/qualité des correspondances candidates (avant verify), pas sur des seuils de validation finaux.

- 2026-04-29 (re-cadrage mission): suite à la demande Tristan, mission officielle basculée en parité complète ZeBlind/Astrometry (local Python pur, sans logique serveur Astrometry), avec ZeNear gelé hors non-régression. Audit transversal lancé sur `astrometry-main` (solver/plot/include/libkd/util), backlog exhaustif formalisé dans `followup.md` + rapport `reports/zeblind_astrometry_full_audit_20260429.md`.

- 2026-04-29 (parité ZeBlind, A1-v1): implémentation d’un backend code-space `kdbox` dans `zeblindsolver.py` (KD-tree local 3D sur codes log + query box rangesearch), cache tuile enrichi, et exposition CLI (`--blind-astrometry-code-rangesearch-backend`, `--blind-astrometry-code-rangesearch-allow-experimental-backends`). Gates techniques OK (py_compile + tests ciblés 2/2). Probe mono-image borné encore en échec solve-rate (`validations=0`) -> blocage principal toujours amont correspondances/resolve, pas un souci de plumbing runtime.

- 2026-04-29 (probe qualité vs logique): exécution réelle ZeBlind via appel explicite `main()` (et non `python -m zeblindsolver.zeblindsolver`, qui peut être no-op). Sur copies RAW sans WCS (`testzeblind/probe_raw`), ZeBlind échoue à la fois sur une image de référence ZeNear-résoluble et sur un cas dur (caps 220/120), avec `validations=0..2`, `best_inliers<=0` pour `linf` et `kdbox` -> blocage confirmé en amont de verify (candidate/hypothesis pipeline), pas seulement qualité image.

- 2026-04-29 (try_permutations v1): ZeBlind patch avec expansion permutations AB-first (`blind_astrometry_try_permutations_max` défaut 12 + helper `_astrometry_permutation_orders`) et nouveaux compteurs `astrometry_try_perm_*`. Probe runtime réel sur image RAW ZeNear-résoluble: permutations et resolve-hit s’activent massivement, mais toujours `no valid solution` (`nummatches=0`, inliers quasi nuls) -> blocage confirmé sur qualité des correspondances/hypothèses, pas sur absence de parcours permutations.

- 2026-04-29 (resolve/reassign v1): ajout d’un scoring de transform one-to-one non index-lock (`_score_transform_with_reassignment`) et intégration dans le pipeline hypothèse + consommation des paires reassigned pour construire les inliers. Probe RAW ZeNear-résoluble (caps 40/40): amélioration marginale du signal (`empty_inliers` 27->21, `too_few_inliers` 3->9, `reassign_eval_better=15`) mais toujours `nummatches=0` / `failed_validations=0`.

- 2026-04-29 (code rangesearch adaptatif): ajout d'un auto-disable `code_rangesearch` en run faible (après N candidats sans validations/inliers), avec compteur `astrometry_code_rangesearch_auto_disabled`. Effet mesuré sur image RAW ZeNear-résoluble (caps 30/30): `nummatches` passe de 0 à 4 et `failed_validations` de 0 à 1; progrès partiel mais pas encore de solution complète, cas dur inchangé.

- 2026-04-29 (portage try_permutations v2): remplacement du set fixe de permutations par une génération récursive Astrometry-like dans le chemin candidat principal (slot-filling interne + gardes `cx<=dx` et `meanx<=0.5`, swap backbone AB). Ajout config `blind_astrometry_cxdx_margin` et compteurs `astrometry_try_perm_meanx_rejects`/`astrometry_try_perm_invalid_code`. Gate OK (py_compile + tests ciblés). Probe RAW ZeNear ref (caps 30/30): `numtries=30`, `nummatches=4`, `failed_validations=1` avec rejets récursifs tracés.

- 2026-04-29 (portage resolve_matches v2): ajout d'un appariement one-to-one global par distance (`_greedy_global_pairing`) et intégration dans `resolve_hit` + `reassign_eval`. Gate OK (py_compile + tests). Sur RAW ZeNear ref, le signal verify augmente à budget élargi (`nummatches` monte à 12 et `failed_validations` à 4 au cap 120), sans solve final pour l'instant.

- 2026-04-29 (portage verify/log-odds v1): alignement de l'acceptation sur une logique Astrometry-like en 2 étages (`logaccept=min(totune,tokeep)` puis gate final `tokeep`), avec nouveau flag `blind_accept_logodds_astrometry_stages` (CLI inclus) et compteurs dédiés (`accept_logodds_precheck_*`, `accept_logodds_below_keep`, `accept_logodds_keep_pass`). Gate OK (py_compile + tests). Probe ZeNear ref caps 120: ON/OFF identiques (`nummatches=12`, `failed_validations=4`).

- 2026-04-29 (portage astrometry counters/stop v1): `nummatches` réaligné sur une sémantique Astrometry (incrément par hit envoyé au verify, plus par volume de paires), suppression de l'incrément en re-verify tune, ajout `maxmatches_active` dans `astrometry_semantics`, et normalisation des `stop_reasons` avec champ `code` stable (`candidate_cap`, `maxquads_reached`, `maxmatches_reached`, `logodds_bail`, etc.). Gate OK (py_compile + tests), probes bornés sans solve final mais diagnostics plus fidèles.

- 2026-04-29 (portage chaîne stricte hit->resolve->fit->verify v1): `_resolve_hit_correspondences` exporte les correspondances indexées (`src_indices/dst_indices`) et ces index sont maintenant propagés dans le flux principal post-hypothesis (`reassign_src_idx/reassign_dst_idx`) avant fit/validate. Nouveaux compteurs `astrometry_post_resolve_attempts/hits` + traces `verify_hit_trace(stage=hit_resolve_chain)`. Gate OK (py_compile + tests). Probe ZeNear ref caps120: pipeline utilisé (`post_resolve_hits=4`) sans solve final.

- 2026-04-29 (portage objets match v1): ajout d'une structure `astrometry_match_objects` (stages `validate_base`/`validate_gate`/`accept`) pour tracer des objets match Astrometry-like avec source explicite des correspondances (`inlier_mask`, `reassign_eval`, `resolve_hit`), score, échelle, qualité et raison de rejet. Export activé dans les stats finaux + compteur de phase `astrometry_match_objects_total`. Gate OK (py_compile + tests). Probe ZeNear ref caps120: 6 objets (3 validate_base fail + 3 validate_gate fail), source `resolve_hit` dominante, toujours sans solve final.

- 2026-04-29 (portage tweak2 verify v1): ajout d'un retry Astrometry-like après `validate_base` en échec (tolérance élargie, refit robuste, re-validation) avec nouveaux flags `blind_tweak2_verify_*` et options CLI associées. Instrumentation `tweak2_verify_attempts/hits`. Gate OK (py_compile + tests). A/B sur ZeNear ref caps120: mécanisme actif (3 tentatives) mais 0 hit et résultat final inchangé (`nummatches=3`, `failed_validations=4`).

- 2026-04-29 (portage validate scale rescue v1): ajout d'un fallback de re-validation ciblé sur les échecs `pixel_scale_out_of_range`, avec bornes d'échelle relâchées de façon contrôlée (`blind_validate_scale_rescue_*`, CLI inclus). Instrumentation `validate_scale_rescue_attempts/hits`. Gate OK (py_compile + tests). A/B sur ZeNear ref caps120: `failed_validations` baisse de 4 à 3 (1 rescue hit), sans solve final.

- 2026-04-29 (probe expansion match-object + guardrail): tentative d'expansion des match-objects `resolve_hit` avant validate (avec soft fallback) implémentée et mesurée. A/B ZeNear ref caps120: ON dégrade (`failed_validations` 3->5). Décision produit appliquée immédiatement: conserver le mécanisme en expérimental mais **désactivé par défaut** (`blind_match_object_expand_enabled=False`, CLI default 0) pour protéger la non-régression. Gate OK (py_compile + tests).


- 2026-04-30 (probe amont hypothèse): re-probe mono-image RAW ZeNear ref avec caps bornés (30/30), depth/scale ladders OFF, puis pipeline hits activé (`code_hit_object=1`, `hit_pipeline_dedicated=1`, lock parité OFF). Résultat inchangé: échec avant verify (`fail_validation_count=0`, `best_fail_reason=hypothesis failed before validation`). Signal dominant mesuré: `fail_stage_counts` pairing=7 / hypothesis=23, surtout `hypothesis_fail_empty_inliers` (scale_only=15, hinted_wide=6). Test d’ablation `preinlier_relax` agressif (max_factor 4.5, max_px 15, refit sigma 4.0) sans effet (`preinlier_relax_hits=0`, `preinlier_refit_hits=0`). Conclusion: blocage principal confirmé sur l’amont `resolve_matches -> seed transform`, pas sur verify ni sur simple relaxation de tolérance.


- 2026-04-30 (A2-v1 seed-rescue transform): patch amont dans `zeblindsolver.py` juste avant `empty_inliers` pour tenter un refit de transform depuis correspondances `resolve_hit` relâchées (tolérance élargie + refit similarity). Validation technique OK (`py_compile` + `pytest` ciblé 2/2). Probe RAW ZeNear ref borné (caps 30/30) sans gain solve: `fail_validation_count=0`, `best_fail_reason=hypothesis failed before validation`, `astrometry_seed_rescue_hits=0`, distribution d’échec inchangée (`hypothesis_fail_empty_inliers` dominant).

- 2026-04-30 (A2-v2/v3 amont): deux itérations supplémentaires testées sur probe RAW ZeNear ref (caps 30/30). A2-v2 = retry adaptatif `resolve_hit` (tolérance x2.5, `max_pairs` élargi). A2-v3 = bootstrap similarity depuis meilleurs résidus (best-4..8) avant gate `empty_inliers`. Gates techniques OK (`py_compile`, `pytest` 2/2). Résultat inchangé: `success=false`, `fail_validation_count=0`, `best_fail_reason=hypothesis failed before validation`, `seed_best4_hits=0`, `seed_rescue_hits=0`, `hypothesis_fail_empty_inliers` reste dominant.

- 2026-04-30 (A2-v4 diagnostics post_resolve): ajout instrumentation ciblée sur échecs `rh_post=None` (compteurs too_few_pairs + paires totales + distance médiane). Gate OK (`py_compile`, `pytest` 2/2). Probe RAW ZeNear ref borné: `post_resolve_attempts=23`, `post_resolve_hits=0`, `post_resolve_fail_too_few_pairs=23`, `post_resolve_fail_pairs_total=2`. Conclusion robuste: blocage principal = manque de correspondances stables avant validate; verify non impliqué à ce stade.

- 2026-04-30 (A2-v5/v6): ajout d’un rescue amont de collecte de paires (passage relâché) + tentative post-resolve relâchée conditionnelle. Gates OK (`py_compile`, `pytest` 2/2). Probe RAW ref: gain massif en paires brutes (`upstream_pair_rescue_pairs_gain=494`, hits rescue=8/15), mais toujours `nummatches=0` et `post_resolve_hits=0` (même en mode relâché). Lecture durable: blocage principal = consistance géométrique des paires candidates, pas seulement volume.

- 2026-04-30 (A2-v7): tentative de stabilisation géométrique des paires après rescue amont via `pre_model_bipartite_reassign` forcé. Gates OK (`py_compile`, `pytest` 2/2). Probe RAW ref: passage tenté 14 fois mais aucun subset retenu (`a2v7_geom_repair_hits=0`), `post_resolve_hits=0`, `nummatches=0`. Le blocage persiste avant verify.

- 2026-04-30 (A2-v8): ajout instrumentation support paires pré-transform + probes comparatives `probe_full` vs `probe_hint_strict` (hinted/hinted_wide only). Gates OK (`py_compile`, `pytest` 2/2). Constat robuste: `post_resolve_hits=0` dans les deux modes, échec dominant `too_few_pairs`; en strict apparaît aussi `best_fail_reason=blind hypothesis scale hard reject` avec `fail_validation_count=1`.

- 2026-04-30 (A2-v9 scale-hard A/B strict): comparaison ON/OFF du `blind_hypothesis_scale_hard_filter_enabled` en mode hinted strict. OFF débloque un léger signal (`hinted post_resolve_hits: 0 -> 1`) mais pas de solve (`nummatches=0`), avec bascule du motif d’échec vers `blind scale prefilter failed`. Conclusion: le gate scale-hard masque un peu de signal mais le verrou principal reste la chaîne de correspondances + cohérence d’échelle.

- 2026-04-30 (A2-v10 scale-prefilter A/B strict): avec `scale_hard_filter=OFF`, comparaison `blind_scale_prefilter` ON/OFF en mode hinted strict. OFF débloque un cran de signal (`nummatches: 0 -> 1`) mais l’hypothèse échoue ensuite sur `pixel_scale_out_of_range[179.816]`. Interprétation: le prefilter précoce masque une piste réelle, mais la validation d’échelle finale reste le verrou suivant.

- 2026-04-30 (A2-v11): prolongation après A2-v10. En strict hinted avec `scale_prefilter=OFF`, le signal `nummatches=1` est reproductible (quasi-candidat `d50_0819`, 12 inliers, RMS~1.36 px, scale ~179.8"/px). Le scale-rescue élargi déplace l’échec de `pixel_scale_out_of_range` vers `validation failed` mais ne solve pas. A/B parité (auto vs nominal-only) ne change pas ce quasi-candidat. Conclusion: verrou actuel sur un sous-gate de validation globale, pas sur la parité ni l’absence totale de match.

- 2026-04-30 (A2-v12): instrumentation fine des sous-gates validation ajoutée (`validation_failed[rms_ok,inliers_ok,...]`) + raison géométrique détaillée sur guardrail. Probe ciblé (strict hinted, `quality_rms=1.4`) confirme le verrou final: quasi-solution dégénérée avec inliers quasi-colinéaires (`cov_area~5e-6`, `cond~5e14`) sur `d50_0819`. Conclusion durable: blocage principal actuel = dégénérescence géométrique des inliers, pas simple seuil RMS/inliers.

- 2026-04-30 (A2-v13): ajout d’un guardrail géométrique précoce sur inliers provisoires (`inlier_count>=8`) pour rejeter plus tôt les hypothèses colinéaires. Gates OK (`py_compile`, `pytest` 2/2). Probe strict hinted: détection effective (`a2v13_geo_early_rejects=1`) mais résultat final inchangé (quasi-candidat dégénéré toujours rejeté). Le levier suivant doit agir en amont sur la construction des correspondances, pas uniquement sur les gates de rejet.

- 2026-04-30 (A2-v14): ajout d’un guard footprint 2D à l’acceptation `post_resolve`. Gates OK (`py_compile`, `pytest` 2/2). Probe strict hinted: le set post-resolve dégénéré est bien rejeté (`a2v14_postresolve_reject_footprint=1`, `post_resolve_hits=0`), mais une autre voie alimente encore la quasi-solution colinéaire (`nummatches=1`, rejet géométrique final inchangé). Prochaine action: imposer la même contrainte en amont sur `reassign_eval`/inlier-mask.

- 2026-04-30 (A2-v15): extension du guard footprint 2D à la voie inliers globale (inlier_mask/reassign_eval/resolve_hit). Gates OK (`py_compile`, `pytest` 2/2). Probe strict hinted: rejet capturé (`a2v15_inlier_footprint_rejects=1`) et `post_resolve_hits=0`, mais pas de solve final (`nummatches=1`, rejet géométrique). Conclusion: les guards sont en place; la prochaine progression doit cibler la construction de correspondances non-colinéaires en amont.

- 2026-04-30 (A2-v16): ajout d’un préfiltre anti-colinéaire avant hypothèse (footprint check + tentative de pruning extrêmes x/y). Gates OK (`py_compile`, `pytest` 2/2). Probe strict hinted: exécution confirmée (`a2v16_prefilter_checks=10`) mais aucun pruning appliqué (`pruned=0`), issue inchangée (`nummatches=1` puis rejet géométrique). Conclusion: il faut agir sur la génération de paires initiale, pas seulement filtrer après coup.

- 2026-04-30 (A2-v17): ajout d’un mécanisme de diversité spatiale dans `_collect_tile_matches` (top-votes + extrêmes x/y) pour tenter d’éviter les sets de paires colinéaires dès la génération. Gates OK (`py_compile`, `pytest` 2/2). Probe strict hinted: issue inchangée (`nummatches=1` puis rejet géométrique colinéaire). Conclusion: nécessité d’un debug ciblé candidat/tile (`d50_0819`) pour comprendre la source exacte de la dégénérescence persistante.

- 2026-04-30 (A2-v18/A2-v18b): debug ciblé sur `d50_0819` confirmé (coverage extrêmement faible, inliers colinéaires) puis skip expérimental après premier hit dégénéré. Gates OK (`py_compile`, `pytest` 2/2). Effet: légère baisse du coût local (`post_resolve_attempts`), mais résultat final inchangé (`nummatches=1` puis rejet géométrique). Conclusion: problème systémique de construction d’hypothèses, pas simple blacklist d’un candidat.

- 2026-04-30 (A2-v19): ajout d’une réparation globale de dispersion (sous-ensemble extrêmes x/y + RANSAC dédié) quand le footprint inliers est insuffisant. Gates OK (`py_compile`, `pytest` 2/2). Probe strict hinted: tentative exécutée (`a2v19_spread_repair_attempts=1`) mais sans réussite (`hits=0`), issue inchangée (`nummatches=1` puis rejet géométrique).

- 2026-04-30 (A2-v20): production d’un packet ZeBlind structuré (`reports/zeblind_a2v20_compare_packet_20260430.json`) + diff face à une WCS Astrometry de référence. Signal clé confirmé: quasi-candidat ZeBlind reste à ~179.8"/px contre ~2.372"/px (écart ~75.8x), aligné avec la dégénérescence colinéaire et les rejets géométriques finaux.

- 2026-04-30 (A2-v21): export ciblé du candidat `d50_0819` (snapshot brut des paires) en NPZ/JSON. Résultat marquant: le set brut capturé (6 paires) présente un footprint large et un conditionnement modéré, donc la dégénérescence colinéaire observée en fin de chaîne semble émerger plus tard (phase de réassignation/filtrage final des inliers), pas au point de collecte brut.

- 2026-04-30 (A2-v22): traçage 3 états sur `d50_0819` (`raw -> post_reassign -> final_guard`) réalisé. Signal clé: `raw` est spatialement étalé (bbox large, cond modéré) alors que `final_guard` est ultra-colinéaire (cov_area ~5e-6, cond ~5e14). Le collapse se produit donc pendant la construction des inliers/matches avant le guard final.

- 2026-04-30 (A2-v23): instrumentation ciblée confirme que le collapse colinéaire est déjà acquis avant validation WCS (`pre_validate_inlier_mask`) et que la source de ce set est `reassign_eval` (12 inliers, cond ~5e14). Conclusion durable: l’anomalie principale est dans la logique de réassignation d’inliers, pas dans `_build_matches_array` ni le geometric guard final.

- 2026-04-30 (A2-v24): instrumentation `reassign_eval` ajoutée. Le set final colinéaire (12 points) est confirmé en provenance logique `reassign_eval`, mais la branche locale principale mesurée peut afficher `in_re0=0`, suggérant que la source de ce set passe parfois par un autre path de réassignation. Le prochain verrou est donc la désambiguïsation stricte des chemins qui écrivent `reassign_src_idx/reassign_dst_idx`.

- 2026-04-30 (A2-v25): désambiguïsation des chemins de réassignation terminée. Le set colinéaire final sur `d50_0819` est confirmé comme issu de `reassign_eval_main` (et non d’un path `resolve_hit`). Cette localisation ferme l’ambiguïté de provenance et cible directement la logique interne de sélection de `reassign_eval_main` pour la suite.

- 2026-04-30 (A2-v26): ajout d’un garde-fou de dispersion avant acceptation `reassign_eval_main`. Sur le cas `d50_0819`, le set reassign dégénéré est bien refusé (`reassign_eval_accept=false`), mais l’échec final persiste via `inlier_mask` déjà colinéaire. Implication durable: la cause racine est en amont de `reassign_eval`, dans la construction du masque d’inliers initial.

- 2026-04-30 (A2-v27): instrumentation de la genèse `inliers_mask` ajoutée (tolérance + quantiles d’erreur + footprint). Constat durable: le set inliers survivant reste ultra-colinéaire (cov_area ~5.3e-6) même quand `reassign_eval` est refusé. La correction doit donc cibler le flux `inlier_mask`/`img_in` lui-même via un garde-fou couverture et fallback candidat.

- 2026-04-30 (A2-v28): ajout d’un guard coverage précoce sur `inlier_mask`/`img_in`. Effet validé: le faux quasi-candidat colinéaire est stoppé avant validation (plus de rejet géométrique final), mais sans alternative immédiate (`nummatches` retombe à 0). Le prochain levier est un fallback explicite vers d’autres candidats après ce rejet coverage.

- 2026-04-30 (A2-v29): fallback policy ajustée pour coverage reject (continuer l’exploration sans consommer le budget d’abort global low-support). Résultat: pas d’amélioration sur le cas cible (`hypothesis failed before validation`, `nummatches=0`). Implication: il faut maintenant injecter de la diversité d’hypothèses locale après reject plutôt que simplement passer au candidat suivant.

- 2026-04-30 (A2-v30): implémentation d’un retry local diversifié après coverage reject (points extrêmes + refit robuste). Effet observé: le pipeline sort du blocage `nummatches=0` et repasse à `nummatches=1` avec rescue actif, mais échoue ensuite en validation RMS massive (~1102 px). Le verrou suivant est de filtrer les rescues non plausibles (RMS/échelle) avant verify.

- 2026-04-30 (A2-v31): ajout d’un garde-fou de plausibilité RMS post-rescue (après A2-v30). Les rescues locaux aberrants sont désormais rejetés (RMS observé ~1116 px), ce qui assainit le flux mais ramène le cas à `nummatches=0`. La suite doit créer des hypothèses alternatives plausibles, pas seulement filtrer les mauvaises.

- 2026-04-30 (A2-v32): diversification des candidats implémentée via `candidate_limit_runtime` adaptatif après rejects coverage. Sur le probe cible, l’échec dominant se rapproche d’un cas solvable (validation borderline RMS 2.45 px avec inliers 4/4) plutôt que des collapses colinéaires extrêmes, ce qui ouvre une piste de rescue fine sur seuils borderline sous garde-fous stricts.

- 2026-04-30 (A2-v33): ajout d’un pass strict de rescue RMS borderline avant acceptation (inliers + géométrie). Sur le cas cible, le mécanisme ne s’est pas activé et le meilleur rejet reste un borderline RMS avec échelle modèle non plausible (~11.62"/px vs attendu ~2.39"/px). Orientation suivante: filtrage de plausibilité d’échelle plus en amont.


- 2026-05-04 (A2-v34): ajout d’un garde-fou de plausibilité d’échelle **avant validation** pour hypothèses low-pairs (`<=10`) dans `zeblindsolver.py`, avec ratio modèle/ancre borné par défaut à `[0.35, 3.20]`. Instrumentation runtime ajoutée (`a2v34_scale_plausibility_*`). Gates techniques OK (`py_compile`, `pytest tests/test_backend_select.py tests/test_failures.py`). Probe runtime ciblé `RAW_Zenear_reference_001_...233459.fit` exécuté (`reports/zeblind_a2v34_probe_20260504.json`): échec solve maintenu (`success=0/1`), mais patch intégré sans régression de compilation/tests.

- 2026-05-04 (A2-v35): itération post A2-v34 orientée exploration plutôt que rejet sec. Ajout d’un mode par défaut où les rejects `a2v34 scale plausibility` ne consomment plus automatiquement `failed_validations` (`blind_a2v35_scale_reject_counts_as_validation=false`) + boost de `candidate_limit_runtime` piloté par le volume de rejects scale (`blind_a2v35_scale_reject_candidate_boost_*`). Gates techniques OK (py_compile + 2 tests ciblés). Probe runtime comparatif (`zeblind_a2v35_probe_20260504.json`) sur le cas RAW ZeNear ref: solve-rate inchangé (0/1), candidats inchangés (40), temps légèrement plus lent; direction suivante = améliorer la génération d’hypothèses plausibles amont.

- 2026-05-04 (A2-v36): ajout d’un mécanisme amont `guided inlier expansion` pour hypothèses à échelle plausible avant échec `empty/too_few inliers` (tolérance guidée + garde coverage). Nouveaux flags `blind_a2v36_guided_inlier_expand_*` et métriques `a2v36_guided_expand_*`. Gates techniques OK (`py_compile`, tests ciblés 2/2). Probes runtime (`zeblind_a2v36_probe_20260504.json`, `zeblind_a2v36_probe2_20260504.json`) sans déblocage solve-rate (0/1 puis 0/2), direction suivante maintenue: contraindre la fabrication d’hypothèse par l’échelle attendue.

- 2026-05-04 (A2-v37): ajout d’un `scale-guided seed refit` pour hypothèses low-inliers à ratio d’échelle plausible (`blind_a2v37_seed_refit_*`) avec instrumentation dédiée (`a2v37_seed_refit_*`). Gates techniques OK (`py_compile`, tests ciblés 2/2). Probe runtime `reports/zeblind_a2v37_probe_20260504.json`: solve-rate inchangé (0/1) mais baisse du temps de tentative sur le cas cible (~57.72s vs ~67.29s sur A2-v36 à protocole identique), indiquant un gain de coût local sans déblocage solve.

- 2026-05-04 (pivot méthodologique appliqué): comparaison ZeBlind vs Astrometry online exécutée sur `RAW_Zenear_reference_001_...233459.fit` avec clé API locale (`~/.zesolver_settings.json`). Résultat mesuré: Astrometry `OK` ~20.44s vs ZeBlind `ERR` ~182.26s (`reports/zeblind_vs_astrometry_online_20260504_single_233459.json`). Trace différentielle locale générée (`reports/zeblind_vs_astrometry_diff_20260504_233459.md`) confirmant un blocage amont `hypothesis/empty_inliers`. Mission files mis à jour (`AGENT.md`, `followup.md`) avec découpage complet des étapes manquantes jusqu’à la parité stricte.

- 2026-05-04 (A2-v38, cohérence logique Astrometry): activation par défaut dans `SolveConfig` des chemins `blind_astrometry_hit_pipeline_first_enabled` et `blind_astrometry_hit_pipeline_dedicated_enabled` pour privilégier la chaîne Astrometry-like avant fallback legacy. Gates OK (`py_compile`, tests ciblés 2/2). Probe `reports/zeblind_a2v38_probe_20260504.json`: solve-rate inchangé (0/1), mais activations pipelines dédiés confirmées (381 sur run direct), donc alignement structurel effectif.

- 2026-05-04 (A2-v39): pour coller à la logique Astrometry et éviter la boucle fine-tuning, désactivation par défaut de trois heuristiques A2 ad-hoc (`a2v34`, `a2v36`, `a2v37`) tout en gardant les pipelines Astrometry-like activés par défaut. Gates OK (`py_compile`, tests 2/2). Probe `zeblind_a2v39_probe_20260504.json`: solve-rate inchangé (0/1), perf proche de A2-v38. Export stage-by-stage oracle ajouté (`zeblind_stage_trace_oracle_233459_20260504.json`) pour M0.2.

- 2026-05-04 (M0.3/M0.4): miroir Astrometry détaillé généré pour le cas oracle (`astrometry_mirror_trace_233459_20260504.json`) avec timeline soumission/job + endpoints calibration/info. Rapport machine-readable de divergence ajouté (`zeblind_astrometry_divergence_233459_20260504.json`) confirmant la divergence la plus précoce côté ZeBlind: collapse `hypothesis/empty_inliers` avant verify. Checklist M0 clôturée dans `followup.md`.

- 2026-05-04 (M1.1a): patch causal unique appliqué pour garder le `code_rangesearch` actif par défaut (`blind_astrometry_code_rangesearch_adapt_disable_enabled=false`). Effet mesuré sur oracle 233459: disparition du cut auto (`auto_disabled 24->0`) et forte remontée du signal amont (`rangesearch_hits 0->1760`, `code_hit_objects_total 0->10312`), avec légère baisse de `empty_inliers` en `scale_only` (21->20). Solve-rate toujours 0/1 à ce stade, coût bench en hausse; prochaine cible = transformer ce signal en validations utiles.

- 2026-05-04 (M1.1b): tentative bootstrap post-resolve renforcée (pass relaxé + refit sparse) avec instrumentation de distribution des paires. Outcome oracle: pas de solve, mais diagnostic causal plus net: chemin `post_resolve` plafonne à `fail_pairs_max=2`, empêchant d’atteindre le seuil inliers pour verify; `refit_inliers_max=0` sur ce chemin. Décision: déplacer l’effort vers M1.2 (fabrication des correspondances `resolve_matches`) plutôt que continuer à ajuster verify/seuils.
- 2026-05-04 (M1.2 retry gate): activation systématique du retry relaxé post-resolve testée; résultat oracle inchangé (0 hit relaxé, `fail_pairs_max=2`). Apprentissage: le verrou n’est pas la condition `upstream_pair_rescue_hits`; la prochaine cible est la qualité/justesse de l’hypothèse (`rot_scale/translation`) avant `resolve_matches`.
- 2026-05-04 (M1.4): ajout d’un bootstrap de refit indexé (pair-order) quand l’hypothèse tombe sous 4 inliers. Impact reproductible sur oracle: baisse `empty_inliers` (20->16), apparition de `post_resolve_hits` (3) et sortie du mode "hypothesis-only" vers des échecs en validation (RMS élevé ~18 px). Interprétation: progression causale réelle, prochain verrou = qualité géométrique des hypothèses refitées.
- 2026-05-04 (M1.5): trois essais orientés baisse RMS (refit post-resolve, refit pré-WCS, assouplissement local retry low-pairs) n’ont pas montré de gain stable sur oracle; rollback effectué pour garder une base propre M1.4. État conservé: entrée verify partielle (post_resolve_hits=3) mais échec dominant en validation RMS (~18.486px pour 4 inliers).
- 2026-05-04 (strict astrometry verify path): ajout d’un mode strict qui court-circuite des garde-fous maison sur le chemin resolve-hit (coverage reject a2v28, match_object_expand, conflict_resolution). Effet: `strict_verify_path_hits=3` et disparition des rejets coverage sur ce chemin, mais échec final inchangé en validation RMS (~18.486 pour 4 inliers). Conclusion durable: l’écart principal restant est la qualité des correspondances low-pairs en entrée `validate_solution`, pas les filtres de vérification ajoutés autour.

- 2026-05-04 (Astrometry exact trace): ajout d’un dump décisionnel exact `zeblind.astrometry_exact_trace.v1` (combos/perm/metrics brut+final/décision), plus gate strict d’échelle quad en mode astrometry strict.
- Probe 233459: rejet massif en amont côté Astrometry (`astrometry_quad_scale_gate_rejects` très élevé), avec reliquat de `skip_zero_inliers`; conclusion durable = dérive d’échelle principalement générée dès la construction des hypotheses quads, avant validate.
- 2026-05-04 (collect trace approfondie): ajout d’un dump exact `_collect_tile_matches` avec provenance des votes (`support_sample` incluant `obs_hash`/`obs_quad_idx`/bucket/indices étoiles). Constat durable: plusieurs `obs_hash` saturés produisent des collisions massives (dizaines de milliers de buckets, >1k tiles), corrélées à la dérive d’échelle des pairsets.
- 2026-05-04 (expérimentation contrôlée): mode opt-in `blind_collect_skip_saturated_hashes_enabled` testé; activé, il coupe quasi toute collecte (`pairing failed` + `no_votes`), ce qui confirme le poids causal des hashes saturés. Gardé désactivé par défaut pour préserver baseline.

- 2026-05-04 (strict probe 233459): added astrometry exact trace enrichment (`obs_quad_idx`, `obs_hash`, `obs_q`) + diagnostic-only quad-scale guard mode. Evidence shows frequent saturated observed hash components (notably q2=65535) on flagged hypotheses and transform precheck model scales far outside expected [1.79,2.99] (roughly 176..4359 arcsec), confirming scale-coherence failure as primary blocker.

- 2026-05-05 (Astrometry-conform strict pass): strict mode now disables non-astrometry hypothesis fallbacks by default (`blind_astrometry_strict_disable_nonastrometry_fallbacks=true`), uses global star pools for reassignment verify in astrometry path, and restores hard quad-scale guard (`blind_astrometry_quad_scale_guard_enabled=true`). On probe 233459 this removed many off-scale hypotheses early and reduced zero-inlier skips, but final blocker remains strict scale prefilter failures.

- 2026-05-05 (audit Astrometry 2e passe): revue approfondie reconduite sur `astrometry-main` avec re-cartographie des appels (`solver/verify/tweak/onefield/engine`). Conclusion validée: pas de manque structurel nouveau sur le cœur `solver_run -> verify_hit -> solver_tweak2/reverify`, mais couverture étendue obligatoire sur les chemins connexes (verify-only/fake_match/predistort+pixel_xscale/entrypoints wrappers).
- 2026-05-05 (mission checklist): `followup.md` enrichi avec une section dédiée `R8.6` pour verrouiller ces coins connexes critiques + item wrappers (`control-program.c`, `solver.i`) afin d’éviter tout angle mort de parité.
- 2026-05-06 (R8.1 bloc 1): implémentation du noyau verify séquentiel Astrometry-like dans `zeblindsolver.py` via `_astrometry_verify_sequence_logodds` + états `THETA_*` + logique conflit/remplacement + cumul log-odds avec seuils bail/stop-looking.
- 2026-05-06 (R8.1 bloc 1): branchement activé sur le chemin strict verify (`blind_astrometry_verify_sequential_enabled`, défaut actif), avec fallback conservé vers l’ancien scoring probabiliste hors strict.
- 2026-05-06 (preuve immédiate): `python3 -m py_compile zeblindsolver/zeblindsolver.py` OK + smoke test synthétique documenté (`reports/r8_1_block1_20260506_0022.md`).
- 2026-05-06 (R8.1 bloc 2): propagation des états `THETA_*` dans les traces ZeBlind via compteurs `prob_theta_*_total` et câblage vers `verify_hit_trace`, `astrometry_match_objects` et `verify_trace` (`theta_counts` transmis à `_update_verify_logodds`).
- 2026-05-06 (R8.1 bloc 2, preuve): compile OK + smoke test synthétique montrant comptage explicite des états (`match/conflict/distractor/bailedout/stoppedlooking`). Rapport: `reports/r8_1_block2_20260506_0026.md`.
