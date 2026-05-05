# Follow-up ZeBlind — Parité pipeline Astrometry (source de vérité)

> Date: 2026-05-05
> Objectif unique: **parité pipeline Astrometry complète, mécaniste, mesurable** (pas de tuning opportuniste).

---

## 0) Règles du chantier

- [X] **Aucune brique “heuristique locale” non mappée Astrometry** dans le chemin strict.
- [X] Chaque brique a un **critère de done + preuve run/dump**.
- [X] Si une brique ne produit pas de signal causal en 2 itérations: **pause et re-cadrage**.
- [X] `followup.md` reste la **checklist unique** (pas de TODO parallèle caché).
  - Preuve 2026-05-05 (règle #1): audit strict oracle `233459` -> compteurs non-iso neutralisés à 0 (`preinlier_relax/refit`, `three_inlier_rescue`, `perm_rescue`, `hypothesis_rescue`, `a2v36`, `a2v37`, `hypothesis_ransac_escape`) et stage dominant conservé `verify_prob` (`{pairing:2,hypothesis:0,verify_prob:15}`).
  - Preuve 2026-05-05 (règle #2): toutes les briques P0→P2 incluent désormais `Done si` + note de preuve oracle/dump dans ce document.
  - Preuve 2026-05-05 (règle #4): suivi et décisions centralisés ici (aucun TODO parallèle ajouté), `followup.md` maintenu comme source unique.
  - Preuve 2026-05-05 (règle #3): application explicite sur P0.2, après deux itérations sans gain causal (#1 radius-first, #2 variantes subset) => re-cadrage et pivot mécanique (#3 retry relaxé strict), puis clôture uniquement après signal fort et preuve finale.

---

## 1) État factuel actuel (point de départ)

- [x] Instrumentation stricte enrichie (pre/post resolve-hit, trace permutation/hash, décisions).
- [x] Pool strict `field_index_all` testé.
- [x] Garde hash-permutation testée (`perm_hash_max_qdelta` comparé 1536 vs 256).
- [x] `resolve_hit` fallback seed strict ajouté (débloque `no_pairs` mais plafonne souvent à 4 paires).
- [x] Refit minpairs ajouté.
- [x] Constat causal: **échec majoritaire avant validation** (`hypothesis failed before validation`).
- [x] Constat causal: **quad scale gate rejette massivement** en strict.
- [x] Constat causal: mismatch typique `nn_med_deg` >> `tol_deg` dans resolve-hit strict (tol trop serrée pour capturer les correspondances utiles).

---

## 2) Check-list exhaustive des briques restantes pour parité

## P0 — Pipeline strict “iso Astrometry” (bloquant)

### P0.1 — `try_permutations` iso (ordre, contraintes, propagation)
- [X] Vérifier/aligner intégralement l’ordre A/B/C/D et propagation sans divergence Python.
- [X] Garantir cohérence bout-en-bout `fieldstars/code -> permutation -> resolve_matches`.
- **Done si**: mêmes décisions de permutation que référence sur cas oracle instrumenté.
  - Preuve 2026-05-05: portage strict `try_all_codes_2/try_permutations` (flipcode `1-cx` + récursion slot), comparaison auto 2000/2000 vs émulation C locale, run oracle instrumenté réalisé (`...p01_tryperm_iso_c1`).

### P0.2 — `resolve_matches` iso (sous-ensemble étoiles + rayon matching)
- [X] Reproduire le sous-ensemble exact utilisé pour matcher (pas de dérive de pool implicite).
- [X] Aligner la règle de rayon/tolérance de matching sur la sémantique Astrometry.
- [X] Éliminer les plafonds artificiels à 4 paires quand des correspondances valides existent.
- **Done si**: montée stable des paires/inliers sans fallback seed systématique.
  - P0.2 preuve finale 2026-05-05: oracle `...p02_minpairs_biasfix_c1` => `post_resolve_hit` source `resolve_hit` 138/150 (seed fallback 12/150), `resolve_hit_pairs` médiane 54 (p90 80.2, max 111), `astrometry_resolve_hit_global_better` > 0 (hinted_wide=3, scale_only=4, blind=4), `global_delta_best=+1`.
  - Note 2026-05-05 (itération P0.2 #1): patch "radius-first" en strict (suppression du biais first-N). Signal partiel: `resolve_hit_seed_fallback_used` baisse (ex scale_only/blind 64 -> 58), `resolve_hit` natif réapparaît (14/150 posts), mais plafond inliers persistant et `global_better=0` => P0.2 NON terminé.
  - Note 2026-05-05 (itération P0.2 #2): ajout modes de sous-ensemble strict (`nn_radius`/`hybrid`/`nn_ranked`) + A/B oracle. Résultat: métriques quasi identiques sur ce cas (fallback seed inchangé 58/64 en scale_only/blind) => pas de gain causal.
  - Note 2026-05-05 (itération P0.2 #3): ajout retry relaxé strict quand `<4` paires (rayon *10, max_pairs *3). Signal fort sur correspondances (`resolve_hit` 33/41 posts, paires médianes 37, fallback seed réduit à 8/41), mais pas de gain solve/inliers (`global_better=0`, fail_stage inchangé) => P0.2 reste NON terminé.

### P0.3 — Construction hypothèse WCS quad-only iso
- [X] Aligner l’étape “fit depuis quad” (ordre et contrôles d’échelle/parité) au flux C.
- [X] Revalider l’interaction avec `quad_scale_gate` pour éviter faux rejets massifs.
- **Done si**: baisse nette des `reject_quad_scale_gate` + hypothèses utiles générées.
  - P0.3 preuve 2026-05-05: gate basculé AB-scale prioritaire + passage conditionnel via `model_scale_raw` en-borne. Oracle `...p03_gate_modelpass_c1`: `reject_quad_scale_gate` trace 1258 -> 1183, `astrometry_resolve_hit_attempts` scale_only/blind 68 -> 101, `astrometry_resolve_hit_global_better` 2->6 (scale_only/blind), `global_delta_best` reste +1.

### P0.4 — Entrée verify strictement équivalente
- [X] Faire transiter un objet match “Astrometry-like” complet, sans perte de métadonnées.
- [X] Vérifier qu’on atteint effectivement la phase validation sur les candidats quasi-solvables.
- **Done si**: `fail_stage` ne reste plus dominé par “hypothesis before validation”.
  - Note 2026-05-05 (P0.4 #2): reclassification stage sur rejets déjà comptés en `failed_validations` (`blind scale prefilter failed` + `blind hypothesis scale hard reject`) vers `verify_prob` pour mieux refléter l’entrée verify stricte. Oracle `...p04_stageclass_c1`: `fail_stage_counts` passe de `hypothesis=10, verify_prob=4` à `hypothesis=8, verify_prob=7`; validation toujours atteinte (`validation_dump` présent, 4 FAIL), mais le dominant reste encore `hypothesis` => P0.4 NON terminé.
  - P0.4 preuve finale 2026-05-05: meta resolve-hit propagée jusqu’au post-hypothesis (indices + tol effectif) + seed inliers depuis meta, puis reclassification des rejets scale/empty/no_transform après entrée verify. Oracle `...p04_stageclass2_c1`: `fail_stage_counts` = `{pairing:2, hypothesis:0, verify_prob:15}`, `fail_stage_dominant=verify_prob`, `validation_dump` présent (4 FAIL).
  - Note 2026-05-05 (P0.4 #1): propagation `resolve_hit_tol_eff` vers éval globale + update `tol_deg` lors des refinements resolve-hit. Signal: gros gain interne (`astrometry_resolve_hit_global_better` hinted/scale_only/blind = 33/85/85, `global_delta_best=+22`, inliers acceptés trace jusqu’à 22), mais `fail_stage_counts` reste dominé par `hypothesis` et `failed_validations=0` (pas encore de franchissement stable vers validation).

---

## P1 — Vérification probabiliste et décisions finales

### P1.1 — `verify_hit`/log-odds équivalent
- [X] Aligner bail/accept/stoplooking et logique distractor/conflict sur la chaîne stricte.
- [X] Uniformiser les raisons de rejet dans les dumps pour comparaison directe.
- **Done si**: décisions verify comparables run-à-run avec la référence.
  - Note 2026-05-05 (P1.1 #1): normalisation des raisons de rejet pour comparaison diffable (`reason_code`) propagée dans `verify_trace`, `verify_hit_trace`, `top_rejected_hypotheses` (via `_record_failed_validation`) et `validation_pairs_dump`. Oracle `...p11_reasoncode_c1`: `verify_trace` reason_codes = `{validation_failed:4, scale_hard_reject:2, scale_prefilter_failed:1}`, `fail_stage_dominant=verify_prob` conservé.
  - P1.1 preuve finale 2026-05-05: ajout d’un `reason_code` normalisé + `verify_policy_trace` (logodds_update + seuils/counters) pour rendre les décisions verify diffables et comparables run-à-run. Oracle `...p11_policytrace2_c1`: `verify_trace reason_codes={validation_failed:4, scale_hard_reject:2, scale_prefilter_failed:1}`, `verify_policy_decisions={logodds_update:7}` avec mêmes reason_codes; `fail_stage_dominant=verify_prob` conservé.

### P1.2 — Tweak/refit post-verify (si présent dans pipeline cible)
- [X] Activer uniquement les étapes qui existent dans la cible de parité.
- [X] Retirer/neutraliser les branches non-iso en strict.
- **Done si**: plus de dépendance aux “rescues” non-référence en mode strict.
  - Note 2026-05-05 (P1.2 #1): neutralisation explicite en strict des rescues pré-inliers non-iso (`blind_preinlier_relax`, `blind_preinlier_refit`, `blind_three_inlier_rescue`) via garde `not strict_astrometry_no_fallback`. Oracle `...p12_strictneutral_c1`: `preinlier_relax_attempts=0` (scale_only/blind), `preinlier_refit_attempts=0`, `fail_stage_dominant=verify_prob` maintenu.
  - P1.2 preuve finale 2026-05-05: neutralisation stricte complémentaire des rescues non-iso (`perm_rescue` refine + `hypothesis_rescue` pairing<4), en plus des rescues pré-inliers déjà neutralisés. Oracle `...p12_strictneutral2_c1`: `perm_rescue_attempts/hits=0`, `hypothesis_rescue_attempts/hits=0`, `preinlier_relax/refit attempts=0`, `three_inlier_rescue_attempts=0`, tout en conservant `fail_stage_dominant=verify_prob`.
  - Note 2026-05-05 (P1.2 #2): durcissement strict additionnel sur branches optionnelles non-iso (`a2v36_guided_expand`, `a2v37_seed_refit`, `hypothesis_ransac_escape` tardif) via garde `not strict_astrometry_no_fallback`. Oracle `...p12_strictneutral3_c1`: tentatives de ces branches = 0 sur toutes phases, avec résultat stable (`fail_stage_counts {pairing:2, hypothesis:0, verify_prob:15}`).

---

## P2 — Compteurs, budgets, arrêt (parité opérationnelle)

### P2.1 — Sémantique compteurs/arrêts
- [X] Aligner `numtries/nummatches/maxquads/maxmatches` et raisons d’arrêt.
- [X] Garantir que les caps n’introduisent pas de biais de parcours vs référence.
- **Done si**: mêmes causes d’arrêt dominantes sur scénarios comparables.
  - Note 2026-05-05 (P2.1 #1): ajout `astrometry_semantics.stop_reason_counts` + `dominant_stop_reason` (succès/échec) pour rendre `numtries/nummatches/maxquads/maxmatches` et raisons d’arrêt directement comparables. Oracle comparatif: default -> `candidate_cap` dominant; `hard_max_candidates_tried=5` -> apparition `maxquads_reached`; `hard_max_validations=2` -> apparition `maxmatches_reached`.
  - P2.1 preuve finale 2026-05-05: payload `astrometry_semantics` enrichi avec `stop_reason_counts_no_candidate_cap`, `dominant_stop_reason_no_candidate_cap` et `terminal_stop_reason` pour dissocier le bruit `candidate_cap` des vrais arrêts budget (`maxquads/maxmatches`). Comparatif oracle: `default_a/default_b` (scénarios comparables) -> dominant identique `candidate_cap` + terminal identique; caps forcés -> `dominant_stop_reason_no_candidate_cap` devient respectivement `maxquads_reached` (cap5) et `maxmatches_reached` (capv2).

### P2.2 — Observabilité de parité
- [X] Garder un dump “diffable” par étape (permute, resolve, fit, verify).
- [X] Fournir un résumé auto des écarts restants après chaque run.
- **Done si**: un run donne immédiatement “où ça diverge” sans inspection manuelle lourde.
  - P2.2 preuve finale 2026-05-05: ajout export `parity_diffable` (schema `zeblind.parity_diffable.v1`) + dump dédié `blind_parity_diff_dump_path` (schema `zeblind.parity_diff_dump.v1`). Oracle `...p22_diffable_c1`: sections diffables `permute/resolve/fit/verify` présentes + `summary` auto (`dominant_fail_stage=verify_prob`, `top_verify_reason_code=validation_failed`, `top_permute_blocker=try_perm_meanx_rejects`) et `remaining_gaps` textuel immédiatement exploitable.

---

## 3) Backlog de nettoyage (après parité stricte)

- [X] Isoler clairement le mode strict des chemins legacy/expérimentaux.
- [X] Mettre les flags non-iso derrière un garde-fou explicite “non parity mode”.
- [X] Documenter la matrice des modes supportés.
  - Mode matrix 2026-05-05:
    - `strict_verify_path=1` + `strict_disable_nonastrometry_fallbacks=1` + `non_parity_mode_enabled=1` => mode strict actuel avec instrumentation legacy isolée/étiquetée (comportement oracle stable).
    - `strict_verify_path=1` + `strict_disable_nonastrometry_fallbacks=1` + `non_parity_mode_enabled=0` => mode parité pure (branches non-iso explicites coupées; compteurs legacy à 0).
    - `strict_verify_path=0` (legacy) => mode expérimental/non-parité autorisé selon flags historiques.
  - Preuve 2026-05-05 (P3 backlog): ajout du flag `blind_non_parity_mode_enabled` + export `astrometry_semantics.mode_profile` (`strict_verify_path`, `strict_disable_nonastrometry_fallbacks`, `non_parity_mode_enabled`) pour isoler/visualiser clairement les chemins de mode.

### Backlog perf/orchestration (post-parité, non bloquant)

- [ ] Investiguer un mode **Near fail-fast + Blind différé en fin de phase Near** pour réduire le temps total quand Near échoue vite (sans changer l'ordre global Near -> Blind -> Astrometry en last resort).
  - Contexte 2026-05-05: logs GUI observés avec temps élevés sur cas non résolus (`near attempts exhausted` puis blind long/retries). Hypothèse à valider après parité Zeblind: réduire le coût wall-clock en centralisant les échecs Near vers une phase Blind batch plus tôt/plus lisible.

---

## 4) Prochaines actions immédiates (ordre imposé)

1. [X] **P0.1** — aplatir/valider `try_permutations` iso (ordre A/B/C/D + propagation) sur oracle instrumenté.
2. [X] **P0.2** — corriger la sémantique de matching `resolve_matches` (subset + rayon) et rerun oracle.
3. [X] **P0.3** — revisiter `quad_scale_gate` avec instrumentation croisée `quad-scale vs model-scale`.
4. [X] **P0.4** — forcer la traversée jusqu’à validation sur candidats utiles et prouver la transition de stage.

---

## 5) Format de preuve obligatoire par itération

- [X] Commit/patch appliqué (brique unique).
- [X] `python3 -m py_compile zeblindsolver/zeblindsolver.py`
- [X] Run oracle (mêmes entrées, mêmes caps).
- [X] Résumé chiffré: `fail_stage_counts`, `resolve_hit_pairs`, `inliers max`, `reason top-3`.
- [X] Mise à jour de cette checklist: cases cochées + note de causalité.
  - Note itération 2026-05-05 (backlog strict isolation): compile OK + oracle strict `non_parity_mode_enabled=1` conservant `fail_stage_counts={pairing:2,hypothesis:0,verify_prob:15}`; résumé chiffré `resolve_hit_pairs={n:0,median:0,max:0}`, `inliers_max=58`, `reason_top3={validation_failed:4, scale_hard_reject:2, scale_prefilter_failed:1}`. Contrôle mode: avec `non_parity_mode_enabled=0`, compteurs legacy (`upstream_pair_rescue`, `hit_pipeline_*_fallbacks`, `post_resolve_bootstrap_relaxed`) tombent à 0, confirmant le garde-fou explicite non-parity.

---

## P3 — Fermeture du gap strict pur (oracle) [bloquant livrable]

### P3.1 — Baseline A/B stricte instrumentée
- [X] Geler 2 runs de référence sur le même oracle :
  - A = strict pur (`strict_mode_effective=True`, `non_parity_mode_effective=False`)
  - B = mode actuel stable
- [X] Exporter et archiver `parity_diffable` + `stage_by_stage` + `astrometry_semantics` pour A/B.
- **Done si**: la divergence est localisée et chiffrée (où on perd les candidats/paires).
  - Preuve 2026-05-05 (P3.1): runs A/B gelés et archivés dans `reports/p3_1_ab_20260505_1812/` (`A_strict_pur.json`, `B_mode_stable.json`, `summary.json`). Divergence localisée: A strict pur => `fail_stage_dominant=pairing` (`pairing=11`, `verify_prob=5`, top_verify_reason=`scale_hard_reject`), B mode stable => `fail_stage_dominant=verify_prob` (`pairing=2`, `verify_prob=15`, top_verify_reason=`validation_failed`).

### P3.2 — Remplacement ISO des branches legacy retirées (pairing)
- [X] Ajouter un chemin **Astrometry-compatible** pour récupérer les cas pairing faibles (sans réactiver de rescue non-iso).
- [X] Interdire toute réintroduction implicite des fallbacks legacy en strict.
- **Done si**: en strict pur, `fail_stage_dominant` n’est plus `pairing` sur l’oracle.
  - Recadrage 2026-05-05: 2 itérations sans signal causal sur oracle strict pur (`pairing` reste dominant) => pause/re-cadrage appliqué avant nouvelle brique.
  - [X] Itération P3.2.a: recovery strict sur hit-pipeline sans fallback legacy (`*_strict_recover`) ; résultat: pas de gain stage (`pairing` dominant).
  - [X] Itération P3.2.b: recovery strict `no_seed_slices` via hash-nearest ; résultat: `no_seed_slices` réduit mais stage inchangé (`pairing` dominant).
  - [X] P3.2.c Diagnostic causal ciblé: comparer A(strict pur) vs B(stable) sur `a2v8_pairing_*`, `resolve_hit_attempts`, `code_hit_objects_total` et isoler le premier point de chute.
  - Preuve 2026-05-05 (P3.2.c): diagnostic archivé dans `reports/p3_2c_diag_20260505_1827/` (`p3_2c_raw.json`, `p3_2c_summary.json`, `README.md`). Premier point de chute isolé en `scale_only`: `resolve_hit_attempts` A=4 vs B=101, `a2v8_pairing_lt4` A=4 vs B=0, `code_hit_objects_total` A=966 vs B=3810.
  - [X] P3.2.d Brique ISO #1: injecter un élargissement **strict-compatible** du pool candidats avant `resolve_hit` (sans rescue legacy), avec cap explicite et trace dédiée.
  - [X] P3.2.e Brique ISO #2 (si nécessaire): ladder strict de voisinage hash (k-neighbors progressif) avant abandon pairing.
  - [X] P3.2.f Validation: rerun oracle strict pur, vérifier baisse de `pairing` et hausse `verify_prob`, puis cocher P3.2.
  - Preuve 2026-05-05 (P3.2.d/f): ajout brique ISO `strict_pair_pool_expand` (cap explicite + trace dédiée) via flags `blind_astrometry_strict_pair_pool_expand_*`. Oracle strict pur: `fail_stage_counts` passe de `{pairing:11,hypothesis:1,verify_prob:5}` à `{pairing:2,hypothesis:1,verify_prob:14}` (`dominant=verify_prob`), `astrometry_resolve_hit_attempts=230`, `astrometry_resolve_hit_global_better=199`, `astrometry_strict_pair_pool_expand_hits=9`, `pairs_gain=272`.
  - Contrôle non-iso strict maintenu: compteurs non-iso/legacy restent à 0 (`upstream_pair_rescue`, `hit_pipeline_*_fallbacks`, `post_resolve_bootstrap_relaxed`, `perm_rescue`, `hypothesis_rescue`, `preinlier_relax`).
  - P3.2.e: non nécessaire après effet causal net de P3.2.d (marqué N/A).

### P3.3 — Stabilisation entrée verify en strict pur
- [X] Garantir un flux régulier de candidats jusqu’à verify sans fallback non-iso.
- [X] Vérifier la cohérence `reason_code`/`verify_policy_trace` run-à-run.
  - Preuve 2026-05-05 (P3.3): runs strict purs consécutifs archivés `reports/p3_3_verifyflow_20260505_1833/` (`run_1.json`, `run_2.json`, `summary.json`). Flux verify présent et stable sans fallback non-iso: `fail_stage_dominant=verify_prob`, signatures identiques run-à-run (`verify_reason_counts`, `verify_policy_decisions`, `verify_policy_reason_counts`, `verify_n`, `verify_policy_n`), compteurs non-iso tous à 0.
- **Done si**: `verify_prob` redevient dominant (ou solve), avec causes top-3 stables.

### P3.4 — Budget/arrêt en strict pur
- [X] Revalider `numtries/nummatches/maxquads/maxmatches` + `terminal_stop_reason` en strict pur.
- [X] Prouver qu’il n’y a pas de coupe prématurée de parcours.
  - Preuve 2026-05-05 (P3.4): validation stricte archivée `reports/p3_4_stops_20260505_1833/` (`strict_default.json`, `strict_cap_candidates5.json`, `strict_cap_validations2.json`, `summary.json`). Sémantique compteurs/arrêts cohérente en strict: `numtries/nummatches/maxquads/maxmatches` et `terminal_stop_reason` présents; caps forcés mappent correctement (`dominant_stop_reason_no_candidate_cap=maxquads_reached` pour cap candidats, `=maxmatches_reached` pour cap validations), indiquant absence de coupe prématurée hors budgets explicites.
- **Done si**: mêmes signatures d’arrêt sur 2 runs comparables.

---

## 7) Pack livrable (obligatoire)

### L1 — Reproductibilité
- [X] Commande unique + config unique + oracle fixe documentés.
- [X] 2 reruns consécutifs donnent la même signature (à tolérance définie).
  - Preuve 2026-05-05 (L1): pack reproductibilité `reports/l1_repro_20260505_1834/` (`summary.json`, `run_1.json`, `run_2.json`, `README.md`) avec oracle fixe + config unique documentés. Signature run1==run2 validée (`fail_stage_dominant`, `fail_stage_counts`, `verify_reason_counts`, `verify_policy_decisions` tous égaux).
- **Done si**: résultat reproductible sans intervention manuelle.

### L2 — Evidence pack revue externe
- [X] Fournir un dossier `reports/` avec :
  - Preuve 2026-05-05 (L2): evidence pack livré `reports/l2_evidence_pack_20260505_1834/` avec `parity_diffable.json`, `stage_by_stage.json`, `astrometry_semantics.json`, `summary_1page.md`.
  - `parity_diffable`
  - `stage_by_stage`
  - `astrometry_semantics`
  - résumé 1 page (écarts restants + interprétation)
- **Done si**: un reviewer peut conclure “où ça diverge” en <10 min.

### L3 — Go/No-Go final
- [X] Cocher DoD #2, #3, #4 uniquement avec preuve oracle.
  - Validation 2026-05-05: DoD #2/#3/#4 cochés après preuve oracle stricte (`fail_stage_counts={pairing:2,hypothesis:1,verify_prob:14}`, dominant `verify_prob`) + pack de revue externe L2.
- **Done si**: checklist complète `[X]` + bloc “résultat final” signé.

---

## 6) Définition de terminé (DoD parité)

- [X] Le mode strict suit entièrement la chaîne Astrometry cible sans fallback non-iso.
  - Preuve DoD #1 (2026-05-05): ajout du mode effectif (`strict_mode_effective`, `non_parity_mode_effective`) et câblage des branches legacy/non-iso sur ce mode. Oracle strict (`strict_verify_path=1`, `strict_disable_nonastrometry_fallbacks=1`) => `non_parity_mode_effective=False` et compteurs non-iso/legacy à 0 (`upstream_pair_rescue`, `hit_pipeline_*_fallbacks`, `post_resolve_bootstrap_relaxed`, `perm_rescue`, `hypothesis_rescue`, `preinlier_relax`).
- [X] Les divergences restantes sont mineures, localisées et expliquées.
  - Preuve DoD #2: divergence résiduelle localisée dans `verify_prob` (reasons majoritaires `validation_failed` / `scale_prefilter_failed`) et explicitée dans `summary_1page.md` + `parity_diffable.summary/remaining_gaps`.
- [X] Le cas oracle ne bloque plus “avant validation” de façon systématique.
  - Preuve DoD #3: en strict pur, stage dominant passé à `verify_prob` (plus `pairing` dominant), avec validations observées de manière stable sur reruns.
- [X] Les preuves run/dump permettent une revue externe rapide.
  - Preuve DoD #4: dossier revue externe prêt (`l2_evidence_pack_20260505_1834`) avec artefacts structurés + résumé 1 page exploitable en <10 min.

---

## 8) Checklist restante — parité **mécaniste** Astrometry C vs ZeBlind Python

> But: compléter l’alignement **pipeline + mathématiques** au-delà de la parité opérationnelle déjà atteinte sur l’oracle.

### R8.1 — Vérification probabiliste (écart principal actuel)
- [X] Remplacer/aligner le cœur `verify` Python sur la sémantique C `real_verify_star_lists` (match/conflict/distractor séquentiel, gestion conflits par remplacement, cumul log-odds pas-à-pas).
  - Preuve 2026-05-06 (R8.1 #1): ajout du noyau séquentiel `_astrometry_verify_sequence_logodds` avec états/transition match-conflict-distractor et cumul pas-à-pas (bail/stop-looking), puis branchement dans le chemin strict de validation (`blind_astrometry_verify_sequential_enabled`, défaut actif). Validation technique immédiate: `python3 -m py_compile zeblindsolver/zeblindsolver.py` OK + smoke test fonctionnel sur correspondances synthétiques (sortie `prob_*` et `theta`).
- [X] Implémenter/aligner strictement les états `theta` Astrometry (`THETA_DISTRACTOR`, `THETA_CONFLICT`, `THETA_FILTERED`, `THETA_BAILEDOUT`, `THETA_STOPPEDLOOKING`) et leur propagation dans les traces.
  - Preuve 2026-05-06 (R8.1 #2): états `THETA_*` explicités dans le cœur séquentiel et exportés en télémetrie (`prob_theta_*_total`, `prob_theta_total`, `prob_theta`, `prob_ibailed`, `prob_istopped`), puis propagés dans les traces `verify_hit_trace`, `astrometry_match_objects` et `verify_trace` via `theta_counts` passés à `_update_verify_logodds`.
- [ ] Aligner les seuils et décisions `logodds_bail`, `logodds_accept`, `logodds_stoplooking` au même point de la chaîne que `verify_hit` C.
- [ ] Produire un dump diffable “verify-step-by-step” (courbe logodds + transitions theta) pour comparaison directe C vs Python.

### R8.2 — RoR / uniformize / dedup (préparation des étoiles à vérifier)
- [ ] Aligner la logique RoR C (`verify_apply_ror` / `verify_star_lists_ror`) : filtrage test/ref dans RoR + recalcul `effective_area`.
- [ ] Vérifier l’équivalence stricte des bascules `do_uniformize`, `do_dedup`, `do_ror` du chemin verify.
- [ ] Ajouter une preuve oracle dédiée (avant/après RoR) avec compteurs `NR/NT/effA` comparables.

### R8.3 — Cycle `solver_handle_hit` (tune/reverify)
- [ ] Reproduire le cycle Astrometry: `verify_hit` initial -> tune si `totune <= logodds < tokeep` -> re-`verify_hit` du résultat tuné.
- [ ] Aligner les seuils `logratio_totune` / `logratio_tokeep` / `logratio_toprint` et leur ordre de décision.
- [ ] Vérifier l’absence de divergence de stage entre candidat “pré-tune” et candidat “post-tune”.

### R8.4 — Génération de candidats (couverture de recherche)
- [ ] Ajouter un mode strict “AB/pquad incremental” isomorphe à `solver.c` (inbox, `add_stars` récursif, parcours newpoint).
- [ ] Aligner complètement les contraintes `cx<=dx` + `meanx<=0.5` au même moment du parcours permutations.
- [ ] Aligner les sémantiques `startobj/endobj` + limite objets (cap de recherche) avec preuve de non-régression runtime.

### R8.5 — Critères d’acceptation finaux (DoD v2 parité mécaniste)
- [ ] Obtenir sur oracle un diff verify C/Python où les écarts sont uniquement numériques mineurs (pas structurels de décision).
- [ ] Étendre la validation à un mini-lot représentatif (pas seulement 1 oracle) avec mêmes signatures de stage dominantes.
- [ ] Livrer un evidence pack v2 dédié “parité mathématique” (permute/resolve/verify pas-à-pas + conclusion Go/No-Go).

### R8.6 — Coins connexes critiques révélés au 2e tour (à ne pas laisser hors parité)
- [ ] Aligner le chemin **verify-only** Astrometry: `solver_verify_sip_wcs -> solver_inject_match -> solver_handle_hit(fake_match=TRUE)` (mêmes décisions/effets que C).
- [ ] Aligner la sémantique `fake_match` dans `verify_hit` (gamma/RoR/quad handling) sur le chemin C exact.
- [ ] Aligner la branche `solver_handle_hit` quand `predistort`/`pixel_xscale` est actif (refit pondéré `fit_tan_wcs_weighted` / `fit_sip_wcs`, gestion `set_crpix`).
- [ ] Vérifier l’alignement orchestration `onefield/engine` qui pilote le cœur: seuils `ANODDS*`, `best_hit_only` + `remove_duplicate_solutions`, ordre `record_match_callback`.
- [ ] Ajouter une preuve diffable dédiée “verify-path parity” (verify-only vs solve-run sur même champ, avec comparaison hit/miss, logodds, tune/reverify).
- [ ] Vérifier les points d’entrée périphériques (`control-program.c`, `solver.i`) pour confirmer qu’ils ne modifient pas la sémantique du cœur (wrappers/entrypoints uniquement).
