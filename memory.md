# ZeSolver Memory Log (accompli uniquement)

> Règle: ce fichier contient uniquement ce qui a été fait/validé.
> Le reste à faire est dans `followup.md`.

## 2026-06-12 (S6, tronc local assaini avant nouveaux audits)

- Le test synthétique cassé n'indiquait pas une régression solveur :
  - `tests/test_synthetic.py::test_synthetic_index_produces_candidate` échouait parce que le fixture synthétique produit un WCS à ~`180"/px`, hors seuil par défaut `scale_max_arcsec=15`
  - le test a été réaligné avec `scale_max_arcsec=300.0`
- Validation locale après correction :
  - `python3 -m py_compile` OK
  - `pytest -q tests/test_zeblindsolver.py tests/test_synthetic.py` => `32 passed`

## 2026-06-12 (S6, candidate-lock `d50_2823` confirme le front causal courant)

- Un replay sentinelle plus propre a été lancé en mode **candidate-scoped** via `blind_forensic_force_tile_key='d50_2823'` :
  - artefact :
    - `reports/r47i_s6_candidate_lock_20260612_154254_d50_2823/`
- Résultat durable :
  - `candidate_order_head` et `candidate_try` restent verrouillés sur `d50_2823`
  - `collect_ready` atteint `24` paires
  - `astrometry_lookup_stage` garde `4` paires avec `lookup_ready=true`
  - le premier front vivant sur cette tuile est bien `reject_perm_hash_gate`
  - distribution utile observée dans l'`exact_trace` :
    - `reject_perm_hash_gate=1884`
    - `reject_not_better_than_best=208`
    - `skip_zero_inliers=142`
    - `accept_new_best=22`
    - `perm_hash_qmax_abs_delta min = 2052`
- Lecture durable :
  - après neutralisation du bruit inter-tuile, le problème causal prioritaire n'est plus le ranking ;
  - c'est bien le pruning `perm_hash_gate` sur la tête naturelle `d50_2823`.

## 2026-06-12 (S6, micro-A/B `perm_hash 2048 -> 2304` sur `d50_2823`)

- Un coupe-circuit forensic lecture-seule a été ajouté dans `zeblindsolver.py` :
  - `blind_forensic_abort_after_astrometry_exact_trace_entries`
  - défaut `0` / OFF
  - rôle : borner les audits candidate-lock centrés sur `astrometry_exact_trace`
- Validation locale après patch :
  - `python3 -m py_compile` OK
  - `pytest -q tests/test_zeblindsolver.py tests/test_synthetic.py` => `32 passed`
- Un micro-A/B diffable a été obtenu sur la même tuile `d50_2823`, avec la même enveloppe forensic :
  - `reports/r47i_s6_candidate_lock_forensic_20260612_155711_q2048_q2304_abort300/`
  - `reports/r47i_s6_candidate_lock_forensic_20260612_155909_q2304_only_abort300/`
- Résultat durable :
  - `q2048` :
    - `reject_perm_hash_gate=471`
    - `reject_not_better_than_best=73`
    - `skip_zero_inliers=17`
    - `accept_new_best=3`
    - `perm_hash_qmax_abs_delta min = 2052`
  - `q2304` :
    - `reject_perm_hash_gate=455`
    - `reject_not_better_than_best=86`
    - `skip_zero_inliers=20`
    - `accept_new_best=3`
    - `perm_hash_qmax_abs_delta min = 2326`
- Lecture durable :
  - le relâchement `2048 -> 2304` libère bien le petit paquet attendu `2052..2304` ;
  - mais il ne change pas encore le nombre de `accept_new_best` ni ne fait émerger un nouvel aval validable ;
  - le front causal vivant reste donc `perm_hash_gate`, seulement un peu plus relâché.

## 2026-06-12 (S6, cran `2560` sur `d50_2823` : le front utile glisse en aval)

- Artefact :
  - `reports/r47i_s6_candidate_lock_forensic_20260612_161712_q2560_only_abort300/`
- Résultat durable :
  - `q2560` continue de réduire `reject_perm_hash_gate` (`444` vs `455` à `2304`)
  - mais `accept_new_best` reste à `3`
  - les candidats nouvellement libérés dans la bande `2305..2560` atteignent bien `pre_resolve/post_resolve`
  - exemples relevés :
    - `qdelta=2396`, `inliers_raw=1`, `rms≈2.42`
    - `qdelta=2472`, `inliers_raw=0`
    - `qdelta=2373`, `inliers_raw=2`, `rms≈1.76`
    - `qdelta=2541`, `inliers_raw=0/1`
  - ces candidats retombent ensuite surtout sur :
    - `reject_not_better_than_best`
    - `skip_zero_inliers`
  - le `transition_dump` reste identique sur `2048/2304/2560` :
    - `fail_reason=post_resolve_too_few_pairs`
- Lecture durable :
  - à ce stade, augmenter encore `perm_hash_max_qdelta` n'est plus la priorité ;
  - le prochain front causal utile est l'aval immédiat des candidats désormais déverrouillés.

## 2026-06-12 (S6, les quasi-bons candidats aval perdent surtout sur les inliers)

- L'observabilité `reject_not_better_than_best` a été enrichie dans `zeblindsolver.py` :
  - `best_rms_px_before`
  - `metric_gap_vs_best`
  - `inlier_gap_vs_best`
  - `rms_gap_vs_best`
- Validation locale après patch :
  - `python3 -m py_compile` OK
  - `pytest -q tests/test_zeblindsolver.py tests/test_synthetic.py` => `32 passed`
- Rejeu utile :
  - `reports/r47i_s6_candidate_lock_forensic_20260612_162453_q2560_gapprobe/`
- Résultat durable :
  - `reject_not_better_than_best=94`
  - `metric_gap_vs_best` médian ≈ `-11.21`
  - `inlier_gap_vs_best` médian = `-5`
  - `rms_gap_vs_best` médian ≈ `+0.09 px`
  - les cas les plus proches du meilleur courant sont tous à `5` inliers contre un meilleur à `6` :
    - `seq 60` : `metric_gap≈-2.91`, `inlier_gap=-1`, `rms_gap≈-0.44`
    - `seq 160` : `metric_gap≈-2.91`, `inlier_gap=-1`, `rms_gap≈-0.17`
    - `seq 244` : `metric_gap≈-2.97`, `inlier_gap=-1`, `rms_gap≈+0.21`
- Lecture durable :
  - le front aval n'est pas principalement un problème de RMS ;
  - il est dominé par un **déficit d'inliers** face au meilleur courant ;
  - la prochaine investigation utile doit viser la cause de ce manque d'inliers.

## 2026-06-12 (S6, comparaison source Astrometry : divergence probable de sémantique aval)

- Référence relue :
  - `doc/code.rst` Astrometry.net
- Résumé durable :
  - `resolve_matches()` appelle `solver_handle_hit()`
  - `solver_handle_hit()` appelle `verify_hit()`
  - `verify_hit()` appelle `real_verify_star_lists()` pour la comparaison vrai/faux match
- Lecture durable :
  - le corridor canonique Astrometry fait porter la décision sur la phase `verify`
  - côté Ze, nous avons encore un comparateur local `hyp_metric` qui rejette déjà des candidats en `reject_not_better_than_best` avant le corridor final de confirmation
  - la divergence la plus plausible n'est donc pas une déficience mathématique brute, mais un **écart de sémantique de sélection aval** entre notre comparateur local et la logique `verify_hit()` d'Astrometry.

## 2026-06-12 (S6, probe near-best : le comparateur local n'est pas le verrou causal principal)

- Un probe-only `near-best override`, défaut OFF, a été ajouté dans `zeblindsolver.py` :
  - `blind_astrometry_probe_near_best_override_enabled`
  - `blind_astrometry_probe_near_best_max_inlier_gap`
  - `blind_astrometry_probe_near_best_min_metric_gap`
  - `blind_astrometry_probe_near_best_max_rms_gap_px`
- Validation locale après patch :
  - `python3 -m py_compile` OK
  - `pytest -q tests/test_zeblindsolver.py tests/test_synthetic.py` => `32 passed`
- Rejeu diffable sur la même enveloppe candidate-lock `d50_2823` / `perm_hash=2560` :
  - `reports/r47i_s6_candidate_lock_forensic_20260612_171213_q2560_nearbest_probe/`
- Résultat durable :
  - le probe laisse bien passer des quasi-`best` auparavant rejetés :
    - baseline `q2560_gapprobe` :
      - `accept_new_best=3`
      - `reject_not_better_than_best=94`
    - variante `nearbest_probe` :
      - `accept_new_best=6`
      - `accept_probe_near_best_override=6`
      - `reject_not_better_than_best=85`
  - mais le front final ne change pas :
    - `stage_by_stage_counts.verify` reste `0`
    - `transition_dump` reste `fail_reason=post_resolve_too_few_pairs`
    - le dernier meilleur accepté reste le même qu'en baseline :
      - `seq 462`
      - `7 inliers`
      - `rms≈1.94`
- Lecture durable :
  - `reject_not_better_than_best` comprimait effectivement trop tôt certaines hypothèses proches ;
  - mais ce comparateur n'est pas, à lui seul, la cause racine active du non-solve ;
  - le prochain front causal utile est désormais l'**entrée effective dans verify** et le support/pairing `post_resolve` du meilleur candidat final.

## 2026-06-12 (S6, le `meta_seed` guard masque un candidat utile ; sans lui on atteint enfin `verify`)

- Un switch probe-only, défaut ON, a été ajouté dans `zeblindsolver.py` :
  - `blind_astrometry_meta_seed_carry_enabled`
- Validation locale après patch :
  - `python3 -m py_compile` OK
  - `pytest -q tests/test_zeblindsolver.py tests/test_synthetic.py` => `32 passed`
- Lecture runtime du candidat nominal `d50_2823` / `6.07"/px` :
  - `astrometry_origin_prescreen` juge encore le **pool courant 4-point**
  - alors que `transform_origin_meta` porte déjà un support `resolve_hit` de `19` paires (`7` inliers origine)
  - sur ce prescreen 4-point, les résiduels explosent :
    - `residual_px_med≈1690`
    - `residual_px_max≈1708`
- Probe `meta_seed carry OFF` :
  - artefact :
    - `reports/r47i_s6_candidate_lock_forensic_20260612_172811_q2560_no_meta_seed_carry/`
  - résultat durable :
    - `resolve_hit_meta_seed_rejected` disparaît ;
    - mais le nominal retombe quand même en `empty_inliers`
    - `verify` reste à `0`
- Probe `meta_seed carry ON + plausibility guard OFF` :
  - artefact :
    - `reports/r47i_s6_candidate_lock_forensic_20260612_173032_q2560_meta_seed_noguard/`
  - résultat durable :
    - le carry transporte bien le vrai support utile :
      - `source_pairs_original=19`
      - `carried_pairs=19`
      - `inlier_count=19`
      - `residual_px_med≈3.01`
      - `residual_px_max≈7.60`
    - le chemin atteint enfin `verify` (`stage_by_stage_counts.verify = 1`)
    - le blocage vivant devient un rejet aval explicite :
      - `validation_failed[rms_ok=0,inliers_ok=1,scale_ok=1,rms=3.660,rms_thr=1.200,inliers=15,inliers_thr=7]`
      - `pairs=18`
      - `inliers=15`
- Lecture durable :
  - le guard de plausibilité `meta_seed` basé sur le prescreen 4-point rejette à tort un candidat utile ;
  - le mécanisme de carry lui-même n'est pas le problème, il est au contraire nécessaire pour exposer le bon support `resolve_hit` ;
  - une fois ce faux verrou levé en probe, le front causal vivant se déplace du corridor `meta_seed` vers un vrai **rejet RMS/validation** plus proche du `verify` canonique.

## 2026-06-13 (S6, relâcher `quality_rms` seul ne ferme pas le front ; l'ancre d'échelle aval reste incohérente)

- Un A/B strict packagé a été rejoué sur la même enveloppe candidate-lock `d50_2823` / `q2560` / `meta_seed guard OFF` :
  - baseline packagée :
    - `reports/r47i_s6_candidate_lock_forensic_20260613_015502_q2560_meta_seed_noguard_baselinepack/`
  - variante RMS :
    - `reports/r47i_s6_candidate_lock_forensic_20260613_015241_q2560_meta_seed_noguard_rms4000/`
- Résultat durable :
  - le meilleur candidat utile reste **strictement** `seq 462` dans les deux runs :
    - `7` inliers
    - `rms≈1.94`
    - `model_scale_arcsec_final≈6.0708`
  - relâcher `quality_rms` de `1.2 -> 4.0` ne déclenche aucun `record_match_callback`
  - le support aval s'élargit (`15 -> 18` inliers) mais le solve reste bloqué
  - les deux runs partagent la même incohérence d'échelle finale :
    - `model_scale_arcsec≈6.078`
    - `scale_anchor_arcsec≈24.214`
    - `bounds_verify_arcsec≈15.436..39.226`
    - `scale_ok=0`
- Lecture durable :
  - le seuil RMS dur n'est pas, à lui seul, la cause racine active ;
  - le prochain front causal utile est désormais la **source de `scale_anchor_arcsec`** dans la validation finale, pas un nouveau relâchement RMS ou `perm_hash`.

## 2026-06-13 (S6, contrat top-down formalisé : le soupçon `step0 non-carried` n'est pas une divergence prouvée)

- Un audit de contrat top-down a été ajouté :
  - `tools/r47i_s6_checkpoint_contract_audit.py`
- Validation locale :
  - `python3 -m py_compile tools/r47i_s6_checkpoint_contract_audit.py` OK
- Artefact produit sur le baseline courant :
  - `reports/r47i_s6_checkpoint_contract_audit_20260613_212249_r47i_s6_theta_core_probe_20260613_205329/summary.json`
- Résultat durable :
  - le contrat de comparaison est maintenant posé checkpoint par checkpoint avec :
    - artefact Ze de référence
    - source Astrometry de référence
    - comparateurs attendus
  - checkpoints méthodologiquement fermés :
    - `input_stars_order` => fermé **par contrat source**
    - `quad_geometry` => fermé
    - `verify_pix2_scale` => fermé
    - `verify_sequence_core` => fermé
  - checkpoint restant ouvert :
    - `verify_support_pre_step0`
- Lecture durable :
  - le fait que `verify` démarre sur une étoile **non portée** par le `carry support` n'est **pas**, à lui seul, une divergence Ze vs Astrometry ;
  - le source C (`verify_get_test_stars`) conserve l'ordre des test-stars après `dedup` + retrait du quad + `RoR` et ne priorise pas le `carry support` ;
  - le prochain cran utile n'est donc plus de re-questionner le tri Ze côté test-stars, mais d'obtenir/comparer un **état Astrometry homologue** du checkpoint `verify_support_pre_step0`.

## 2026-06-13 (S6, premier pendant Astrometry blind runtime extrait sur le FITS sentinelle)

- Un chemin runtime Astrometry local exploitable a été remis en service sur le FITS sentinelle `232102` :
  - génération `.axy` via `solve-field --just-augment`
  - exécution du binaire repo `astrometry-net-main/solver/astrometry-engine`
  - config locale :
    - `reports/r47i_s6_astrometry_verify_entry_probe_20260613_2130/backend.cfg`
  - logs :
    - `reports/r47i_s6_astrometry_verify_entry_probe_20260613_2130/engine.log`
- Un parseur dédié a été ajouté :
  - `tools/r47i_s6_astrometry_engine_verify_audit.py`
- Validation locale :
  - `python3 -m py_compile tools/r47i_s6_astrometry_engine_verify_audit.py` OK
- Artefact synthèse :
  - `reports/r47i_s6_astrometry_engine_verify_audit_20260613_213601_r47i_s6_astrometry_verify_entry_probe_20260613_2130/summary.json`
- Résultat durable :
  - le runtime Astrometry instrumenté émet bien des blocs `C_VERIFY_ENTRY` et `C_VERIFY_TERM`
  - `entry_count = 27`
  - l'entrée retenue juste avant `Got a new best match` est :
    - `NT=123`
    - `NR=19`
    - `quad_field_head=[4,0,3,5]`
    - `testperm_head=[1,2,7,12,11,6,8,9,16,19,13,17,10,14,23,21]`
    - `refperm_head=[13,42,26,17,41,12,10,8,39,43,28,31,29,36,0,7]`
- Lecture durable :
  - un `testperm` Astrometry **canonique** n'est pas forcément monotone ;
  - `verify_apply_ror()` peut réordonner les test-stars via `uniformize`, pas seulement filtrer ;
  - cela invalide toute lecture trop forte du type « ordre monotone = forcément canonique » ;
  - en revanche, ce pendant runtime est encore **même FITS mais pas même étage** que le sentry Ze `d50_2823`, donc il ne suffit pas encore à clore `verify_support_pre_step0`.

## 2026-06-13 (S6, l'ancre `24.21\"/px` vient bien de `candidate_pairset_local`, mais cette pairset locale est elle-même incohérente)

- Probe ciblé rejoué sur la même enveloppe candidate-lock `d50_2823` / `q2560` / `meta_seed guard OFF` :
  - `reports/r47i_s6_candidate_lock_forensic_20260613_020005_q2560_meta_seed_noguard_anchorprobe/`
- Résultat durable :
  - `pairset_scale_precheck` montre bien :
    - `scale_anchor_source = candidate_pairset_local`
    - `scale_anchor_arcsec ≈ 24.214`
    - `approx_scale_arcsec ≈ 2.393`
    - `pairs = 12`
  - la `pairset_local` disponible est très large :
    - `median ≈ 24.214"/px`
    - `p10 ≈ 3.65"/px`
    - `p90 ≈ 40.36"/px`
    - `span_implied_scale_arcsec ≈ 18.993`
  - cette ancre candidate-locale reste incompatible avec le candidat utile transporté par `meta_seed` :
    - `resolve_hit/model_scale_arcsec ≈ 6.071`
    - `validation model_scale_arcsec ≈ 6.078`
- Lecture durable :
  - l'ancre aval incohérente n'est pas due à une fuite de source ou à une mauvaise consommation ;
  - elle est **déjà mauvaise à la source** côté `candidate_pairset_local` ;
  - le prochain front causal utile est donc la **construction/robustesse de la pairset-local scale anchor**, pas le seul gate RMS final.

## 2026-06-13 (S6, le clamp d'ancre puis l'ouverture RMS atteignent enfin le vrai rejet `verify/logodds`)

- Un switch probe-only, défaut OFF, a été ajouté dans `zeblindsolver.py` :
  - `blind_astrometry_probe_resolve_hit_scale_anchor_enabled`
- Validation locale après patch :
  - `python3 -m py_compile` OK
  - `pytest -q tests/test_zeblindsolver.py tests/test_synthetic.py` => `32 passed`
- Probes durables :
  - clamp d'ancre seul :
    - `reports/r47i_s6_candidate_lock_forensic_20260613_020311_q2560_meta_seed_noguard_anchorclamp/`
    - effet durable :
      - `scale_ok` passe de `0 -> 1` à support inchangé
      - l'ancre finale devient `anchor_source=probe_resolve_hit_model_scale`, `anchor_arcsec≈6.078`
      - le rejet vivant retombe bien sur le RMS pur (`rms=3.660`, `rms_thr=1.200`)
  - clamp d'ancre + `quality_rms=4.0` :
    - `reports/r47i_s6_candidate_lock_forensic_20260613_020420_q2560_meta_seed_noguard_anchorclamp_rms4000/`
    - le candidat utile reste rejeté sur le RMS (`rms=4.860 > 4.000`)
  - clamp d'ancre + `quality_rms=5.0` :
    - `reports/r47i_s6_candidate_lock_forensic_20260613_020530_q2560_meta_seed_noguard_anchorclamp_rms5000/`
    - `validation_pairs` passe enfin `quality=GOOD`
    - `record_match_callback.json` apparaît enfin
    - le rejet final devient :
      - `terminal_decision = reject_callback_no_positive_verify_match`
      - `record_match_callback_reason = no_positive_verify_match`
      - `onefield_final_logodds = -1.3862943611198908`
      - `prob_matches = 0`
      - `prob_theta_match_total = 0`
- Lecture durable :
  - la chaîne causale est maintenant beaucoup plus propre :
    - ancre `candidate_pairset_local` incohérente
    - puis gate RMS aval
    - puis, une fois ces deux verrous probe levés, **vrai rejet `verify/logodds` sans support positif**
  - le prochain front causal utile n'est donc plus le scale gate ni le seul RMS, mais le **corridor `verify` lui-même** (`prob_matches` / `MatchObj` / logodds).

## 2026-06-13 (S6, snapshot verify diffable: une ouverture tardive existe, mais aucun préfixe positif ne se stabilise)

- Un nouvel outil lecture-seule a été ajouté :
  - `tools/r47i_s6_verify_snapshot_audit.py`
  - rôle : résumer directement un report `verifydump` (`verify_debug_sets.json`, `verify_step_dump.json`, `record_match_callback.json`) pour classer vite le front vivant du corridor `verify`
- Validation locale après ajout :
  - `python3 -m py_compile tools/r47i_s6_verify_snapshot_audit.py` OK
  - `pytest -q tests/test_zeblindsolver.py tests/test_synthetic.py` => `32 passed`
- Rejeu utile sur le dernier artefact :
  - source :
    - `reports/r47i_s6_candidate_lock_forensic_20260613_021153_q2560_anchorclamp_rms5000_verifydump/`
  - synthèse :
    - `reports/r47i_s6_verify_snapshot_audit_20260613_085226_r47i_s6_candidate_lock_forensic_20260613_021153_q2560_anchorclamp_rms5000_verifydump/`
- Résultat durable :
  - `pool_source = field_native_astrometry_vs_tile_world`
  - `test/ref = 38 / 12`
  - `prob_logodds = -1.3862943611198908`
  - le corridor Ze n'est pas un "zéro gate" absolu :
    - `gate_pass_count = 1`
    - `first_gate_pass_step = 17`
  - mais aucun préfixe positif n'émerge quand même :
    - `prob_matches = 0`
    - `prob_theta_match_total = 0`
    - `terminal_decision = reject_callback_no_positive_verify_match`
  - distribution NN très défavorable sur ce snapshot :
    - `nsig2 min ≈ 23.68`
    - `nsig2 median ≈ 1636.77`
    - `nsig2 max ≈ 8168.33`
- Lecture durable :
  - le callback n'est plus le suspect principal ;
  - le front vivant est bien le **support verify natif lui-même** :
    - soit les entrées `verify` (pool / ordre / sigma / géométrie) ne sont pas encore canoniques,
    - soit la logique séquentielle `theta/logodds` diverge encore du C sur ce même snapshot.

## 2026-06-13 (S6, le ref pool `verify` apparaît beaucoup plus compact que le test pool utile)

- L'outil `tools/r47i_s6_verify_snapshot_audit.py` a été enrichi avec des métriques géométriques :
  - empreinte spatiale `test/ref`
  - distances au `quad_center`
  - ratio de compacité `ref vs test`
- Validation locale après enrichissement :
  - `python3 -m py_compile tools/r47i_s6_verify_snapshot_audit.py` OK
  - `pytest -q tests/test_zeblindsolver.py tests/test_synthetic.py` => `32 passed`
- Rejeu utile :
  - `reports/r47i_s6_verify_snapshot_audit_20260613_090357_r47i_s6_candidate_lock_forensic_20260613_021153_q2560_anchorclamp_rms5000_verifydump/`
- Résultat durable :
  - `test_xy_use` couvre quasiment tout le champ :
    - span ≈ `954 x 1873 px`
  - `ref_xy_use` est beaucoup plus compact :
    - span ≈ `137 x 242 px`
    - `ref_compact_vs_test_span_ratio ≈ 0.129`
  - le `quad_center` du snapshot est à `(~1003.44, ~532.16)` ;
  - les `ref_xy_use` gardés sont tous assez loin de ce centre :
    - distance min/médiane/max ≈ `558.76 / 677.55 / 738.14 px`
  - en parallèle, `validation` reste bonne :
    - `pairs=18`
    - `inliers=18`
    - `rms≈4.86 px`
  - mais le verify séquentiel reste quasi intégralement distractor-only :
    - `gate_pass_count = 1`
    - `prob_matches = 0`
    - `prob_theta_match_total = 0`
- Lecture durable :
  - le verrou vivant se resserre encore côté **géométrie d'entrée verify** ;
  - avant d'accuser davantage `theta/logodds`, il faut expliquer pourquoi le ref pool consommé par `verify` est si compact et si décentré par rapport au test pool utile, alors que la validation de la solution transportée reste bonne.

## 2026-06-13 (S6, cause amont confirmée: le hard-cap natif à 12 refs cassait le support verify)

- Comparatif source utile confirmé :
  - Astrometry `verify_hit` ne pose pas de hard-cap minuscule sur le ref pool avant :
    - in-image filtering
    - quad exclusion
    - sweep ordering
    - RoR
  - côté Ze, le chemin `field_native_astrometry_vs_tile_world` gardait encore un cap dur hérité :
    - `blind_astrometry_verify_refstar_max_keep = 12`
- Patch appliqué dans `zeblindsolver.py` :
  - nouveau helper :
    - `_native_verify_ref_pool_can_hard_cap(...)`
  - règle durable :
    - sur le chemin canonique/natif (`astrometry_native_verify_semantics_mode=True`), le cap précoce du ref pool est désormais désactivé
    - il reste autorisé pour les chemins heuristiques non canoniques
- Test ajouté :
  - `tests/test_zeblindsolver.py::test_native_verify_ref_pool_hard_cap_disabled_on_canonical_paths`
- Validation locale :
  - `python3 -m py_compile zeblindsolver/zeblindsolver.py tests/test_zeblindsolver.py tools/r47i_s6_verify_snapshot_audit.py` OK
  - `pytest -q tests/test_zeblindsolver.py tests/test_synthetic.py` => `33 passed`
- Rejeu ciblé :
  - run :
    - `reports/r47i_s6_refpool_uncap_probe_20260613_102802/`
  - audit snapshot :
    - `reports/r47i_s6_verify_snapshot_audit_20260613_102849_r47i_s6_refpool_uncap_probe_20260613_102802/`
- Résultat durable :
  - avant patch :
    - `ref_n = 12`
    - `ref_compact_vs_test_span_ratio ≈ 0.129`
    - `gate_pass_count = 1`
    - `prob_matches = 0`
    - `prob_theta_match_total = 0`
    - `terminal_decision = reject_callback_no_positive_verify_match`
  - après patch :
    - `ref_n = 377`
    - `ref_compact_vs_test_span_ratio ≈ 1.055`
    - `gate_pass_count = 17`
    - `prob_matches = 1`
    - `prob_theta_match_total = 6`
    - `terminal_decision = accept_keep`
    - `success = true`
- Lecture durable :
  - le premier point de rupture amont était bien la **compaction artificielle du ref pool natif** ;
  - une fois ce verrou levé, le corridor `verify` redevient capable de former un support positif réel sur `d50_2823` ;
  - le prochain écart amont évident restant est `mo_scale_native`, toujours reconstruit depuis `quadpix_median_px` au lieu d'une vraie source canonique `arcsec/pix`.

## 2026-06-13 (S6, `mo_scale_native` et `verify_pix2` sont maintenant recâblés sur une vraie échelle canonique)

- Patch appliqué dans `zeblindsolver.py` :
  - nouveau helper :
    - `_resolve_native_mo_scale_arcsec_px(...)`
  - règle durable :
    - sur le chemin natif/canonique, `mo_scale_native` et `verify_pix2` préfèrent désormais :
      - `model_scale_arcsec`
      - puis `pix_scale_arcsec`
    - le fallback `quadpix_median_px` ne reste autorisé que hors chemin canonique
- Tests ajoutés :
  - `test_resolve_native_mo_scale_prefers_wcs_scale_on_native_path`
  - `test_resolve_native_mo_scale_keeps_pixel_geom_on_noncanonical_path`
  - `test_resolve_native_mo_scale_prefers_pix_scale_if_model_missing_on_native_path`
- Validation locale :
  - `python3 -m py_compile zeblindsolver/zeblindsolver.py tests/test_zeblindsolver.py` OK
  - `pytest -q tests/test_zeblindsolver.py tests/test_synthetic.py` => `36 passed`
- Rejeu utile après fix ref-pool seul :
  - `reports/r47i_s6_refpool_uncap_probe_20260613_102802/`
  - audit :
    - `reports/r47i_s6_verify_snapshot_audit_20260613_102849_r47i_s6_refpool_uncap_probe_20260613_102802/`
  - état durable intermédiaire :
    - `verify_scale_arcsec_px_for_pix2 = 180.689`
    - `verify_scale_source_for_pix2 = mo_scale_native`
    - `mo_scale_source = quadpix_median_px`
    - `prob_logodds ≈ -1.12448`
    - `nsig2 min/median/max ≈ 1.3966 / 12.5812 / 144.94`
- Rejeu utile après fix `mo_scale_native` + `verify_pix2` :
  - `reports/r47i_s6_verifypix2_probe_20260613_103813/`
  - audit :
    - `reports/r47i_s6_verify_snapshot_audit_20260613_103857_r47i_s6_verifypix2_probe_20260613_103813/`
- Résultat durable :
  - `mo_scale = 6.077758799729469`
  - `mo_scale_source = model_scale_arcsec_native`
  - `verify_scale_arcsec_px_for_pix2 = 6.077758799729469`
  - `verify_scale_source_for_pix2 = model_scale_arcsec_native`
  - légère amélioration mesurée du corridor `verify` à pool comparable :
    - `prob_logodds ≈ -1.12448 -> -1.10100`
    - `nsig2 min ≈ 1.3966 -> 1.3820`
    - `nsig2 median ≈ 12.5812 -> 12.4497`
    - `nsig2 max ≈ 144.94 -> 143.42`
  - `callback` reste positif :
    - `accept_keep`
    - `prob_matches = 1`
    - `prob_theta_match_total = 6`
- Lecture durable :
  - le gros faux écart sémantique de scale est maintenant refermé ;
  - le gain est réel mais modeste, ce qui indique que le verrou suivant n'est plus un mauvais `arcsec/pix` grossier ;
  - le prochain front amont utile est le modèle `testsigma` / gamma-like (`quad_center`, `Q2`, variance radiale), avant de descendre davantage dans `theta/logodds`.
- 2026-06-13 (S6, quad geometry native): le comparatif `verify.c` a confirmé un écart de sémantique encore vivant dans `quad_center/Q2`.
  - Astrometry `verify_get_quad_center(vf, mo, ...)` prend strictement :
    - `centerpix = midpoint(A,B)` avec `A=mo->field[0]`, `B=mo->field[1]`
    - `Q2 = dist2(A, centerpix)`
  - Côté Ze, le chemin natif utilisait encore la moyenne des `4` ancres puis la moyenne des distances au centre.
  - Patch durable appliqué :
    - nouveau helper `zeblindsolver.py::_resolve_verify_quad_geometry_px(...)`
    - en mode natif/canonique :
      - `quad_center/Q2` basculent sur `midpoint(AB)` + `dist2(A, midpoint(AB))`
    - hors chemin canonique :
      - conservation du comportement centroid/mean-radius existant
  - Tests ajoutés :
    - `test_resolve_verify_quad_geometry_uses_ab_midpoint_on_native_path`
    - `test_resolve_verify_quad_geometry_keeps_centroid_on_noncanonical_path`
    - `test_resolve_verify_quad_geometry_requires_two_points`
  - Validation locale :
    - `python3 -m py_compile zeblindsolver/zeblindsolver.py tests/test_zeblindsolver.py tools/r47i_s6_verify_snapshot_audit.py` OK
    - `pytest -q tests/test_zeblindsolver.py tests/test_synthetic.py` => `39 passed`
  - Rejeu utile :
    - `reports/r47i_s6_quadab_probe_20260613_110322/`
  - Lecture durable :
    - la géométrie native bascule bien sur la forme canonique :
      - avant : `quad_center_px=[1003.44, 532.16]`, `quad_q2_px2=15234.68`, `quad_center_source=collect_pref_pair_idx_quad`
      - après : `quad_center_px=[1020.88, 616.40]`, `quad_q2_px2=15382.70`, `quad_center_source=matchobj_ab_midpoint_native`
    - sur ce rejeu, le support `verify` observé est :
      - `test/ref = 43 / 223`
      - `ref_dist_to_quad_center min ≈ 38.41 px`
      - `prob_logodds ≈ -1.38629`
      - `gate_pass_count = 11`
      - `theta_match_count = 5`
      - `nsig2 min/median/max ≈ 0.333 / 25.32 / 244.01`
  - Leçon durable :
    - le patch `AB midpoint` est correctement branché et déplace la géométrie du corridor natif dans le sens Astrometry ;
    - mais le verdict causal sur l’amélioration effective du score `verify` reste provisoire tant qu’on n’a pas un A/B strictement identique sur code courant avec la vieille géométrie centroid forcée comme témoin.
- 2026-06-13 (S6, A/B borné midpoint vs centroid): l’A/B strictement identique sur code courant a finalement été rejoué avec un runner borné dédié.
  - Outil ajouté :
    - `tools/r47i_s6_bounded_verify_probe.py`
    - usage durable :
      - lancer un `blind_solve` S6 avec dumps `verify_*`
      - éviter les solveurs pendants une fois les artefacts utiles écrits
  - Artefacts utiles :
    - midpoint canonique :
      - `reports/r47i_s6_quadab_bounded_midpoint_long_20260613_112416/`
    - centroid forcé témoin :
      - `reports/r47i_s6_quadab_bounded_centroid_long_20260613_112416/`
  - Invariant A/B obtenu :
    - même `runtime_cfg`
    - même `verify_pix2_input = 2.5870715472637387`
    - même `verify_scale_arcsec_px_for_pix2 = 6.077758799729469`
    - même `test/ref = 38 / 395`
    - seul delta utile :
      - `quad_center/Q2`
  - Résultat durable :
    - `midpoint AB` :
      - `quad_center_source = matchobj_ab_midpoint_native`
      - `prob_logodds ≈ -0.98933`
      - `gate_pass_count = 15`
      - `theta_match_count = 6`
      - `nsig2 min/median/max ≈ 0.272 / 12.510 / 141.713`
    - `centroid` forcé :
      - `quad_center_source = forced_quad_geometry_json`
      - `prob_logodds ≈ -1.15296`
      - `gate_pass_count = 19`
      - `theta_match_count = 8`
      - `nsig2 min/median/max ≈ 0.238 / 10.931 / 143.424`
  - Leçon durable forte :
    - la géométrie canonique `midpoint AB` améliore bien le score final `verify` **malgré moins de matches instantanés et moins de gate passes**
    - donc le résiduel prioritaire n’est plus un simple problème de “quantité de matches”
    - le prochain front utile descend désormais dans la **qualité séquentielle `theta/logodds`** et la façon dont les matches/distractors s’additionnent.
- 2026-06-13 (S6, diff séquentiel verify): un outil de diff pas-à-pas a été ajouté pour comparer deux snapshots `verify`.
  - Outil ajouté :
    - `tools/r47i_s6_verify_sequence_diff.py`
  - Artefact utile :
    - `reports/r47i_s6_verify_sequence_diff_20260613_113505_r47i_s6_quadab_bounded_midpoint_long_20260613_112416_vs_r47i_s6_quadab_bounded_centroid_long_20260613_112416/summary.json`
  - Lecture durable :
    - le premier avantage du mode `midpoint AB` n’apparaît **pas** au premier flip `match/distractor`, mais dès le **step 0** :
      - `midpoint` :
        - `nsig2 ≈ 4.8949`
        - `delta_logodds ≈ -0.98933`
        - `logfg ≈ -15.53413`
      - `centroid` :
        - `nsig2 ≈ 5.4294`
        - `delta_logodds ≈ -1.15296`
        - `logfg ≈ -15.69775`
    - c’est ce premier match qui fixe déjà le meilleur préfixe :
      - `prob_logodds midpoint ≈ -0.98933`
      - `prob_logodds centroid ≈ -1.15296`
    - les divergences plus tardives existent mais deviennent secondaires comme cause racine :
      - premier écart `gate_pass` à `i=3`
      - premier `match_flip` à `i=7` (`midpoint=distractor`, `centroid=match`)
  - Lecture durable sur `testsigma²` :
    - le gain vient d’abord du profil radial consommé sur les toutes premières étoiles utiles :
      - step `0` :
        - `sigma2 midpoint ≈ 145.80`
        - `sigma2 centroid ≈ 131.44`
      - step `7` :
        - `sigma2 midpoint ≈ 281.13`
        - `sigma2 centroid ≈ 317.55`
    - le recâblage `midpoint AB` ne rend pas tout le préfixe plus permissif ;
    - il améliore surtout la **première étoile utile**, ce qui suffit à relever le meilleur préfixe.
  - Leçon durable forte :
    - le front prioritaire n’est plus “plus de matches” ;
    - il est maintenant le **profil `testsigma²` et/ou l’ordre des premières test-stars** qui déterminent la qualité du tout premier préfixe `verify`.
- 2026-06-13 (S6, quad leakage fix): une divergence source-vs-source plus amont a été confirmée puis refermée dans `verify_get_test_stars` côté Ze.
  - Astrometry enlève explicitement les `mo->field[i]` du quad après déduplication.
  - Côté Ze, le retrait des quad stars passait encore par `img_in` générique, ce qui laissait survivre au moins une vraie étoile du quad d’hypothèse dans `test_xy_use`.
  - Preuve runtime avant patch :
    - `reports/r47i_s6_testperm_sigma_probe_20260613_173945/verify_debug_sets.json`
    - `mo_quadpix[0] = [1000.323974609375, 494.0926208496094]`
    - apparaissait encore exactement dans `test_xy_use`
  - Patch durable appliqué :
    - le retrait des quad stars préfère désormais les **vraies ancres d’hypothèse** (`collect_pref_pair_idx_quad`) au lieu de `img_in`
    - instrumentation ajoutée en plus :
      - `teststarid_px`
      - `teststarid_use`
  - Validation locale :
    - `python3 -m py_compile ...` OK
    - `pytest -q tests/test_zeblindsolver.py tests/test_synthetic.py` => `40 passed`
  - Rejeu utile :
    - `reports/r47i_s6_quadremoval_probe_20260613_174218/`
  - Résultat durable :
    - après patch, aucune des `mo_quadpix[:4]` n’apparaît plus exactement dans `test_xy_use`
    - mais `prob_logodds` reste encore négatif (`≈ -1.38629`)
    - donc la fuite du quad était une divergence réelle, mais pas le dernier verrou causal
  - Leçon durable :
    - le front vivant retombe bien sur l’**ordre des premières test-stars utiles** et/ou leur **profil `testsigma²` précoce**
    - pas sur un retour à la vieille géométrie ni sur le quad leakage lui-même.

## 2026-06-07 (S6, le bloqueur vivant revalidé est bien le pairset scale gate produit)

- Une réinstrumentation courte du corridor `post-collect -> pre-resolve_hit` a été ajoutée dans `zeblindsolver.py` :
  - événements `collect_ready`, `astrometry_lookup_stage`, `astrometry_no_transform`
  - enrichissement des `candidate_skip` sur `strict_pairset_scale_gate_reject`
- Probe borné produit rejoué :
  - `reports/r47i_s6_m106_first_divergence_audit_20260607_002531_postcollect_probe_20260607/`
- Résultat durable :
  - sur le sentinelle M106 `...232102.fit`, `product` atteint `collect_ready` pour `hinted/S/d50_2725` avec `19` paires ;
  - il est ensuite coupé avant Astrometry par `strict_pairset_scale_gate_reject` ;
  - le garde voit un pairset réduit à `10` paires avec `span_implied_scale_arcsec ≈ 14.287`, au-dessus de la borne haute `≈ 11.963`.
- Comparatif borné `product` vs `matched_probe` :
  - `reports/r47i_s6_m106_first_divergence_audit_20260607_002630_pairset_compare_20260607/`
  - `matched_probe` garde le pairset (`keep_pairset`) puis entre bien dans `astrometry_lookup_stage`, `astrometry_origin_prescreen`, `resolve_hit_meta_seed_*` sur le même `d50_2725`.
- Ablation minimale validée :
  - `reports/r47i_s6_m106_first_divergence_audit_20260607_002850_no_pairset_gate_probe_20260607/`
  - avec **seulement** `blind_pairset_scale_gate_enabled=False`, le chemin produit entre lui aussi dans Astrometry et publie des `pre_resolve_hit/post_resolve_hit` sur `d50_2725`.
- Conclusion durable :
  - le premier bloqueur causal vivant n'est plus “entre collect et resolve_hit” de façon vague ;
  - c'est bien le **pairset scale gate produit** qui élimine `d50_2725` avant Astrometry ;
  - une fois ce garde levé isolément, le résiduel suivant visible remonte à `reject_perm_hash_gate`.

## 2026-06-07 (S6, rescope minimal appliqué: le front pairset gate est franchi en produit)

- Un rescope minimal a été appliqué dans `zeblindsolver.py` :
  - le `pairset scale gate` reste observé/dumpé ;
  - mais il ne peut plus **hard-reject** quand le corridor strict Astrometry est prêt (`lookup_ready=true`).
- Helper dédié ajouté :
  - `_pairset_scale_gate_can_hard_reject(...)`
- Validation locale :
  - `python3 -m py_compile` OK
  - `pytest -q tests/test_zeblindsolver.py` => `28 passed`
- Probe produit borné rejoué :
  - `reports/r47i_s6_m106_first_divergence_audit_20260607_005919_scoped_pairset_gate_20260607/`
- Résultat durable :
  - `product_pairset_scale_precheck.json` montre sur `hinted/S/d50_2725` :
    - `decision=keep_pairset`
    - `lookup_ready=true`
    - `pairset_gate_hard_reject_enabled=false`
    - `pairset_gate_hard_reject_suppressed=true`
  - `product_phase_handoff.json` atteint bien `astrometry_lookup_stage` sur `d50_2725`
  - `product_resolve_hit_preresolve.json` et `product_transition_dump.json` confirment l'entrée réelle dans `resolve_hit`
  - `product_astrometry_exact_trace.raw.json` fait alors remonter comme premier résiduel aval visible :
    - `reject_perm_hash_gate`
  - `product_validation_pairs.json` montre ensuite un rejet de validation sur le candidat nominal `resolve_hit` (`pairs=18`, `rms≈4.254`)
- Conclusion durable :
  - le front `preastrometry_scale_gate_gap` est maintenant franchi sur le chemin produit ;
  - la priorité causale suivante devient `perm_hash_gate`, puis la validation du candidat `resolve_hit`.

## 2026-06-07 (S6, le `perm_hash_gate` est relâché modérément mais pas supprimé)

- L'outil `tools/r47i_s6_m106_first_divergence_audit.py` a été enrichi avec deux variantes minimales :
  - `product_permhash2048`
  - `product_no_permhash`
- A/B borné sur le sentinelle courant `d50_2725` :
  - baseline utile : `reports/r47i_s6_m106_first_divergence_audit_20260607_010508_product_permhash_ab_20260607/`
  - lecture sur `product` avant delta :
    - `reject_perm_hash_gate=473`
    - `reject_not_better_than_best=25`
    - `skip_zero_inliers=16`
    - `accept_new_best=3`
    - le candidat nominal utile reste `resolve_hit`, puis échoue en validation (`pairs=18`, `rms≈4.254`)
- Variante `product_permhash2048` :
  - `reports/r47i_s6_m106_first_divergence_audit_20260607_010715_product_permhash2048_only_20260607/`
  - résultat durable :
    - `reject_perm_hash_gate` baisse (`411` dans cette fenêtre bornée)
    - `accept_new_best` monte légèrement (`5`)
    - le corridor reste structuré et continue d'émettre `pre_resolve_hit/post_resolve_hit`
- Variante `product_no_permhash` :
  - `reports/r47i_s6_m106_first_divergence_audit_20260607_010801_product_no_permhash_only_20260607/`
  - résultat durable :
    - le front `reject_perm_hash_gate` disparaît
    - mais la fenêtre se diffuse surtout en `reject_not_better_than_best` et `skip_zero_inliers`
    - sans réexposer rapidement un meilleur candidat aval propre
- Décision produit appliquée :
  - `blind_astrometry_strict_perm_hash_max_qdelta` passe par défaut de `1536` à `2048`
- Validation après patch :
  - `py_compile` OK
  - `pytest -q tests/test_zeblindsolver.py` => `28 passed`
  - probe produit :
    - `reports/r47i_s6_m106_first_divergence_audit_20260607_010959_product_permhash2048_default_20260607/`
  - lecture durable :
    - le front reste dominé par `reject_perm_hash_gate`, mais plus relâché :
      - `reject_perm_hash_gate=458`
      - `reject_not_better_than_best=34`
      - `skip_zero_inliers=20`
      - `accept_new_best=5`
    - le même candidat nominal aval reste visible (`resolve_hit_direct`, validation échouée `pairs=18`, `rms≈4.254`)
- Conclusion durable :
  - `perm_hash_gate` n'est pas un simple garde à supprimer ;
  - un relâchement modéré (`2048`) garde une structure causale lisible ;
  - la suppression totale n'est pas retenue comme direction produit par défaut.

## 2026-06-07 (S6, le garde de plausibilité `meta_seed` redevient le défaut produit)

- L'outil `tools/r47i_s6_m106_first_divergence_audit.py` a été enrichi avec une variante minimale :
  - `product_meta_seed_guard`
- Probe dédié :
  - `reports/r47i_s6_m106_first_divergence_audit_20260607_011916_product_meta_seed_guard_only_20260607/`
- Résultat durable utile :
  - le début du corridor reste identique sur `d50_2725` :
    - `candidate_try`
    - `collect_ready`
    - `astrometry_lookup_stage`
  - mais le vieux faux aval visible ne revient plus rapidement dans la fenêtre lue :
    - pas de `transition_dump` ni `validation_pairs` précoces pour `d50_2725`
    - l'`exact_trace` partielle écrite sur `d50_2725` est nettement plus courte (`228` lignes au moment de lecture, contre `517` sur la base produit comparable)
    - distribution lue :
      - `reject_perm_hash_gate=197`
      - `reject_not_better_than_best=15`
      - `skip_zero_inliers=12`
      - `accept_new_best=4`
  - le `phase_handoff` montre déjà que le budget recommence à passer au candidat suivant `d50_2822`.
- Lecture causale combinée avec la preuve antérieure du 2026-06-05 :
  - le garde `blind_astrometry_meta_seed_carry_plausibility_guard_enabled` coupe bien le carry permissif `resolve_hit_meta_seed` qui ramenait la fausse validation nominale `d50_2725`.
- Décision produit appliquée :
  - `blind_astrometry_meta_seed_carry_plausibility_guard_enabled` passe par défaut à `True`
- Validation après patch :
  - `py_compile` OK
  - `pytest -q tests/test_zeblindsolver.py` => `28 passed`
  - probe produit relancé :
    - `reports/r47i_s6_m106_first_divergence_audit_20260607_012238_product_meta_seed_guard_default_20260607/`
  - lecture utile au stade partiel :
    - le corridor produit démarre sainement sur `d50_2725`
    - aucun signal précoce de régression de câblage n'apparaît avant arrêt du probe
- Conclusion durable :
  - le faux carry permissif `meta_seed` n'est plus traité comme mode produit par défaut ;
  - le prochain front utile à traiter devient le premier candidat/résiduel réexposé après cette coupure, probablement `d50_2822` dans la fenêtre bornée actuelle.

## 2026-06-01 (S6, M106 end-to-end requalifie le risque produit)

- Un runner dédié de validation oracle M106 a été ajouté :
  - `tools/r47i_s6_m106_blind_oracle_compare.py`
  - rôle : copier les FITS du lot propre M106, retirer le WCS sur les copies, lancer ZeBlind courant, puis comparer directement aux WCS oracle par nom de fichier.
- Validation end-to-end M106 lancée en **baseline produit** sur le lot validé/oracle ASTAP :
  - au moins `5` FITS échantillonnés couvrant début et fin de série ont été exécutés ;
  - résultats observés :
    - `...232102.fit` -> `no valid solution` (~`98.6s`)
    - `...232144.fit` -> `no valid solution` (~`96.6s`)
    - `...232205.fit` -> `no valid solution` (~`83.4s`)
    - `...232247.fit` -> `no valid solution` (~`83.7s`)
    - `...233644.fit` -> `no valid solution` (~`106.4s`)
- Test ciblé complémentaire sur `...232102.fit` avec les **3 hypothèses S5.5 toutes activées** :
  - `skip_hinted_wide_reentry_after_hinted`
  - `skip_blind_carryover_after_scale_only`
  - `skip_blind_carryover_after_hinted`
  - résultat : toujours `no valid solution` (~`91.6s`).
- Conclusion durable :
  - la fermeture `S5` en probe-only strict ne se traduit pas encore en parité solve blind produit sur M106 ;
  - les trois hypothèses produit issues de `S5.5`, prises seules même cumulées, ne suffisent pas à rétablir un solve réel sur le case de tête M106 ;
  - `S6` doit donc être traité comme une vraie réouverture de l'écart end-to-end, pas comme une simple formalité de confirmation.

## 2026-06-02 (S6, l'écart sentinelle M106 se requalifie d'abord en enveloppe de config amont)

- Sur le case sentinelle `...232102.fit`, une ablation courte Ze-vs-Ze a confirmé que le tout premier drift amont ne vient pas du solveur profond tout entier mais d'abord de l'enveloppe de config utilisée :
  - baseline produit : `row0 = pairs_collected_initial=19`, `code_lookup_hits=33646`, `hypothesis=2`, `verify=0`, échec ;
  - variante `downsample=3` seule : `row0 = 9 / 12150`, `hypothesis=5`, `verify=6`, échec conservé ;
  - paquet `A_detect_amont` (`downsample`, `max_stars=180`, `max_quads=2800`, `max_candidates=20`, `detect_k_sigma=3.2`, `pixel_tolerance=6.0`) : `row0 = 3 / 10930`, soit alignement sur le probe strict, mais échec final encore présent.
- Lecture durable :
  - `single_pass_newpoint` n'explique pas le gap produit ;
  - le drift `code_lookup_hits / pairs_collected_initial` observé dès la première ligne est d'abord un **config envelope gap** amont ;
  - l'échec final sentinelle ne se réduit pas à ce drift-là, puisqu'il survit même une fois le front amont réaligné.
- Outillage consolidé :
  - `tools/r47i_s6_m106_first_divergence_audit.py` exporte désormais le diff de config amont pertinent et peut classer ce cas en `config_envelope_gap` quand le tuple/callsite de tête est partagé mais que l'enveloppe de config diverge.

## 2026-06-02 (S6, noyau canonique resserré: `native_verify + tosolve` suffit déjà)

- Micro-ablations sentinelle M106 poursuivies sur `...232102.fit` après séparation des paquets `A_detect_amont` et `B_guards_accept`.
- Résultat durable supplémentaire :
  - `blind_astrometry_native_verify_semantics_enabled=True` + `blind_anodds_tosolve=1e-3` suffit déjà à restaurer un succès sentinelle en enveloppe produit :
    - `solution found (level=S, parity=mirror) (phase=hinted)`
    - `row0` reste produit : `pairs_collected_initial=19`, `code_lookup_hits=33646`
  - `blind_prob_verify_enabled=False` n'est pas suffisant, même combiné avec la famille `blind_anodds_*`.
- Lecture causale durable :
  - le vrai levier côté `G2` est `native_verify_semantics`, pas la simple désactivation `prob_verify`;
  - le singleton `anodds_tosolve` est particulièrement puissant car, en mode Astrometry strict, il rabaisse aussi les seuils dérivés `tokeep` puis `toprint` via les clamps internes.
- Rejeu avec drift amont réaligné :
  - `native_verify + tosolve + A_detect_amont` réussit aussi ;
  - `row0` retombe alors sur le probe (`pairs_collected_initial=3`, `code_lookup_hits=10930`).
- Conclusion durable :
  - le succès sentinelle peut être rétabli **sans** corriger d'abord `A_detect_amont` ;
  - `A_detect_amont` reste donc un écart d'enveloppe amont séparé, pas le bloqueur causal principal du solve sur ce case.

## 2026-06-02 (S6, le singleton gagnant est réel mais trop permissif en multicase)

- Vérification des singletons restants sur le sentinelle `...232102.fit` :
  - `native_verify + totune` : échec
  - `native_verify + tokeep` : échec
  - `native_verify + toprint` : échec
- Conséquence durable :
  - `native_verify + tosolve` est bien le **singleton minimal unique** observé à ce stade sur le sentinelle.
- Généralisation courte sur `5` FITS M106 anciennement en échec baseline (`232102`, `232144`, `232205`, `232247`, `233644`) :
  - avec `native_verify + tosolve` seul :
    - `5/5 success`
    - `5/5 false_positive` contre l'oracle WCS
  - avec `native_verify + tosolve + A_detect_amont` :
    - encore `5/5 success`
    - encore `5/5 false_positive`
    - mais avec des dérives centre/échelle souvent moins extrêmes.
- Lecture durable :
  - le noyau minimal rétablit une **acceptation**, pas encore une **solution correcte** ;
  - `A_detect_amont` aide à calmer le front amont, mais ne règle pas la permissivité de l'acceptation.
- Direction pratique validée :
  - le prochain cran utile n'est plus d'ouvrir davantage le solveur ;
  - c'est de **resserrer le corridor d'acceptation** autour de `native_verify + tosolve`, probablement via un `tosolve` moins permissif ou un garde-fou compatible `native_verify`.

## 2026-06-02 (S6, M1 refermé proprement sur le chemin `native_verify`)

- Le correctif `M1` a été implémenté de façon volontairement étroite dans `zeblindsolver.py` :
  - en mode `blind_astrometry_native_verify_semantics_enabled=True`, la résolution initiale d'échelle ne laisse plus `config.pixel_scale_arcsec` primer devant les indices locaux ;
  - l'ordre de priorité a été resserré vers les hints locaux (`header`, `range hint`, optique), puis seulement la config globale ;
  - une ancre locale `candidate_pairset_local` peut désormais réaligner `scale_anchor_arcsec` avant la verify sur le chemin canonique native.
- L'observabilité de provenance a été ajoutée dans `scale_policy` :
  - `approx_scale_source`
  - `scale_anchor_initial_source`
  - `scale_anchor_current_source`
- Le point dépendant `M2` a été traité sans élargir la portée :
  - l'interdiction du fallback `empty_inliers_fallback_all_finite` est maintenant centralisée et testée explicitement sur le chemin `native_verify` ;
  - aucun changement n'a été propagé au strict/product non-native à ce stade, pour éviter une dérive latérale non auditée.
- Vérification accomplie :
  - `py_compile` OK ;
  - tests ciblés `resolve_scale_arcsec / pairset_local_scale_summary / empty_inliers_fallback / onefield` passés ;
  - replay sentinelle `reports/r47i_s6_false_positive_logodds_validity_audit_20260602_184636_sentinel232102_m1_native_anchor/summary.json` :
    - `success=false`
    - `message=no valid solution`
    - `oracle_status=fail_no_wcs`
- Conclusion durable :
  - le patch `M1` ne réouvre pas le faux positif historique sur `...232102.fit` ;
  - la prochaine vérification utile n'est plus “est-ce que ça réaccepte à tort ?”, mais “est-ce que `candidate_pairset_local` est bien la source réellement consommée sur le candidat causal quand elle existe ?”.

## 2026-06-02 (S6, la source locale est consommée mais l'ancre reste partagée entre candidats)

- Un nouvel audit dédié a été ajouté :
  - `tools/r47i_s6_scale_anchor_source_audit.py`
- Rejeu sentinelle publié :
  - `reports/r47i_s6_scale_anchor_source_audit_20260602_200546_sentinel232102_anchor_source_d502725/summary.json`
- Résultat durable :
  - `candidate_pairset_local` n'est pas ignoré ; il est bien consommé sur le chemin `native_verify`.
  - Mais l'ancre d'échelle finale reste **globale et mutable** à l'échelle du run, donc elle peut être écrasée par un autre candidat plus tard.
- Lecture causale utile sur le sentinelle :
  - pour `d50_2725`, le pairset local publie `scale_anchor_arcsec ≈ 9.7089846` avec `scale_anchor_source = candidate_pairset_local`;
  - pourtant la `scale_policy` finale du run finit à `scale_anchor_current_arcsec ≈ 24.2138844`, également marquée `candidate_pairset_local`;
  - cette valeur finale correspond en pratique au pairset local d'un autre candidat du même run (`d50_2823`), pas à `d50_2725`.
- Conclusion durable :
  - le vrai résiduel après `M1` n'est plus "source locale ignorée" ;
  - c'est désormais un problème plus précis d'**ancre locale partagée entre candidats** ;
  - la prochaine correction utile doit donc scoper l'ancre par candidat/tentative avant toute propagation au strict non-native.

## 2026-06-02 (S6, la contamination inter-candidats est fermée, mais l'état final du run ne suffit pas pour lire la consommation locale)

- Le chemin `native_verify` a été refermé sur ce point :
  - l'ancre d'échelle de travail est maintenant réinitialisée par candidat/tentative, au lieu de rester mutable à l'échelle du run.
- Rejeu sentinelle après ce scoping :
  - `reports/r47i_s6_scale_anchor_source_audit_20260602_221106_sentinel232102_anchor_source_d502725_scoped/summary.json`
- Résultat durable :
  - la contamination inter-candidats disparaît ;
  - mais, sur un run sans solution finale, la `scale_policy` exportée en fin de run retombe sur l'ancre initiale `header_scale_arcsec`, donc elle ne permet plus à elle seule de conclure si `candidate_pairset_local` a été réellement consommé pendant la tentative du candidat causal.
- Pour lever cet angle mort, l'observabilité a été étendue dans `hypothesis_probe_trace` avec :
  - `approx_scale_arcsec/source`
  - `scale_anchor_arcsec/source`
- Audit candidate-level publié :
  - `reports/r47i_s6_scale_anchor_source_audit_20260602_221809_sentinel232102_anchor_source_d502725_scoped_probe/summary.json`
- Lecture durable sur `d50_2725` :
  - classification : `pairset_local_consumed_candidate_trace_only`
  - l'ancre locale `candidate_pairset_local` est bien consommée pendant les tentatives du candidat causal ;
  - les lignes `hypothesis_probe_trace` montrent `scale_anchor_arcsec ≈ 9.7089846` et `scale_anchor_source = candidate_pairset_local` ;
  - le résumé final du run restant à `header_scale_arcsec ≈ 2.392674...` est donc un **angle mort d'observabilité finale**, pas une preuve que `M1` serait ignoré.
- Conclusion durable :
  - la piste “M1 non consommé” est désormais fermée ;
  - le prochain résiduel utile à traiter redevient un problème de transform / géométrie d'inliers sur `d50_2725`, pas un problème de provenance d'ancre.

## 2026-06-02 (S6, le résiduel canonique devient une incohérence seed-vs-residual sur `d50_2725`)

- Audit ciblé publié :
  - `tools/r47i_s6_effective_fit_geometry_audit.py`
  - `reports/r47i_s6_effective_fit_geometry_audit_20260602_223944_sentinel232102_d502725_fitgeom/summary.json`
- Résultat durable :
  - la classification utile est `plausible_scale_but_geometry_no_inliers`
  - `d50_2725` échoue avant `verify`, avec `0` ligne `verify_hit_trace`
  - le ratio `model_scale / scale_anchor_local` n'est pas la cause dominante :
    - nominal `≈ 1.0737`
    - mirror `≈ 1.1742`
  - en revanche les résidus du fit explosent immédiatement par rapport à la tolérance `2.5 px` :
    - nominal : `residual_px_med ≈ 26.66`, `max ≈ 542.94`
    - mirror : `residual_px_med ≈ 949.11`, `max ≈ 1233.75`
- Point causal à retenir :
  - la même trace indique pourtant `transform_origin_source = astrometry`
  - avec un seed apparemment “bon” :
    - nominal : `transform_origin_inliers = 8`, `transform_origin_rms_px ≈ 1.85`
    - mirror : `transform_origin_inliers = 4`, `transform_origin_rms_px ≈ 1.28`
  - le résiduel vivant n'est donc plus un problème d'ancre, ni même d'échelle brute ;
  - c'est une **incohérence entre le seed de transform retenu et le contrôle résiduel final** ensuite appliqué dans Ze.
- Lecture source utile :
  - côté Astrometry C, le seed passe par `MatchObj.quadpix/quadxyz` puis entre dans `verify_hit()`
  - côté Ze, le candidat `d50_2725` casse avant l'étape équivalente de `verify_hit`
  - la prochaine comparaison source doit viser le portage du transform “astrometry” et sa continuité jusqu'au test résiduel, pas la logique probabiliste aval.

## 2026-06-02 (S6, l'incohérence seed-vs-residual est maintenant localisée à un changement de pool)

- Audit ciblé publié :
  - `tools/r47i_s6_astrometry_seed_residual_incoherence_audit.py`
  - `reports/r47i_s6_astrometry_seed_residual_incoherence_audit_20260602_230608_sentinel232102_d502725_seed_residual_forced/summary.json`
- Résultat durable sur `d50_2725` :
  - classification : `astrometry_seed_residual_incoherence`
  - le seed `astrometry` retenu est réellement bon dans son propre régime d'évaluation :
    - `resolve_hit_source = resolve_hit`
    - `resolve_hit_kept = 38`
    - `inliers_final = 8`
    - `rms_px_final ≈ 1.84699`
    - `model_scale_arcsec_final ≈ 12.11577`
  - le screen aval Ze retombe pourtant ensuite sur :
    - `8` paires globales seulement
    - `0` inlier
    - `residual_px_med ≈ 26.66`
    - `residual_px_max ≈ 542.94`
- Conclusion durable :
  - le seed n'est pas le premier coupable ;
  - l'incohérence vivante est maintenant localisée dans un **changement de pool/correspondance** entre :
    - le seed `astrometry` scoré sur `resolve_hit`
    - et le contrôle résiduel final Ze recomputé sur `img_points/tile_points` globaux
  - la prochaine correction utile doit préserver le même pool/correspondance jusqu'au screen résiduel quand `transform_origin_source = astrometry`, avant toute reprise des réglages produit.

## 2026-06-05 (S6, l'audit `matched-envelope` devient un outil borné rejouable)

- Après réévaluation de trajectoire, le sentinelle M106 n'a pas encore livré de divergence amont interprétable entre `product` et `probe` :
  - l'audit `reports/r47i_s6_m106_first_divergence_audit_20260605_104603_loop_reaudit_20260605/` retombe toujours en `config_envelope_gap`.
- Pour éviter de rester bloqué sur des one-liners inline lents et opaques, `tools/r47i_s6_m106_first_divergence_audit.py` a été durci :
  - sélection de variantes via `--variants product,matched_probe,probe`
  - budgets runtime via `--hard-max-candidates` et `--hard-max-validations`
  - publication d'un artefact JSON par variante (`<variant>_result.json`) même si le run complet n'est pas allé au bout.
- Rejeu canonique micro publié :
  - `python3 tools/r47i_s6_m106_first_divergence_audit.py --label matched_envelope_micro_cli_20260605 --variants product,matched_probe --trace-max 8 --hard-max-candidates 1 --hard-max-validations 2`
  - artefact : `reports/r47i_s6_m106_first_divergence_audit_20260605_110512_matched_envelope_micro_cli_20260605/`
- Lecture durable déjà acquise sans attendre la fin de `matched_probe` :
  - `product_result.json` confirme que la tête bornée reste `hinted/S/nominal/d50_2725` avec enveloppe produit (`phase_local_rank`, `33646`, `19`) ;
  - `matched_probe` reste nettement plus coûteux que `product` même sous cap `1/2`, donc la suite doit être pilotée comme audit borné outillé, pas comme boucle interactive d'attente.

## 2026-06-05 (S6, le `matched-envelope` réaligne bien la tête; le drift vivant descend d'un cran)

- L'outil `tools/r47i_s6_m106_first_divergence_audit.py` a été encore durci pour publier aussi, par variante, des artefacts **live** utilisables avant la fin du solve :
  - `<variant>_runtime_cfg.json`
  - `<variant>_meta_seed_probe.json`
  - `<variant>_astrometry_exact_trace.raw.json`
  - `<variant>_validation_pairs.json`
- Sur le micro-run aligné :
  - dossier : `reports/r47i_s6_m106_first_divergence_audit_20260605_110512_matched_envelope_micro_cli_20260605/`
  - `product_result.json` garde en tête :
    - `hinted / nominal / d50_2725`
    - `candidate_vote_score = 33646`
    - `pairs_collected_initial = 19`
  - `matched_probe_meta_seed_probe.json` montre déjà, pendant l'exécution :
    - `phase = hinted`
    - `parity = nominal`
    - `tile = d50_2725`
    - `score = 33646`
    - puis un carry `resolve_hit_meta_seed_carried` à `38` paires
  - `matched_probe_astrometry_exact_trace.raw.json` démarre aussi sur `hinted / nominal / d50_2725`.
- Conclusion durable :
  - le gros drift de tête vu sur l'ancien `probe` (`33646 -> 10930`, `19 -> 3`) n'est pas le résiduel causal sous le sous-espace `matched-envelope` ;
  - une fois l'enveloppe réalignée, la divergence utile descend **après** la sélection du top-candidat `d50_2725`, probablement dans le corridor `prescreen / resolve_hit / verify / accept`.

## 2026-06-05 (S6, la première divergence utile sous `matched-envelope` est un saut `hinted -> hinted_wide`)

- Un deuxième run live dédié `product` a été publié :
  - `reports/r47i_s6_m106_first_divergence_audit_20260605_1127_product_live_probe/`
- Un diff synthétique a été produit :
  - `reports/r47i_s6_m106_first_divergence_audit_20260605_110512_matched_envelope_micro_cli_20260605/checkpoint_drift_compare.json`
- Lecture durable :
  - même tête alignée amont :
    - `product_head0 = hinted / nominal / d50_2725 / score=33646`
    - `matched_probe_first_live = hinted / nominal / d50_2725 / score=33646`
  - mais `product` diverge dès le **premier checkpoint live** :
    - `product_first_live = hinted_wide / nominal / d50_2726 / score=14108`
    - `product_first_validation = hinted_wide / nominal / d50_2726 / pairs=23 / rms≈4.181`
  - en face, `matched_probe` reste sur le corridor attendu :
    - `hinted / nominal / d50_2725`
    - `first_validation = pairs=38 / rms≈6.166`
- Conclusion durable :
  - sous `matched-envelope`, la première divergence exploitable n'est plus un `config_envelope_gap` ni un simple drift de score ;
  - c'est maintenant un **`wiring_gap` inter-phase / inter-tuile** :
    - le produit saute de `hinted/d50_2725` vers `hinted_wide/d50_2726` avant validation, alors que `matched_probe` reste sur le top-candidat publié.

## 2026-06-05 (S6, le `wiring_gap` se resserre entre `collect` et `resolve_hit`)

- Instrumentation d'audit ajoutée dans le solveur et branchée dans le runner :
  - `blind_phase_handoff_dump_path`
  - `blind_collect_matches_exact_dump_path`
  - `blind_astrometry_transition_dump_path`
- Artefacts publiés :
  - `reports/r47i_s6_m106_first_divergence_audit_20260605_1145_product_handoff_probe/`
  - `reports/r47i_s6_m106_first_divergence_audit_20260605_1157_product_transition_probe/`
  - synthèse :
    - `reports/r47i_s6_m106_first_divergence_audit_20260605_1157_product_transition_probe/hinted_d2725_collect_to_transition_gap.json`
- Lecture durable :
  - `product` partage bien avec `matched_probe` tout le préfixe utile :
    - `phase_start(hinted)`
    - `phase_variant_select(first_pass)`
    - `candidate_order_head`
    - `candidate_try(hinted / nominal / d50_2725 / score=33646)`
  - `_collect_tile_matches` confirme ensuite pour ce même candidat :
    - `stage=scalar_final`
    - `pairs_selected=19`
    - `votes_total=35`
  - malgré cela, `product` n'émet ensuite ni :
    - `resolve_hit transition` pour `hinted/d50_2725`
    - ni `exact_trace` pour `hinted/d50_2725`
  - la première transition/exact-trace visible repart directement sur :
    - `hinted_wide / nominal / d50_2726`
- Conclusion durable :
  - le résiduel vivant n'est plus “saut de phase” au sens large ;
  - il est désormais localisé plus finement dans le corridor **post-collect / pré-resolve_hit** du chemin `product` sur `hinted/d50_2725`.

## 2026-06-01 (S5.10/S5.11, strict28 refermé en probe-only)

- Validation multicase stricte relancée sur le pool complet couvert (`28` FITS) :
  - strict28 initial :
    - `reports/r47i_s5_upstream_trace_probe_20260601_2205_strict28/summary.json`
    - `reports/r47i_s5_multicase_head_residual_audit_20260601_221221/summary.json`
    - `reports/r47i_s5_upstream_diff_audit_20260601_2212/summary.json`
  - strict28 bis après ciblage `hinted -> blind` :
    - `reports/r47i_s5_upstream_trace_probe_20260601_2227_strict28_hintedblind/summary.json`
    - `reports/r47i_s5_multicase_head_residual_audit_20260601_223243/summary.json`
    - `reports/r47i_s5_upstream_diff_audit_20260601_2232/summary.json`
- Résultat durable :
  - strict28 initial : `28/28` FITS résolus, `27/28` têtes entièrement C-like ;
  - strict28 bis : `28/28` FITS résolus, `0/28` résiduel de tête ;
  - `S5.10` est donc fermé sur ce protocole probe-only strict.
- Résiduel strict-lot ciblé puis refermé :
  - FITS : `...232924.fit`
  - candidat : `d50_2823`
  - séquence utile : `hinted:first_pass(nominal) -> blind:first_pass(nominal)` sur le même `(candidate, level)`
  - audits ciblés :
    - `reports/r47i_s5_tuple_semantic_diff_audit_d50_2823_20260601_221304/summary.json`
    - `reports/r47i_s5_same_candidate_transition_audit_d50_2823_20260601_221304/summary.json`
- Outillage probe-only prolongé :
  - ajout du flag expérimental `blind_s5_skip_blind_carryover_after_hinted_enabled`
  - plomberie CLI associée dans `tools/r47i_s5_upstream_trace_probe.py`
- Vérification clé :
  - la fermeture single-fit est confirmée par `reports/r47i_s5_multicase_head_residual_audit_20260601_222545/summary.json`
  - `d50_2625` peut remonter ensuite, mais il est classé `c_like_parity_pair`, pas résiduel causal de tête.
- Conclusion durable :
  - le dernier résiduel strict-lot visible relevait encore d’un carryover d’orchestration Ze `hinted -> blind`, pas d’un nouveau manque de lecture du source Astrometry ;
  - `S5.11` est fermé sur ce protocole probe-only strict ;
  - la question restante n’est plus d’isoler un nouveau résiduel strict, mais de décider quels skips probe-only méritent ou non une hypothèse de correction produit.

## 2026-06-01 (S5.5, tri des skips probe-only)

- Audit de décision publié :
  - `reports/r47i_s5_skip_hypothesis_audit_20260601_234255/summary.md`
  - `reports/r47i_s5_skip_hypothesis_audit_20260601_234255/summary.json`
- Résultat durable :
  - trois flags ont un support causal multicase propre via fermeture par delta à un seul flag :
    - `skip_hinted_wide_reentry_after_hinted`
    - `skip_blind_carryover_after_scale_only`
    - `skip_blind_carryover_after_hinted`
  - ils deviennent des **hypothèses de correction produit** crédibles, mais restent OFF par défaut tant qu’ils ne sont pas validés hors probe-only.
- Le reste de la batterie S5 est reclassé en **compression-only / investigation-only** :
  - `skip_blind_reentry_after_scale_only`
  - `skip_hinted_widened_reentry_after_first_pass`
  - `skip_scale_only_widened_reentry_after_first_pass`
  - `skip_blind_widened_reentry_after_first_pass`
  - `skip_blind_carryover_after_hinted_wide`
  - `skip_hinted_widened_only_tuples`
  - `skip_scale_only_widened_only_tuples`
  - `skip_blind_widened_only_tuples`
- Lecture durable :
  - les flags retenus comme hypothèses produit sont tous des re-émissions inter-phase `first_pass` sur le même `(candidate, level)` ;
  - les flags relégués servent surtout à compresser des widened-pass ou des sentinelles locales pour rendre le diff C-vs-Ze lisible ;
  - l’investigation S5 amont peut donc être considérée close côté probe-only, avec relais naturel vers `S6`.

## 2026-06-01 (S5, audit multicase fiabilisé puis broad12 refermé à 12/12)

- Les audits multicase ciblés ont été corrigés :
  - `tools/r47i_s5_tuple_semantic_diff_audit.py`
  - `tools/r47i_s5_same_candidate_transition_audit.py`
  - ils acceptent désormais `--result-index` et `--fits-substring` au lieu de retomber silencieusement sur `results[0]`.
- Conséquence immédiate :
  - le résiduel large-lot `d50_2824` a été requalifié sur le vrai FITS `...233356.fit` :
    - `reports/r47i_s5_tuple_semantic_diff_audit_d50_2824_20260601_190506/summary.json`
    - `reports/r47i_s5_same_candidate_transition_audit_d50_2824_20260601_190506/summary.json`
  - lecture durable corrigée :
    - carryover `scale_only:first_pass(mirror) -> blind:first_pass(nominal)` ;
    - même `candidate_key`, même `level`, même `perm_head` ;
    - pas un widened-pass.
- Nouveau flag probe-only ajouté dans `zeblindsolver.py` :
  - `blind_s5_skip_blind_carryover_after_scale_only_enabled`
  - plomberie CLI dans `tools/r47i_s5_upstream_trace_probe.py` :
    - `--skip-blind-carryover-after-scale-only`
- Vérification single-fit utile :
  - sur `...233356.fit`, activer ce skip retire bien `d50_2824` de la tête sans casser le solveur ; le succès retombe sur `scale_only`.
- Replay broad12 publié :
  - `reports/r47i_s5_upstream_trace_probe_20260601_1907_broad12_scaleblind/summary.json`
  - `reports/r47i_s5_multicase_head_residual_audit_20260601_191051/summary.json`
  - `reports/r47i_s5_upstream_diff_audit_20260601_1910/summary.json`
- Résultat durable :
  - `12/12` FITS résolus ;
  - `0/12` résiduel de tête ;
  - audit multicase `closed`.
- Conclusion durable :
  - avec l’audit multicase fiabilisé et le skip `scale_only -> blind`, le broad12 est fermé sur ce protocole probe-only ;
  - le prochain cran utile n’est plus un audit source Astrometry générique, mais une validation multicase stricte.

## 2026-06-01 (S5, validation multicase large après fermeture de d50_2823)

- Validation multicase élargie relancée sur `12` FITS avec la batterie probe-only cumulée :
  - `reports/r47i_s5_upstream_trace_probe_20260601_1814_broad12/summary.json`
  - `reports/r47i_s5_multicase_head_residual_audit_20260601_182050/summary.json`
  - `reports/r47i_s5_upstream_diff_audit_20260601_1820/summary.json`
- Résultat durable :
  - `12/12` FITS résolus ;
  - `11/12` têtes entièrement C-like ;
  - diversité des phases terminales conservée (`scale_only=6`, `hinted=2`, `hinted_wide=2`, `blind=2`).
- Nouveau résiduel survivant isolé :
  - `d50_2824`
  - classe `candidate_carryover`
  - `blind:first_pass`
  - présent comme premier résiduel sur `1/12` FITS (`...233356.fit`).
- Qualification ciblée publiée :
  - `reports/r47i_s5_tuple_semantic_diff_audit_d50_2824_20260601_182134/summary.json`
  - `reports/r47i_s5_same_candidate_transition_audit_d50_2824_20260601_182134/summary.json`
- Lecture durable :
  - le lot large ne réouvre ni le résiduel `d50_2823` ni un widened-pass dominant ;
  - le survivant `d50_2824` est un carryover `scale_only:first_pass(mirror) -> blind:first_pass(nominal)`, avec `perm_head` stable ;
  - la prochaine itération utile doit cibler `d50_2824`, pas rouvrir un audit générique du source Astrometry.

## 2026-06-01 (S5, skip probe-only hinted widened-only)

- Nouveau flag expérimental ajouté dans `zeblindsolver.py` :
  - `blind_s5_skip_hinted_widened_only_tuples_enabled`
  - portée volontairement étroite : mode opt-in, saut des tuples `hinted:widened_pass` sans équivalent `hinted:first_pass`, sans changer le comportement produit par défaut.
- Le probe S5 sait maintenant activer ce mode via `tools/r47i_s5_upstream_trace_probe.py --skip-hinted-widened-only-tuples`.
- Audit de qualification du dernier widened-only avant patch :
  - `reports/r47i_s5_tuple_semantic_diff_audit_d50_2823_20260601_174342/summary.json`
  - `reports/r47i_s5_same_candidate_transition_audit_d50_2823_20260601_174342/summary.json`
  - lecture durable : `d50_2823` n'était plus un same-candidate transition, mais un pur `hinted:widened_pass` sans contrepartie `first_pass`.
- Expérience cumulée publiée :
  - `reports/r47i_s5_upstream_trace_probe_20260601_1745/summary.json`
  - `reports/r47i_s5_remaining_head_order_audit_20260601_174555/summary.json`
- Résultat durable :
  - la tête de trace tombe de `10` à `8` lignes ;
  - `d50_2823` disparaît bien comme widened-only opportuniste ;
  - `d50_2822` ne reste plus qu'en `hinted:first_pass` ;
  - il ne reste plus qu'un seul résiduel non C-like : `d50_2725` en carryover `hinted_wide:first_pass(nominal) -> blind:first_pass(mirror)`.
- Conséquence durable :
  - la compression probe-only S5 est presque au plancher sur ce FITS ;
  - le prochain diff utile ne porte plus sur un widened-pass, mais sur le dernier carryover inter-phase/inter-parité.

## 2026-06-01 (S5, d50_2725 devient le dernier résiduel)

- Artefacts publiés :
  - `reports/r47i_s5_tuple_semantic_diff_audit_d50_2725_20260601_174627/summary.json`
  - `reports/r47i_s5_same_candidate_transition_audit_d50_2725_20260601_174627/summary.json`
- Lecture durable :
  - `d50_2725` apparaît deux fois seulement dans la tête `1745` :
    - `hinted_wide:first_pass` / `nominal`
    - `blind:first_pass` / `mirror`
  - `code_lookup_hits` et `pairs_collected_initial` restent identiques (`3005`, `52`) ;
  - le différentiel restant porte surtout sur `phase`, `parity` et `use_ra_filter`.
- Conséquence durable :
  - le prochain questionnement S5 n'est plus “faut-il supprimer un widened-pass ?” ;
  - c'est “ce carryover `hinted_wide -> blind` du même candidat est-il encore un artefact Ze ou déjà une divergence acceptable/attendue vis-à-vis du `solver.c` ?”.

## 2026-06-01 (S5, fermeture same-FITS complète en probe-only)

- Nouveau flag expérimental ajouté dans `zeblindsolver.py` :
  - `blind_s5_skip_blind_carryover_after_hinted_wide_enabled`
  - portée volontairement étroite : mode opt-in, saut des tuples `blind:first_pass` déjà vus en `hinted_wide:first_pass` pour le même `(candidate_key, level)`, sans changer le comportement produit par défaut.
- Le probe S5 sait maintenant activer ce mode via `tools/r47i_s5_upstream_trace_probe.py --skip-blind-carryover-after-hinted-wide`.
- Expérience cumulée publiée :
  - `reports/r47i_s5_upstream_trace_probe_20260601_1753/summary.json`
  - `reports/r47i_s5_remaining_head_order_audit_20260601_175404/summary.json`
- Résultat durable :
  - la tête same-FITS du sentinelle M106 est désormais entièrement classée C-like par `remaining_head_order_audit` (`residual_row_count=0`) ;
  - `d50_2725` disparaît comme dernier carryover inter-phase visible ;
  - un nouveau candidat `d50_2625` remonte en fin de tête, mais sans ouvrir de résiduel non C-like dans ce protocole.
- Audit complémentaire publié :
  - `reports/r47i_s5_stage_callsite_mapping_audit_20260601_1754/summary.json`
  - lecture durable : `phase_order_break_count=0`, `callsite_contract_break_count=0`.

## 2026-06-01 (S5, smoke multicase court après fermeture same-FITS)

- Nouvel outil ajouté : `tools/r47i_s5_multicase_head_residual_audit.py`.
  - but : classer, par FITS, la tête résiduelle d’un probe S5 multi-résultats et isoler le résiduel dominant qui généralise réellement.
- Smoke multicase court relancé sur `3` FITS avec le protocole probe-only maximal :
  - `reports/r47i_s5_upstream_trace_probe_20260601_1755/summary.json`
  - `reports/r47i_s5_multicase_head_residual_audit_20260601_175730/summary.json`
  - `reports/r47i_s5_upstream_diff_audit_20260601_1756/summary.json`
- Résultat durable :
  - `3/3` FITS résolus ;
  - `1/3` FITS reste entièrement C-like sur la tête ;
  - `2/3` FITS rouvrent le même premier résiduel : `d50_2823`, classe `same_tuple_reentry`, en `hinted_wide:first_pass`.
- Lecture durable :
  - le nettoyage same-FITS sur M106 est réel, mais pas encore complètement généralisé ;
  - le prochain vrai sujet S5 n’est plus `d50_2725`, mais la réémission multicase `hinted:first_pass -> hinted_wide:first_pass` du tuple `d50_2823`.

## 2026-06-01 (S5, d50_2823 ciblé puis multicase court refermé)

- Nouveau flag expérimental ajouté dans `zeblindsolver.py` :
  - `blind_s5_skip_hinted_wide_reentry_after_hinted_enabled`
  - portée volontairement étroite : mode opt-in, saut des tuples `hinted_wide:first_pass` déjà vus en `hinted` pour le même `(candidate_key, parity, level)`, sans changer le comportement produit par défaut.
- Le probe S5 sait maintenant activer ce mode via `tools/r47i_s5_upstream_trace_probe.py --skip-hinted-wide-reentry-after-hinted`.
- Lecture causale validée avant patch :
  - le résiduel dominant multicase `1755` était bien `d50_2823`, classe `same_tuple_reentry`, séquence `hinted:first_pass -> hinted_wide:first_pass`, sur `2/3` FITS.
- Smoke multicase court relancé après ajout du skip :
  - `reports/r47i_s5_upstream_trace_probe_20260601_1805/summary.json`
  - `reports/r47i_s5_multicase_head_residual_audit_20260601_180609/summary.json`
  - `reports/r47i_s5_upstream_diff_audit_20260601_1806/summary.json`
- Résultat durable :
  - `3/3` FITS résolus ;
  - `0/3` FITS avec résiduel de tête ;
  - l’audit multicase passe à `status=closed`.
- Conclusion durable :
  - le next-step “diff C-vs-Ze sur la réémission `hinted -> hinted_wide` de `d50_2823`” était bien le bon ;
  - il a permis d’identifier un problème d’orchestration Ze, pas un nouveau besoin d’audit générique du source Astrometry.

## 2026-06-01 (compaction documentaire du backlog)

- `followup.md` a été réécrit en version compacte pour redevenir un vrai plan vivant :
  - S1/S2/S3/S4 résumés en statut synthétique,
  - S5 recentré sur le seul résiduel amont utile (`d50_2822` en `hinted:widened_pass`),
  - l'historique détaillé fermé retiré du backlog actif.
- Décision durable de doc :
  - `followup.md` doit rester court et orienté prochain delta causal,
  - les chronologies fermées et les expérimentations déjà démontrées vivent dans `memory.md` + `reports/`, pas dans le backlog courant.

## 2026-06-01 (S5, d50_2822 précisé par audit same-candidate)

- Correctif outillage appliqué dans `tools/r47i_s5_tuple_semantic_diff_audit.py` :
  - les rangs `candidate_rank_global=0` et `newpoint=0` ne sont plus écrasés par des fallbacks faux ;
  - l'audit détecte maintenant explicitement les transitions intra-phase `first_pass -> widened_pass`.
- Artefact propre publié : `reports/r47i_s5_tuple_semantic_diff_audit_d50_2822_20260601_173403/summary.json`.
- Lecture durable obtenue sur la tête `1315` :
  - `d50_2822` n'est plus un vague résiduel “hinted” ;
  - le premier écart utile est précisément `hinted:first_pass(S) -> hinted:widened_pass(S)` ;
  - sur cette transition, le `perm_head` reste identique et seul `levels_to_use` s'élargit (`[S] -> [S,M]`).
- Conséquence durable :
  - le prochain diff S5 doit comparer cette transition intra-phase même-candidat avant de rouvrir des questions plus larges de réentrée inter-phase.

## 2026-06-01 (S5, audit transitionnel d50_2822)

- Nouvel outil ajouté : `tools/r47i_s5_same_candidate_transition_audit.py`.
  - but : publier les deltas exacts entre lignes consécutives d'un même tuple S5, sans rebrancher l'instrumentation runtime.
- Artefact publié : `reports/r47i_s5_same_candidate_transition_audit_d50_2822_20260601_173552/summary.json`.
- Lecture durable :
  - transition 1 utile = `rank 0 -> 1`, `first_pass:S -> widened_pass:S`, même `perm_head`, widening `[S] -> [S,M]`, `pairs_collected_initial 57 -> 59`;
  - transition 2 = `rank 1 -> 2`, toujours `widened_pass` mais `level S -> M`, `code_lookup_hits 2797 -> 1`, avec changement de `perm_head`.
- Conséquence durable :
  - le premier diff causal à mener dans le code n'est pas le saut vers `level M`, mais bien le passage `first_pass:S -> widened_pass:S` dans `hinted`.

## 2026-05-24 (S1 forced replay harness, critères S1 systématiques)

- Patch ciblé `tools/r47i_s1_forced_payload_full_replay.py` pour rendre la sortie `summary` robuste même quand OFF termine en échec global:
  - extraction systématique des métriques S1 depuis `refpool_trace` / `forensic_rows`,
  - fallback `verify_entry_nt/nr` via forensic (`prob_verify_nt/nr`) si `hit_off` absent,
  - publication explicite dans `summary.{json,md}` de `prob_distractor_rate_effective`, `ror_enabled/applied`, `prob_match_nsig2_cap`.
- Rejeu validé: `reports/r47i_s1_forced_payload_full_replay_case055_20260524_1145/summary.{json,md}`.
  - ON/OFF alignés au terminal: `reason_code=accept_logodds_gate`, `accept_logodds=11.010719844475592`, `verify_entry_nt/nr=51/12`.
  - Critères S1 visibles côté ON/OFF: `distractor≈0.25`, `RoR=true/true`, `nsig2_cap=25.0`.
- `first_divergence.key=None` sur ce case.

## 2026-06-01 (S5 canonical upstream, filtre code-continuous neutralisé)

- Patch causal appliqué dans `zeblindsolver.py` pour la parité amont S5:
  - nouveau `canonical_disable_code_continuous_filter` actif en `astrometry_native_verify_semantics_mode`,
  - effet: le préfiltre Ze `blind_astrometry_code_continuous_filter_enabled` n’élimine plus les candidats sur le chemin canonique natif.
- Validation syntaxique: `python3 -m py_compile zeblindsolver/zeblindsolver.py zeblindsolver/verify.py zesolver.py` OK.
- Probe same-FITS publié: `reports/r47i_s5_upstream_trace_probe_20260601_1201/summary.json`.
  - Changement observé: le case de référence se ferme désormais avec `success=true` et `message=... phase=scale_only` sur ce protocole S5.
  - Le rejet antérieur `perm_null_reason=code_continuous_log_delta_reject` disparaît de la tête de trace.
- Instrumentation complémentaire ajoutée juste après dans `zeblindsolver.py`:
  - `perm_null_reason="skip_no_seed_slices"` sur le `continue` où aucun `obs_slices` ne survit avant permutation.
- Probe final post-instrumentation publié: `reports/r47i_s5_upstream_trace_probe_20260601_1203/summary.json`.
  - Résultat durable: `pre_perm_n=5`, tous concentrés sur `pass_tag='widened_pass'` avec `perm_null_reason='skip_no_seed_slices'`.
- Instrumentation seed-path ajoutée ensuite dans `zeblindsolver.py` et probe publié: `reports/r47i_s5_upstream_trace_probe_20260601_1208/summary.json`.
  - Constat clé: sur les lignes `pre_perm`, `seed_code_hit_count=0`, `seed_ordered_hit_count=0`, `seed_strict_recover_attempted=true`, `seed_strict_recover_added=0`.
  - Cause racine: en mode canonique, désactiver le filtre continu coupait aussi le calcul de `obs_code`, donc plus de matière pour le `rangesearch`/`strict_hash_recover`.
- Correctif causal appliqué dans `zeblindsolver.py`:
  - `obs_code` reste calculé dès qu’il est requis par le seed-path (`code_rs_enabled` ou `strict_mode_effective`), même si le filtre Ze `code_continuous_filter` reste désactivé pour la parité.
- Probe post-correctif publié: `reports/r47i_s5_upstream_trace_probe_20260601_1209/summary.json`.
  - Résultat durable: `pre_perm_n=0` sur la tête de trace same-FITS.
  - Lecture durable: le résiduel `skip_no_seed_slices` était un artefact de l’itération de parité, pas un verrou solver amont fondamental sur ce case.
- Suite logique validée:
  - arrêter de micro-itérer ce candidat same-FITS;
  - passer à une validation multicase stricte S5 pour voir si ce nettoyage amont tient hors du case de référence.

## 2026-06-01 (S5 callsite/permutation, fermeture de l’angle mort sur trace frais)

- Instrumentation runtime S5 enrichie dans `zeblindsolver.py`:
  - ajout d’un champ `perm_callsite_label` qui encode explicitement `phase`, `pass_tag`, `level`, `agg_levels`, `levels_to_use`, `use_px_spec` et `use_ra_filter`.
- Outillage S5 consolidé:
  - `tools/r47i_s5_upstream_trace_probe.py` republie maintenant `trace_head_firstfit.json` pour le premier FITS du probe;
  - `tools/r47i_s5_callsite_permutation_audit.py` et `tools/r47i_s5_math_permutation_audit.py` prennent le dernier trace S5 disponible par défaut au lieu de dépendre du dump figé `20260527_1008`.
- Validation syntaxique: `python3 -m py_compile zeblindsolver/zeblindsolver.py tools/r47i_s5_upstream_trace_probe.py tools/r47i_s5_callsite_permutation_audit.py tools/r47i_s5_math_permutation_audit.py` OK.
- Probe same-FITS relancé: `reports/r47i_s5_upstream_trace_probe_20260601_1215/summary.json` avec trace frais `trace_head_firstfit.json`.
- Audit callsite/permutation relancé: `reports/r47i_s5_callsite_permutation_audit_20260601_1216/summary.json`.
  - Résultat durable: `first_callsite_gap=null`, `preperm_rows_missing_reason=[]`, statut `closed`.
- Audits de consolidation relancés sur le même trace:
  - `reports/r47i_s5_stage_callsite_mapping_audit_20260601_1218/summary.json` => mapping stage/callsite fermé (`status=closed`).
  - `reports/r47i_s5_permutation_callsite_audit_20260601_1221/summary.json` => variantes widened/aggregate reclassées comme attendues, `perm_inconsistency_count=0`, `status=closed`.
  - `reports/r47i_s5_tuple_semantic_diff_audit_20260601_1221/summary.json` => `pass_tag` déjà disponible sur les tuples ciblés; le prochain diff utile est C-vs-Ze sur tuple identique, sans nouvelle instrumentation.
- Audit maths/permutation relancé: `reports/r47i_s5_math_permutation_audit_20260601_1216/summary.json`.
  - Résultat durable: toutes les checks source/runtime utiles restent vertes sur le trace frais.
- Lecture durable:
  - le gap historique `same_candidate_same_parity_perm_shift` était un faux positif créé par un vieux trace S5 incomplet et un audit qui mélangeait des callsites distincts;
  - avec le trace frais + le label de callsite, l’angle mort permutation/callsite est refermé et le prochain step utile redevient le diff C-vs-Ze à candidat identique.

## 2026-06-01 (S5 tuple C-vs-Ze, reclassement du résiduel en phase re-entry)

- Audit tuple S5 enrichi dans `tools/r47i_s5_tuple_semantic_diff_audit.py`:
  - support par défaut du dernier `trace_head_firstfit.json`,
  - résumé par parité du tuple cible,
  - détection explicite des cas `phase_reentry_without_perm_shift`.
- Audit relancé: `reports/r47i_s5_tuple_semantic_diff_audit_20260601_1231/summary.json`.
  - Sur `d50_2628`, les `perm_head` restent stables par parité (`nominal=[1,0,2,3]`, `mirror=[0,1,2,3]`).
  - Le tuple est réémis 4 fois par parité via `scale_only/first_pass`, `scale_only/widened_pass`, `blind/first_pass`, `blind/widened_pass`.
  - Le résiduel utile n’est donc plus un drift de permutation mais une **ré-entrée multi-phase du même tuple** incompatible avec la lecture la plus stricte de la boucle monotone `solver.c`.
- Suite logique durable:
  - le prochain delta causal S5 doit cibler l’isolation/suppression de cette ré-émission post-`scale_only` dans un protocole contraint, avant tout retuning produit.

## 2026-06-01 (S5 expérience probe-only, suppression de la ré-entrée blind après scale_only)

- Nouveau flag expérimental ajouté dans `zeblindsolver.py`:
  - `blind_s5_skip_blind_reentry_after_scale_only_enabled`
  - portée volontairement étroite: mode opt-in, saut des tuples déjà vus en `scale_only` lorsqu’ils reviennent en phase `blind`, sans changer le comportement produit par défaut.
- Le probe S5 sait maintenant activer ce mode via `tools/r47i_s5_upstream_trace_probe.py --skip-blind-reentry-after-scale-only`.
- Expérience publiée: `reports/r47i_s5_upstream_trace_probe_20260601_1234/summary.json`.
  - `d50_2628` ne revient plus en `blind`; la queue de trace est remplacée par d’autres tuples (`d50_2824`, `d50_2725`).
- Audit tuple relancé: `reports/r47i_s5_tuple_semantic_diff_audit_20260601_1236/summary.json`.
  - `target_rows_count` passe de `8` à `4`,
  - `target_seen_in_blind=false`,
  - `perm_head` reste stable par parité,
  - le résiduel utile sur ce case se réduit maintenant à `scale_only:first_pass -> widened_pass`.
- Lecture durable:
  - la ré-émission post-`scale_only` du tuple cible est bien un résiduel d’orchestration isolable;
  - le prochain delta causal S5 doit viser la ré-émission intra-`scale_only`, pas les maths de permutation.

## 2026-06-01 (S5 expérience probe-only, suppression de la ré-entrée widened après first_pass)

- Nouveau flag expérimental ajouté dans `zeblindsolver.py`:
  - `blind_s5_skip_scale_only_widened_reentry_after_first_pass_enabled`
  - portée volontairement étroite: mode opt-in, saut d’un tuple déjà vu en `scale_only:first_pass` lorsqu’il revient en `scale_only:widened_pass`, sans changer le comportement produit par défaut.
- Le probe S5 sait maintenant activer ce mode via `tools/r47i_s5_upstream_trace_probe.py --skip-scale-only-widened-reentry-after-first-pass`.
- Expérience cumulée publiée: `reports/r47i_s5_upstream_trace_probe_20260601_1240/summary.json`.
  - `d50_2628` ne survit plus qu’en `scale_only:first_pass`, une fois par parité.
  - la solution du probe reste `success=true`, mais le terminal de ce run passe en `phase=blind`, preuve que l’espace libéré est repris par d’autres tuples au lieu d’être simplement supprimé du solve.
- Audit tuple relancé: `reports/r47i_s5_tuple_semantic_diff_audit_20260601_1240/summary.json`.
  - `target_rows_count=2`,
  - `target_seen_in_blind=false`,
  - `target_phase_reentry_detected=false`,
  - `perm_head` reste stable par parité.
- Lecture durable:
  - sur le tuple sentinelle `d50_2628`, l’angle mort S5 `callsite/permutation/phase_reentry` est désormais refermé dans ce protocole contraint;
  - le prochain diff utile doit se déplacer du tuple sentinelle vers les nouveaux tuples qui prennent la relève dans la tête de trace (`d50_2824`, `d50_2625`, `d50_2725`).

## 2026-06-01 (S5 expérience probe-only, suppression de la ré-entrée blind:widened après blind:first_pass)

- Nouveau flag expérimental ajouté dans `zeblindsolver.py`:
  - `blind_s5_skip_blind_widened_reentry_after_first_pass_enabled`
  - portée volontairement étroite: mode opt-in, saut d’un tuple déjà vu en `blind:first_pass` lorsqu’il revient en `blind:widened_pass`, sans changer le comportement produit par défaut.
- Le probe S5 sait maintenant activer ce mode via `tools/r47i_s5_upstream_trace_probe.py --skip-blind-widened-reentry-after-first-pass`.
- Expérience cumulée publiée: `reports/r47i_s5_upstream_trace_probe_20260601_1309/summary.json`.
  - `d50_2625` ne survit plus qu’en `blind:first_pass`, une fois par parité.
  - la widened blind du même run est reprise par un nouveau tuple `d50_2821`, ce qui confirme que le solve continue à remplir la tête résiduelle au lieu de simplement perdre des lignes.
- Audit tuple relancé: `reports/r47i_s5_tuple_semantic_diff_audit_20260601_1310/summary.json` (cible `d50_2625`).
  - `target_rows_count=2`,
  - `target_seen_in_blind=true`,
  - `target_phase_reentry_detected=false`,
  - `perm_head` reste stable par parité.
- Lecture durable:
  - les tuples sentinelles `d50_2628` et `d50_2625` sont maintenant ramenés à des flux `first_pass` simples dans ce protocole contraint;
  - le prochain diff utile doit se déplacer vers les successeurs qui occupent encore la tête résiduelle (`d50_2824`, `d50_2821`) et le cas transversal `d50_2822`.

## 2026-06-01 (S5 audit de la tête résiduelle, sans nouveau patch runtime)

- Nouvel outil ajouté: `tools/r47i_s5_remaining_head_order_audit.py`.
  - but: classer chaque ligne de la tête S5 résiduelle comme `c_like_first_occurrence`, `same_tuple_widened_reentry`, `widened_candidate_carryover`, `widened_new_candidate`, etc., sans rouvrir l’instrumentation runtime.
  - point pratique fixé dans l’outil: les paires de parité `first_pass` sont reclassées comme **non résiduelles**, et les artefacts tuple multi-cibles ont maintenant un nom de sortie stable par `candidate_key`.
- Audit publié: `reports/r47i_s5_remaining_head_order_audit_20260601_134051/summary.json` sur le trace `reports/r47i_s5_upstream_trace_probe_20260601_1309/trace_head_firstfit.json`.
- Résultat durable:
  - premier résiduel causal restant = `d50_2822`, `row_index=1`, `phase=hinted`, `pass_tag=widened_pass`, classe `same_tuple_widened_reentry`;
  - compteurs résiduels utiles:
    - `same_tuple_widened_reentry=2`
    - `widened_candidate_carryover=5`
    - `widened_new_candidate=2`
  - candidats dominants du résiduel:
    - `d50_2822` (4 lignes),
    - `d50_2824` (2),
    - `d50_2821` (2),
    - `d50_2725` (1).
- Lecture durable:
  - après nettoyage probe-only de `d50_2628` et `d50_2625`, le prochain diff utile S5 doit se concentrer sur `d50_2822` côté `hinted:first_pass -> hinted:widened_pass`;
  - `d50_2824` et `d50_2821` deviennent des successeurs résiduels secondaires, pas le premier écart causal.

## 2026-06-01 (S5 expérience probe-only, suppression des widened-only tuples)

- Nouveaux flags expérimentaux ajoutés dans `zeblindsolver.py`:
  - `blind_s5_skip_scale_only_widened_only_tuples_enabled`
  - `blind_s5_skip_blind_widened_only_tuples_enabled`
  - portée volontairement étroite: mode opt-in, saut des tuples émis uniquement en `widened_pass` sans équivalent `first_pass` dans la même phase.
- Le probe S5 sait maintenant activer ce mode via `tools/r47i_s5_upstream_trace_probe.py --skip-scale-only-widened-only-tuples --skip-blind-widened-only-tuples`.
- Expérience cumulée publiée: `reports/r47i_s5_upstream_trace_probe_20260601_1315/summary.json`.
  - La tête de trace passe de `16` lignes (`1309`) à `10`.
  - `newpoint` reste monotone et l’audit global reste vert: `reports/r47i_s5_newpoint_permutation_gap_audit_20260601_1316/summary.json` (`first_causal_gap=null`).
  - La structure résiduelle devient:
    - `d50_2822` dans `hinted` seulement (`first_pass` + `widened_pass`);
    - `d50_2628` dans `scale_only:first_pass` seulement;
    - `d50_2824` et `d50_2725` dans `blind:first_pass`.
- Audits tuple publiés sur cette tête résiduelle:
  - `reports/r47i_s5_tuple_semantic_diff_audit_d50_2822_20260601_131622/summary.json`
  - `reports/r47i_s5_tuple_semantic_diff_audit_d50_2824_20260601_131622/summary.json`
  - `reports/r47i_s5_tuple_semantic_diff_audit_d50_2725_20260601_131622/summary.json`
- Lecture durable:
  - les widened-only opportunistes n’étaient pas indispensables pour produire une tête diffable;
  - le prochain diff utile peut désormais se faire directement sur la tête `1315`, sans ajouter de nouvelle instrumentation ni de nouveau flag de skip dans l’immédiat.

## 2026-05-26 (S3.1 démarrage, onefield accept-machine)

- Premier bloc S3.1 traité dans `zeblindsolver.py` (alignement `onefield`):
  - `record_match_callback` intègre désormais `nsolves_sofar` avec seuil configurable `blind_nsolves_required` (et CLI associée `--blind-nsolves-required`).
  - La sélection `best_solution` est passée en logique `logodds-first` (priorité `solve_logodds`, puis `accept_logodds`, puis `prob_logodds`), avec fallback inliers/RMS seulement à égalité.
  - Le dedup runtime multi-solution privilégie maintenant le score logodds et utilise `fieldfile/fieldnum` quand disponibles (fallback local sinon).
- Contrôle de non-régression syntaxique validé: `python3 -m py_compile zeblindsolver/zeblindsolver.py` OK.
- Statut durable: S3.1 lancé concrètement; reste à valider par artefact runtime ciblé et à finir la parité stricte `best_match_solves/remove_duplicate_solutions`.

## 2026-05-24 (S1 micro-runner verify-only)

- Nouveau micro-runner ajouté: `tools/r47i_s1_verify_only_replay.py`.
  - Entrée: bundle forced d’un artefact S1 (`forced_test/ref/perms` + `on/off_verify_sets` + `on/off_refpool_trace`).
  - Comportement: rejeu **verify/accept uniquement** via `_astrometry_verify_sequence_logodds` (sans repasser par la recherche candidat amont).
- Artefact de preuve publié: `reports/r47i_s1_verify_only_replay_case055_20260524_1156/summary.{json,md}`.
  - ON/OFF strictement identiques en verify-only (`verify_logodds=11.010719844475592`, `accept_band=below_toprint`, `reason_code=accept_logodds_gate`).
  - `first_divergence=None`.
- Conséquence durable: sur ce case et ce bundle figé, la divergence n’est plus amont; le comportement résiduel est purement séquentiel verify/accept et déjà aligné ON/OFF.

## 2026-05-24 (S1 bundle pré-verify enrichi)

- Harness `tools/r47i_s1_forced_payload_full_replay.py` enrichi pour produire un bundle pré-verify plus complet: `forced_preverify_bundle.json`.
- Artefact de référence: `reports/r47i_s1_forced_payload_full_replay_case055_20260524_1252/`.
  - Bundle contient: `test_xy_use/ref_xy_use/testsigma2_use`, `refperm/testperm`, `quad_center/q2`, `pix2`, `fieldcenter/radius`.
  - `refstarid` désormais alimenté (head IDs dérivés depuis `on_refpool_trace.trace[*].gid_head`).
  - `mo_star/mo_field` restent `null` (non exportés nativement par le runtime actuel).

## 2026-05-23 (S1 harness same-hit candidate replay)

- Instrumentation durable ajoutée dans `zeblindsolver.py` pour S1:
  - `object_index` propagé dans `candidate_ctx`, `verify_hit_trace` et `astrometry_match_objects`;
  - export explicite des champs utiles de rejeu/parité dans `astrometry_match_objects` et `verify_hit_trace` (`verify_entry_*`, `verify_refperm_*`, `prob_testsigma_*`, `prob_quad_q2_px2`, `prob_verify_steps`).
- Harness de rejeu S1 ajouté: `tools/r47i_s1_samehit_candidate_replay.py`.
  - Capacité validée: auto-sélection d’un candidat `accept_logodds_gate` compatible avec le `manifest` de l’index actif, puis replay OFF/ON et publication `ab_summary.{json,md}`.
- Artefact durable publié:
  - `reports/r47i_s1_samehit_candidate_replay_case055_20260523_1000b/ab_summary.{json,md}`.
- Résultat factuel de ce run:
  - candidat ON trouvé/rejoué: `phase=hinted`, `level=S`, `parity=nominal`, `tile=d50_2823`, `reason_code=accept_logodds_gate`;
  - métriques ON exportées: `NT/NR=51/1`, `testsigma_mode=gamma_like`, `quad_q2_px2=15324.2865`, séquence `theta_head`, `best/worst logodds`;
  - OFF ne reproduit pas encore le même hit (terminal `no valid solution`), donc protocole S1 iso-candidat toujours ouvert.

## 2026-05-22 (S0 mode exclusivity audit + angle mort)

- Audit S0/S1 publié: `reports/r47i_s0_mode_exclusivity_audit_20260522_0752/summary.{md,json}`.
- Portée scannée: 6 artefacts `ab_summary.json` (`r47i_s0*` + `r47i_s1*`), 12 lignes OFF/ON contrôlées.
- Verdict validé: **PASS**. Sur toutes les probes, `strict_mode_effective=true`, `non_parity_mode_effective=false`, `exact_preverify_effective=false`, `verify_only_injected_effective=false`, avec cohérence OFF (`native_effective=false`) / ON (`native_effective=true`).
- Mise à jour backlog: item S0 « mode exclusif utilisé dans les probes de parité » coché `[x]` dans `followup.md`.
- Angle mort ajouté dans `followup.md`: le gate `a2v34 scale plausibility` n’est pas explicitement forcé OFF par `astrometry_native_verify_semantics` (il dépend encore d’un flag config), ce qui peut réintroduire un rejet Ze pré-verify en mode canonique si mal paramétré.
- Correctif appliqué ensuite dans `zeblindsolver.py`: ajout d’un disable canonique explicite `canonical_disable_a2v34_scale_plausibility` en mode natif, instrumentation `mode_profile` associée (`canonical_disable_a2v34_scale_plausibility` + note `disabled_a2v34_scale_plausibility_in_native_mode`), et garde runtime du gate `a2v34`.
- Validation du delta: `python3 -m py_compile zeblindsolver/zeblindsolver.py zeblindsolver/verify.py` => OK.
- Artefact de patch publié: `reports/r47i_s0_a2v34_canonical_disable_20260522_0756/summary.{md,json}`.
- Correctif suivant appliqué dans `zeblindsolver.py`: disable canonique explicite du `scale hard filter` (`canonical_disable_scale_hard_filter`) avec forçage runtime `hard_filter_enabled_runtime=False` en mode natif + télémétrie `mode_profile` (`canonical_disable_scale_hard_filter`, note `disabled_scale_hard_filter_in_native_mode`).
- Validation du delta: `python3 -m py_compile zeblindsolver/zeblindsolver.py zeblindsolver/verify.py` => OK.
- Artefact de patch publié: `reports/r47i_s0_scalehard_canonical_disable_20260522_0815/summary.{md,json}`.
- Probe rapide ON/OFF publié: `reports/r47i_s0_scalehard_modeprofile_probe_fast_20260522_0819/ab_summary.{md,json}`.
- Constat durable associé: sur ce chemin d'échec rapide, `mode_profile` n'est pas présent dans `sol.stats` (clés canoniques absentes), donc l'observabilité runtime des disables canoniques reste incomplète tant que ce point n'est pas corrigé.
- Correction durable de ce constat: le `mode_profile` runtime est exporté sous `sol.stats["astrometry_semantics"]["mode_profile"]` (et non en top-level `sol.stats["mode_profile"]`) sur ces chemins d'échec.
- Audit runtime OFF/ON publié: `reports/r47i_s0_canonical_disable_runtime_audit_20260522_0859/ab_summary.{md,json}`.
  - Résultat validé: `on_all_true=true`, `off_all_false=true` pour tous les flags `canonical_disable_*` attendus.
  - Lecture durable: le chemin canonique s'exécute bien avec les courts-circuits ciblés neutralisés; le blocage restant S0 n'est plus l'observabilité des disables mais le court-circuit `validate_solution()` avant accept gate natif.
- Audit global post-compaction relancé (mission + blocage S0): confirmation causale que `validate_solution()` peut encore court-circuiter la chaîne native `verify -> accept` via plusieurs callsites (`main`, `validate_scale_rescue`, `a2v33`, `tweak2`, `posthit_tan_refit`, `posthit_sip`).
- Angle mort complémentaire ajouté au backlog: `preverify_center_prior` peut encore rejeter pré-transaction en présence de hints/config (pas neutralisé explicitement par `astrometry_native_verify_semantics`).
- Plan S0 enrichi dans `followup.md`: neutralisation canonique explicite des rescues/réécritures non Astrometry (`validate_scale_rescue`, `a2v33_borderline_rms`, `tweak2_verify`, `posthit_tan_refit`, `posthit_sip`) + neutralisation explicite `preverify_center_prior` avec trace `mode_profile`.
- Clarification d'usage index: `astrometry-indexes/` (fichiers `.fits` téléchargés) ne contient pas de `manifest.json` et ne peut pas être consommé directement par `solve_blind` (qui attend un index Ze avec manifest), d'où usage d'un index Ze local `reports/forensic_case055_subset96/index` pour les probes runtime.

## 2026-05-21 (reprise mission + probe S1 same-hit)

- Reprise du chantier après interruption avec relecture croisée de `followup.md`, des artefacts `r47i_s0_*` et du code actif `zeblindsolver.py`.
- Validation de base après reprise: `python3 -m py_compile zeblindsolver/zeblindsolver.py zeblindsolver/verify.py` => OK.
- Probe ciblé S1 publié dans `reports/r47i_s1_verify_entry_forced_tile_case055_20260521_1430/` pour essayer d'obtenir un comparatif OFF/ON sur un même hit via tile-lock forcé `d50_3018`.
- Résultat durable de ce probe: le tile-lock seul ne suffit pas encore à fabriquer un protocole `same-hit verify-entry` stable.
  - OFF retombe sur `strict_pairset_scale_gate_reject` (`pairing=10`, `verify_hit_trace_count=0`).
  - ON natif ne retrouve même pas le hit post-pairset attendu et échoue plus tôt en `hypothesis failed before validation` (`hypothesis=10`, `verify_hit_trace_count=0`).
  - Aucun dump `blind_verify_forensic_dump_path` n'a été produit, preuve que l'entrée verify n'a pas été atteinte sur ce harness.
- Lecture durable: la prochaine brique causale n'est pas un nouveau tuning verify, mais la construction d'un **harness de rejeu du même candidat** (capture/replay du candidat natif qui atteint `accept_logodds_gate`) avant de poursuivre S1/S2.

## 2026-05-20 (audit élargi Astrometry vs ZeBlind + refonte followup)

- Relecture consolidée de l'audit source-à-source Astrometry vs ZeBlind avec recentrage sur les écarts **encore ouverts** réellement structurants.
- Conclusion durable reformulée: le trou restant n'est pas un simple paramètre, mais un ensemble cohérent d'écarts de sémantique entre notre chemin Ze courant et le vrai chemin Astrometry.
- Relecture complémentaire validée ensuite sur `solver.c / verify.c / onefield.c / engine.c` : certains écarts de parité devaient être explicités noir sur blanc dans `followup.md`, en particulier
  - le chemin `solver_verify_sip_wcs()` / `fake_match` (injection verify sur tous les index, `distance_from_quad_bonus=FALSE`, sémantique fake-match distincte),
  - la politique exacte `record_match_callback / best_match / best_match_solves / remove_duplicate_solutions`,
  - la matérialisation du hit avant verify (`quadfile_get_stars -> startree_get -> fit_tan_wcs -> set_center_and_radius`) et la collecte ref `bounding circle -> inside image` avec sémantique `NRall / NRimage / NR`.
- Ordre de mise en oeuvre désormais clarifié et validé dans `followup.md` : **verrouiller d'abord l'aval canonique sur candidat figé** (`S0 -> S4`), puis seulement **remonter vers l'amont solver** (`S5`) et finir par validation end-to-end / smoke Seestar (`S6`). Lecture durable: commencer par `A2/A3` trop tôt compliquerait inutilement le chantier et brouillerait les causalités.
- Premier delta S0 implémenté dans `zeblindsolver.py` : quand `blind_astrometry_native_verify_semantics_enabled=1` sous chemin strict, le runtime force désormais un **profil canonique unique** pour éviter les mélanges silencieux avec `blind_astrometry_exact_validation_preverify_parity_enabled`, `blind_non_parity_mode_enabled`, `blind_verify_only_injected_wcs_enabled` et le relâchement `strict_disable_nonastrometry_fallbacks`; les artefacts exportent maintenant `requested vs effective` + `mode_profile_notes` pour rendre ces overrides visibles.
- Deuxième delta S0 implémenté dans `zeblindsolver.py` : plusieurs courts-circuits Ze restés actifs même en mode natif ont été explicitement neutralisés côté runtime effectif et exposés dans le `mode_profile`, notamment `strict_pair_pool_expand`, `pair_scale_prefilter`, `scale_prefilter`, `strict_require_perm_hash_match`, `strict_slot_align`, `strict_resolve_hit_relaxed_retry`, `resolve_hit_pool_adaptive_fallback`, `strict_resolve_hit_seed_fallback` et `empty_inliers_fallback`.
- Gap de backlog clarifié dans `followup.md` : il ne suffisait pas de publier `requested vs effective` au niveau macro, il fallait aussi tracer quels **courts-circuits Ze résiduels** sont forcés OFF dans le profil canonique natif.
- Rerun S0 court publié dans `reports/r47i_s0_canonical_ab_case055_short_20260520_200006/ab_summary.json` : OFF et ON restent alors bloqués avant verify (`verify_hit_trace_count=0`), avec dominance `pairing`, preuve que le prochain écart causal n’était pas encore dans l’aval verify.
- Diagnostic ciblé S0 ensuite : le vrai prochain bloqueur du chemin canonique natif était encore `strict_pairset_scale_gate_reject` (documenté dans `reports/r47i_s0_pairset_gap_and_canonical_shift_20260520_2012.md`). Correction appliquée dans `zeblindsolver.py` : désactivation runtime explicite de `blind_pairset_scale_gate_enabled` et `blind_pairset_scale_recenter_enabled` quand `blind_astrometry_native_verify_semantics_enabled=1`, avec publication dans `mode_profile`.
- Signal causal post-patch validé : après neutralisation du pairset gate, le chemin natif ON ne meurt plus sur `strict_pairset_scale_gate_reject` mais atteint des rejets plus aval de type `blind accept-logodds below toprint`; la prochaine cible utile se situe donc désormais dans la chaîne verify/accept native, plus dans le pairset gate.
- Artefact S0 de lecture rapide publié dans `reports/r47i_s0_mode_profile_off_on_20260520_2016.md`, pour figer le comparatif OFF/ON `requested vs effective` du profil canonique sans devoir replonger dans le JSON brut.
- Familles d'écarts désormais verrouillées comme backlog principal:
  - entrées verify encore mixtes (`inlier_pairs` vs `full field stars / full projected index stars`),
  - ordre/composition des pools (`refperm`, exclusion quad stars, ordre test-stars),
  - géométrie native du quad (`quad_center`, `Q2`, RoR),
  - modèle probabiliste (`distractor=0.25`, `testsigma²` Astrometry réel),
  - machine d'acceptation exacte (`verify -> tune -> re-verify -> accept`) sans gates Ze parasites,
  - fidélité de traversée amont `newpoint/add_stars/try_permutations/startobj/endobj`,
  - orchestration solver autour de `solver.c / control-program.c / onefield.c / engine.c`.
- `followup.md` a été entièrement remanié pour refléter cette checklist de parité complète et séparer clairement :
  - ce qui relève du **chemin canonique Astrometry**,
  - de ce qui relève plus tard du tuning produit Ze.
- Vérification utile sur les indexes locaux: le dossier `projects/ZeSolver/astrometry-indexes/` ne contient actuellement que `4100/index-4110..4119`.
- Lecture durable sur couverture Seestar: ce set couvre seulement les quads ~`60..2000` arcmin, ce qui est insuffisant/au mieux limite pour un FOV Seestar typique (~`76 x 43` arcmin, diagonale ~`88` arcmin) ; les plages plus petites `4107/4108/4109` et probablement `5206/5205` (voire `5204`) restent les candidats naturels à ajouter.

## 2026-05-18 (oracle ASTAP sur EQ/IRCUT)

- Mise en place d'un lot oracle local à partir de `/home/tristan/zemosaic/example/organized/EQ/IRCUT` : 30 copies FITS nettoyées WCS dans `reports/eq_ircut_cleanbench_20260518_230249/data`.
- Référence terrain validée: `/home/tristan/zemosaic/example/backuplightsastap` contient les **mêmes 30 fichiers** avec WCS corrects, exploitable comme oracle image-par-image.
- Ajout de l'outil `tools/wcs_oracle_compare.py` pour comparer un dossier candidat à un dossier de référence WCS (centre, coins, échelle, parité, classification `true_match / near_miss / false_positive`).
- Validation clé du run `blind_minuq3` en cours: les **10 premiers solves marqués `success=true`** sont en réalité **10/10 faux positifs** contre l'oracle ASTAP.
- Signatures validées par phase sur ces 10 cas:
  - `hinted` -> dérive centre médiane ~6.8° et échelle typique ~3.9x la référence,
  - `hinted_wide` -> dérive centre médiane ~7.9° et échelle typique ~0.90x,
  - `scale_only` -> dérive centre ~110° avec solutions complètement hors champ.
- Lecture durable: le pipeline solve actuellement "actif" peut produire des succès trompeurs; pour ce lot, le KPI pertinent devient la **concordance oracle ASTAP**, pas le `success=true` brut.

## 2026-05-18 (cause racine probable des faux positifs oracle)

- Forensic ciblé sur trois faux positifs représentatifs du lot oracle EQ/IRCUT:
  - `232102` (`hinted`, tile `d50_2725`) validé comme `GOOD` malgré `rms=972.595 px`.
  - `232205` (`hinted_wide`, tile `d50_2627`) validé comme `GOOD` avec `3` paires, `rms=2466.221 px`, résidu WCS médian `~4119"`.
  - `232534` (`scale_only`, tile `d50_0814`) validé comme `GOOD` avec seulement `2` paires; cas dégénéré typique (fit quasi parfait localement mais centre réel faux de `~110°`).
- Cause racine très probable identifiée: dans `zeblindsolver/verify.py`, `validate_solution()` force `success=True` quand `astrometry_parity_mode=True`.
- Ce mode est activé par défaut via `blind_astrometry_exact_validation_preverify_parity_enabled=True` dans `zeblindsolver.py`.
- Effet produit validé: des validations diagnostiques `validation_metrics_only[...]` (avec `rms_ok=0` et/ou `inliers_ok=0`) deviennent `quality=GOOD`, ce qui laisse passer des WCS fantaisistes.
- Effet aggravant validé: plusieurs garde-fous géométriques/low-pairs sont désactivés quand ce mode parity preverify est actif.

## 2026-05-19 (patch produit du bypass parity preverify)

- Correction appliquée: `blind_astrometry_exact_validation_preverify_parity_enabled` est maintenant **désactivé par défaut** côté produit (`False` dans la config + `--blind-astrometry-exact-validation-preverify-parity-enabled` par défaut à `0`).
- Le mode reste disponible en opt-in pour du forensic/debug seulement.
- Validation ciblée après patch en forçant les anciennes tuiles faux-positives:
  - `232534` + `d50_0814` ne passe plus; rejet sur `preverify_geom_guard_failed[...]`.
  - `232205` + `d50_2627` ne passe plus; rejet sur `strict_pairset_scale_gate_reject`.
- Lecture durable: couper ce bypass réactive bien les garde-fous normaux et supprime au moins deux faux positifs reproductibles qui passaient auparavant en `success=true`.

## 2026-05-19 (rerun oracle post-patch, état partiel)

- Lot oracle frais relancé dans `reports/eq_ircut_oracle_postpatch_20260519_001639` (30 copies neuves nettoyées WCS).
- Le comparateur `tools/wcs_oracle_compare.py` sait désormais classer proprement les runs sans WCS candidat en `fail_no_wcs`.
- Mesure partielle post-patch sur les 12 premières images du lot: **`12/12 fail_no_wcs`** dans `oracle_compare_partial.md`.

## 2026-05-19 (native verify semantics Astrometry, première implémentation)

- Ajout d’un nouveau flag canonique `blind_astrometry_native_verify_semantics_enabled` (+ CLI `--blind-astrometry-native-verify-semantics-enabled`) pour forcer un chemin verify plus fidèle à Astrometry sans réactiver le vieux bypass forensic produit.
- Quand ce mode est actif, le solveur force notamment: verify inputs `field_all_vs_index_all`, `distractor=0.25`, RoR actif, `testsigma` promu au moins en `gamma_like`, validate gate metrics-only, et bypass des principaux garde-fous/coherence gates Ze non-Astrometry sur ce chemin.
- Amélioration supplémentaire appliquée juste après: le calcul `quad center / Q2` pour le mode gamma/RoR ne part plus par défaut des premières étoiles de verify; il préfère le support courant d’hypothèse `img_in`, avec nouveaux compteurs de traçabilité `prob_quad_center_px`, `prob_quad_q2_px2`, `prob_quad_center_source`.
- Gate technique validée après ces patches: `python3 -m py_compile zeblindsolver/zeblindsolver.py zeblindsolver/verify.py` => OK.
- Tentatives A/B subset96 (case055) relancées, mais le protocole propre courant ne rentre pas encore assez loin dans verify pour une mesure OFF/ON exploitable; les échecs dominants observés restent en amont (`preverify_geom_guard_failed`, `strict_pairset_scale_gate_reject`, `strict_lowpairs_scale_guard_failed`).
- Lecture durable: après désactivation du bypass parity preverify, les faux succès rapides ont disparu sur ce sous-lot, mais le solveur retombe pour l’instant sur un profil beaucoup plus coûteux (timeouts / absence de solution dans le budget imparti).

## 2026-05-19 (mode canonique, corrections de câblage)

- Correction durable appliquée: `blind_astrometry_native_verify_semantics_enabled` ne s’auto-active plus via `blind_astrometry_exact_validation_preverify_parity_enabled`; les deux modes sont maintenant séparés proprement pour permettre un vrai A/B.
- Correction durable appliquée: le chemin `native_verify_semantics` ne force plus d’uncaps implicites sur les pools test/ref et respecte de nouveau les caps configurés.
- Lecture durable: après ces correctifs, le principal verrou n’est plus le câblage du mode canonique lui-même mais la qualité du protocole de repro servant à l’évaluer.

## 2026-05-19 (audit source-à-source math/semantics vs Astrometry)

- Audit relancé directement contre les sources Astrometry `solver/verify.c`, `solver.c`, `control-program.c` pour identifier les écarts **forcément suspects** dans le chemin Ze utilisant des index Astrometry.
- Constat durable n°1: Ze a introduit un hybride de modèles (validation WCS maison + verify probabiliste + garde-fous Ze), alors qu’Astrometry garde une sémantique d’acceptation centrée sur `verify_hit`, éventuel `tweak`, puis re-verify.
- Constat durable n°2: le tri `refperm` par `skdt->sweep` natif index et l’exclusion des étoiles du quad sont des invariants Astrometry importants; quand Ze ne les reproduit pas strictement, la parité mathématique avec les index Astrometry est cassée à la racine.
- Constat durable n°3: Ze reste par défaut en `testsigma` constant alors qu’Astrometry supporte explicitement un modèle radial/gamma (`verify_pix2 * (1 + R²/Q²)`), donc le poids statistique des matches n’est pas encore aligné par défaut même sur le verify strict.
- Rapport formel rédigé dans `reports/astrometry_math_audit_20260519_0114.md`.
- Constat durable n°4: le strict verify Ze courant peut encore fonctionner en **chemin legacy `inlier-only pairs`** au lieu du problème verify Astrometry complet (`full field stars` vs `full projected index stars`).
- Constat durable n°5: hors mode exact parity, Ze utilise un distractor ratio effectif `0.65` (`1 - prob_prior`) au lieu du `0.25` par défaut d’Astrometry; c’est un écart probabiliste majeur, pas un simple détail de tuning.
- Constat durable n°6: le RoR Ze courant se base sur une approximation médiane/percentile au lieu du vrai centre/Q2 du quad utilisé par Astrometry, donc même la géométrie du filtre de pertinence n’est pas alignée.

## 2026-05-19 (forensic case055, requalification du repro subset96)

- Découverte critique: l’index `reports/forensic_case055_subset96/index` ne contient pas les tuiles hinted historiques attendues (`d50_1025`, `d50_1026`, `d50_1127`, `d50_1128`) ; il ne peut donc pas servir de repro fidèle du `case055` historique utilisé pour la parité.
- Découverte critique complémentaire: certains inputs subset96 portent déjà des centres WCS recadrés sur des tuiles subset96, ce qui peut créer des succès artificiels hors protocole brut.
- Découverte critique complémentaire: les IDs de tuiles historiques observés dans les anciens forensics ne sont plus stables vis-à-vis de l’index live actuel `/home/tristan/zesolver_index` (ex. `d50_1025` pointe maintenant vers une géométrie incompatible avec M106).
- Lecture durable: les anciens succès `subset96` doivent être requalifiés avec prudence; aucune conclusion forte de parité ne doit plus être tirée de ce repro tant qu’un couple input+index propre n’a pas été reconstruit.

## 2026-05-19 (hygiène repo)

- `followup.md` a été remanié pour revenir à un plan court orienté action: repro propre `case055` d’abord, A/B verify canonique ensuite, tuning produit seulement après.
- `followup.md` a ensuite été compacté une nouvelle fois après clôture de l’ancienne checklist: la todo historique est désormais résumée ici, et le nouveau plan est recentré uniquement sur la suite utile pour rendre **ZeBlind seul fonctionnel et juste**.

## 2026-05-19 (reconstruction forensic M106 / case055 propre)

- Un nouvel index forensic M106 figé a été construit dans `reports/forensic_m106_reference_v1/index` à partir du live actuel autour de M106 (`tile_count=24`).
- Validation durable: aucun snapshot local cohérent avec les tuiles historiques `d50_1025/1026/1127/1128` n’a été retrouvé; la lecture correcte est donc **reconstruction d’une référence stable**, pas récupération historique locale.
- Les `hash_tables` du nouvel index figé ont bien été générées (`quads_L.npz`, `quads_M.npz`, `quads_S.npz`). Point pratique important: l’appel CLI `python3 -m zeblindsolver.db_convert ... --quads-only` peut finir `EXIT:0` sans rien écrire; l’appel explicite `build_index_from_astap(...)` fonctionne.
- Le vrai input source du repro `case055` a été ré-identifié: toutes les variantes inspectées pointent vers `Light_mosaic_M 106_20.0s_IRCUT_20250518-233828.fit` (hash image identique).
- Un repro propre a été figé dans `reports/forensic_case055_repro_v1/input_nowcs.fit` en retirant les mots-clés WCS/solver du frame source `233828` sans toucher aux pixels.
- Sanity check durable publié dans `reports/forensic_case055_repro_v1/summary.json`: `subset96/input_nowcs.fit` réutilisait bien les bons pixels, mais avec une WCS stale (`CRVAL1/2 = 194.24586757029 / 53.984245181255`), donc ce n’était pas un input forensic propre.
- Première mesure A/B saine relancée: run OFF exécuté sur `forensic_case055_repro_v1 + forensic_m106_reference_v1`; il atteint les phases `hinted`, `scale_only`, `blind` mais échoue encore en `no valid solution`, donc la suite prioritaire reste le protocole OFF/ON court et reproductible avant toute conclusion de parité verify.
- Mesure A/B courte complétée ensuite en OFF **et** ON (`blind_astrometry_native_verify_semantics_enabled=0/1`) dans `reports/forensic_case055_repro_v1/ab_short_results.json`: comportement identique des deux branches sur ce protocole (`total_candidates_tried=100`, `best_fail_inliers=0`, `verify_hit_trace_count=0`, champs verify clés `null`). Lecture durable: sur ce repro propre et ce budget court, le blocage est encore **avant** l’entrée verify exploitable.
- Essais complémentaires accomplis pour forcer une entrée verify utile (hint centré sur le frame source `233828`, puis forçage de tuile `d50_2627`) : toujours `no valid solution` / `verify_hit_trace_count=0`.
- Preuve formelle ajoutée dans `reports/forensic_case055_repro_v1/ab_stage_dump.json`: `stage_by_stage.verify=[]`, `verify_hit=[]`, `hypothesis=[]` en OFF comme en ON.
- Localisation du premier coupe-circuit causal publiée dans `reports/forensic_case055_repro_v1/ab_first_cutoff_summary.json`: la divergence OFF/ON n’existe pas encore, les deux branches meurent identiquement sur `strict_pairset_scale_gate_reject` pendant la validation pairing/candidate; `preverify_guard_reject_total=0`, donc le blocage est plus amont que preverify.
- Test causal minimal exécuté ensuite: désactiver **uniquement** `blind_pairset_scale_gate_enabled`. Résultat publié dans `reports/forensic_case055_repro_v1/ab_disable_pairset_gate{,_summary}.json`.
- Lecture durable de ce test: ce seul delta fait franchir le blocage amont aux deux branches et les fait entrer dans `hypothesis`; OFF va jusqu’à `verify` (`verify_count=21`), tandis que ON reste sur `verify_hit` / `verify_prob` (`verify_count=0`, `verify_hit_count=64`, `best_fail_reason=blind verify logodds_accept failed`). Le prochain écart structurel utile est donc **post-pairset**, pas au niveau des index ni du gating pairing initial.
- Correctif causal appliqué ensuite dans `zeblindsolver.py`: le garde-fou Ze `strict_lowpairs_scale_guard` est désormais **désactivé** sur le chemin `blind_astrometry_native_verify_semantics_enabled`, comme les autres garde-fous non-Astrometry déjà bypassés sur ce mode.
- Validation du correctif: `python3 -m py_compile zeblindsolver/zeblindsolver.py` OK, puis rerun publié dans `reports/forensic_case055_repro_v1/ab_disable_pairset_gate_after_lowpairs_patch.json`.
- Lecture durable après patch: en ON, le premier `verify_hit.validate_base` sur `d50_2628` passe de `FAIL strict_lowpairs_scale_guard_failed[...]` à `OK validation_metrics_only[...]`; l’écart structurel restant n’est plus `lowpairs` mais le chemin `verify/logodds` natif (`best_fail_reason=blind verify logodds_accept failed`, `verify_count` encore nul côté ON).
- Divergence `verify/logodds` formalisée dans `reports/forensic_case055_repro_v1/native_verify_logodds_divergence_summary.json`: après bypass lowpairs, le chemin natif ON ne meurt plus sur `validate_base`, mais plusieurs candidats `validation_metrics_only[GOOD]` sont rejetés juste après à `validate_gate` par `blind verify logodds_accept failed`, avant toute matérialisation de ligne `verify`.
- Audit chiffré complémentaire effectué via `reports/forensic_case055_repro_v1/on_pairset_off_after_lowpairs_patch_solveblind.log`: le chemin natif rejetait ces candidats avec `logodds=-1.386` contre un seuil dur `min=12.000`, preuve qu’un `prob gate` Ze restait appliqué trop tôt sur ce mode.
- Correctif causal appliqué ensuite dans `zeblindsolver.py`: bypass du `blind_prob_verify` gate quand `blind_astrometry_native_verify_semantics_enabled=True`.
- Validation après correctif via `reports/forensic_case055_repro_v1/ab_disable_pairset_gate_after_native_prob_bypass.json`: l’échec ON se déplace de `blind verify logodds_accept failed` vers `blind accept-logodds below toprint`. Lecture durable: le prochain blocage utile est désormais dans la chaîne d’acceptation `accept-logodds`, plus dans le `prob gate` Ze.
- Forensic suivant figé dans `reports/forensic_case055_repro_v1/native_accept_toprint_gap_summary.json`: après bypass `lowpairs` + `prob gate`, le chemin natif ON atteint bien `accept_gate`, mais échoue avant toute ligne `verify` avec `accept_logodds=-1.3862943611` contre des seuils Astrometry-like `toprint/totune≈13.8155`, `tokeep≈20.7233`. Signal probabiliste observé au rejet: `prob_matches=0`, `prob_conflicts=0`, `prob_distractors=1`.
- Chaîne cible Astrometry reconstituée dans `reports/forensic_case055_repro_v1/astrometry_accept_chain_target_note.md`: `validate/verify_hit -> verify séquentiel -> banding totune/tokeep -> tune_reverify éventuel -> accept_gate(toprint -> logaccept -> tokeep -> accept)`, avec clamps `tokeep=min(tokeep,tosolve)` et `toprint=min(toprint,tokeep)`.
- Traces candidates ON figées dans `reports/forensic_case055_repro_v1/native_accept_candidate_traces.json`: sur `d50_3017`, `d50_2627` et `d50_2628`, le chemin natif atteint `validate_base=OK` puis échoue de façon stable à `validate_gate` avec `accept_logodds=prob_logodds=-1.3862943611`, `prob_matches=0`, `prob_conflicts=0`, `prob_distractors=1`.
- Comparatif seuils/sémantiques publié dans `reports/forensic_case055_repro_v1/native_accept_thresholds_comparison.json`: l’écart causal prioritaire n’est probablement pas un seuil `toprint/totune` trop dur, mais un **signal probabiliste d’entrée mal alimenté** (mêmes `prob_matches=0`, `prob_conflicts=0`, `prob_distractors=1` sur toutes les tuiles représentatives).
- Vérification dédiée des objets probabilistes dans `reports/forensic_case055_repro_v1/native_accept_prob_objects_check.json`: `accept_logodds_source=null`, `verify_refperm_seed_ids_n=0` et `verify_refperm_final_ids_n=0` sur les tuiles ON représentatives; le prochain suspect causal concret est donc l’alimentation/refperm du verify natif plutôt qu’un simple seuil d’acceptation.
- Delta causal d’instrumentation seule validé dans `reports/forensic_case055_repro_v1/native_accept_source_instrumentation_check.json`: `accept_logodds_source` est maintenant visible et vaut `prob_logodds_astrometry_seq` sur `d50_3017`, `d50_2627`, `d50_2628`, avec `prob_match_ratio=0.0`; la prochaine cible causale durable est donc le calcul probabiliste séquentiel amont, pas le gate `accept-logodds` lui-même.
- Audit amont publié dans `reports/forensic_case055_repro_v1/native_accept_probfeed_rootcause_summary.json`: le premier neutraliseur causal du score natif est le préfixe `distractor` du verify séquentiel (bestlogodds figé à `-1.386` dès `i=0`), sur fond de `verify_refperm_* = 0`; même quand un match apparaît (`d50_2627`, `i=3`, `nsig2≈18.05`), il arrive trop tard pour inverser le meilleur préfixe.
- Fil de preuve post-pairset consolidé dans `reports/forensic_case055_repro_v1/post_pairset_forensic_index.md`, pour garder une chaîne causale unique depuis le bypass `pairset` jusqu’au diagnostic actuel sur le feed probabiliste natif.
- Validation élargie du garde-fou `blind_pairset_scale_recenter_min_unique_obs_quads=3` synthétisée dans `reports/forensic_case055_repro_v1/min_unique_obs_quads_validation_note.json`: compatible avec le sample `eq_ircut_cleanbench` disponible (`10/10`), mais pas promouvable par défaut car le run `eq_ircut_oracle_postpatch_20260519_001639` reste entièrement en timeouts (`0/12`). Décision durable: **rester default-off**.

## 2026-05-19 (résumé compacté de l’ancienne todo `followup.md`)

- Le repro forensic de référence est désormais propre et gelé: `reports/forensic_case055_repro_v1/input_nowcs.fit` + `reports/forensic_m106_reference_v1/index`.
- Le couple OFF/ON court a été revalidé sur ce repro sain, puis poussé juste assez loin pour isoler la chaîne causale post-pairset.
- Les anciens coupe-circuits prioritaires ont été franchis/neutralisés un par un dans l’enquête canonique:
  - `strict_pairset_scale_gate_reject`
  - fuite `strict_lowpairs_scale_guard`
  - `blind_prob_verify` gate Ze appliqué trop tôt sur le chemin natif
- Après ces corrections/bypass ciblés, le chemin natif ON atteint `accept_gate`, ce qui ferme le faux diagnostic "bloqué très en amont".
- La chaîne cible Astrometry de l’acceptation native a été reconstituée et figée (`verify séquentiel -> banding totune/tokeep -> tune_reverify éventuel -> accept_gate`).
- Les comparaisons ON candidates (`d50_3017`, `d50_2627`, `d50_2628`) montrent un pattern stable: `validate_base=OK`, puis rejet `accept-logodds below toprint` avec `accept_logodds=prob_logodds=-1.3862943611`.
- Le diagnostic causal prioritaire a été déplacé des **seuils** vers le **feed probabiliste amont**:
  - `accept_logodds_source=prob_logodds_astrometry_seq`
  - `prob_match_ratio=0.0`
  - `prob_matches=0`, `prob_conflicts=0`, `prob_distractors=1`
  - `verify_refperm_seed_ids_n=0`, `verify_refperm_final_ids_n=0`
- Le neutraliseur le plus proche identifié à ce stade est le préfixe `distractor` du verify séquentiel, qui fige `bestlogodds` dès `i=0`; même quand un match tardif apparaît (`d50_2627`), il ne suffit pas à renverser ce préfixe.
- Le fil de preuve post-pairset a été consolidé dans `reports/forensic_case055_repro_v1/post_pairset_forensic_index.md` pour éviter de redisperser l’enquête.
- Conséquence durable: pour rendre ZeBlind seul **fonctionnel et juste**, la suite prioritaire n’est plus de retoucher les seuils d’acceptation, mais de **réparer l’alimentation du verify séquentiel natif** (`refperm`, pool ref/test, `testsigma`, ordre des test-stars, gate `5sigma`).
- Forensic refperm complété via `reports/forensic_case055_repro_v1/refperm_pool_construction_summary.json`: sur `d50_3017`, `d50_2627`, `d50_2628`, `verify_refperm_* = 0` ne signifie pas que les IDs sont perdus avant verify. Les IDs sont visibles dans `inside_image` (ou `inside_image_fallback_unfiltered`), mais la branche active `field_native_astrometry_vs_tile_world` est non-strict et tronque directement le pool à `12` refstars sans exporter de `refperm` debug. Lecture durable: le prochain suspect causal concret est la **qualité/ordre** de ce pool final de `12`, pas une disparition brute d’IDs.
- Comparatif `test_xy/ref_xy/tests sigma` publié dans `reports/forensic_case055_repro_v1/test_ref_testsigma_comparison_summary.json`: sur ON, le verify séquentiel compare `51` test-stars plein champ à un pool ref capé à `12` étoiles en `gamma_like`. Sur `d50_3017`, le tout premier test est déjà hors gabarit (`nsig2≈4736`), ce qui empoisonne immédiatement le préfixe; sur `d50_2627`, un match tardif existe (`i=3`, `nsig2≈18.05`), ce qui rend moins crédible un blocage purement dû au gate `5σ` et renforce l’hypothèse **pool/order/projection ref**.
- Diagnostic causal affiné dans `reports/forensic_case055_repro_v1/distractor_prefix_rootcause_ordering_summary.json`: le premier sous-composant qui force le préfixe `distractor` n’est probablement pas le gate `5σ`, mais l’**ordre actuel des test-stars plein champ** contre un pool ref local compact. Preuve clé: sur `d50_2627`, les trois premiers tests sont des distractors massifs, puis un vrai match arrive déjà à `i=3` (`gate_pass=true`, `nsig2≈18.05`) mais trop tard pour sauver le meilleur préfixe.
- Premier delta causal order-only testé dans `reports/forensic_case055_repro_v1/reorder_collect_delta_summary.json`: `blind_astrometry_mirror_testperm_from_collect_enabled=True` réordonne une partie de la tête des test-stars, mais échoue à déplacer le tout premier outlier. Résultat durable: `accept_logodds` reste `-1.3862943611`, `prob_match_ratio=0.0`, `best_i=0`. Lecture produit: le prochain levier doit agir sur la **sélection initiale** des test-stars, pas seulement un reorder partiel.
- Delta plus agressif de sélection locale-first validé dans `reports/forensic_case055_repro_v1/forced_localfirst_delta_ab.json`: en forçant un sous-ensemble de test-stars locales en tête sur `d50_2627`, le premier `match` passe à `i=0`, `prob_match_ratio` passe à `1.0` et `accept_logodds` remonte de `-1.3862943611` à `-0.1570168371`. Lecture durable: l’ordre/sélection initiale des test-stars est bien un levier causal réel; le prochain verrou n’est plus le tout premier `distractor`, mais ce qui empêche encore ce match initial de franchir `toprint`.
- Test de composition locale refpool dans `reports/forensic_case055_repro_v1/localfirst_ref24_delta_ab.json`: augmenter `blind_astrometry_verify_refstar_max_keep` de `12` à `24` (en gardant le même test local-first forcé) dégrade au contraire le score (`accept_logodds: -0.157 -> -0.850`) sans générer de match supplémentaire utile. Lecture durable: le prochain levier ne sera pas un élargissement naïf du pool ref; il faut plutôt resserrer/intelligemment sélectionner les test-stars autour du support projeté utile.
- Test de sous-ensemble ultra-serré dans `reports/forensic_case055_repro_v1/localfirst_top1_delta_ab.json`: réduire le verify au seul meilleur candidat local (donc zéro distractor ultérieur) laisse `accept_logodds` inchangé à `-0.1570168371`. Lecture durable très forte: même un match “propre” reste intrinsèquement trop faible; le prochain levier doit viser la **qualité géométrique / scoring local du match** (projection ref ou `testsigma`), pas la simple pollution de la séquence.
- Test de scoring local dans `reports/forensic_case055_repro_v1/top1_sigma2_delta_ab.json`: sur ce même cas `top1`, augmenter `blind_astrometry_verify_sigma_parity_factor` à `2.0` transforme le match unique de `accept_logodds=-0.1570168371` à `+5.2259867559` (`nsig2≈18.05 -> 4.51`). Lecture durable: le calibrage `testsigma` n’est pas un détail, c’est désormais un levier causal confirmé de l’acceptation native.
- Confirmation sur cas moins artificiel dans `reports/forensic_case055_repro_v1/localfirst16_sigma2_delta_ab.json`: le même levier `blind_astrometry_verify_sigma_parity_factor=2.0` fait aussi monter `localfirst16` de `accept_logodds=-0.1570168371` à `+5.2259867559`, malgré le retour des distractors. Lecture durable: le problème restant n’est plus “est-ce que le levier sigma aide ?” mais “pourquoi il manque encore ~quelques unités de logodds / ou un meilleur alignement global pour franchir le seuil utile ?”.
- Quantification du résiduel publiée dans `reports/forensic_case055_repro_v1/localfirst16_sigma2_remaining_gap.json`: le cas ON amélioré est au-dessus du `blind_accept_logodds_min` par défaut (`0.25`) mais reste à `-6.774` sous le seuil forensic observé `toprint=12.0`. Lecture durable: le chantier restant n’est pas un micro-ajustement, il manque encore au moins un levier structurel au-dessus du seul recalage `testsigma`.
- Probe A/B OFF/ON publiée dans `reports/forensic_case055_repro_v1/off_on_verify_entry_probe_ab.json`: avec le meilleur probe ON actuel (`localfirst16 + sigma_parity_factor=2.0`), ON matérialise enfin un vrai préfixe verify (`accept_logodds=+5.2259867559`, source `prob_logodds_astrometry_seq`), alors que OFF ne remonte toujours pas de métriques verify comparables sur ce test ciblé et échoue plus tard sur `validation_failed[rms_ok=0,...]`. Lecture durable: le prochain écart unique n’est pas encore purement numérique; c’est une **asymétrie de point d’observation / de chemin effectif** entre OFF et ON.
- A/B isomorphe obtenu sans patch code dans `reports/forensic_case055_repro_v1/off_on_validate_gate_isomorphic_summary.json`: l’extraction montre qu’OFF exposait déjà `validate_gate`/`validate_base` dans `verify_hit_trace`. Lecture durable mise à jour: l’asymétrie d’observation est levée; le prochain écart causal réel est désormais le contenu du gate lui-même — OFF échoue sur une géométrie fausse (`rms≈51 px`, `scale≈4.60"/px`), ON sur une preuve probabiliste encore insuffisante (`+5.226 < 12`).
- Décision d’ancrage produit publiée dans `reports/forensic_case055_repro_v1/off_on_anchor_decision_summary.json`: pour ce case055, le chemin utile à renforcer est **ON/natif-probabiliste**, pas OFF. Lecture durable: OFF ressemble à une hypothèse géométriquement fausse, alors qu’ON matérialise déjà un support local plausible; la suite doit donc renforcer la preuve native ON jusqu’à la zone d’acceptation forensic.
- Renforcement ON via ref support local publié dans `reports/forensic_case055_repro_v1/localfirst16_sigma2_ref4local_delta_ab.json`: en gardant `localfirst16 + sigma_parity_factor=2.0` mais en resserrant le ref support aux `4` étoiles les plus proches du match utile, `accept_logodds` monte de `+5.2259867559` à `+6.3245990446`. Lecture durable: le support ref local est bien un levier causal additionnel, mais il ne suffit pas encore à lui seul à franchir la bande forensic `toprint≈12`.
- Poursuite du resserrement ref support dans `reports/forensic_case055_repro_v1/localfirst16_sigma2_ref2local_delta_ab.json`: passer de `4` à `2` ref stars fait encore monter `accept_logodds` à `+7.0177462251`. Lecture durable: le levier “support ref local plus serré” n’a pas encore saturé; il reste causalement actif.
- Test ultime du support ref local dans `reports/forensic_case055_repro_v1/localfirst16_sigma2_ref1local_delta_ab.json`: avec un seul ref star local, `accept_logodds` monte encore à `+7.7108934057`, mais reste sous `toprint≈12`. Lecture durable: même en poussant ce levier à l’extrême, il manque encore ~`4.29` logodds; la suite doit donc viser une correction plus structurelle que le simple resserrement du support local.
- Plafond pratique formalisé dans `reports/forensic_case055_repro_v1/current_forensic_probe_ceiling_summary.json`: la meilleure combinaison forensic actuelle (`localfirst16 + sigma2 + ref1local`) reste ~`4.29` logodds sous `toprint≈12`. Lecture durable: il faut maintenant sortir du micro-tuning de forçages externes et prototyper une correction **structurelle native** inspirée de ces gains.
- Prototype structurel natif introduit dans `zeblindsolver.py` via `blind_astrometry_native_local_support_enabled`, validé par compilation et premier A/B dans `reports/forensic_case055_repro_v1/native_local_support_proto_ab.json`. Lecture durable: sans JSON forcé, le flux natif passe de `accept_logodds=-1.3862943611` à `+1.0671036725`, donc la direction est bonne; mais la tête de séquence reste trop proche du flux original (trois distractors initiaux), signe que le comportement `nearest_nsig2` attendu n’est pas encore pleinement appliqué côté natif.
- Correctif du prototype natif validé dans `reports/forensic_case055_repro_v1/native_local_support_proto_v2_ab.json`: en laissant le shaping local s’appliquer réellement côté natif ON, le flux réel atteint `accept_logodds=+5.2259867559` sans JSON forcé, avec un `match` en tête et `prob_verify_nt=8`. Lecture durable: une part importante de la recette forensic est désormais récupérée **structurellement** dans le chemin natif réel.
- Extension structurelle confirmée dans `reports/forensic_case055_repro_v1/native_local_support_ref4_ab.json`: avec le même flux natif réel, réduire `blind_astrometry_verify_refstar_max_keep` de `12` à `4` fait monter `accept_logodds` de `+5.2259867559` à `+6.3245990446` sans JSON forcé. Lecture durable: le gain `ref-local` se transpose lui aussi dans le chemin natif réel.
- Test natif `refstar_max_keep=2` dans `reports/forensic_case055_repro_v1/native_local_support_ref2_ab.json`: aucun gain supplémentaire observé et `prob_verify_nr` reste à `4`. Lecture durable: un plancher interne `refstar_max_keep>=4` empêche encore de reproduire structurellement les probes forensic `ref2local/ref1local` dans le flux natif réel.
- Plancher natif levé puis revalidation dans `reports/forensic_case055_repro_v1/native_local_support_ref2_postfloor_ab.json`: quand `blind_astrometry_native_local_support_enabled=True`, autoriser `refstar_max_keep<4` permet bien de récupérer le gain structurel `ref2local`, avec `accept_logodds=+7.0177462251` et `prob_verify_nr=2` dans le flux natif réel. Lecture durable: la transposition native de la recette forensic continue de tenir.
- Test ultime natif réel dans `reports/forensic_case055_repro_v1/native_local_support_ref1_ab.json`: passer de `ref2` à `ref1` dégrade le score de `+7.0177462251` à `+5.9656457493`. Lecture durable: dans le flux natif réel, le meilleur support ref local observé est actuellement `ref2`, pas `ref1`; un resserrement excessif fait perdre la bonne identité/géométrie du support.
- Plafond structurel natif formalisé dans `reports/forensic_case055_repro_v1/native_structural_ceiling_summary.json`: la meilleure recette native réelle actuelle (`native_local_support + sigma2 + ref2`) plafonne à `+7.0177462251`, encore ~`4.98` sous `toprint≈12`. Lecture durable: la famille “ordre/support local” a donné l’essentiel de ce qu’elle pouvait; la suite doit cibler la **géométrie/projection native amont**.
- Premier probe géométrique amont publié dans `reports/forensic_case055_repro_v1/native_projection_shift_ref2_probe_ab.json`: appliquer un simple shift local au support ref `ref2` fait monter le score natif réel de `+7.0177462251` à `+9.2741788765`, avec `d2min` du premier match qui tombe à `0`. Lecture durable très forte: le gap résiduel vient en grande partie d’un **décalage projectif local** encore non compensé dans la projection native actuelle.
- Correction structurelle native prototypée dans `zeblindsolver.py` via `blind_astrometry_native_local_ref_recenter_enabled`, validée par compilation et A/B dans `reports/forensic_case055_repro_v1/native_local_ref_recenter_proto_ab.json`: sans JSON forcé, le flux natif réel monte bien de `+7.0177462251` à `+9.2741788765`. Lecture durable: le recentrage local du support ref projeté récupère structurellement l’essentiel du gain du probe géométrique.
- Gap résiduel quantifié dans `reports/forensic_case055_repro_v1/native_local_ref_recenter_remaining_gap.json`: après support local + sigma2 + ref2 + recentrage natif, il ne reste plus qu’environ `2.73` logodds sous `toprint≈12`. Lecture durable: on est probablement à portée d’un dernier levier local étroit avant de devoir conclure à un manque plus global.
- Dernier levier local testé dans `reports/forensic_case055_repro_v1/native_recenter_sigma25_ab.json`: pousser `sigma_parity_factor` de `2.0` à `2.5` dégrade le score (`+9.2741788765 -> +8.8278917739`). Lecture durable: le meilleur point local observé reste `sigma=2.0 + ref2 + recentrage local`, et le résiduel `~2.73` ne se ferme pas par un simple cran de sigma.

## 2026-05-19 (audit Astrometry -> cran géométrie locale native, reprise compacte)

- Un contrat de parité exécutable pour le prochain cran a été figé dans `reports/forensic_case055_repro_v1/next_cran_parity_contract.md`. Lecture durable: pour la suite immédiate, les invariants à préserver sont les entrées verify natives, le support local explicite, une variable de travail limitée à la projection/géométrie locale, `sigma_parity_factor=2.0` fixé, et une comparaison toujours faite au même étage `validate_gate`.
- Instrumentation durable ajoutée au trace natif dans `zeblindsolver.py`: le `verify_hit_trace` exporte maintenant les champs de shaping local utiles (`verify_teststar_filter_mode`, cardinalités `test/ref`) ainsi que ceux du recentrage natif (`verify_local_ref_recenter_mode`, `shift_xy`, ancres test/ref, `d2min_before/after`). Validation publiée dans `reports/forensic_case055_repro_v1/native_local_ref_recenter_instrumentation_check.json`.
- Test de nouveau mode de recentrage `local_centroid` publié dans `reports/forensic_case055_repro_v1/native_local_ref_recenter_mode_ab.json`: échec fort (`accept_logodds: +9.2741788765 -> -1.3862943611`). Lecture durable: un shift global par centroïde détruit la bonne accroche locale; le problème n’est pas un simple décalage moyen du nuage ref.
- Test de nouveau mode de choix d’ancre `head_objective` publié dans `reports/forensic_case055_repro_v1/native_local_ref_recenter_head_objective_ab.json`: échec partiel (`+9.2741788765 -> +6.5790565224`). Lecture durable: optimiser un coût sur la tête locale déplace le premier match utile trop tard (`i=2`) et reste moins bon que le recentrage `anchor_pair`.
- Plafond actuel de la famille “géométrie locale / recentrage” formalisé dans `reports/forensic_case055_repro_v1/current_native_geometry_family_gap_summary.json`: le meilleur mode observé reste `anchor_pair` à `+9.2741788765`, encore ~`2.73` sous `toprint≈12`. Lecture durable: le prochain cran utile doit probablement sortir de la micro-famille “recentrage local” et viser une correction plus globale de projection/géométrie native.

## 2026-05-19 (état consolidé après audit Astrometry + refonte du cran suivant)

- L’audit `reports/astrometry_math_audit_20260519_0114.md` reste désormais le document de référence pour la suite ZeBlind. Lecture durable: le problème n’est pas un simple seuil mais un **hybride de sémantique** entre verify Astrometry et garde-fous/logique Ze.
- Invariants P0 validés comme structurants pour toute reprise future:
  1. entrées verify Astrometry natives (`full field stars` vs `full projected index stars`),
  2. ordre/support ref cohérent avec Astrometry,
  3. géométrie locale projetée fidèle,
  4. `testsigma²` réellement aligné,
  5. chemin d’acceptation comparé sans gates Ze parasites.
- Sur `case055`, le meilleur chemin natif réel obtenu à ce stade est `accept_logodds≈+9.2741788765`, encore sous `toprint≈12` mais nettement au-dessus de la baseline historique `-1.3862943611`.
- Lecture durable importante: la famille “ordre/support local” a déjà donné l’essentiel de ses gains; le prochain cran utile doit cibler la **géométrie/projection locale native amont**.
- `followup.md` a été remanié à nouveau pour devenir une checklist courte, ancrée sur l’audit Astrometry, avec un prochain cran unique: figer un contrat de parité exécutable, instrumenter le recentrage/projection locale, tester un seul patch amont, puis décider si le résiduel restant est encore local ou devient global.

## 2026-05-18 (garde-fou `min_unique_obs_quads`)

- Ajout d'un garde-fou expérimental `blind_pairset_scale_recenter_min_unique_obs_quads` dans `zeblindsolver.py`, laissé à `0` par défaut pour ne pas changer le comportement produit.
- Validation causale du faux candidat `d50_2631`: avec `min_unique=3`, le `recenter_keep` est rejeté proprement car seulement **2 quads observés distincts** soutiennent les paires sauvées.
- Mini-validation de non-régression exécutée sur des originaux M106 avec `blind_reuse_existing_solved_wcs=False`:
  - `233828` résout avec `min_unique=0` et `3`
  - `232945` résout avec `min_unique=0` et `3`
  - `233130` résout avec `min_unique=0` et `3`
- Point durable important: sur `233130`, le solve gagnant utilise un vrai `recenter_keep` avec `unique_obs_quads=3` exactement. Lecture produit: **3** est actuellement le plus haut seuil observé compatible avec ce mini-lot; au-delà, risque de casser des solves réels.

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
- 2026-05-06 (R8.1 bloc 3): alignement du placement des seuils verify en strict séquentiel: `logodds_bail`/`logodds_stoplooking` appliqués dans la boucle séquentielle, `logodds_accept` pris comme gate de décision (`effective_min_prob = verify_logodds_accept`) au point de validation; bypass des dynamiques legacy sur ce chemin.
- 2026-05-06 (R8.1 bloc 3, preuve): compile OK + run oracle capé documenté (`reports/r8_1_block3_20260506_0035.{json,md}`).
- 2026-05-06 (R8.1 bloc 4): ajout d’un dump diffable verify pas-à-pas (`prob_verify_steps`, `prob_theta_transitions`) avec schéma `zeblind.verify_step_diffable.v1`; artefact de preuve: `reports/r8_1_verify_step_diffable_20260506_0035.json`.
- 2026-05-06 (R8.2 bloc 1): ajout d’un filtre RoR strict (`_verify_ror2`, `_apply_verify_ror_filter`) branché avant la vérification séquentielle, avec recalcul `effective_area` et compteurs `NR/NT` avant/après.
- 2026-05-06 (R8.2 bloc 2): alignement des bascules verify finalisé: ajout du toggle explicite `blind_verify_do_ror` (config + CLI `--blind-verify-do-ror`) et export des 3 flags effectifs `verify_do_uniformize` / `verify_do_dedup` / `verify_do_ror` dans les stats pour diff run-à-run. Preuve technique: `python3 -m py_compile zeblindsolver/zeblindsolver.py` OK.
- 2026-05-06 (R8.2 bloc 3): preuve oracle A/B RoR ajoutée avec compteurs comparables `NR/NT/effA` via dump paires validation + replay `_apply_verify_ror_filter` (`reports/r8_2_block3_validation_pairs_20260506_0630.json`, `reports/r8_2_block3_ror_oracle_pairs_ab_20260506_0633.json`).
- 2026-05-06 (R8.3): cycle tune/reverify aligné côté `solver_handle_hit` avec seuils explicites `totune/tokeep/toprint` (config + CLI), décision par bandes (`below_totune`, `between_tune_keep`, `above_keep`) et compteurs de divergence pré/post tune (`tune_reverify_stage_band_changed/unchanged`). Preuves: `reports/r8_3_block1_20260506_0858.json` et `reports/r8_3_block1_20260506_0905.md`.
- 2026-05-06 (R8.4): ajout du mode strict `blind_astrometry_strict_abpquad_incremental_enabled` (config+CLI) pour traversée candidats plus iso Astrometry (rerank non-iso coupé, cap objet strict), + traçage `startobj/endobj/objects_cap` et instrumentation `astrometry_perm_constraints_prealign_hits` pour positionner explicitement les contraintes `cx<=dx/meanx` avant gates aval. Preuves: `reports/r8_4_block1_abpquad_ab_20260506_0930.json`, `reports/r8_4_block123_20260506_0940.json`, `reports/r8_4_block123_20260506_0945.md`.
- 2026-05-06 (R8.5 pré-check): mini-lot strict 4 images exécuté (`reports/r8_5_minilot_20260506_0955.json`) ; signature dominante encore mixte (2x `verify_prob`, 2x `pairing`), donc R8.5 non clos pour l’instant malgré `top_verify_reason_code=validation_failed` stable.
- 2026-05-06 (audit R8.5): correction de cap importante — le mode `blind_astrometry_strict_abpquad_incremental_enabled` n'implémente pas encore la vraie mécanique `newpoint + add_stars` de `solver.c`; il ne fait qu'un durcissement d'orchestration. Conséquence: R8.4 #1/#3 réouverts et R8.5 bloqué tant que ce gap amont n'est pas fermé (`reports/r8_5_pipeline_audit_20260506_1725.md`).
- 2026-05-06 (audit R8.5 pass2): `startobj/endobj` sont désormais branchés fonctionnellement (config+CLI + fenêtre runtime réelle sur la traversée strict incremental), preuve `reports/r8_5_startendobj_ab_20260506_1732.json`; mais le vrai gap amont persiste: pas encore de boucle `newpoint + add_stars` récursive type `solver.c` (`reports/r8_5_pipeline_audit_20260506_1740.md`).
- 2026-05-06 (R8.4 complétion): implémentation du parcours observé `newpoint` strict (`_build_newpoint_quad_order`) + branchement runtime fonctionnel `startobj/endobj` (config+CLI). Preuves: `reports/r8_4_newpoint_ab_20260506_1748.json`, `reports/r8_4_newpoint_default_20260506_1756.json`, `reports/r8_5_startendobj_ab_20260506_1732.json`, synthèse `reports/r8_4_completion_20260506_1802.md`.
- 2026-05-06 (R8.5 progression): après complétion R8.4, le mini-lot strict devient homogène en stage dominant (`verify_prob` 4/4, `reports/r8_5_post84_minilot_20260506_1746.json`), mais l'oracle garde des `remaining_gaps` structurels avec blocker `try_perm_meanx_rejects` (`reports/r8_5_post84_oracle_20260506_1742.json`).
- 2026-05-06 (R8.5 #1 itération): correctif mécaniste `try_permutations` aligné C sur la source de code (contraintes cxdx/meanx appliquées côté index `origcode`/`dst4`, pas sur `src4`). Oracle post-fix: `reports/r8_5_permcode_src_vs_dst_20260506_1829.json` (gaps structurels encore ouverts). Diagnostic: désactiver cxdx/meanx est une fausse bonne idée (`reports/r8_5_meanx_gate_ab_20260506_1808.json`).
- 2026-05-06 (R8.5 itération amont): ajout d'une trace structurée `validation_failed_meta` (rms/inliers/seuils + `rms_ratio`/`inliers_margin`) dans `verify_trace`/rejets, et audit d'attrition par phase (`reports/r8_5_upstream_phase_attrition_20260506_1858.json`). Constat renforcé: pertes majoritaires en scale_only/blind et échecs verify surtout RMS-catastrophiques.
- 2026-05-06 (R8.5 garde gradué): implémentation d'un garde géométrique pré-verify strict (`blind_astrometry_preverify_geom_guard_*`) avant `validate_solution`, avec compteurs dédiés. Oracle A/B: runtime réduit (~62s->~46s) et validations réduites (12->8), mini-lot conserve `verify_prob` 4/4; gap structurel #1 encore ouvert. Preuves: `reports/r8_5_preverify_guard_oracle_20260506_1918.json`, `reports/r8_5_preverify_guard_minilot_20260506_1922.json`.
- 2026-05-06 (R8.5 calibration): calibration du garde pré-verify RMS testée en sweep oracle+mini-lot; compromis retenu provisoire `soft/hard=70/200` (mêmes signatures structurelles, coût moyen inférieur à 90/260). Rejoué avec défauts calibrés (`reports/r8_5_post_calib_default_oracle_20260506_1959.json`), R8.5 #1 reste ouvert.
- 2026-05-06 (parité principe): décision explicite avec Tristan de viser un fonctionnement **identique** Astrometry C (pas un simple mimétisme comportemental). Audit de gaps pipeline restants ajouté: verify-only chain, solver_handle_hit predistort/pixel_xscale, orchestration onefield/engine, ANODDS runtime (`reports/r8_5_pipeline_mapping_gap_audit_20260506_2010.md`).
- 2026-05-06 (R8.7 ANODDS): patch runtime pour aligner les seuils log-odds stricts depuis des odds Astrometry (`blind_anodds_semantics_enabled` + paramètres `*_toprint/tokeep/totune/bail/stoplooking`). Oracle confirme `accept=log(1e6)=13.8155` et `bail=log(1e-100)=-230.2585` (`reports/r8_6_r8_7_anodds_runtime_oracle_20260506_2037.json`).
- 2026-05-06 (R8.6 progression aval): ajout de la télémétrie `fake_match` dans `verify_hit_trace` et `astrometry_match_objects` + flag scaffold `blind_verify_only_injected_wcs_enabled` pour préparer le mode verify-only injecté. Preuve: `reports/r8_6_fake_match_trace_scaffold_20260506_2049.json`.
- 2026-05-06 (R8.6 verify-only): ajout d'un mode runtime `blind_verify_only_injected_wcs_enabled` qui bypass la génération candidats quand un WCS header valide est injecté, avec sortie `SOLVMODE=BLIND_VERIFY_ONLY_INJECTED` et traces `fake_match=true` (verify_hit_trace / astrometry_match_objects / stage_by_stage). Preuve: `reports/r8_6_verify_only_injected_smoke_20260506_2100.json`.
- 2026-05-11 (M5/P0 verify-only diffable): le chemin `BLIND_VERIFY_ONLY_INJECTED` exporte désormais `solvmode`, `onefield_solution_records`, `solver_handle_hit_runtime_trace` et un bloc `parity_diffable.solver_handle_hit` complet (fingerprint CRC32). Validation A/B sur image ASTAP solved: fingerprint stable `a16b2de9`, `transaction_runtime_count=1`, `trace_invalid_count=0` (`reports/r8_p0_verifyonly_ab_diffable_20260511_0834.md`).
- 2026-05-11 (R8/P1 qualité correspondances pré-verify): patch ciblé dans `zeblindsolver.py` pour plafonner la promotion de tolérance issue de `resolve_hit` (`blind_astrometry_resolve_hit_tol_promote_cap_factor=1.75`, compteur `astrometry_resolve_hit_tol_promote_capped`). A/B contrôlé sur `apod2` à config identique (caps runtime stricts): **cap actif** => `accept_new_best=30`, `accept_inliers_max=5`, `accept_rms_median=1.63`; **no-cap (factor=1000)** => `accept_new_best=52`, `accept_inliers_max=12`, `accept_rms_median=37.16`. Conclusion durable: la dérive de `tol_deg` est un amplificateur direct du pattern "inliers apparents élevés + géométrie fausse" avant verify. Rapport: `reports/r8_p1_delta_prehit_tol_promotion_cap_20260511_0919.md`.
- 2026-05-11 (R8/P1 extension apod3): A/B court reproduit le même effet que sur apod2 pour le cap de promotion `resolve_hit->tol_deg`. Sur `apod3` avec mêmes caps runtime: **cap actif** (`factor=1.75`) => `tol_promote_capped=944`, `accept_inliers_max=5`, `accept_rms_median=1.79`; **no-cap** (`factor=1000`) => `accept_inliers_max=12`, `accept_rms_median=33.72`. Conclusion consolidée sur 2 images: la dérive de tolérance pré-verify est un facteur causal robuste du pattern "inliers élevés + géométrie fausse". Rapport: `reports/r8_p1_delta_prehit_tol_promotion_cap_apod3_20260511_0934.md`.
- 2026-05-11 (R8/P1 consolidation pack3): extension A/B du cap de promotion `resolve_hit->tol_deg` à `apod4` puis consolidation `apod2/apod3/apod4`. Pattern confirmé sur 3/3: sans cap, inliers apparents montent (`accept_inliers_max` 10-12) mais RMS médian des `accept_new_best` devient massif (~32-37 px); avec cap actif, RMS médian reste ~1.6-1.8 px. Conclusion durable renforcée: le contrôle de promotion de tolérance est un garde-fou causal sur la qualité géométrique pré-verify. Rapport de synthèse: `reports/r8_p1_delta_prehit_tol_promotion_cap_pack3_20260511_0938.md`.
- 2026-05-11 (R8/P1 delta #2 pool adaptatif): ajout d'un fallback adaptatif `resolve_hit` global->local (seuils mutualité + meddist/tol) avec compteurs `astrometry_resolve_hit_pool_adaptive_*`. Vérification A/B ON/OFF sur apod2/apod4 en protocole court: checks actifs (688/1026) mais `swaps=0`, métriques identiques (pas de gain RMS/inliers). Conclusion durable: ce levier n'est pas dominant sur ce protocole; garder l'instrumentation mais reprioriser le mapping amont comme prochain verrou.
- 2026-05-11 (R8/P1 delta #3 mutualité): ajout d'un gate optionnel de mutualité `resolve_hit` (ratio `mutual_kept/kept`) avec instrumentation dédiée. Calibration sur `apod2` en stress no-cap (`tol_promote_cap_factor=1000`): seuil 0.90 réduit fortement le bruit géométrique (`accept_rms_median` 37.16 -> 2.65). En mode nominal cap-actif, effet marginal (RMS déjà ~1.63). Décision: conserver le mécanisme désactivé par défaut en attendant validation solve-rate mini-lot. Rapport: `reports/r8_p1_delta_prehit_mutual_ratio_gate_calib_20260511_0956.md`.
- 2026-05-11 (R8/P1 mini-lot cap-actif + mutualité 0.90): vérification sur `apod2/apod3/apod4` avec cap `tol_deg` actif: RMS médian des `accept_new_best` reste stable (~1.6-1.8 px), `accept_new_best` baisse légèrement (-1/-2), pas de régression visible sur ce protocole court. Décision: mécanisme conservé en mode durcissement optionnel (pas activé par défaut tant que solve-rate complet non prouvé). Rapport: `reports/r8_p1_delta_prehit_mutual_ratio_gate_capactive_minilot_20260511_1002.md`.
- 2026-05-11 (R8/P1 delta #4 self-rms): ajout d'un gate optionnel sur le RMS interne des correspondances `resolve_hit` (`self_rms_px`). Calibration apod2 en stress no-cap: `accept_rms_median` chute de 37.16 à 1.74 (seuil 4.0), avec réduction du support apparent (`accept_new_best` 52 -> 34). Vérification mini-lot cap-actif (apod2/3/4): RMS médian déjà propre inchangé, léger durcissement du support (-1/-2 accepts). Décision: conserver désactivé par défaut (mode durci optionnel), cap `tol_deg` reste le garde principal. Rapport: `reports/r8_p1_delta_prehit_self_rms_gate_calib_20260511_1010.md`.
- 2026-05-11 (R8/P1 delta #5 ranking high-RMS): ajout d'une pénalité soft dans `_hypothesis_quality_metric` pour les hypothèses `rms_px` > 8 px (`blind_hypothesis_rank_high_rms_*`, activé par défaut). A/B apod2 stress no-cap: RMS médian `accept_new_best` 37.16 -> 12.24 (effet net mais moins fort qu'un gate dur). Mini-lot cap-actif apod2/3/4: pas de régression visible (RMS déjà propre, écarts minimes). Rapport: `reports/r8_p1_delta_prehit_hypothesis_rank_high_rms_penalty_20260511_1022.md`.
- 2026-05-11 (R8/P1 delta #6 pool adaptive self-RMS trigger): enrichissement du fallback global->local `resolve_hit` avec critère `self_rms` (nouveaux knobs `blind_astrometry_resolve_hit_pool_adaptive_max_self_rms_px`, `..._self_rms_min_pairs`) + instrumentation `astrometry_resolve_hit_pool_adaptive_self_rms_*`. Résultat: le mécanisme est enfin actif (swaps observés: apod2=1, apod4=5), mais sur protocole court cap-actif et no-cap les métriques `accept_*` et le statut solve restent inchangés (`success=false`, `tx_runtime_count=0`). Rapport: `reports/r8_p1_delta_prehit_pool_adaptive_selfrms_trigger_20260511_1050.md`.
- 2026-05-11 (R8/P1 delta #7 dst prefilter resolve_hit): implémentation d'un préfiltre destination (`dst`) en strict path dans `_resolve_hit_correspondences` avec nouveaux knobs `blind_astrometry_strict_resolve_hit_dst_prefilter_*`, câblage sur tous les callsites et télémétrie `dst_eval_ratio`. Sur mini-lot cap-actif: réduction massive du pool destination évalué (~0.90 -> ~0.04/0.05) avec effet aval neutre à légèrement positif selon image. En stress no-cap: effet non robuste (RMS médian plutôt en recul). Décision: conserver OFF par défaut tant que solve-rate/hit-bearing n'est pas amélioré. Rapport: `reports/r8_p1_delta_prehit_resolve_hit_dst_prefilter_20260511_1134.md`.
- 2026-05-11 (R8/P1 delta #8 candidate_pairs-guided retry): ajout d'un retry optionnel `resolve_hit` guidé par `candidate_pairs` (top-k sur erreur pairwise prédite) avec nouveaux knobs `blind_astrometry_resolve_hit_pool_guided_*` et compteurs `astrometry_resolve_hit_pool_guided_*`. Validation technique OK (py_compile + tests). Probes cap-actif (apod2/apod4) et no-cap (apod2): mécanisme exécuté (`guided_checks` élevés) mais aucun swap retenu (`guided_swaps=0`) et impact aval nul sur `accept_*`/solve status. Décision: conserver la brique instrumentée, prochain focus sur critère guided plus discriminant (score structurel/pool quality). Rapport: `reports/r8_p1_delta_prehit_candidate_pairs_guided_retry_20260511_1152.md`.
- 2026-05-11 (R8/P1 delta #9 guided structural scoring): enrichissement du retry `candidate_pairs-guided` par un score structurel (cohérence d’échelle radiale log) via `blind_astrometry_resolve_hit_pool_guided_structural_*`. Validation technique OK. Probes cap-actif (apod2/apod4) + no-cap (apod2): score structurel bien exécuté (`guided_structural_scored` non nul) mais toujours aucun swap guided (`guided_swaps=0`) ni gain aval sur métriques solve-run court. Rapport: `reports/r8_p1_delta_prehit_candidate_pairs_guided_structural_score_20260511_1203.md`.
- 2026-05-11 (R8/P1 delta #10 unified compare): implémentation d'un comparateur unifié global/local/guided dans la branche `weak_global` (`blind_astrometry_resolve_hit_pool_unified_compare_*`) avec score explicite (kept, med_ratio, self_rms) et compteur `astrometry_resolve_hit_pool_unified_*`. Validation technique OK. Sur probes courts: des décisions de mode explicites apparaissent (guided/local/unified_selected non nuls selon cas), mais sans amélioration aval du solve status (`success=false`, `tx_runtime_count=0`). Rapport: `reports/r8_p1_delta_prehit_unified_compare_global_local_guided_20260511_1238.md`.
- 2026-05-11 (R8/P1 delta #11 unified-score sweep): ajout de knobs de calibration du score unifié (`blind_astrometry_resolve_hit_pool_unified_score_kept_weight`, `..._med_ratio_weight`, `..._self_rms_weight`) puis mini-sweep marge/poids (def/m0/aggr) sur apod2/apod4 cap/no-cap. Constat: les décisions pré-verify changent bien (`unified_selected` varie), mais aucun impact aval sur métriques solve-run court (`accept_*` inchangés, `success=false`, `tx_runtime_count=0`). Rapport: `reports/r8_p1_delta_prehit_unified_score_calibration_sweep_20260511_1254.md`.
- 2026-05-11 (R8/P1 oracle première divergence): ajout d'un comparateur diffable Python "first divergence" (OFF vs ON comparateur unifié) sur `apod2`, avec artefact JSON dédié. Première divergence décisionnelle localisée à `seq=61` (tile `d50_0716`, `obs_quad_idx=129`), transition `reject_not_better_than_best -> skip_zero_inliers`, corrélée à un basculement `resolve_hit` adaptive (`adaptive_pool_fallback_used`, `kept`, `med`, `self_rms`). Statut followup mis à [~] pour la brique C/Python: composant Python prêt, extension C↔Python encore ouverte. Rapport: `reports/r8_p1_first_divergence_py_oracle_unified_off_on_apod2_20260511_1304.md`.
- 2026-05-11 (R8/P1 C↔Python first-divergence v1): ajout du script `reports/tools/cp_first_divergence_compare.py` et exécution sur `r8_p1_c_trace_jpg_20260507_1936.json` vs `r8_p1_py_trace_20260507_1933.json`. Divergence détectée sur 3/3 images, première divergence au statut terminal (`solved=true` côté C vs `success=false` côté Python, ex. apod2). Limite actée: trace C actuelle sans granularité verify/tune/reverify, donc comparateur v1 terminal-level. Références: `reports/r8_p1_first_divergence_cp_oracle_v1_20260511_1332.json`, `reports/r8_p1_first_divergence_cp_oracle_v1_20260511_1334.md`.
- 2026-05-11 (R8/P1 audit pack3 first-gate Python): mini-lot `apod2/apod3/apod4` en config nominale cap-active + unified-default avec traces diffables complètes. Résultat 3/3: `success=false`, `tx_runtime_count=0`, `verify_prob` dominant; raisons verify dominantes `pixel_scale_out_of_range` (apod2/apod4) et `validation_failed` (apod3). Les `accept_new_best` existent avec RMS médian propre (~1.7-1.8), suggérant un blocage plutôt scale/verify que dérive RMS locale.
- 2026-05-11 (R8/P1 scale-gate focus): extraction dédiée du pack3 montre des rejets verify `pixel_scale_out_of_range` avec overshoot majeur sur apod2/apod4 (`scale` 54.545 et 46.339 pour `max=15.0`, soit ~x3.64 et ~x3.09). apod3 reste dominé par `validation_failed` mais contient aussi des rejets scale. Direction recommandée: delta d'instrumentation "scale origin" (tile/parity/level/index/model_scale) avant tout retuning de seuil.
- 2026-05-11 (R8/P1 probe scale-rescue max_arcsec): A/B rapide `blind_validate_scale_rescue_max_arcsec=30` vs `80` sur apod2/apod4. Avec 80, le rescue scale devient actif (`hits=1`) et la raison dominante bascule vers `validation_failed`, mais `tx_runtime_count` reste 0 et les RMS explosent (ex. 244 px / 88 px). Décision: ne pas élargir ce plafond en nominal; privilégier instrumentation de l’origine de dérive d’échelle.
- 2026-05-11 (R8/P1 scale-origin clusters): extraction des événements `pixel_scale_out_of_range` depuis `validation_pairs` du pack3 -> 12 événements / 8 clusters (tile,parity). Hotspot principal identifié: `d50_3503/nominal` (overshoot médian ~x14.99, max ~x22.89) présent sur apod2+apod4. Insight: la dérive d'échelle est structurée par clusters locaux, pas seulement un bruit global; piste delta suivante = pénalité de ranking clusterisée plutôt qu'assouplissement de seuils.
- 2026-05-11 (R8/P1 delta #12 scale-hotspot ranking): ajout d'une pénalité douce de ranking hypothèses pilotée par hotspots `tile/parity` et overshoot d'échelle (`blind_hypothesis_rank_scale_hotspot_penalty_*`, fallback scale max 15"). A/B court apod2/apod4 cap-actif: mécanisme actif (`checks/hits` non nuls) mais pas de gain aval (`success=false`, `tx_runtime_count=0`), avec léger durcissement sur apod4 (`accept_new_best` -1). Décision: conserver OFF par défaut.
- 2026-05-11 (R8/P1 C↔Python first-divergence v2 verbose): ajout d'un extracteur C verbose `reports/tools/c_verbose_trace_extract.py` (logodds updates, tweak2, essais index/solve), capture pack3 JPG `reports/r8_p1_c_trace_verbose_pack3_20260511_1429.json`, puis extension du comparateur `cp_first_divergence_compare.py` pour exploiter cette granularité. Résultat: divergence maintenue 3/3 avec stage classé `pre_hit_runtime_absent_py_after_c_verify_progress` (C: verify/tweak actif, Python: `tx_runtime_count=0`). Limite actée: classification encore heuristique tant que la trace Python n'expose pas un flux verify/tune/reverify équivalent.

- 2026-05-11 (post-audit, briques manquantes): ajout d'une instrumentation explicite de sémantique d'arrêt dans `astrometry_semantics.limits` (numtries/nummatches limits, reached flags, stop-code presence, consistency checks). Objectif: fermer le gap M3/P1 sur les conditions d'arrêt C diffables avant tout tweaking de seuils.
- 2026-05-11 (audit profond pipeline R8/P1): audit de fond consolidé avec preuves fraîches C verbose pack3 (`r8_p1_c_trace_verbose_pack3_20260511_1429.json`) et Python strict pack3 (`r8_p1_py_trace_pack3_strict_20260511_1442/py_trace.json`). Constat stable: C progresse verify/tweak et solve 3/3, Python reste `success=false` + `tx_runtime_count=0` (stage `verify_prob`). Oracle v2 mis à jour (`r8_p1_first_divergence_cp_oracle_v2_verbose_20260511_1444.json`), divergence classée `pre_hit_runtime_absent_py_after_c_verify_progress`. Priorisation décidée: P0 observabilité (flux step-events Python C-like) avant nouveaux deltas heuristiques.
- 2026-05-11 (P0 observabilité, étape suivante): export step-events Python pack3 strict ajouté (`verify_trace`, `verify_policy_trace`, `verify_hit_trace`) via artefact `reports/r8_p1_py_step_events_pack3_strict_20260511_1608.json`, puis oracle C↔Python v3 (`reports/r8_p1_first_divergence_cp_oracle_v3_with_py_step_events_20260511_1611.json`). Gain: meilleure observabilité côté Python; verrou inchangé: divergence 3/3 et `tx_runtime_count=0`.
- 2026-05-11 (alignement C↔Python strict-ish): outil `cp_event_alignment_check.py` ajouté, résultat pack3 `r8_p1_cp_event_alignment_v1_20260511_1618.json` / `.md` -> mismatch 3/3 `missing_python_runtime_after_verify`. Interprétation: Python exécute des vérifications (`verify_trace_count>0`) mais ne franchit pas la transition runtime (`tx_runtime_count=0`). Prochaine cible: tracer la raison exacte de non-transition verify->runtime.
- 2026-05-11 (probe non-transition runtime): outil `py_step_events_extract.py` ajouté et run v2 pack3 (`r8_p1_py_step_events_pack3_strict_v2_20260511_1628.json`). Signal neuf: terminal stage `hit_resolve_chain/resolve` sur apod2/apod3 et `validate_gate/fail` sur apod4, mais `tx_runtime_count=0` partout. Alignement v2 (`r8_p1_cp_event_alignment_v2_20260511_1630.json`) confirme mismatch 3/3 `missing_python_runtime_after_verify`.
- 2026-05-11 (delta instrumentation runtime-transition): patch `zeblindsolver.py` pour exposer `verify_hit_runtime_transition_block_counts` et `verify_hit_terminal_event` dans `astrometry_semantics`. Re-run pack3 strict v3: divergence 3/3 inchangée mais blocage localisé (beaucoup de `validate_gate:fail` après `hit_resolve_chain:resolve`).
- 2026-05-11 (validate_gate reason mix): sur pack3 strict v3, les `validate_gate:fail` se répartissent en `validation_failed` (9), `pixel_scale_out_of_range` (8), `accept_logodds_below_toprint` (1). Cible de delta suivante confirmée: gate de validation post-hit.
- 2026-05-11 (import pipeline first, itération 1): bloc post-hit TAN refit ajouté dans `zeblindsolver.py` (`blind_solver_posthit_tan_refit_enabled`, `blind_solver_posthit_tan_refit_min_pairs`) avec CLI et instrumentation (`posthit_tan_refit_attempts/hits/errors`). Probe A/B OFF/ON pack3: aucun impact observable à ce stade car le flux n'atteint pas encore un runtime post-hit (`tx_runtime_count=0`).
- 2026-05-11 (import pipeline first, itération 2): câblage complet de `blind_validate_lowpairs_inliers_floor` finalisé (SolveConfig + CLI + mapping runtime). Le levier n'est plus ad-hoc et devient testable/reproductible dans la méthode standard.
- 2026-05-11 (import pipeline first, itération 3): probe capé `numtries/nummatches` exécuté (`r8_p1_caps_semantics_probe_20260511_1904.json`) ; cohérence runtime Python validée sur cas atteints (`maxquads_reached`/`maxmatches_reached`, flags `*_stop_consistent=true`). Reste à fermer: comparer C↔Python sur même protocole capé.
- 2026-05-11 (import pipeline first, itération 4): câblage complet CLI+SolveConfig des branches solveur Astrometry-like (`set_crpix`, `pixel_xscale`, `predistort`, `predistort_posthit_sip`) + fix robuste d'un `NameError` latent en postverify runtime (`refit_rms`/`model_scale_arcsec`), puis smoke strict confirmant l'activation runtime des branches (`xscale_only` vs `predistort_only`).
- 2026-05-11 (import-first, itération 5): en poursuivant l'instrumentation caps, deux bugs runtime ont été révélés puis corrigés (`NameError` postverify sur `refit_rms/model_scale_arcsec`, puis `UnboundLocalError` `hit_diag` en branche non-parity). Probe décisionnelle caps produite: à `hard_max_candidates_tried=25`, le sweep `hard_max_validations {1,2,4}` bascule de `tx=0` (fingerprint `15129b7c`) à `accept_keep`/`tx=1` (fingerprint `e81dcc26`) quand `nummatches=4`.
- 2026-05-11 (itération vérification structurelle avant poursuite): audit statique de présence des blocs P0/P1 exécuté (`r8_structural_presence_matrix_20260511_1854.json`) sans manque explicite (`missing=[]`). Probe runtime court associé: en strict parity le verrou persiste (`tx_runtime_count=0`), alors qu'un mode non-parity hit-bearing atteint `tx_runtime_count=1` (`accept_keep`). Conclusion durable: blocage actuel = convergence comportement strict, pas absence brute de code.
- 2026-05-11 (itération déblocage strict hit-bearing): preuve A/B pack3 que le pipeline strict peut atteindre la transaction runtime quand on ouvre temporairement les gates (strict flags conservés, seuils quality/scale/logodds permissifs). Baseline strict: `tx_runtime_count=0` (0/3); profil forced-hitbearing: `tx_runtime_count=1` (3/3, terminal `accept_keep`). Implication durable: le blocage restant est de calibration/ordre des gates nominales, pas un manque structurel de chemin runtime.
- 2026-05-11 (itération isolation barrière stricte): depuis un profil strict hit-bearing forcé, rollback unitaire sur apod2 montre que le retour des seuils ANODDS est le premier coupe-circuit (`accept_logodds_gate`, `tx=0`), puis les seuils quality recassent aussi (`validation_failed`, `tx=0`), tandis que le rollback scale seul conserve `tx=1`. Le toggle minimal `blind_anodds_semantics_enabled=0` en strict baseline fait apparaître un hit-bearing (apod4) sans toucher au code structurel.
- 2026-05-11 (itération strict frontier): la remise en route de `tune_reverify` a révélé un bug latent (`NameError img_points/tile_points/tile_world_matches`) corrigé en basculant sur les buffers locaux runtime (`local_img_in/local_tile_in/local_world_in`). Ensuite, isolation des gates: avec ANODDS semantics active et odds valides (`anodds_totune>0`), apod2 reste bloqué en `accept_logodds_gate` (`tx=0`). En mode `blind_anodds_semantics_enabled=0`, une frontier logodds exploitable apparaît: apod2 passe vers `thr<=-1.0`, apod3 nécessite ~`thr<=-1.5`; à `thr=-1.0` sur pack3, apod2 tx=1 success, apod3 tx=0, apod4 tx=1 mais échec final de cohérence de parité.
- 2026-05-11 (resserrage quality, apod3 corridor strict): en maintenant le corridor logodds hit-bearing (`blind_anodds_semantics_enabled=0`, `blind_logodds_totune=tokeep=-1.5`), la transition runtime tient pour `quality_rms >= 100` mais retombe à `tx=0` dès `quality_rms<=20`. À `quality_rms=100`, `quality_inliers=4` passe (`tx=1`) tandis que `12/20` recassent (`tx=0`). Cela localise la barrière quality après la barrière accept-gate.
- 2026-05-11 (continuation resserrage strict): corridor strict hit-bearing affiné et stabilisé sur apod3 avec `blind_anodds_semantics_enabled=0`, `blind_logodds_totune=tokeep=-1.5`, `quality_rms=45`, `quality_inliers=4`. Frontière RMS robuste observée entre 45 (pass tx=1) et 40 (fail tx=0), confirmée en répétition. À `rms=45`, la frontière inliers montre `4` success tx=1, `6` tx=1 mais échec final low-coverage, `>=8` tx=0. Sur pack3 avec ce corridor: apod2/apod3 success tx=1, apod4 atteint tx=1 mais échoue en `final coherence parity mismatch`; les toggles parity/permhash testés montrent que ce verrou est post-runtime et spécifique cohérence finale.
- 2026-05-11 (fix structurel parité): découverte d'un appel `SimilarityTransform` sans `parity` dans le chemin best-transform strict (default +1), susceptible de produire `parity_label=mirror` vs `parity_value=+1` et rejet final coherence. Patch appliqué en forçant `parity` depuis `parity_label`. Après patch, l'ancien corridor (`thr=-1.5,rms45,inliers4`) n'accepte plus le faux hit apod4 (tx=0), mais un corridor strict 3/3 est retrouvé à `thr=-2.5` (ANODDS semantics OFF), avec `tx=1` sur apod2/apod3/apod4.
- 2026-05-11 (post-fix parité, calibration ANODDS): après correction du champ `parity` sur le best-transform strict, l'ancien corridor `thr=-1.5/rms45/inliers4` n'accepte plus le faux hit apod4 (tx=0). Un corridor strict 3/3 est retrouvé à `thr=-2.5` (ANODDS OFF). Ensuite, ANODDS semantics ON redevient fonctionnel en calibrant les odds à `exp(-2.5)` (`blind_anodds_totune=tokeep=toprint`), donnant 3/3 success et 3/3 tx=1 sur pack3; les defaults ANODDS restent 0/3.
- 2026-05-11 (resserrage post-fix parité): frontière stricte ANODDS apod4 mesurée de façon stable entre `thr=-2.45` (succès tx=1, 2/2) et `thr=-2.40` (échec tx=0, 2/2). En conservant `thr=-2.45` (ANODDS semantics ON), le corridor quality a pu être resserré à `quality_rms=42.5` et `quality_inliers=5` tout en gardant pack3 à 3/3 success et 3/3 tx=1; `rms=40` ou `inliers=6` cassent apod3.
- 2026-05-11 (toward-nominal pass): revalidation de la frontière ANODDS (`-2.45` stable, `-2.40` casse apod4), puis démonstration que `thr=-2.4` devient tenable en budget élevé (stable 2/2) mais `-2.35/-2.3` cassent apod3. Resserage réussi sur le bloc scale-rescue jusqu'à quasi-default `factor=1.1, max_arcsec=30` tout en gardant pack3 à 3/3 sous corridor `thr=-2.45, rms=42.5, inliers=5`. Verrou principal restant vers nominal: le scale-guard ratio `0.75/1.35` reste bloquant (0/3, `pixel_scale_out_of_range`).
- 2026-05-11 (dernière itération scale-guard): verrou final fortement réduit. En gardant ANODDS ON calibré (`exp(-2.45)`), quality (`rms=42.5`, `inliers=5`) et rescue quasi-default (`1.1/30`), la frontière de scale-guard a été localisée sur apod4 entre `min_ratio=0.08` (pass) et `0.09` (fail). La borne haute peut revenir au nominal `max_ratio=1.35`. Un corridor near-nominal stable est validé sur pack3 (`min=0.08,max=1.35,lowpairs_guard=off`) avec 3/3 success et 3/3 tx=1, confirmé sur 2 répétitions.
- 2026-05-12 (formalisation): création d'un preset strict near-nominal gelé (`r8_strict_near_nominal_preset_v1_20260512_0002.json`) validé par un Go/No-Go documenté (`r8_strict_near_nominal_gonogo_20260512_0004.md`) sur base de runs 3/3 success et 3/3 tx répétés. Compromis final explicite conservé pour ce pack: `blind_scale_guard_min_ratio=0.08` avec `blind_scale_guard_max_ratio=1.35` nominal.
- 2026-05-12 (scope check): après formalisation Go/No-Go pack3, un probe sur images demo hors pack (`apod1`, `apod5`, `sdss`) donne 0/3 avec le preset near-nominal (`scale_hard_reject`/`validation_failed`). Le preset est donc validé comme profil stable **ciblé pack3**, pas encore généralisé; généralisation à traiter en chantier dédié.
- 2026-05-12 (M6 probe-1): tentative de généralisation hors pack3 par simple élargissement scale-rescue (`factor=2.0`, `max_arcsec=120`) sur `apod1/apod5/sdss` => 0/3, sans transaction runtime. Le levier rescue seul est insuffisant; il faudra viser un autre bloc (ex: scale guard logic / candidate path) pour récupération hors pack.

- 2026-05-12 (audit de contrôle import): audit ciblé P0/P1 relancé sur `zeblindsolver.py`; couverture structurelle confirmée (solver_handle_hit runtime, onefield runtime callback/best_hit_only/dedup, ANODDS, set_crpix/pixel_xscale/predistort, caps numtries/nummatches). Gate technique `py_compile` OK. Verdict: pas de manque d'import structurel détecté sur ce périmètre; verrou restant = preuve décisionnelle/runtime C≈Python hors pack3. Preuve: `reports/r8_control_audit_imports_20260512_1546.md`.

- 2026-05-12 (contrôle C↔Python capé): tentative hors pack3 réalisée sur `apod1/apod5/sdss` (non discriminante: C et Python échouent 3/3). Fallback capé exploitable sur `apod2/apod3/apod4` (`hard_max_candidates_tried=40`, `hard_max_validations=40`) confirme divergence terminale 3/3 avec `tx_runtime_count=0` côté Python. Rapport: `reports/r8_control_cp_probe_20260512_1606.md`.

- 2026-05-12 (J+1 Run A baseline diffable): sur `testzeblind/probe_raw/RAW_...233828.fit`, référence C Astrometry confirmée solvable (`solved=true`) mais ZeBlind Python capé strict échoue (`success=false`) avec `numtries=0`, `nummatches=0`, `tx_runtime_count=0`; stage dominant `hypothesis` et `best_fail_reason=no_failed_validation_recorded`. Conclusion durable: verrou amont avant verify/runtime. Preuve: `reports/r8_j1_runA_baseline_20260512_1912/result.json`.

- 2026-05-12 (isolation verrou ZeBlind, J+1): correction de protocole importante — le cap `max_quads=5` masquait la génération candidats (faux 0-try). En baseline corrigée, pipeline actif (`numtries=10`). Cascade causale isolée sur RAW testzeblind: `preverify_geom_guard_failed` -> `accept/logodds` -> `scale guard` -> `center prior`. Déblocage contrôlé validé avec gates assouplies (`preverify guard off`, logodds permissif, `scale_guard_min_ratio=0.08`, center prior tolérance large): `success=true`, `numtries=1`, `nummatches=1`, `tx_runtime_count=1`. Preuve: `reports/r8_j1_lock_isolation_progress_20260512_1951.md`.

- 2026-05-12 (corridor robuste candidat ZeBlind): sweep ciblé sur les 4 RAW `testzeblind/probe_raw` aboutit à un corridor 4/4 avec `tx_runtime_count>0` sur 4/4 (déblocage stable), au prix d'un profil permissif (ANODDS off, guards relaxés, center prior large, quality_rms=900, inliers=3). Mesures RMS: ~5.9 / 4.1 / 262 / 759 (cas `ref028` encore très dégradé). Conclusion durable: corridor de déblocage trouvé, mais resserrement qualité encore requis avant promotion produit. Preuve: `reports/r8_corridor_robust_candidate_20260512_2140.md`.

- 2026-05-12 (backlog hygiene followup): tâches pending devenues non utiles ont été nettoyées/archivées dans `followup.md` (Run C désormais atteint; hunt hors-pack3 initial reclassé après correction de la config index C). Pile active recentrée en tête sur 3 axes: corridor qualité 4/4, comparaison Astrometry-like WCS/scale/parité/centre, et fermeture parité décisionnelle C↔Python P0/P1.

- 2026-05-13 (mission confiée + cadrage anti faux-positifs): Tristan a explicitement relancé la mission avec exigence de respecter la checklist `followup.md` et la méthode d’itération. Stratégie opérationnelle validée pour le prochain delta: traiter centre+scale comme **prior doux** (pas verrou dur global), garder un verify aval **strict** (inliers/RMS/logodds/géométrie) et conserver un fallback anti-0/4 par paliers (budget blind réservé + unlock progressif; lock RA/Dec limité à `hinted`).

- 2026-05-13 (ZeBlind delta #1 centre+scale soft-prior): implémentation effective d’un rerank candidat Astrometry-like **non bloquant** dans `_attempt_level` (`_apply_center_scale_soft_prior_rank`). Le prior combine proximité centre (RA/Dec hint -> distance angulaire) et cohérence d’échelle (ratio local deg/px estimé sur top-K), avec boost doux configurable (`blind_center_scale_soft_prior_*`) et cache `center_scale_prior_cache`. Décision conservée: pas de lock dur hors phases hinted; validation aval stricte inchangée. Gate technique: `py_compile` OK.
- 2026-05-13 (ZeBlind delta #1, mesures initiales): A/B capé exécuté après implémentation du prior doux centre+scale. Sur `ref028` avec header reuse OFF, le prior est effectivement actif (`reordered=true` en phases `hinted_wide` et `scale_only`) et réduit fortement le coût de recherche (`elapsed 330.5s -> 184.7s`, `candidates_tried 55 -> 20`, `failed_validations 22 -> 6`), sans gain solve-rate sur ce cas (échec conservé). Références: `reports/r8_center_scale_softprior_ab_20260513_0958/result.json`, `reports/r8_center_scale_softprior_probe028_20260513_1001/result.json`, synthèse `reports/r8_center_scale_softprior_delta1_20260513_1005.md`.
- 2026-05-13 (delta #1 centre+scale, A/B complet probe_raw): exécution d'un A/B complet 4 RAW `testzeblind/probe_raw` (header reuse OFF, caps 20/20). Soft-prior ON ne change pas le solve-rate capé (`0/4` inchangé) mais apporte un gain de coût substantiel (`496.1s -> 373.0s`, -24.8%) avec activation/reorder confirmés en `hinted_wide` et `scale_only` sur 4/4. Cas marquant `ref028`: `candidates_tried 55 -> 14`, `failed_validations 22 -> 5`, `elapsed 242.5s -> 118.5s`. Réf: `reports/r8_center_scale_softprior_ab_full_probe_raw_20260513_1014/result.json` + `summary.md`.
- 2026-05-13 (audit Astrometry gap): audit ciblé du flux `solver_handle_hit/verify` Astrometry vs pipeline ZeBlind. Conclusion durable: la non-parité actuelle vient surtout d'un sur-filtrage en amont (notamment `blind scale prefilter failed` en mode strict + guards preverify) avant la vraie décision verify/logodds. Rapport: `reports/r8_astrometry_gap_audit_20260513_1044.md`.
- 2026-05-13 (delta #2 strict prefilter soft): implémentation d'un bypass soft des misses de scale prefilter en mode strict (hard reject conservé sur écarts extrêmes, ratio défaut 12.0). A/B 4 RAW probe_raw: solve-rate inchangé `0/4`, mais suppression des rejets prefilter `10 -> 0` et gain temps `409.8s -> 349.5s` (-14.7%). Réf: `reports/r8_delta2_strict_prefilter_soft_ab_20260513_1359/result.json`.
- 2026-05-13 (side-by-side Astrometry vs ZeBlind, probe_raw 4 RAW): comparaison instrumentée exécutée avec Astrometry local (`solve-field --no-verify --depth 10-120 --cpulimit 120 --index-dir astrometry-indexes/4100`) vs ZeBlind delta #2 (`strict_soft_mode=1`, caps 20/20). Résultat: 0/4 des deux côtés sous cap, mais Astrometry pousse une recherche très profonde (1,134,803 quads / 17,671,167 codes, CPU-limit 4/4) alors que ZeBlind n'atteint aucun `tx_runtime` positif (0/4) avec `prefilter_reject=0`. Conclusion durable: la divergence prioritaire est désormais en verify/accept aval, pas en prefilter d'échelle strict.
- 2026-05-13 (Astrometry parity, étape 1 verify/accept order): patch ZeBlind appliqué pour aligner l’ordre de transaction runtime sur Astrometry (`verify` avant `accept/tune/keep`). Validation compile OK. Side-by-side post-patch (r10) sur probe_raw 4 RAW: résultat solve-rate inchangé (0/4 capé) et `tx_runtime_positive=0/4`, indiquant que la divergence restante est surtout en amont de ce runtime (entrée candidate/chemin hit) plutôt que dans l’ordre verify/accept lui-même.
- 2026-05-13 (Astrometry parity, callback to-solve): patch ciblé dans `zeblindsolver.py` pour fermer un écart décisionnel majeur de `record_match_callback` (retour systématique True). Ajout d’un vrai gate `logratio_tosolve` via `blind_anodds_tosolve` (strict mode: score priorisé sur `prob_logodds`; fallback hors strict: `accept_logodds`/`blind_logodds_tosolve`). En parallèle, bypass des guards pré-accept custom en strict verify séquentiel (`preverify_guard_bypassed`) et alignement de la source `cand_logodds` sur `prob_logodds` en strict-seq. Gates validées: `py_compile` OK + smoke CLI module sur copie FITS avec sortie `SOLVED=1` et header WCS écrit. Rapport: `reports/r10_verify_accept_tosolve_strictseq_alignment_20260513_231339.md`.
- 2026-05-13 (side-by-side post to-solve): exécution d’un side-by-side capé `probe_raw` 4 RAW après patch callback `to-solve`, avec protocole aligné r10 et copies FITS nettoyées WCS (header reuse OFF). Résultat durable: ZeBlind reste à `0/4` (Astrometry capé `0/4` aussi), mais l’entrée runtime post-hit est désormais confirmée sur `4/4` (`tx_runtime_positive` dérivé via `phase_perf.*.astrometry_match_objects_total`, total=43), avec `prefilter_reject_total=0` et `prefilter_soft_miss_total=13`. Conclusion: le verrou principal n’est plus l’absence d’entrée runtime; il reste sur la conversion runtime->solve sous caps.
- 2026-05-13 (callback hit-rate check): vérification dédiée post-patch `to-solve` sur 4 RAW capés: `tx_runtime_positive=4/4` (total 43) mais `onefield_solution_records` reste `0/4` (total 0). Fait durable: le callback `record_match_callback` n’est pas encore atteint dans ce protocole; ajuster `blind_anodds_tosolve` seul ne peut pas débloquer le solve tant que le flux n’atteint pas ce point.
- 2026-05-13 (A/B preverify guard): essai causal ON/OFF de `blind_astrometry_preverify_geom_guard_enabled` sur 4 RAW capés. Résultat stable: aucun solve (`0/4`) et aucun callback (`onefield_solution_records=0`) dans les deux modes; OFF augmente le coût (362.6s -> 415.7s) et le volume tx-runtime (43 -> 62) sans bénéfice décisionnel. Conclusion durable: le verrou n’est pas ce guard; il faut cibler la qualité des candidats avant l’entrée callback.
- 2026-05-13 (diag relax validation ref001): test A/B avec relâchement fort (`quality_rms=50`, `quality_inliers=3`, `blind_validate_lowpairs_inliers_floor=3`) n’atteint toujours pas le callback (`0 record`) et reste en `validation_failed` sur RMS énorme (ordre centaines px). Fait durable: le verrou est la géométrie des candidats en amont, pas les seuils d’inliers/callback.

- 2026-05-14 (r12d consolidation runtime->callback): audit post r11/r12/r12c publié (`reports/r12d_runtime_to_callback_gap_audit_20260514_0710.md`). Fait durable confirmé: ZeBlind entre bien en runtime post-hit sur probe_raw capé (`tx_runtime_positive=4/4`, total=43) mais n’atteint toujours pas `record_match_callback` (`onefield_solution_records=0`). Les verrous observés se partagent entre `preverify_geom_guard_failed` (2/4) et `validation_failed` (2/4); le prochain delta doit cibler la qualité géométrique pré-callback plutôt que le gate `tosolve`.

- 2026-05-14 (r12e A/B anti-RMS rerank guard): essai causal strict-mode sur 4 RAW capés (`hard_max_candidates_tried=20`, `hard_max_validations=20`) avec prefilter soft strict ON et WCS/header reuse OFF. Résultat durable: **NO-GO** décisionnel (`success=0/4`, `tx_runtime_total=43`, `callback_records_total=0` inchangés). Le delta expérimental a été retiré après test; le verrou reste la qualité géométrique pré-callback (RMS extrême) plutôt qu’un problème d’entrée runtime.

- 2026-05-14 (cleanup followup + alignement plan): section r11→r12e remaniée pour supprimer les “next causal” redondants et ne garder qu’un focus actif unique sur le premier étage de divergence `runtime -> callback`. Plan aligné: side-by-side C vs ZeBlind centré sur le premier rejet pré-callback, taxonomie commune des raisons de rejet, puis un seul delta causal avant re-mesure.

- 2026-05-14 (r13 side-by-side premier rejet pré-callback): exécution C vs ZeBlind sur `testzeblind/probe_raw` (4 RAW capés, même profil que r11/r12). Résultat durable: `tx_runtime_total=43` et `callback_records_total=0` côté ZeBlind, `success=0/4` des deux côtés (Astrometry capé). Taxonomie premier rejet ZeBlind: `validation_failed@hinted_wide` 2/4, `preverify_geom_guard_failed@hinted` 1/4, `preverify_geom_guard_failed@scale_only` 1/4. Décision de suite: cibler en premier la branche majoritaire `validation_failed@hinted_wide` avec un seul delta causal et re-mesure stricte.

- 2026-05-14 (r14+r15 sur branche majoritaire validation_failed@hinted_wide): deux deltas testés et clôturés NO-GO. r14 (patch code local min-pairs tweak2 en hinted_wide strict) actif en métriques mais sans ouverture callback (`0/4`, `tx=43`, `callback=0`) puis rollback propre. r15 (config-only tweak2 agressif: min_pairs=4, tol_factor=2.8, max_iters=2) idem (`tweak2_hits=0`). Décision durable: avant nouveau patch, passer par r16 mesure causale fine via dumps `validation_pairs`/`wcs_coherence` centrés sur les premiers `validation_failed` de hinted_wide.

- 2026-05-14 (r16 causal dumps sur validation_failed majoritaires): dumps `blind_validation_pairs_dump_path` + `blind_wcs_coherence_dump_path` collectés sur `ref001` et `ref027` (`reports/r16_validation_failed_hinted_wide_dumps_20260514_1120`). Fait durable: les échecs dominants arrivent avec pair-support très faible (2–3 paires), avec deux régimes: (a) RMS transform faible mais inliers insuffisants, (b) fit WCS dégénéré avec RMS massif (ordre 10^3 px / arcsec énorme). Conclusion opérante: prochain delta doit cibler l’augmentation de pair-support avant validation, pas callback/ANODDS.

- 2026-05-14 (r17 pair-support delta): patch strict `hinted_wide` all-levels pool-expand testé sur 4 RAW capés. Effet local confirmé (`strict_pool_expand_pairs_gain_total=66`, hits all-levels=6) mais aucun effet décisionnel (`success=0/4`, `tx=43`, `callback=0`). Conclusion durable: augmenter le volume de paires seul ne suffit pas; le verrou reste la qualité/cohérence des low-pairs. Delta rollbacké après test.

- 2026-05-14 (r18 low-pairs preverify guard): patch strict `hinted_wide` pour activer le preverify guard dès 2 paires testé sur 4 RAW capés. Effet: déplacement des rejets vers preverify et baisse du flux runtime (`tx 43 -> 38`) sans ouverture callback (`0`) ni gain solve (`0/4`). Conclusion durable: hard reject précoce contre-productif ici; privilégier une réallocation de budget/priorisation des candidats cohérents plutôt qu’une coupure brute. Delta rollbacké.

- 2026-05-14 (r19 soft budget hinted_wide): patch de ranking (bonus support paires + pénalité douce dégénérés) testé sur 4 RAW capés. Aucun effet décisionnel (`success=0/4`, `tx=43`, `callback=0`), activation dégénérée quasi nulle (`hits=1`). Conclusion: le verrou n'est pas principalement dans le ranking hypothèse; il faut intervenir plus amont sur la qualité de correspondances `match_objects/resolve_matches`. Delta rollbacké.

- 2026-05-14 (r20 resolve_hit lowpairs prune): patch strict `hinted_wide` pour pruner les branches `resolve_hit` avec peu de paires (< min_pairs_hit) testé sur 4 RAW capés. Aucun effet décisionnel (`success=0/4`, `tx=43`, `callback=0`) et compteur d'activation nul (`resolve_hit_hinted_wide_lowpairs_pruned_total=0`). Conclusion: la divergence est plus amont que cette branche `resolve_hit` ciblée (probablement dans la qualité des paires collectées). Delta rollbacké.

- 2026-05-14 (r21 lowpairs premodel after collect): passe one-to-one légère en `hinted_wide` strict sur low-pairs [6..11] testée sur 4 RAW capés. Aucune évolution décisionnelle (`success=0/4`, `tx=43`, `callback=0`), avec instrumentation active mais sans effet (`attempts=12`, `hits=0`). Conclusion: la divergence est encore plus amont, dans la sélection vote/pair pool de `_collect_tile_matches`. Delta rollbacké.

- 2026-05-14 (r22 collect vote-cap): assouplissement local du vote-threshold en `hinted_wide` strict (`vote_percentile` cap 40→30) testé sur 4 RAW capés. Effet causal net (activation `vote_cap_hits_total=32`) mais pas de déblocage (`success=0/4`, `callback=0`) et baisse du flux runtime utile (`tx 43 -> 37`). Conclusion durable: relâcher le seuil de vote ajoute surtout du bruit/dégénérescence; la piste suivante doit préserver les paires fortes plutôt que d'ouvrir plus bas. Delta rollbacké.

- 2026-05-14 (r23 collect uniformize bypass): bypass `uniformize_pairs` en `hinted_wide` strict testé sur 4 RAW capés pour préserver les paires fortes sans relâcher le seuil vote. Effet activé (`uniformize_bypass_hits_total=32`) mais aucun déblocage décisionnel (`success=0/4`, `callback=0`, `tx 43 -> 42`). Conclusion: la divergence ne vient pas du post-filter uniformize; remonter à la composition du vote-pool (unsaturated vs saturated vote mass). Delta rollbacké.

- 2026-05-14 (r24 unsaturated vote preference off): désactivation locale de la préférence `unsaturated hash votes` en collect `hinted_wide` strict testée sur 4 RAW capés. Effet activé (`unsat_prefer_disabled_hits_total=32`) mais aucun changement décisionnel (`success=0/4`, `tx=43`, `callback=0`). Conclusion durable: la divergence n'est pas pilotée par ce switch de composition du vote-pool à ce stade; elle persiste dans la sélection/consommation de candidats avant callback. Delta rollbacké.

- 2026-05-14 (pivot méthode validé): ajout dans `AGENT.md` d'un protocole mission "First-divergence mirror-trace" (RAW sentinelle, trace miroir C↔ZeBlind candidate-by-candidate, patch interdit avant preuve de première divergence). Exécution r25: trace ZeBlind sentinelle ref001 produite (`zeblind_preresolve_dump` + top rejects), verrou pré-callback confirmé; prochaine étape obligatoire = trace Astrometry équivalente pour alignement et première divergence explicite.

- 2026-05-14 (r26 trace Astrometry miroir): exécution `solve-field -v` + `astrometry-engine -v -v` sur sentinelle ref001, avec extraction `c_trace.json` et `c_engine_vv_trace.json`. Résultat: C non résolu aussi, mais trace riche objet/index-level disponible (389 rows, premier non-zéro obj10, max quads_tried=84986, matched=1269115). Étape suivante verrouillée: alignement d'identité C↔ZeBlind pour isoler la première décision divergente explicite (r27).

- 2026-05-14 (r27 alignment seed): agrégation des traces sentinelle ZeBlind (r25) et C (r26) dans `reports/r27_alignment_ref001_20260514_1425/`. Constat durable: impossibilité d'extraire la première divergence candidate-id exacte avec les traces actuelles car espaces d'identité non isomorphes (ZeBlind tile/parity/candidate vs C object/index/quad). Blocage explicite: besoin d'une join-key candidate-level côté C avant tout patch (r28 instrumentation C).

- 2026-05-14 (r28 breakthrough): première divergence primaire objectivée sur sentinelle ref001 en **amont** du solveur: composition du source-pool. Preuve: C `simplexy found 91 sources` alors que ZeBlind preresolve référence des `obs_combo` jusqu'à id 418 (pool minimal >=419). Conclusion durable: espaces candidats non isomorphes avant quad-generation; inutile d'itérer sur heuristiques solveur tant que l'entrée source n'est pas alignée (r29).

- 2026-05-14 (r29 validation divergence amont): test sentinelle ref001 avec `max_stars=91` (aligné C simplexy=91). Confirmation causale: source-pool ZeBlind passe de `obs_id_max=418` à `80` (entrée nettement réalignée). Effet utile observé sur le profil d'échec (`inliers 3 -> 11`, RMS 815 -> 310), mais no-solve persistant. Conclusion: divergence amont corrigée partiellement; prochaine isolation = divergence solveur résiduelle sur base d'entrée alignée (r30).

- 2026-05-14 (r30): après alignement source (`max_stars=91`), jonction d'identité C↔ZeBlind rétablie sur sentinelle (49 ids ZeBlind, tous inclus côté C).
- 2026-05-14 (r31): projection ZeBlind `obs_combo` en triplets vs traces C `trying quad [a b c]`: aucune candidate ZeBlind totalement hors support C (0 no-hit). Le résiduel se déplace vers l'ordonnancement/priorisation des candidats plutôt qu'un espace de recherche disjoint.

- 2026-05-14 (r32 ordering proof): après réalignement source, comparaison de rang triplets partagés C↔ZeBlind sur ref001 => corrélation de rang quasi nulle (Spearman ~0.00013, 126 triplets communs). Conclusion durable: divergence résiduelle = politique de priorisation/ordonnancement candidats (et non espace de recherche disjoint).

- 2026-05-14 (r33 NO-GO): test delta unique de ranking `blind_center_scale_soft_prior_enabled=False` sur base alignée (`max_stars=91`) n'a produit aucun effet (fail identique, corrélation de rang C↔Ze inchangée). Ce levier est non causal pour la divergence résiduelle.

- 2026-05-14 (r34 NO-GO): désactivation `blind_hypothesis_scale_prior_enabled` testée sur base alignée (`max_stars=91`) sans aucun effet (fail + ordre C↔Ze inchangés). Le levier n'agit pas sur la divergence résiduelle observée.

- 2026-05-14 (r35 NO-GO aligné): re-test de `blind_collect_prefer_unsaturated_hash_votes_enabled=False` après alignement source n'a eu aucun effet (fail + ordre C↔Ze invariants). Le levier est désormais classé non causal même en contexte aligné.

- 2026-05-14 (r36 NO-GO runtime): test delta unique `blind_astrometry_strict_require_perm_hash_match_enabled=False` sur base alignée (`max_stars=91`) a provoqué 2 timeouts à 300s (baseline ~171s). Conclusion durable: retirer globalement le strict perm-hash gate est non retenable (explosion coût/non-terminaison); poursuivre via instrumentation fine locale collect-order avec garde-fou ON.

- 2026-05-14 (r37 instrumentation): ajout d'instrumentation de rejet `perm_hash_gate` dans la trace exacte ZeBlind. Probe borné (8/8) validé sans timeout; preuve locale d'inversion d'ordre C↔Ze sur triplet `(5,12,24)` avec rejets explicites `perm_hash_qmax_abs_delta > max_qdelta`.
- 2026-05-14 (r38 NO-GO runtime): micro-delta `blind_astrometry_strict_perm_hash_max_qdelta=4096` testé sur base alignée => timeout 300s (baseline ~171s) et ordre partiel dégradé. Conclusion: assouplissement global du seuil qdelta non retenable.

- 2026-05-14 (r39 copie réelle): implémentation d'un mode dédié `blind_astrometry_mirror_try_permutations_exact` (pas un simple tweak), qui force la discipline permutations Astrometry (try_permutations + cxdx/meanx) et neutralise le gate hash Ze-only sur cette étape. Effet observé: profil d'échec déplacé vers `preverify_geom_guard_failed` avec support 16 paires (vs verrou précédent), mais coût runtime plus élevé aux caps hauts; mode gardé derrière flag pour stabilisation budget (r40).

- 2026-05-14 (r40 stabilisation miroir): calibration runtime du mode `blind_astrometry_mirror_try_permutations_exact` sur sentinelle. Avec caps 20/20 et `max_orders=1`, une borne `blind_astrometry_endobj=4` permet de tenir <200s (`197.26s`) tout en conservant le profil miroir (`preverify_geom_guard_failed`, 16 paires). Décision: garder en mode expérimental flaggé et valider sur mini-lot (r41) avant promotion.

- 2026-05-14 (r41 mini-lot verdict): validation sur 4 RAW du mode miroir borné (`max_orders=1`, `endobj=4`, caps 20/20) => NO-GO promotion. Résultats: succès inchangé `0/4` vs baseline, mais coût runtime en hausse marquée (`459.1s` baseline vs `723.6s` miroir). Le mode miroir reste utile comme sonde topologique (fails homogénéisés en preverify), pas comme trajectoire perf/robustesse en l'état.

- 2026-05-14 (r42 extraction utile validée): sans activer le mode miroir, le levier `blind_astrometry_try_permutations_max=1` dans le path baseline conserve exactement la même topologie d'échec sur mini-lot 4 RAW (0/4, mêmes classes/raisons) tout en réduisant fortement le coût runtime (~459.1s -> ~302.7s, ~-34%). Candidat solide de promotion perf sous validation lot élargi (r43).

- 2026-05-14 (r43 lot8): validation élargie baseline vs baseline+perm1 (8 images testzeblind) confirme absence de régression de robustesse (mêmes succès/fail classes) avec gain runtime global ~-14.5% (877.84s -> 750.951s).
- 2026-05-14 (r44 promotion): changement code appliqué, default `blind_astrometry_try_permutations_max` passé de 12 à 1 (blind path), `py_compile` OK, avec validation sentinelle post-changement + lot8.

- 2026-05-14 (r46 parity audit validation/preverify): audit code+logs confirme non-parité à cette étape. ZeBlind ajoute un gate bloquant `preverify_geom_guard_failed[...]` et une réussite strictement `rms_ok && inliers_ok` (verify.py), alors qu'Astrometry opère surtout via `verify: logodds ...` + best-match policy. Conclusion opérationnelle: avant large-lot, implémenter un mode de parité validation/preverify dédié (r47) plutôt que poursuivre les benchmarks à l'aveugle.
- 2026-05-14 (r47 exact parity validation/preverify): implémentation d’un mode dédié `blind_astrometry_exact_validation_preverify_parity_enabled` (config+CLI) qui: (1) passe `validate_solution` en metrics-only via `astrometry_parity_mode`, (2) bypass le `preverify_geom_guard` Ze-only, (3) bypass le `strict_lowpairs_scale_guard` Ze-only. Gate `py_compile` OK. Sur sentinelle ref001 cap20, le fail principal bascule de `preverify_geom_guard_failed[...]` vers `blind verify logodds_accept failed` et `preverify_guard_reject_total` passe de 1 à 0, cohérent avec la trajectoire de parité demandée.
- 2026-05-14 (r47b mini-lot 4 RAW cap12): validation comportementale du mode parité exact. Résultats agrégés: `preverify_guard_reject_total 5 -> 0`, raisons d’échec basculées vers logodds-like `0/4 -> 4/4`, temps total `249.0s -> 190.9s`, succès inchangé `0/4 -> 0/4`. Conclusion durable: la parité ciblée preverify/validation est effective dans le flux décisionnel; la parité globale de robustesse reste à fermer.

- 2026-05-15 (reprise mission ZeBlind, r47c): patch blind-only appliqué pour enlever deux garde-fous Ze-only restants en mode parité exact (`blind_astrometry_exact_validation_preverify_parity_enabled=1`): (1) neutralisation de `blind_solver_handle_hit_strict_order` dans l’accept-gate runtime, (2) neutralisation du rejet `reject_trace_invalid` lié à la cohérence de trace stricte. Gate technique validée (`python3 -m py_compile zeblindsolver/zeblindsolver.py` OK). Prochaine action: rerun mini-lot cap12 en A/B baseline vs parity+patch et publication matrice per-image.

- 2026-05-15 (r47c, parité exacte + strict-order neutralisé): mini-lot 4 RAW cap12 rejoué avec `blind_reuse_existing_solved_wcs=0`. Résultats: succès inchangé `0/4 -> 0/4`, mais shift décisionnel complet confirmé (`preverify_guard_reject_total 13 -> 0`, raisons d’échec `logodds-like 0/4 -> 4/4`) et coût total légèrement réduit (`239.7s -> 222.9s`). Conclusion durable: les garde-fous Ze-only ne sont plus le verrou principal; prochain axe = fermeture stricte du gate `verify/logodds accept` vs Astrometry C.

- 2026-05-15 (r47d, alignement verify/accept Astrometry): lecture source `astrometry-main/solver/solver.c` et alignement ZeBlind du gate d’acceptation: usage de `post_tune_logodds` pour l’accept-gate (au lieu du logodds pré-tune) + maintien du gate `toprint` en mode parité exact comme côté C. Validation mini-lot cap12: succès inchangé `0/4`, mais trajectoire décisionnelle parité maintenue (`preverify_guard_reject_total 13 -> 0`, raisons logodds-like `0/4 -> 4/4`).

- 2026-05-15 (r47e, audit candidat→verify Astrometry): audit source C effectué (`solver.c`, `onefield.c`) et alignement ZeBlind ajouté en mode parité exact sur le bruit verify: `sigma2_use` inclut désormais `index_jitter/scale` (fallback `index_jitter=1.0 arcsec` si metadata absente), en écho à `match_distance_in_pixels2` Astrometry. Validation mini-lot cap12: parité décisionnelle maintenue (preverify 13->0, fails logodds-like 4/4) mais aucun gain solve-rate (0/4).

- 2026-05-15 (r47f, alignement distractor Astrometry): audit source C (`solver.c`, `verify.c`) puis patch ZeBlind en mode parité exact pour forcer un `distractor_use=0.25` (ratio fixe Astrometry) dans le verify séquentiel. Validation mini-lot cap12: décisionnel parité maintenu (`preverify 13->0`, fails logodds-like 4/4), mais solve-rate inchangé (`0/4`).

- 2026-05-15 (r47g, RoR parity force): alignement supplémentaire du flux candidat→verify en mode parité exact avec RoR forcé ON (comme verify.c). Validation mini-lot cap12: parité décisionnelle maintenue (`preverify 13->0`, fails logodds-like 4/4), solve-rate inchangé (`0/4`). Prochain verrou confirmé: modèle `testsigma` par étoile (gamma) encore non isomorphe C↔Ze.

- 2026-05-15 (r47h, testsigma gamma-like): alignement du principe math `verify.c` ajouté en parité exact côté ZeBlind avec `testsigma2` par étoile selon `sigma2=verify_pix2*(1+R2/Q2)` (approximation locale des ancres quad). Validation mini-lot cap12: décision logodds maintenue, solve-rate inchangé (`0/4`). Prochain verrou: diff séquentielle pas-à-pas C↔Ze (theta/logodds) sur hypothèse identique.

- 2026-05-15 (r47i in progress): instrumentation verify montre un pattern stable en parité (`prob_logodds=-1.386`, tout distractor, `prob_steps`≈paires inlier). Hypothèse forte: mismatch structurel d’entrée verify (NT trop court vs Astrometry champ complet), indépendamment des alignements RoR/distractor/tests sigma déjà faits.

- 2026-05-15 (r47i.b): tentative de rapprochement structurel des entrées verify vers Astrometry (NT field_all, NR index_all projeté) en mode parité exact. Mini-lot cap12: pas de gain solve (`0/4`), mais signature d’échec parité inchangée (logodds-like). Prochain levier validé: diff pas-à-pas `theta/logodds` sur hypothèse contrôlée.

- 2026-05-15 (r47i.c): verify parité branché sur `image_positions`+`tile_world` avec preuves diffables NT/NR. Effet confirmé: NT augmente (ex. 81) mais NR reste petit (ex. 5) sur hypothèses testées; conséquence, échec logodds persistant et solve-rate inchangé (0/4). Nouvelle priorité: expliquer/lever la compression NR avant RoR.

- 2026-05-15 (r47i.d): bug structurel identifié/corrigé en parité verify: `tile_world` hors portée entraînait un fallback silencieux vers `world_in` (NR réduit). Après correction (passage explicite `tile_world_all`), NR complet est confirmé (ex. 2000 sur ref001). Malgré cela, mini-lot cap12 reste à 0/4: verrou déplacé vers divergence math/ordering intra-verify, pas seulement composition NT/NR.

- 2026-05-15 (r47i.f): essai permutation relevance-first des test stars en verify parité (proxy testperm Astrometry). Effet mesuré sur ref001: `nsig2_min` chute fortement (~458→~39) sans encore passer sous 25; aucun match foreground ni solve. Cela confirme que l’ordre influence le score, mais qu’un ajustement sigma/échelle reste nécessaire.

- 2026-05-16 (r47i.f.1): ajout d’un hook `sigma parity factor` côté verify. Le sweep montre un effet mécanique correct (nsig2 baisse sous 25 sur ref001 à partir ~1.3), mais mini-lot cap12 reste bloqué à 0/4. Conclusion durable: le gate verify n’est plus le seul verrou; il faut améliorer la qualité des hypothèses amont en parallèle.

- 2026-05-16 (r47i.g): combinaison parité testée (sigma factor 1.3 + centre RoR issu de l’hypothèse). Le `nsig2_min` peut passer sous 25 localement, mais la décision reste tout-distractor/logodds fail (0/4 mini-lot). Insight: le seuil distance seul n’explique plus tout; il faut décomposer `logfg-logd` terme par terme.

- 2026-05-16 (r47i.h): la parité verify génère maintenant des pas foreground sur certaines hypothèses (logfg>logd), mais le cumul de preuves reste insuffisant pour le seuil d’acceptation strict (logodds max observé ~2.43 sur ref001). Nouveau focus: mesurer le déficit cumulatif de logodds (positif/négatif) vs seuil, plutôt qu’un simple check de passage 5σ.

- 2026-05-16 (r47i.i): le gap à l’acceptation est maintenant chiffré: meilleur logodds ~2.43 vs seuil ~13.82 (manque ~11.38). Le problème dominant est structurellement un excès de contributions négatives (beaucoup de distractors) plutôt qu’un manque d’amplitude sur le meilleur pas positif.

## 2026-05-16 (nettoyage mission + synthèse r47 récente)

- Nettoyage demandé par Tristan: `followup.md` a été refactoré en version courte orientée exécution (objectif unique, discipline, KPI, itération active), et l’historique détaillé a été sorti du flux principal.
- Conformité maths Astrometry confirmée sur le bloc `verify -> accept` (ordre C, `logaccept=min(tokeep,totune)`, gates `toprint/tokeep`, clamps ANODDS, composantes sigma/distractor alignées en mode parité).
- Série r47 (r47b→r47i.i) consolidée: verrou résiduel principalement amont verify (qualité/structure des correspondances), plus qu’un écart de formule des gates aval.
- r47i.j implémenté: filtre test-stars `nearest_nsig2` (priorisation des étoiles test prometteuses avant verify séquentiel), avec instrumentation dédiée (`verify_teststar_*`).
- Mini-lot cap12 r47i.j.1: pas de solve additionnel (`0/4`), mais déplacement partiel du profil d’échec (non mono-cause logodds).
- A/B ciblé r47i.j.2 sur cas 055:
  - baseline: `best_logodds=-1.386`, `gap_to_accept=15.202`, `pos/neg=1/19`, ~54.1s.
  - filtre `nearest_nsig2`: `best_logodds=7.166`, `gap_to_accept=6.649`, `pos/neg=3/23`, ~39.8s.
  - verdict: gain net (logodds/gap/runtime), raison d’échec inchangée (`verify logodds_accept failed`) -> filtre conservé (pas de rollback).
- Direction validée: continuer en micro-deltas amont (densité de pas foreground), sans retoucher les gates aval déjà alignées.
- 2026-05-16 (r47i.j.3, essai NO-GO): ajout d’un seuil optionnel `max_nsig2_keep` dans le filtre test-stars de parité pour renforcer la densité foreground. Probe 055 (`reports/r47i_j3_case055_threshold_probe_20260516_0946/result.json`) sans gain KPI cœur (`best_logodds` inchangé à ~7.166, raison d’échec inchangée) et runtime plus lent (~76.8s). Décision appliquée: garder la capacité en option mais neutraliser par défaut (`blind_astrometry_verify_teststar_max_nsig2_keep=1e9`) pour éviter une régression globale implicite.
- 2026-05-16 (r47i.j.4, NO-GO + stop rule): essai micro-delta d’ordre test-stars "interleave center" (mix nearest-nsig2 + proximité centre) évalué sur mini-lot cap12. Comparatif vs baseline r47i.j.1: KPI strictement inchangés (`best_logodds`, `gap_to_accept`, `pos/neg`, reason mix), avec runtime plus lent. Rollback appliqué du bloc j.4 pour garder baseline `nearest_nsig2`. Conséquence méthodologique: stop rule activée après deux itérations sans gain (j.3 + j.4), bascule vers audit causal amont (génération hypothèses/paires).
- 2026-05-16 (audit amont post-stop-rule): audit comparatif C vs ZeBlind sur le bloc pré-verify documenté dans `reports/r47i_upstream_code_audit_20260516_1015.md`. Résultat durable: l’écart résiduel plausible est la composition/lineage des candidats avant verify (ordre + filtres + provenance), plus que les formules de gates aval déjà alignées. Décision: prochaine itération `r47i.k1` dédiée à l’instrumentation diffable de lineage candidat.
- 2026-05-16 (r47i.k1 lineage): instrumentation diffable de lineage candidat produite sur 055 + mini-lot cap12 (`reports/r47i_k1_candidate_lineage_case055_minilot4_cap12_20260516_1021`). Insight durable: les meilleurs candidats pré-verify portent souvent une échelle modèle très hors cible (~5.4 à ~13.5 arcsec/pix vs attendu ~2.39), ce qui explique des échecs aval variés malgré des logodds parfois élevés. Direction actée: prochain micro-delta amont sur prior de cohérence d’échelle.
- 2026-05-16 (r47i.k2, NO-GO): test d’un prior de cohérence d’échelle avant verify (`blind_scale_coherence_prior_*`). Mini-lot cap12 reste à 0/4; le prior coupe des cas hors-échelle mais n’améliore pas le cas pivot 055 (`best_logodds` inchangé 7.166, même raison finale). Décision: garder la brique en option mais la désactiver par défaut pour éviter une régression implicite de comportement.
- 2026-05-16 (r47i.k3 recadrage appliqué): followup basculé en mode "pipeline parity audit only" (amont→aval, sans tuning). Premier snapshot pivot 055 produit (`reports/r47i_k3_pipeline_parity_audit_case055_20260516_1057`): candidat dominant récurrent `d50_1025` avec perte structurelle `pairs=11 -> inliers=3` avant échec verify. Prochaine action: isoler la perte exacte dans le maillon resolve/pairing vs attendu C.


## 2026-05-16 11:02 — r47i.k3.1
- Exécuté autopsie `pairs->inliers` case055 tile `d50_1025`.
- Rapport: `reports/r47i_k3_1_pairs_inliers_autopsy_case055_20260516_1101`.
- Résultat stable sur hinted/hinted_wide: hypothèse avec `pairs=11` mais `inliers=3`, puis validate_base/gate sur 3 inliers seulement; fail final `verify_prob/logodds_accept`.
- Lecture causale: goulot amont confirmé (construction des correspondances et support verify), pas un problème primaire des gates aval.
- Next: instrumentation snapshot C-like à l'entrée verify (NT/NR, ordre effectif, provenance des paires), comparaison diffable Ze↔C.


## 2026-05-16 11:07 — r47i.k3.2
- Instrumentation ajoutée dans `zeblindsolver.py`: snapshot verify-entry C-like.
- Champs ajoutés dans `verify_hit_trace` / `astrometry_match_objects` (events validate):
  - `verify_entry_nt`, `verify_entry_nr`
  - `verify_entry_order_img_idx_head`, `verify_entry_order_tile_idx_head`
  - `verify_entry_provenance_source`, `verify_entry_provenance_path`
  - `verify_entry_obj_index`
- Run case055 régénéré (`reports/r47i_k3_2_verify_entry_snapshot_case055_20260516_1106`) pour valider la présence des champs.
- Prochaine étape: extraction miroir côté C (engine/solver) avec mêmes clés pour diff Ze↔C.


## 2026-05-16 11:09 — r47i.k3.3
- Ajout patch `verify.c` côté Astrometry pour snapshot C-like pré-verify:
  - `NT/NR/NTall/NRall`, `fieldnum`, `quadno`, `dimquads`
  - `quad_field_head`, `testperm_head`, `refperm_head`
- Objectif: comparaison diffable directe avec snapshot Ze (`verify_entry_*`).
- Rapport de mapping produit: `reports/r47i_k3_3_c_like_verify_diff_spec_20260516_1109`.
- Next opérationnel: rebuild + run case055 avec logs VERB pour extraction JSON C et diff Ze↔C.


## 2026-05-16 11:13 — r47i.k3.4
- Exécution C réussie sur case055 via `solve-field` + cfg local indexes; log verify observé (`matches=12`, `field_objects=48`).
- Le marqueur patché `[C_VERIFY_ENTRY]` n'apparaît pas car exécution via binaire système non patché.
- Build local Astrometry pour binaire patché actuellement bloqué par dépendances build (`pkg-config` absent).


## 2026-05-16 11:21 — k3.5/k3.6 verify-entry parity
- Capture C patchée obtenue via binaire local `astrometry-engine`:
  NT=51, NR=12, ordre test/ref complet.
- Diff Ze sélectionné (`d50_1025`, validate_base, hinted) montre un décalage structurel:
  Ze entre verify avec set inliers réduit (`verify_entry_nt/nr=3`) alors que C vérifie sur large pool (`NT=51, NR=12`).
- Implication: divergence principale avant/à l'entrée verify (pipeline d'input verify, pas juste gate logodds).


- 2026-05-16 (2026-05-16 11:26, r47i.k3 clôture): parité d’entrée verify instrumentée et comparée Ze↔C sur case055.
  - Ze: snapshot verify-entry initial montrait `NT/NR=3/3`, source `inlier_mask`.
  - C (binaire local patché `verify.c`): capture `[C_VERIFY_ENTRY]` effective `NT=51`, `NR=12`, `NTall=51`, `NRall=15`, ordre `testperm/refperm` exporté.
  - Diff final (`reports/r47i_k3_6_ze_vs_c_verify_entry_diff_20260516_1121`) confirme mismatch structurel d’entrée verify (taille de pool + ordre + provenance), donc verrou principal amont verify et non gate aval uniquement.
- 2026-05-16: run C avec `solve-field` + cfg local index validé; limitation opérationnelle constatée: le binaire système non patché ne suffit pas pour traces C custom, nécessité d’exécuter le `astrometry-engine` local compilé pour capter `[C_VERIFY_ENTRY]`.


- 2026-05-16 (2026-05-16 12:13, r47i.k4 impl): parité verify avancée concrètement.
  - Fix d’un bug de wiring snapshot parité (référence variable hors scope) qui empêchait la provenance C-like d’être reflétée.
  - Bascule du pool ref parité vers `tile_points` (candidate-level) au lieu de `tile_positions` global: effet immédiat sur case055 avec `NR` aligné quasi C (`11` vs `12`).
  - Enrichissement `verify_hit_trace` avec `prob_verify_nt/nr/input_mode/pool_source` pour lecture causale directe Ze↔C.
  - Case055 post-fix: `prob_verify_input_mode=field_all_vs_index_all`, `prob_verify_nr=11`, `prob_verify_nt=64` (delta restant sur NT vs C=51).

- 2026-05-16 (2026-05-16 12:21, r47i.k4 NT convergence): réduction contrôlée du pool test verify en mode parité (`blind_astrometry_verify_teststar_keep_frac=0.60`, `max_keep=52`).
  - Case055 post-run: `prob_verify_nt=52` (vs C=51), `prob_verify_nr=11` (vs C=12), provenance stable `field_all_vs_tile_xy_inverse_transform`.
  - Résultat: quasi-parité d’entrée verify obtenue sur cardinalités, sans casser la cohérence de provenance.

- 2026-05-16 (2026-05-16 12:30, consolidation k4): stabilité confirmée après tuning NT.
  - Case055 reste en échec `blind verify logodds_accept failed` malgré quasi-parité d'entrée verify (`NT=52` vs C=51, `NR=11` vs C=12).
  - Règle opérationnelle explicitée: **quasi n'est pas identique**; conserver ce delta comme hypothèse active de dérive potentielle tant qu'on n'a pas parité stricte 1:1.

- 2026-05-16 (2026-05-16 12:40, r47i.k4 strict parity): passage de quasi à strict sur case055 pour les cardinalités verify effectives.
  - Ze: `prob_verify_nt=51`, `prob_verify_nr=12` (match C), entrée snapshot cohérente (`verify_entry_nr=12`, provenance parité).
  - Malgré cette parité stricte d'entrée/pool, l'échec aval persiste (`blind verify logodds_accept failed`), confirmant un delta désormais focalisé sur la dynamique de scoring/steps verify plutôt que sur la composition de pool.

- 2026-05-16 (2026-05-16 21:36, r47i.k4 sigma test): mini-lot avec `blind_astrometry_verify_sigma_parity_factor=3.0`.
  - 055 améliore fortement la dynamique verify (matchs présents, `prob_logodds=17.744`, `prob_matches=8`) mais n'atteint pas l'acceptation stricte.
  - Interprétation durable: le verrou principal a quitté "zero-match early"; il reste un écart de seuil/politique d'acceptation aval malgré parité de pool et steps redevenus actifs.


- 2026-05-16 (r47i.k4 forensic verify C↔Ze, case055):
  - Instrumentation C `verify.c` enrichie avec `[C_VERIFY_TERM]` (d2/sig2/logbg/logd/logfg/delta/cum/mu) + rebuild local `astrometry-engine` validé.
  - Extraction C terms réalisée sur `NT=51/NR=12` (`reports/r47i_k4_27_c_term_forensic_case055_20260516_2206/c_block_51_12.json`).
  - Extraction Ze terms sigma=3 réalisée (`reports/r47i_k4_28_ze_terms_sigma3_case055_20260516_2207/ze_terms.json`) puis comparaison (`r47i_k4_29.../summary.json`) confirmant un écart massif initial (`med_c_d2=0.381` vs `med_ze_d2=424.201`, ratio ~1112x).
  - A/B effectué sur la WCS utilisée par verify (`verify_wcs` entrée brute candidate vs `final_wcs`) : impact négligeable sur `d2/logodds` des candidats communs => **non-causal principal**.
  - Nouveau levier validé: cap refs verify `center` vs `head` (ordre natif). En mode `head`, amélioration significative sur plusieurs candidats (ex `d50_1125`: `prob_logodds=18.417`, `matches=6`, `d2_0≈3.7e-06`).
  - Patchs intégrés dans `zeblindsolver.py`:
    - `blind_astrometry_verify_refstar_cap_mode` (ajouté, défaut `head`),
    - `blind_astrometry_verify_use_entry_wcs` (ajouté, défaut `False`).
  - Rapport A/B de référence: `reports/r47i_k4_37_refcap_compare_case055_20260516_232131/summary.json`.
  - État: amélioration partielle de la géométrie verify, mais parité C↔Ze non fermée globalement; piste prioritaire restante = provenance/référentiel exact des ref stars vs sémantique `verify.c`.

- 2026-05-16 (r47i.k4 step1 lock): candidat Ze de référence stabilisé sur 2 runs consécutifs sous gate `NT=51/NR=12`.
  - Artefact: `reports/r47i_k4_38_ref_candidate_lock_case055_20260516_232803`.
  - Candidat locké: `d50_1125` (`hinted_wide`, `prob_logodds=18.417`, `matches=6`).
  - Parité cardinalités C↔Ze validée (`51/12`), mais parité step-level encore non atteinte (`theta_match_ratio_0_11=0.083`, `median_d2_ratio_z_over_c_0_11≈1650`).

- 2026-05-16 (r47i.k4 step2 A/B provenance/référentiel):
  - Test `ref pool mode` (`tile_world` vs `tile_xy_inverse_transform`) sur candidat locké `d50_1125` (NT/NR 51/12): impact négligeable sur la parité step-level (`theta_ratio~0.083`, `median_d2_ratio~1650`).
  - Test ordre/filtrage test-stars: `strict_head_noreorder` augmente l’alignement theta local (0.5) mais dégrade fortement les distances (`d2`) et logodds; `head+reorder` échoue sévèrement (matches=0).
  - Conclusion durable: la fermeture de parité C↔Ze ne se joue pas uniquement sur pool/order; next prioritaire = alignement plus strict du modèle `testsigma2/gamma` C-like sur candidat verrouillé.

- 2026-05-16 (r47i.k4 step2b testsigma): ajout d’un switch `blind_astrometry_verify_testsigma_mode` (`constant`/`gamma_like`) pour tester l’hypothèse C-like `fake=1` (sigma constant).
  - A/B case055 (`r47i_k4_41_testsigma_ab_case055_20260516_235148`):
    - `constant` améliore la cohérence step-level discrète (`theta_ratio_0_11` 0.417 vs 0.083; `median_abs_delta_diff` 2.756 vs 3.743),
    - mais l’écart de distances reste massif (`median_d2_ratio_z_over_c` ~1650, ordre inchangé).
  - Conclusion durable: sigma/gamma influe la dynamique de décision, mais la cause racine du `d2` gonflé reste probablement géométrique (repère/projection des distances).

- 2026-05-17 (r47i.k4 step3 same-hypothesis probe):
  - Scan des hypothèses Ze 51/12 (`r47i_k4_43...`) confirme qu’aucune candidate courante n’atteint la géométrie C (min `median_d2_ratio~1647`).
  - Probe contrôlée en injectant l’hypothèse C (WCS final C + RDLS C + AXY C) dans le verify Ze (`r47i_k4_44...`) donne un profil quasi-aligné: `logodds~99.31`, `matches=12`, `theta` mappé 0..11 à 100% sur steps 0..11, `median_abs_delta_diff~0.082`.
  - Le ratio `d2` all-steps est pollué par les distractors (logs C avec `d2` apparemment stale/répété). Sur matched-only: `median_d2_ratio~0.139`, `median_abs_delta_diff~0.250`.
  - Le verrou principal migre de "verify math" vers "capture de la bonne hypothèse candidate" dans la boucle ZeBlind.

- 2026-05-18 (r47i.k4, reprise active case055):
  - Migration opérationnelle validée: repo ZeSolver déplacé dans le workspace OpenClaw (`/home/tristan/.openclaw/workspace/projects/ZeSolver`), permettant l’usage direct de `apply_patch`.
  - Patch ajouté dans `zeblindsolver.py`: nouveau flag `blind_astrometry_refperm_collect_ids_strict_enabled` (défaut OFF) pour tester un chemin refperm plus strict côté pool verify.
  - A/B initial sur case055 (fichier avec WCS): OFF vs ON => métriques identiques sur sémantique verify (`NT`, `NR`, `step0`, `pool_source`), donc **pas d’effet causal démontré**.
  - Incident technique corrigé: un patch intermédiaire utilisait `phase_name` hors scope dans la branche world-pool (`prob_verify_pool_error: name 'phase_name' is not defined`), puis fix appliqué et compilé.
  - Validation sur entrée réellement sans WCS (copie + strip header):
    - OFF: solve OK en `hinted_wide`.
    - ON strict global: échec `no valid solution`.
    - Conclusion validée: le strict refperm global est **trop agressif** en l’état (risque de régression solve-rate), non activable en prod tel quel.
  - Décision durable: conserver ce chemin en expérimental (OFF par défaut) et ne poursuivre qu’avec garde-fous/fallback explicites.

- 2026-05-23 (S0 validate short-circuit instrumentation v2):
  - Extension d’instrumentation appliquée dans `zeblindsolver.py` autour de `validate_solution()`:
    - ajout `blocked_stage` normalisé par callsite;
    - ajout `fail_class` normalisé;
    - ajout export agrégé `validate_short_circuit_inventory` (total, `by_stage`, `by_fail_class`, `by_stage_fail_class`) dans `stats` succès + échec.
  - Validation technique immédiate: `python3 -m py_compile zeblindsolver/zeblindsolver.py zeblindsolver/verify.py` OK.
  - Constat runtime provisoire: probes courts bornés (`reuse_existing_wcs=OFF` + caps durs + timeout) restent majoritairement en timeout amont, empêchant encore la publication d’un inventaire runtime diffable stable des branches réellement bloquantes.
  - Décision de pilotage: marquer l’étape “instrumentation” comme faite dans `followup.md`, conserver séparément l’étape “publication inventaire runtime diffable” comme ouverte avec blocage explicite.

- 2026-05-23 (S0 validate runtime inventory + semantics split):
  - Inventaire runtime diffable finalement publié sur `case055` OFF/ON: `reports/r47i_s0_validate_blocking_inventory_case055_20260523_0230/ab_summary.{md,json}`.
  - Signal capturé côté ON: `validate_callsite_counts={\"validate_base\": 18}` ; `validate_short_circuit_count=0` sur ce probe, avec export agrégé stable `validate_short_circuit_inventory`.
  - Scission sémantique appliquée dans `_validate_solution_traced(...)`:
    - `metrics_only` (passthrough non bloquant) pour échecs métriques en mode canonique natif;
    - `hard_fail_structural` conservé pour cas structurels (pas de matches / WCS inutilisable / résidu non exploitable).
  - Champs runtime ajoutés: `validate_semantics_mode`, `validate_reason_code`, `validate_blocked_stage`, `validate_fail_class`, `validate_metrics_only_passthrough`, `validate_original_*`.
  - Probe de contrôle post-split: `reports/r47i_s0_validate_semantics_split_case055_20260523_0238/ab_summary.{md,json}`.

- 2026-05-23 (S0 neutralisation rescues de validation non-Astrometry):
  - En mode canonique natif, neutralisation explicite appliquée sur:
    - `validate_scale_rescue`
    - `a2v33_borderline_rms`
    - `tweak2_verify`
    - `posthit_tan_refit`
    - `posthit_sip`
  - Implémentation: garde runtime `not astrometry_native_verify_semantics_mode` ajoutée sur ces branches dans `zeblindsolver.py`.
  - Vérification probe ON: `reports/r47i_s0_native_disable_validation_rescues_case055_20260523_0245/summary.{md,json}`.
  - Résultat: compteurs agrégés de rescues neutralisés tous à `0` sur ce cas (`validate_callsite_counts` reste concentré sur `validate_base`).

- 2026-05-23 (S0 neutralisation preverify_center_prior en canonique):
  - Ajout d’un disable canonique explicite `canonical_disable_preverify_center_prior` dans `zeblindsolver.py`.
  - Gate `preverify_center_prior` désormais court-circuité quand `astrometry_native_verify_semantics_mode=True`.
  - Trace ajoutée dans `astrometry_semantics.mode_profile`: `canonical_disable_preverify_center_prior`.
  - Probe ON `case055`: `reports/r47i_s0_disable_center_prior_case055_20260523_0250/summary.{md,json}`.
  - Vérification runtime: `native_verify_semantics_effective=true` et `canonical_disable_preverify_center_prior=true`.

- 2026-05-23 (S0 transaction/accept-gate guardrails):
  - Passage transactionnel canonique renforcé: dans `_solver_handle_hit_transaction_runtime(...)` + `_solver_handle_hit_postverify_transaction_runtime(...)`, un échec `metrics_only` peut être promu pour laisser dérouler `postverify -> stage_pipeline -> accept_gate` quand verify est calculable.
  - Champs diagnostics ajoutés: `validate_metrics_passthrough_count`, `validate_metrics_passthrough_trace_head`, et marqueurs de passthrough transaction/postverify.
  - Garde-fou natif ajouté avant `validate_gate`: si `quality=FAIL` provient d’un motif validate-like et qu’aucun `accept_gate` n’a encore été exécuté, promotion contrôlée + ré-exécution `stage_pipeline` (anti-court-circuit pré-accept).
  - Probes de contrôle:
    - `reports/r47i_s0_force_stage_pipeline_metrics_passthrough_case055_20260523_0300/summary.{md,json}`
    - `reports/r47i_s0_preaccept_validate_guard_case055_20260523_0310/summary.{md,json}`
  - Sur `case055`, les triggers restent à `0` (garde-fous non activés), mais la mécanique est déployée et traçable.

- 2026-05-23 (S0 A/B same-hit post-guard, clôture bloc validate):
  - A/B same-hit OFF/ON publié: `reports/r47i_s0_samehit_ab_post_guard_case055_20260523_0320/ab_summary.{md,json}`.
  - Baseline de comparaison intégrée: `reports/r47i_s0_validate_blocking_inventory_case055_20260523_0230/ab_summary.json`.
  - Résultat durable:
    - `validate_short_circuit_count` reste `0` avant/après (OFF et ON);
    - divergence OFF/ON documentée (native effectif ON uniquement côté canonique);
    - décision finale + provenance runtime publiées (terminal `validate_gate:fail` sur ce cas ON, sans court-circuit validate amont).
  - `followup.md` mis à jour: item global S0 “sortir les rejets Ze non Astrometry” coché sur base de la preuve artefactée.

- 2026-05-23 (S1 same-hit forced payload continuation, post-crash):
  - Artefact de rejeu OFF forcé publié: `reports/r47i_s1_forced_payload_off_probe_case055_20260523_1028/summary.{json,md}`.
    - Candidat: `hinted/S/nominal/d50_2823`.
    - ON: `reason_code=accept_logodds_gate` (`accept_logodds=-1.386...`).
    - OFF forcé: `reason_code=validation_failed` (échec RMS), divergence causale conservée.
  - Nouveau harness dédié créé: `tools/r47i_s1_forced_payload_full_replay.py` pour rejouer ON->OFF avec payload verify complet.
  - Blocage observabilité confirmé: sur runs ON de contrôle (`tmp_dbg_sets_manual_20260523_1042/1046`), `on_refpool_trace.json` est bien produit (avec `verify_input_final`) mais `on_verify_sets.json`/`on_forensic_rows.json` restent absents.
  - Patch instrumentation ajouté dans `zeblindsolver.py`: dump `pre_verify_call` branché sur `blind_astrometry_verify_debug_dump_sets_path` avant l’appel verify; malgré cela le dump attendu ne sort pas sur ce cas, ce qui indique un court-circuit amont/latéral restant à localiser.

## 2026-05-23 (S1 full-payload unblock + nouveau blocage OFF pré-verify)

- Correctif runtime appliqué dans `zeblindsolver.py` (`_solver_handle_hit_postverify_transaction_runtime`): remplacement de l’usage hors-scope `phase_name` dans les dumps pre/post-verify par un label local dérivé de `stats`.
- Effet durable: disparition du crash de dump `verify_dump_error[... name 'phase_name' is not defined]` observé sur `case055` en mode forensic strict.
- Validation technique: `python3 -m py_compile zeblindsolver/zeblindsolver.py tools/r47i_s1_forced_payload_full_replay.py` OK.
- Artefact de preuve S1 full-payload publié: `reports/r47i_s1_forced_payload_full_replay_case055_20260523_1255/summary.{json,md}`.
  - `on_verify_sets.json` et `on_forensic_rows.json` générés avec payload complet (`test=51`, `ref=12`, perms `51/12`).
  - Divergence OFF/ON toujours présente au premier point causal: `reason_code` (`ON=accept_logodds_gate`, `OFF=pixel_scale_out_of_range[scale=1.328,min=1.525,max=15.000]`).
- Lecture durable: le verrou S1 n’est plus l’observabilité des dumps; le nouveau verrou est un gate Ze d’échelle pré-verify côté OFF qui empêche l’entrée verify malgré payload forcé.

- 2026-05-25 (S1 strict: gel des façonnages verify non C-like):
  - Patch appliqué dans `zeblindsolver/zeblindsolver.py` pour renforcer `blind_astrometry_s1_strict_enforce`.
  - Nouveau garde-fou runtime: en mode canonique natif S1 (hors forced replay), lever `s1_invalid[verify_pool_noncanonical_shaping:*]` si un shaping local Ze modifie le pool verify (`verify_local_ref_recenter`, `verify_refstar_cap`, `verify_teststar_filter`, `verify_testperm_reorder`).
  - Effet recherché: empêcher silencieusement les dérives non-Astrometry dans la phase S1 aval et rendre ces écarts immédiatement diffables via `s1_invalid_reason`.
  - Validation technique: `python3 -m py_compile zeblindsolver/zeblindsolver.py` OK.

- 2026-05-25 (validation runtime du garde-fou S1 pool-shaping):
  - Probe artefacté: `reports/r47i_s1_poolshape_probe_case055_20260525_0720/summary.json`.
  - Injection volontaire de shaping local non C-like (`verify_local_ref_recenter` actif) en mode canonique natif S1.
  - Le candidat ciblé est rejeté avec `s1_invalid[verify_pool_noncanonical_shaping:verify_local_ref_recenter]` au stage `validate_gate` (trace `verify_hit_trace`).
  - Point d’observabilité: sur ce run global, le drapeau top-level `stats.s1_invalid` peut rester `false`; la preuve robuste est la raison candidate-stage dans la trace.

- 2026-05-25 (S1 canonical: suppression fallback inlier-only):
  - Durcissement appliqué dans `zeblindsolver.py`: en `astrometry_native_verify_semantics_mode`, le fallback verify via `matches_array` est retiré (plus de chemin fonctionnel `inlier_fallback_matches_array`).
  - En cas de pool natif indisponible, le run strict est explicitement invalidé (`s1_invalid[verify_native_pool_unavailable]`) au lieu d’un fallback silencieux.
  - Validation harness: `reports/r47i_s1_forced_payload_full_replay_case055_20260525_0734/summary.json` conserve la parité OFF/ON (`first_divergence=None`, `accept_logodds_gate` des deux côtés).

- 2026-05-25 (S1 strict collect-ref hardening):
  - Renforcement du garde-fou `blind_astrometry_s1_strict_enforce` sur la collecte ref canonique.
  - Les fallbacks heuristiques de pool sont désormais explicitement classés non-canoniques et bloquants (`s1_invalid`) quand la collecte native stricte est attendue:
    - `refpool_source_from_collect_n`
    - `refpool_source_cap_n`
    - `strict_no_seed_mapping`
  - Objectif: rapprocher l’item S1 « collecte ref verify_hit C stricte sans fallback heuristique » d’une clôture artefactable.
  - Validation: `python3 -m py_compile zeblindsolver/zeblindsolver.py` OK.

- 2026-05-25 (S1: collecte ref canonique durcie):
  - En mode S1 strict natif, la collecte ref est désormais contrainte au parcours canonique scope WCS -> projection WCS -> inside-image, avec invalidation explicite si signaux non canoniques.
  - Nouveaux marqueurs non canoniques injectés dans `verify_ref_collect_noncanonical`: `verify_ref_mode_noncanonical[...]`, `mirror_scope_source_noncanonical[...]`, `external_ref_pool_enabled`.
  - Le scope global ref est forcé actif en strict (pas de dépendance à un flag config externe pour activer le comportement attendu en S1).
  - Validation ON/OFF verte sur case055: `reports/r47i_s1_forced_payload_full_replay_case055_20260525_0752/summary.json` (`first_divergence=None`, `accept_logodds_gate` des deux côtés).

- 2026-05-25 (S1 strict MatchObj guard):
  - Ajout d’un garde-fou bloquant sur la matérialisation MatchObj en mode canonique natif.
  - Si `mo_native_source` vaut `nearest_*`/`none`, on lève `s1_invalid[verify_matchobj_not_materialized:*]`.
  - Interprétation durable: l’export `mo.star/mo.field` ne suffit plus; la clôture S1 exige désormais une provenance non-proxy explicite.

- 2026-05-25 (S1: matérialisation MatchObj avant verify):
  - `zeblindsolver.py` utilise maintenant en priorité `collect_pref_pair_idx` pour matérialiser les ancres `mo.star/mo.field` (quad) au lieu d’un proxy nearest.
  - Nouvelles sorties observabilité ajoutées: `mo_quadpix`, `mo_quadxyz`, `mo_scale` (en plus de `mo_star/mo_field/mo_native_source`).
  - Sur case055, la source est bien `mo_native_source=collect_pref_pair_idx` et la parité ON/OFF reste intacte (`reports/r47i_s1_forced_payload_full_replay_case055_20260525_0848/summary.json`).

- 2026-05-25 (S1 strict MatchObj completeness):
  - Après le guard source non-proxy, ajout d’un guard de complétude structurelle MatchObj.
  - Les runs S1 natifs échouent désormais si le bundle `mo` n’est pas complet (4/4 + scale), ce qui verrouille la preuve attendue avant clôture.
  - Instrumentation `verify_matchobj_source/field_n/star_n/quadpix_n/quadxyz_n/scale` publiée dans les stats verify.

- 2026-05-25 (S1 strict closure revalidée):
  - `r47i_s1_canonical_strict_closure_case055_20260525_1011/summary.json` passe `all_ok=true`.
  - Ajustement nécessaire du checker de clôture: accepter le `pool_source` canonique runtime actuel (`field_mirror_global_refscope_tile_world_gidperm`) en plus de l’ancien libellé strict.
  - La parité OFF/ON reste inchangée sur case055 (first_divergence none, fingerprint identique, verify-only divergence none).

- 2026-05-25 (lisibilité backlog):
  - `followup.md` enrichi d’un bloc en tête `Backlog opératoire lisible (pilotage)` avec files `NOW/NEXT/LATER`.
  - Objectif: pilotage rapide sans perdre l’historique détaillé S0..S5 plus bas dans le fichier.
  - Convention pratique: exécuter d’abord `NOW`, ne remonter à `NEXT` qu’après artefact vert sur `NOW`.

- 2026-05-25 (audit S2 vs Astrometry):
  - Revue S2 effectuée, couverture globalement solide, mais 3 angles morts explicites ajoutés au backlog:
    1) assertions runtime d’invariants canonique S2 (anti-fallback silencieux),
    2) validation multi-cas (au-delà de case055),
    3) audit stabilité numérique verify/log-odds (ties, NaN/inf, clipping).
  - `followup.md` mis à jour dans la section S2.

- 2026-05-25 (S2 item complété):
  - `quad_center`/`Q2` sont désormais calculés prioritairement depuis le vrai quad d'hypothèse (indices image issus de `collect_pref_pair_idx`) au lieu de `img_in` head-only.
  - Nouvelle provenance runtime: `quad_center_source=collect_pref_pair_idx_quad`.
  - Non-régression parité validée sur case055: `reports/r47i_s1_forced_payload_full_replay_case055_20260525_2355/summary.json` (`first_divergence=None`, `accept_logodds_gate` ON/OFF).

- 2026-05-25 (S2.2 item complété):
  - `pix2` verify est maintenant calculé explicitement selon la forme Astrometry auditable: `verify_pix2 = verify_pix^2 + (index_jitter/scale)^2` (plus de formulation implicite via `hypot`).
  - Nouvelles traces exportées dans les dumps pre/post verify: `verify_pix2_base`, `verify_pix2_jitter_term`, `verify_pix2_input`, `verify_index_jitter_arcsec`, `verify_scale_arcsec_px_for_pix2`.
  - Probe de validation: `reports/r47i_s1_forced_payload_full_replay_case055_20260525_2358/summary.json` reste vert en parité (`first_divergence=None`, `s1_invalid_any=false`).
  - Angle mort ouvert en S2: verrouiller la source stricte de `mo.scale` pour ce calcul (éviter fallback implicite quand la valeur native MatchObj est disponible).

## 2026-05-26 00:20 — S2 audit rapide vs Astrometry
- Relecture croisée `astrometry-net-main/solver/verify.c` vs `zeblindsolver.py` (zone verify/tests sigma).
- Confirmation runtime de `S2.1` (gamma-like avec vrai quad hypothèse `collect_pref_pair_idx_quad`) via artefact `r47i_s1_forced_payload_full_replay_case055_20260525_2355`.
- Confirmation runtime de `S2.2` (`verify_pix2 = verify_pix^2 + (index_jitter/mo.scale)^2`).
- `S2.3` confirmé (`first_divergence=None` même candidat OFF/ON).
- Backlog mis à jour: `S2.1`/`S2.3` cochés; reste S2.4/S2.5 comme prochains blocs critiques.

- 2026-05-26 (S2.4 partiel):
  - RoR est maintenant rebranché sur le `Q2` exact du quad hypothèse quand disponible (`quad_q2_px2`), au lieu d’une estimation percentile `test_xy`.
  - Ajout de traçabilité runtime: `ror_q2_source` (`hypothesis_quad_q2` / fallback explicite), exporté dans `on_refpool_trace.json`.
  - Validation non-régression ON/OFF: `reports/r47i_s1_forced_payload_full_replay_case055_20260526_0034/summary.json` conserve `first_divergence=None`.
  - Vérification trace: `on_refpool_trace.json` expose `ror_q2_source=hypothesis_quad_q2` et `ror_q2_px2` cohérent avec le quad forcé.

## 2026-05-26 01:02 — S2.4 audit RoR/effective-area
- Audit artefacté publié: `reports/r47i_s2_ror_effective_area_audit_20260526_0102/summary.{md,json}`.
- Confirmé: formule `verify_get_ror2` et calcul `effective_area_px2` alignés code-level avec Astrometry.
- Ouvert: preuve runtime diffable du sous-chemin `uniformize bins -> effective area` (équivalence non encore démontrée).
- Backlog maintenu lisible: S2.4 reste ouvert, S2.5 inchangé.

- 2026-05-26 (S2.4 clôturé):
  - Instrumentation RoR enrichie dans `zeblindsolver.py` avec entrées/sorties explicites de `verify_get_ror2` (`ror_input_area_px2`, `ror_input_sigma2_px2`, `ror_input_distractor_rate`, `ror_input_nr`, `ror2_px2`).
  - Audit runtime publié: `reports/r47i_s2_ror_effective_area_audit_20260526_0106/summary.{md,json}`.
  - Résultats: `ror_q2_source=hypothesis_quad_q2` ON/OFF, recomputation `verify_get_ror2` conforme au runtime ON/OFF, `effective_area_px2` aligné ON/OFF sur case055.

## 2026-05-26 01:11 — S2.4 uniformize/bins gap verrouillé
- Audit dédié publié: `reports/r47i_s2_uniformize_bins_gap_audit_20260526_0111/summary.{md,json}`.
- Conclusion claire: le runtime verify ZeBlind n’a pas encore l’équivalent de la branche Astrometry `uniformize bins -> effective area`.
- Décision de pilotage: garder S2.4 ouvert; prochain delta causal unique = implémenter cette branche puis revalider OFF/ON diffable.

- 2026-05-26 (S2.5 partiel):
  - Harness `r47i_s1_forced_payload_full_replay.py` enrichi avec dump `on/off_verify_steps.json` (trace séquentielle log-odds + seuils accept/bail/stoplooking).
  - Audit publié: `reports/r47i_s2_logodds_chain_audit_20260526_0118/summary.{md,json}`.
  - Résultat: séquence log-odds ON/OFF alignée sur case055 (`first_divergence=None`, seuils identiques, pas de bail/stoplooking déclenché).
  - Gap restant identifié: preuve explicite `min_validations` incomplète en canonique strict (branche `verify_policy_trace` non alimentée côté ON sur ce case).

- 2026-05-26 (S2.5 clôturé):
  - `zeblindsolver.py` exporte désormais `verify_logodds_min_validations` dans `blind_verify_step_dump_path` (entrée diffable ON/OFF de la chaîne log-odds).
  - Harness `r47i_s1_forced_payload_full_replay.py` conserve `on/off_verify_steps.json` pour audit runtime explicite des seuils `accept/bail/stoplooking`.
  - Audit final S2.5: `reports/r47i_s2_logodds_chain_audit_20260526_0122/summary.{md,json}` avec alignement ON/OFF sur `accept/bail/stoplooking/min_validations` (`min_validations=8`) et `first_divergence=None` sur case055.

## 2026-05-26 01:32 — S2.4 premier item manquant traité
- Vérification backlog: incohérence détectée (S2.4 marqué fait en tête alors que gap bins encore ouvert).
- Correctif code appliqué dans `zeblindsolver.py`:
  - nouvelle branche runtime `uniformize bins -> effective area` dans `_apply_verify_ror_filter`,
  - sizing bins HEALPix approx + filtrage RoR bin-centers + `effA` proportion de bins,
  - exports runtime `ror_uniformize_bins_used/nw/nh/goodbins_n`.
  - nouveau paramètre config `blind_astrometry_verify_index_cutnside`.
- Probe publié: `r47i_s1_forced_payload_full_replay_case055_20260526_0132`.
- Résultat: champs présents dans `on/off_refpool_trace.json`, mais bins non activés sur ce cas (`ror_uniformize_bins_used=false`) ; S2.4 reste ouvert pour preuve active.

- 2026-05-26 (S2 angle mort mo.scale -> pix2 résorbé):
  - `zeblindsolver.py` sélectionne désormais explicitement la source de scale pour `pix2` (`model_scale_arcsec` prioritaire, `pix_scale_arcsec_fallback` explicite seulement en secours) au lieu d’un fallback implicite.
  - Nouvelle traçabilité exportée: `verify_scale_source_for_pix2` dans `on/off_verify_sets.json`.
  - Validation run: `reports/r47i_s1_forced_payload_full_replay_case055_20260526_0137/summary.json` (`first_divergence=None`, `s1_invalid_any=false`).

## 2026-05-26 01:40 — S2.4 fermé sur preuve runtime active
- Harness patché pour injecter `blind_astrometry_verify_index_cutnside` via env `ZB_VERIFY_INDEX_CUTNSIDE`.
- Probe validé: `r47i_s1_forced_payload_full_replay_case055_20260526_0140`.
- Résultat clé: `ror_uniformize_bins_used=true` côté ON/OFF (trace runtime), sans régression terminale (`first_divergence=None`, `accept_logodds_gate` des deux côtés).
- Décision: `S2.4` clôturé; prochain manque S2 = verrouillage source native `mo.scale` pour `verify_pix2` (éviter fallback implicite).

- 2026-05-26 (S2 term-pollution check validé):
  - Audit diffable publié: `reports/r47i_s2_term_pollution_audit_20260526_0142/summary.{md,json}`.
  - Constat clé: `verify_pix2_input` et `logbg` alignés ON/OFF; `index_jitter` identique; source scale explicite `model_scale_arcsec` (pas de fallback implicite).
  - Les écarts restants (`verify_pix2_jitter_term`, `verify_scale_arcsec_px_for_pix2`, `field_radius_px`) proviennent de la géométrie/scale runtime du candidat, pas d’une pollution par defaults Ze.

## 2026-05-26 01:48 — S2 verify_pix2 source hardening
- `verify_pix2` durci pour prioriser `mo_scale_native` quand disponible (source exportée `verify_scale_source_for_pix2`).
- Nouveau garde-fou runtime: en natif canonique strict, fallback silencieux de scale interdit si une source native est attendue (`s2_invalid[verify_pix2_scale_fallback_under_native]`).
- Probe validé: `r47i_s1_forced_payload_full_replay_case055_20260526_0148` avec `first_divergence=None` et terminal `accept_logodds_gate` ON/OFF.

- 2026-05-26 (S2 invariants runtime validés):
  - Audit publié: `reports/r47i_s2_runtime_invariants_audit_20260526_0206/summary.{md,json}`.
  - Les invariants de source scale/pix2 sont maintenant contrôlés par garde-fou runtime (`s2_invalid[...]`) et vérifiés sur le run de référence sans régression parité.

## 2026-05-26 02:11 — S2 clôture opérationnelle
- Validation multicase publiée: `r47i_s2_multicase_native_probe_20260526_0209` (3 cas, invariants verify cohérents).
- Audit stabilité numérique publié: `r47i_s2_numeric_stability_audit_20260526_0211` (pass: aucun NaN/inf/logodds/nsig2 invalide sur les traces scannées).
- Décision backlog: S2 marqué clos; focus bascule sur S3.1 (onefield accept machine).

## 2026-05-26 08:30 — S3.1 itération callback onefield (logodds-first + identité champ)
- Patch ciblé appliqué dans `zeblindsolver.py`:
  - `record_match_callback` enregistre maintenant `fieldfile` et `fieldnum` dans `onefield_solution_records` (au lieu d’un enregistrement centré `tile` uniquement).
  - insertion callback triée par `solve_logodds` décroissant (fallback append si score non fini), au lieu d’un append brut.
- Vérification locale: `python3 -m py_compile zeblindsolver/zeblindsolver.py` OK.
- Impact attendu: rapprochement de la sémantique onefield `best_match`/dedup par identité de champ avant la finalisation stricte `best_match_solves` + `remove_duplicate_solutions`.

## 2026-05-26 08:38 — S3.1 gap de câblage fermé (best_match_solves)
- `blind_best_match_solves_enabled` n’était utilisé qu’au runtime via `getattr`, sans exposition explicite config/CLI.
- Correctif appliqué:
  - ajout du champ dans `SolveConfig`,
  - ajout de l’argument CLI `--blind-best-match-solves-enabled`,
  - injection explicite dans la construction `SolveConfig(...)`.
- Complément dedup: tri final des solutions dédupliquées par score logodds décroissant pour un ordre stable onefield-like.
- Vérifications: `py_compile` OK + `SolveConfig().blind_best_match_solves_enabled == True`.

## 2026-05-26 08:25 — S3.1 itération best_match_solves + dedup strict
- Patch complémentaire `zeblindsolver.py`:
  - propagation explicite de `record_match_callback_solve_pass` dans `solution.stats` (accept/reject tosolve).
  - `_is_solution_better_onefield(...)` priorise les solutions `solve_pass` (flag `blind_best_match_solves_enabled`, défaut True), puis logodds/inliers/RMS.
  - `_remove_duplicate_solutions_runtime(...)` ne déduplique plus sur fallback `tile`; dedup seulement si identité `fieldfile/fieldnum` valide.
- Validation rapide:
  - `python3 -m py_compile zeblindsolver/zeblindsolver.py` OK.
  - probe S1 republié: `reports/r47i_s1_forced_payload_full_replay_case055_20260526_0824/summary.json` (pas de régression invariants S1, mais cas non discriminant pour multi-solves car rejet `accept_logodds_gate`).

## 2026-05-26 08:52 — S3.1 preuve intermédiaire via probe onefield renforcé
- `tools/r47i_s3_onefield_multisol_probe.py` renforcé pour ajouter un discriminant synthétique contrôlé de la policy onefield (dedup + tri).
- Artefact publié: `reports/r47i_s3_onefield_multisol_probe_20260526_0851/summary.{md,json}`.
- Résultat:
  - runtime réel case055: `onefield_solution_records_n=0` (toujours non discriminant),
  - discriminant synthétique: `pass=true` (`output_n=3`, gagnant `A/1` à `solve_logodds=-0.7`, ordre décroissant, unresolved identity conservée).
- Conclusion opérationnelle: la logique policy est validée sur jeu contrôlé, mais la clôture S3.1 reste dépendante d’un cas runtime avec multi-solutions effectives (`n>1`).

- 2026-05-26 (S3.1 blocage runtime confirmé):
  - scan borné `reports/tmp_*.fit` avec config permissive onefield (`blind_best_hit_only=0`, seuils très bas) sur index `forensic_case055_subset96/index`.
  - résultats observés avant timeout: `onefield_solution_records=0` et `collected_hits_runtime=0` sur tous les fichiers traités.
  - implication: pour clôturer S3.1, il faut changer de couple `input/index_root` (subset forensic actuel trop pauvre pour générer des multi-hits exploitables).

## 2026-05-26 09:31 — S3 recommandations pratiques exécutées (11/12/13)
- S3.11 audit couverture réalisé: `reports/r47i_s3_index_coverage_audit_20260526_0925.json`.
  - `tmp_case055_nowcs.fit`: RA=184.775, Dec=47.447, scale~2.393"/px, FOV~1.276°.
  - subset96: seulement 2 tuiles candidates (`d50_2725`, `d50_2823`) -> couverture serrée.
- S3.12 index dédié local construit depuis `~/zesolver_index`:
  - sortie: `reports/r47i_s3_case055_dedicated_index_v1/index`,
  - cône élargi ~4.59°, 9 tuiles sélectionnées autour du cas.
- S3.13 probe rejoué sur couple FITS+index dédié:
  - artefact: `reports/r47i_s3_onefield_multisol_probe_20260526_0929/summary.json`,
  - résultat: toujours `onefield_records_n=0`.
- Tentative complémentaire sur index complet `~/zesolver_index` lancée mais non concluante en temps interactif (timeout long), à reprendre avec exécution bornée instrumentée.

- 2026-05-26 (S3.12 extension index dédié):
  - nouvel index local `r47i_s3_case055_dedicated_index_v2` construit depuis `~/zesolver_index` avec cônes cumulés (~4.59°, ~7.66°, 12.0°) pour augmenter la densité locale.
  - couverture résultante: `selected_tile_n=38` (vs 9 pour v1).
  - probe S3 sur v2 lancé; exécution plus longue que la fenêtre interactive courte, résultat final à récupérer/valider.

- 2026-05-26 (S3 index dédié v3 + probe rapide):
  - index `r47i_s3_case055_dedicated_index_v3` construit depuis `~/zesolver_index` avec seeds multi-FITS historiques M106 + case055.
  - couverture v3 reste `selected_tile_n=38` (mêmes tuiles utiles que v2 sur cette zone).
  - smoke probe rapide publié: `reports/r47i_s3_onefield_multisol_probe_fast_20260526_1023/summary.json` -> `onefield_records_n=0`.
  - implication: le blocage S3.1 n’est probablement plus un problème de couverture locale brute; il faut maintenant un cas FITS/candidat qui passe la chaîne jusqu’au callback onefield.

## 2026-05-26 11:37 — Audit profondeur S3 (verrou pré-callback identifié)
- Drilldown publié: `reports/r47i_s3_gap_drilldown_20260526_1137.json`.
- Résultat: `verify_hit_trace_len=0` et `reject_reason_class_counts={'other':58}` sur un FITS M106 historisé.
- Top rejets: `strict_transform_scale_precheck_reject` avec `scale_ratio_hint` très élevé (jusqu’à ~400), avant entrée dans `verify_hit`.
- Lecture: le blocage S3 est principalement un verrou de precheck scale/transform (alignement pipeline), plus qu’un problème simple de qualité image ou de couverture locale index.

## 2026-05-26 08:29 — Probe runtime post-patch S3.1
- Artefact: `reports/r47i_s1_forced_payload_full_replay_case055_20260526_0829`.
- OFF/ON restent alignés (`first_divergence=None`) avec même terminal `accept_logodds_gate`, même `accept_logodds=-1.3862943611198908`, même `verify_entry_nt/nr=49/83`, `s1_invalid_any=false`.
- Interprétation: les ajustements callback/best-match/dedup appliqués ce matin n’ont pas cassé la parité runtime sur le case de référence.

## 2026-05-26 08:40 — S3.1 preuve multi-solutions: gap dataset
- Outil ajouté: `tools/r47i_s3_onefield_multisol_probe.py` pour forcer un run `best_hit_only=0` et auditer `best_match_solves` + dedup `fieldfile/fieldnum`.
- Run probe: `reports/r47i_s3_onefield_multisol_probe_20260526_0840` => `raw_solution_records=0` (`collected_hits_runtime=0`) même avec fallback permissif.
- Audit des artefacts existants: aucune trace `raw_solution_records > 1` dans `reports/**`; les traces historiques trouvées sont à `1` uniquement.
- Conclusion durable: la clôture S3.1 nécessite un cas/index qui produit réellement plusieurs callbacks acceptables; sans cela, on ne peut pas prouver runtime `best_match_solves/remove_duplicate_solutions` au-delà du câblage code.

## 2026-05-26 09:12 — S3.1 preuve unitaire multi-solutions (fallback pragmatique)
- Les probes runtime sur datasets courants n’ont pas fourni de cas `raw_solution_records>1`; impossible de conclure exclusivement par artefact run-time multi-hit à ce stade.
- Refactor ciblé: extraction de la logique onefield score+dedup en fonctions pures réutilisées par le runtime:
  - `_onefield_logodds_score_from_row`
  - `_onefield_dedup_rows`
- Tests unitaires ajoutés et passants (`pytest -k onefield`) pour valider explicitement un scénario multi-solutions simulé (dedup par `fieldfile/fieldnum`, conservation du meilleur score, tri score décroissant).
- Décision: conserver ce filet de preuve déterministe tant qu’un dataset runtime multi-hit robuste n’est pas identifié.

## 2026-05-26 09:49 — Audit couverture index S3
- Script ajouté: `tools/r47i_s3_index_coverage_audit.py` (calcule RA/Dec/rayon depuis WCS FITS puis compte les tuiles sélectionnées dans chaque manifest).
- Artefact: `reports/r47i_s3_index_coverage_audit_20260526_0949`.
- Constat durable: sur les 3 FITS `forensic_oracle_*`, les 2 index testés (`forensic_case055_subset96`, `forensic_m106_reference_v1`) donnent exactement la même couverture: `2/3` couverts, `1/3` hors couverture (`tiles_selected=0`), et un cas à couverture minimale (`1` tuile).
- Interprétation: ce n’est pas seulement un problème de tuning solver; la qualité de preuve runtime S3 dépend d’un couple dataset/index mieux aligné (couverture multi-tuile suffisante).

## 2026-05-26 10:08 — S3 runtime: coverage filtrée mais toujours 0 callback
- Nouveau script: `tools/r47i_s3_select_fits_by_coverage.py` (sélectionne automatiquement les FITS ayant au moins `N` tuiles candidates pour un manifest).
- Sur `forensic_m106_reference_v1` avec `min_tiles=2`, un seul FITS passe (`reports/r47i_s3_select_fits_by_coverage_20260526_1004`).
- Probe onefield relancé sur ce FITS avec réutilisation WCS désactivée dans la config probe (`blind_reuse_existing_solved_wcs=False`) ; artefact `r47i_s3_onefield_multisol_probe_20260526_1008`.
- Résultat: toujours `raw_solution_records=0`; la couverture minimale seule ne déclenche pas de callback multi-hit.
- Conclusion durable: pour fermer S3 runtime, il faut un couple dataset/index conçu pour générer des candidats valides concurrents, pas seulement un index qui couvre géométriquement le champ.

## 2026-05-26 10:48 — Batch probe S3 robuste (timeouts par FITS)
- Outil créé: `tools/r47i_s3_batch_probe_from_coverage.py`, piloté par `summary.json` d’audit coverage.
- Ajout clé: exécution `solve_blind` en process enfant avec timeout par FITS (`--per-fit-timeout-s`) pour garantir un verdict même si un solve est trop long.
- Run: `reports/r47i_s3_batch_probe_from_coverage_20260526_1048` (2 FITS, `min_tiles=1`, timeout 90s) => 2 timeouts, aucun callback (`raw_solution_records` non observé).
- Leçon durable: le blocage S3 runtime n’est pas seulement la couverture géométrique; le coût de solve sur ces FITS/index empêche même d’atteindre la phase callback dans une fenêtre probe raisonnable.

## 2026-05-26 11:47 — Probe ultracourt S3
- Le batch probe coverage supporte maintenant `--profile ultrashort` pour stress-test callback sous timeout court.
- Run `r47i_s3_batch_probe_from_coverage_20260526_1147` (2 FITS, timeout 45s/FITS) => 2 timeouts, aucun `raw_solution_records`.
- Conclusion durable: sur ces FITS/index, même la stratégie ultracourte n’atteint pas la phase callback; il faut un index S3 reconstruit/focalisé pour réduire drastiquement le coût de recherche.

## 2026-05-26 12:25 — Index S3 focalisé validé mais callbacks toujours inatteignables
- Rebuild focalisé `s3_focused_index_20260526_1200` corrigé avec `sampler=pairwise_multiscale_v1`; l’erreur de compatibilité précédente est levée.
- Coverage audit sur ce nouvel index: 2/2 FITS couverts (`tiles 6` et `1`).
- Malgré cela, batch probe ultracourt (45s/FITS) reste à 2 timeouts, sans `raw_solution_records`.
- Leçon durable: le verrou S3 courant est un verrou de coût/latence avant callback; la prochaine étape doit réduire le coût warmup/search ou utiliser un sous-lot runtime encore plus léger pour atteindre au moins un callback observé.

## 2026-05-26 13:22 — S3 focused index: verdict callback négatif sans timeout
- Probe `r47i_s3_batch_probe_from_coverage_20260526_1322` sur index focalisé (`min_tiles=2`, 1 FITS, timeout 120s) terminé sans timeout.
- Résultat runtime: `success=false`, `raw_solution_records=0`, `dedup_solution_records=0`, `collected_hits_runtime=0`.
- Conséquence: même quand le solve aboutit temporellement, aucun hit n’atteint la machine onefield callback sur ce setup; la prochaine étape doit instrumenter/ouvrir explicitement la trajectoire pré-callback (pas seulement retoucher index/timeout).

## 2026-05-26 14:56 — S3: chaîne de verrous pré-callback confirmée
- Le batch probe expose maintenant un diagnostic pré-callback (`reason_counts`, `stage_counts`, `terminal_event`).
- Séquence observée sur index focalisé:
  1) baseline probe -> `validation_failed` (validate_gate),
  2) RMS relâché -> `verify_prob_failed`,
  3) prob verify désactivé -> terminal `center_prior` fail.
- Aucun run n’atteint encore `record_match_callback` (`raw_solution_records=0`), mais l’ordre causal des verrous est désormais explicite.
- Étape logique suivante: profil probe “pre_callback_open_path” pour ouvrir simultanément ces verrous et obtenir un premier callback runtime avant toute optimisation produit.

## 00:50 — S5 upstream diff audit (multicase)
- Artefact publié: `reports/r47i_s5_upstream_diff_audit_20260527_0047/{summary.md,summary.json}`.
- Résultat robuste: `first_divergence.index=0` sur toutes les paires de FITS sentinelles (4 FITS).
- Lecture durable: ce signal confirme une variabilité amont immédiate inter-images (candidate_key/hits/pairs), mais ne constitue pas une preuve de divergence Ze-vs-C au sens strict car la comparaison est faite entre FITS différents.
- Règle de suite: pour isoler un écart causal S5 exploitable, comparer le même FITS (Ze vs référence C, ou Ze vs Ze contraint en ordre canonique single-pass newpoint).

## 2026-05-27 09:00 — S5 single-pass newpoint trace mode
- Ajout d’un mode d’instrumentation upstream ZeBlind pour S5: `blind_s5_upstream_single_pass_newpoint_enabled`.
- Effet validé sur même FITS M106 + même index S3 élargi: `newpoint` ne reset plus entre phases dans la trace head (`0..15` sur les 16 premières lignes), avec source explicite `single_pass_trace_cursor`.
- Artefacts: `r47i_s5_upstream_trace_probe_20260527_0859` + compare `r47i_s5_singlepass_newpoint_compare_20260527_0900`.
- Limite connue: l’audit `r47i_s5_newpoint_permutation_gap_audit.py` attend encore `upstream_solver_trace_head` en top-level et ne lit pas directement `results[].trace_head`; adaptation nécessaire pour l’automatisation stricte.
## 2026-05-27 09:57 — S5 audit auto aligné probe format
- `tools/r47i_s5_newpoint_permutation_gap_audit.py` supporte désormais les deux entrées: legacy `upstream_solver_trace_head` et probe `results[0].trace_head`.
- Validation auto sur `r47i_s5_upstream_trace_probe_20260527_0859`: verdict `closed` avec `newpoint_global_non_decreasing=true` et `first_causal_gap=null`.
- Conséquence: le verrou “reset newpoint inter-phases” est désormais traité et vérifiable automatiquement sur le protocole S5 courant.
## 2026-05-27 10:04 — S5 permutation/callsite audit
- Trace upstream ZeBlind enrichie avec marqueurs callsite (`use_px_spec`, `use_ra_filter`, `agg_levels`).
- Probe même FITS/index relancé (`r47i_s5_upstream_trace_probe_20260527_1004`).
- Audit dédié `r47i_s5_permutation_callsite_audit_20260527_1004`:
  - tuples répétés détectés (`multi_seen_count=5`),
  - incohérences callsite sur ces tuples (`callsite_inconsistency_count=5`),
  - aucune divergence de tête de permutation (`perm_inconsistency_count=0`).
- Implication: le prochain différentiel S5 doit prioriser la sémantique callsite/pipeline plutôt qu’un tuning permutation head.
## 2026-05-27 10:18 — S5 callsite drift classifié (pas de régression)
- Nouvel audit `r47i_s5_callsite_drift_audit_20260527_1018` sur le probe S5 courant.
- Les 5 dérives callsite détectées sont classées `expected_widening` (élargissement monotone de `agg_levels/levels_to_use`), avec `unexpected_count=0`.
- Conséquence: pas de régression callsite pipeline sur ce périmètre; prochain différentiel S5 doit cibler le mapping stage/callsite C-vs-Ze.
## 2026-05-27 10:25 — S5 stage/callsite mapping validé
- Audit dédié `r47i_s5_stage_callsite_mapping_audit_20260527_1025` exécuté sur le probe S5 courant.
- Résultat: aucun break d’ordre de phase et aucun break de contrat callsite (`0/0`), statut clos.
- Implication: le prochain différentiel S5 doit viser la sémantique C-vs-Ze intra-tuple candidat, pas une incohérence de transitions phase/callsite.
## 2026-05-27 10:33 — S5 tuple semantic diff (d50_2628)
- Audit `r47i_s5_tuple_semantic_diff_audit_20260527_1033` publié sur le trace S5 courant.
- `d50_2628` apparaît 8 fois (`scale_only=4`, `blind=4`), ce qui confirme une réutilisation inter-phases cohérente avec l’orchestration Ze (widening de level sets), pas un artefact isolé de permutation.
- L’écart C-vs-Ze restant à ce stade est structurel de flux (single-pass C vs passes phasées Ze), donc le next step utile est d’introduire un tag trace `first_pass/widened_pass` pour corréler précisément les issues d’acceptation.
## 2026-05-27 10:42 — S5 pass-tag activé et auditable
- `pass_tag` (`first_pass`/`widened_pass`) est maintenant injecté dans la trace upstream et dans `stage_by_stage.hypothesis`.
- Artefacts: `r47i_s5_upstream_trace_probe_20260527_1041` + `r47i_s5_pass_tag_acceptance_audit_20260527_1042`.
- Résultat de l’audit: `first_pass` et `widened_pass` sont tous deux visibles côté hypothèses, mais l’export `hypothesis` est encore principalement `prevalidate`; il faut propager le tag sur `verify/verify_hit` pour corréler directement avec l’acceptation finale.
## 2026-05-27 10:58 — S5 pass_tag verify_hit: blocage de propagation callsite
- Malgré propagation `pass_tag` dans `candidate_ctx` + `verify_trace` + `verify_hit` builders, l’audit `r47i_s5_pass_tag_acceptance_audit_20260527_1058` donne encore `verify_hit_rows_by_pass_tag={'unknown':30}`.
- Interprétation durable: sur ce protocole, les callsites `_emit_hit_event` majoritaires alimentent `verify_hit` sans contexte candidat taggé.
- Next utile: patch ciblé des callsites `_emit_hit_event` dominants pour forcer `context.pass_tag`, puis re-run audit pour obtenir une mesure finale first_pass vs widened_pass.
## 2026-05-27 11:30 — S5 pass_tag: fallback insuffisant, callsites à patcher explicitement
- Même après fallback runtime renforcé dans `_emit_hit_event`, audit `r47i_s5_pass_tag_acceptance_audit_20260527_1130` reste à `verify_hit_rows_by_pass_tag={'unknown':30}`.
- La matrice `unknown_stage_phase_counts` isole les cibles exactes (`validate_base`, `accept`, `hit_resolve_chain` sur hinted/hinted_wide/scale_only/blind).
- Conclusion durable: next patch doit injecter `pass_tag` explicitement au niveau des callsites `_emit_hit_event` dominants, pas seulement via fallback global.
## 2026-05-27 11:59 — S5 fermeture angle mort pre_perm->perm_emitted
- Patch runtime trace dans `zeblindsolver.py`: `perm_null_reason` par défaut passé à `no_quad_survived_preperm_gates` pour les lignes restant en stage `pre_perm`.
- Reprobe same-FITS/index: `projects/ZeSolver/reports/r47i_s5_upstream_trace_probe_20260527_1159/summary.json`.
- Outcome explicite extrait du trace head: `PRE_PERM_N=5`, `PRE_PERM_REASON_COUNTS={'no_quad_survived_preperm_gates': 5}`, `PRE_PERM_LEVELS={'M': 5}`.
- Conclusion durable: le non-pass `pre_perm -> perm_emitted` observé sur ce protocole est expliqué par un filtrage pré-permutation (aucun quad survivant), pas par une dérive de permutation/callsite.
## 2026-05-27 21:01 — S5 pass_tag verify_hit: loup fermé
- Re-run complet same-FITS/index: `projects/ZeSolver/reports/r47i_s5_upstream_trace_probe_20260527_2101/summary.json` + `projects/ZeSolver/reports/r47i_s5_pass_tag_acceptance_audit_20260527_2101/summary.json`.
- Résultat clé: `verify_hit_rows_by_pass_tag` n’est plus `unknown` et devient `{'first_pass':16,'widened_pass':14}`; `verify_hit_unknown_stage_phase_counts={}` et `verify_hit_unknown_callsite_counts={}`.
- Conclusion durable: la chaîne de propagation `pass_tag` vers `verify_hit` est désormais opérationnelle sur le protocole S5 courant; l’angle mort d’observabilité est levé.
## 2026-05-27 22:52 — S5 guardrail widened-pass: verdict no-go par défaut
- Implémentation opt-in ajoutée (`blind_s5_widened_pass_code_log_relax_enabled`, `blind_s5_widened_pass_code_log_relax_multiplier`) + support A/B dans `tools/r47i_s5_upstream_trace_probe.py`.
- A/B OFF vs ON(x1.35) sur same-FITS: aucun effet (`success` inchangé, `pre_perm` inchangé).
- Sweep ON (`x1.35/x1.75/x2.25`) montre un effet seulement à `x2.25` (disparition `pre_perm` sur ce cas), mais audit runtime OFF vs ON(x2.25) montre surtout plus de widened-pass (`verify_hit_n 30 -> 36`, `widened_pass 14 -> 20`) sans gain de succès.
- Décision durable: ne pas activer ce guardrail en produit par défaut (default OFF), le garder uniquement comme levier d’investigation.

## 2026-06-02 11:33 — S6 audit Astrometry ciblé: divergences onefield/callback à fermer avant retuning
- Pour le lot M106, le problème utile n'est plus “obtenir un success” mais comprendre pourquoi `native_verify + tosolve` produit `5/5 false_positive`.
- Une passe d'audit source Astrometry focalisée sur `solver_handle_hit` / `record_match_callback` / `compare_matchobjs` / `remove_duplicate_solutions` a mis en avant trois divergences runtime Ze à traiter avant tout nouveau sweep de seuils:
  - identité onefield non canonique: fallback `fieldfile = tile_key`, donc dédup/ranking par tuile au lieu de par field;
  - tri onefield non canonique: insertion sur le seul `solve_logodds`, alors qu'Astrometry trie par `(fieldfile, fieldnum, logodds desc)`;
  - score callback non unifié: le callback Ze choisit explicitement `prob_logodds` en mode strict au lieu de consommer un unique `mo->logodds` final comme Astrometry.
- Règle durable S6: ne plus retweaker `blind_anodds_tosolve` à l'aveugle tant que ces divergences de sémantique onefield ne sont pas fermées.

## 2026-06-02 11:39 — S6 R1/R2 fermés côté runtime onefield
- `R1` implémenté: le runtime onefield injecte désormais une identité canonique issue de `input_fits` (`fieldfile=str(input_fits)`, `fieldnum=0`) au lieu de retomber sur `tile_key`.
- `R2` implémenté: insertion et dédup des `onefield_solution_records` basculées sur une clé Astrometry-like `(fieldfile, fieldnum, logodds desc)`.
- Vérification locale: `python3 -m py_compile` OK et `pytest -q tests/test_zeblindsolver.py -k onefield` => `4 passed`.
- Prochaine marche utile: `R3`, c'est-à-dire unifier le score final consommé par gate d'acceptation, callback et ranking.

## 2026-06-02 11:41 — S6 R3 amorcé sans retuning
- Un `onefield_final_logodds` explicite est maintenant publié au niveau finalize/post-tune.
- Le callback onefield et le ranking/dedup runtime le consomment désormais en priorité avant `solve_logodds` / `accept_logodds` / `prob_logodds`.
- Conclusion pratique: le chantier R3 a quitté le stade “conceptuel”; le prochain travail utile est de vérifier sur run réel que gate d'acceptation, callback et artefacts exportés lisent bien cette même valeur.

## 2026-06-02 11:50 — S6 R3 validé sur run réel, faux positifs inchangés sur sous-lot 2
- Run réel sentinelle `232102` publié dans `reports/r47i_s6_r3_score_probe_20260602_1150`:
  - `accept_logodds = onefield_final_logodds = accept_gate_logodds_used = record_match_callback_logodds`
  - `accept_gate_logodds_matches_onefield_final=true`
  - `onefield_final_logodds_consistent_with_accept=true`
- Conclusion durable: la divergence de sémantique gate/callback est bien fermée sur le sentinelle.
- Replay partiel `R4` sur `232102 + 232144` publié dans `reports/r47i_s6_r4_subset2_after_r3_20260602_1150/summary.json`:
  - `2/2 success`
  - `2/2 false_positive`
  - même cohérence parfaite du score unique sur les deux FITS.
- Lecture durable: après fermeture de `R1/R2/R3`, le résiduel S6 n'est plus une divergence de sémantique onefield; c'est désormais un problème plus directement math/logodds sur la mauvaise solution acceptée.

## 2026-06-02 13:30 — S6 audit math/logodds: faux positif accepté mais non soutenu
- Audit dédié publié: `reports/r47i_s6_false_positive_logodds_validity_audit_20260602_133050_sentinel232102/summary.json`
- Classification durable: `threshold_permissive_unsupported_accept`.
- Constat clé:
  - la chaîne de score est cohérente (`accept = final = gate = callback = -1.386294`);
  - mais la verify maths ne soutient pas la solution:
    - `prob_matches=0`
    - `prob_theta_match_total=0`
    - `prob_theta_distractor_total=18`
    - head `prob_verify_steps` = distractors only
    - support MatchObj dégénéré (`field/star/quadpix/quadxyz = 2`)
    - géométrie aberrante malgré `quality=GOOD` (`inliers=2`, `rms_px≈790.7`)
- Règle durable: ne plus formuler le résiduel S6 comme “mauvais seuil de callback”; le vrai prochain cran est d'expliquer où un support verify dégénéré cesse d'être rejeté et reste pourtant éligible à `accept_keep`.

## 2026-06-02 13:45 — S6 point causal source: `verify.py` force déjà le faux positif en `GOOD`
- Le premier point causal utile a été isolé plus bas que le wrapper runtime:
  - dans `zeblindsolver/verify.py`, `validate_solution(...)` force `success=True` quand `astrometry_parity_mode=True`;
  - en mode parity metrics-only, la fonction supprime aussi la raison (`reason=None`) même si `rms_ok=0` ou `inliers_ok=0`.
- Conséquence durable observée sur le sentinelle `232102`:
  - le faux positif arrive déjà avec `quality=GOOD` **avant** tout passthrough runtime additionnel, malgré:
    - `inliers=2`
    - `rms_px≈790.7`
    - `prob_matches=0`
    - `prob_theta_match_total=0`
    - `prob_theta_distractor_total=18`
    - `verify_matchobj_*_n=2`
- Lecture durable:
  - le wrapper `metrics_only_passthrough` n'est plus le suspect racine principal sur ce cas;
  - le prochain patch doit viser d'abord la requalification source dans `verify.py`, en séparant clairement:
    - progression canonique éventuelle,
    - et validation réellement `GOOD`.

## 2026-06-02 14:01 — S6 patch de séparation de contrat: plus de faux `GOOD`, mais le sentinelle retombe sans solution
- Correctif appliqué dans `zeblindsolver/verify.py`:
  - en mode `astrometry_parity_mode`, un échec métrique (`rms/inliers`) ne force plus `success=True`;
  - le validateur publie maintenant explicitement:
    - `validation_metrics_only=true`
    - `validation_progress_eligible=true`
  - tout en gardant `quality=FAIL`.
- Correctif compagnon appliqué dans `zeblindsolver/zeblindsolver.py`:
  - les wrappers natifs ne transforment plus ces cas `metrics_only` en `quality=GOOD`;
  - la progression canonique se fait désormais via une éligibilité explicite, séparée de la qualité de validation ;
  - si un candidat franchit quand même l'accept gate, il peut être marqué `accepted_despite_validation_fail=true` sans falsifier `quality`.
- Vérification ciblée réussie:
  - `python3 -m py_compile` OK sur les fichiers touchés ;
  - `pytest -q tests/test_synthetic.py::test_validate_solution_parity_mode_keeps_fail_quality_but_marks_progress tests/test_zeblindsolver.py -k onefield` => `4 passed`.
- Replay sentinelle outillé publié:
  - `reports/r47i_s6_false_positive_logodds_validity_audit_20260602_140132_sentinel232102_contractshift_rerun/summary.json`
- Résultat durable:
  - le sentinelle `232102` ne produit plus le faux positif précédemment accepté ;
  - il retombe maintenant à `success=false`, `message=no valid solution`, `oracle_status=fail_no_wcs`.
- Lecture durable:
  - la promotion silencieuse `FAIL -> GOOD` faisait bien partie du chemin causal du faux positif ;
  - la retirer change le comportement de recherche de façon réelle (run plus long, plus aucune solution acceptée sur ce case) ;
  - le prochain cran utile n'est pas un retuning de seuils, mais l'identification de l'étage précis où ce sentinelle cesse désormais de survivre.

## 2026-06-02 14:15 — S6 comparaison post-contrat: le faux positif fermé expose un résiduel transform/scale sur `d50_2725`
- Comparaison directe cadrée entre :
  - ancien faux positif accepté :
    - `reports/forensic_oracle_232102_20260518_2342/summary.json`
    - `success=true`, `tile_key=d50_2725`, `phase=hinted`, `prob_logodds=24.176`
    - oracle faux positif avec `center_sep_arcsec≈14576.9` et `scale_ratio≈3.88`
  - run forensique post-séparation de contrat :
    - `reports/r47i_s6_contractshift_forensic_20260602_141607/validation_pairs.json`
    - `reports/r47i_s6_contractshift_forensic_20260602_141607/wcs_coherence.json`
- Constat durable:
  - le candidat/famille `d50_2725` remonte encore après patch ;
  - il n'est plus promu artificiellement en `GOOD`, mais il apparaît maintenant comme un échec mathématique/cohérence clair.
- Signaux utiles relevés sur `d50_2725` après patch :
  - branche nominale :
    - `pairs=5`
    - `quality=FAIL`
    - `validate_rms_px≈191.0`
    - `model_scale_arcsec≈13.55`
    - `anchor_scale_arcsec≈2.39`
    - `wcs_residual_max_arcsec≈3638.8`
  - branche mirror :
    - `pairs=6`
    - `quality=FAIL`
    - `validate_rms_px≈2184.6`
    - `model_scale_arcsec≈23.98`
    - `anchor_scale_arcsec≈2.39`
    - `wcs_residual_max_arcsec≈61084.3`
- Lecture durable:
  - après fermeture du faux `GOOD`, le résiduel S6 le plus crédible n'est plus un simple problème de seuil d'acceptation ;
  - il reste vraisemblablement une **approximation mathématique préjudiciable** dans le support transform/scale/cohérence WCS du candidat `d50_2725` ;
  - la suite utile doit cibler ce différentiel mathématique avant tout nouveau retuning `anodds/tosolve`.

## 2026-06-02 14:30 — S6 audit `d50_2725`: deux sources d'erreur concrètes avant tout patch math plus profond
- Source d'erreur `E1` isolée :
  - ancien faux positif accepté `d50_2725` évalué avec une ancre d'échelle `scale_anchor_current_arcsec=9.56`, alors que :
    - le pairset du même case publie déjà `approx_scale_arcsec=2.39`
    - le run post-contrat recale l'ancre autour de `2.392674`
  - Lecture durable :
    - une ancre d'échelle non canonique a contribué à laisser `model_scale_arcsec=9.216` passer comme `scale_ok`.
- Source d'erreur `E2` isolée :
  - l'ancien faux positif accepté passe par le statut `empty_inliers_fallback_all_finite` avec `pairs=15`, `inliers=15`, `model_scale_arcsec=9.216`.
  - Lecture code durable :
    - en chemin strict Astrometry, `zeblindsolver.py` peut encore promouvoir `finite_mask_empty` quand `inlier_count<=0`, donc transformer une absence d'inliers géométriques en support verify dense.
  - Lecture durable :
    - ce fallback est un vrai générateur de faux support mathématique et a probablement alimenté le faux positif historique `d50_2725`.
- Règle de pilotage durable :
  - avant tout patch “math plus profond” sur transform/scale, fermer d'abord :
    - la provenance d'ancre d'échelle non canonique (`E1`)
    - puis la promotion `empty_inliers_fallback_all_finite` (`E2`)
  - seulement après cela, réévaluer le résiduel transform/scale restant sur `d50_2725`.

## 2026-06-03 01:20 — S6 `astrometry`/Ze: résiduel vivant resserré jusqu'au carry effectif des paires `resolve_hit`
- Le résiduel S6 n'est plus à cadrer comme un simple problème d'ancre, de scale global ou de source Astrometry ignorée.
- L'audit causal durable de référence reste :
  - `reports/r47i_s6_astrometry_seed_residual_incoherence_audit_20260602_230608_sentinel232102_d502725_seed_residual_forced/summary.json`
  - lecture utile :
    - seed `astrometry` bon sur `d50_2725` (`resolve_hit.kept=38`, `inliers_final=8`, `rms_px≈1.85`)
    - effondrement aval Ze (`pairs=8`, `inliers=0`, `residual_px_med≈26.66`, `residual_px_max≈542.94`)
- Nouvelle lecture code durable :
  - dans le fallback `meta_seed_inliers`, Ze réestimait bien le transform à partir des couples `resolve_hit_src_indices/dst_indices`,
  - mais ne réinjectait pas ces indices `dst` dans la chaîne aval ;
  - si `post_resolve` ne sauvait pas ensuite le candidat, le solveur pouvait rechuter vers un masque source-only puis un appariement index-locké faux.
- Correctif minimal appliqué :
  - conservation locale `meta_seed_src_idx/meta_seed_dst_idx`
  - réinjection explicite en `reassign_src_idx/reassign_dst_idx`
  - nouvelle provenance aval : `resolve_hit_meta_seed`
- Règle de pilotage durable :
  - le prochain audit doit maintenant vérifier si ce carry explicite suffit à empêcher la rechute index-lockée ;
  - si l'échec persiste, le front suivant devient strictement `post_resolve / reassign_eval`, sans rouvrir d'audit Astrometry large.

## 2026-06-04 13:10 — S6 `resolve_hit` pair-pool carry: le rejet rapide change de forme, mais le chemin doit être borné
- Le correctif `meta_seed` a été renforcé : quand le seed `astrometry` fournit des couples `resolve_hit_src_indices/dst_indices`, le contrôle aval ne garde plus seulement les indices en `reassign_*`; il restreint aussi le pool local à ces couples appariés (`img_points[src] -> tile_points[dst]`) et remappe les indices en coordonnées locales `0..N-1`.
- Vérifications courtes accomplies :
  - `python -m py_compile zeblindsolver/zeblindsolver.py zeblindsolver/verify.py` OK ;
  - `python -m pytest tests/test_zeblindsolver.py -q` => `19 passed`.
- Rejeu causal borné :
  - `timeout 150s python tools/r47i_s6_astrometry_seed_residual_incoherence_audit.py --tile d50_2725 --label sentinel232102_d502725_pairpoolcarry`
  - artefact partiel : `reports/r47i_s6_astrometry_seed_residual_incoherence_audit_20260604_130200_sentinel232102_d502725_pairpoolcarry/`
  - résultat : timeout `150s` sans `summary.json/stats_full.json`.
- Lecture durable :
  - le carry du pool `resolve_hit` modifie bien le comportement par rapport au rejet rapide historique, mais ne ferme pas encore S6 ;
  - le prochain travail doit borner/observer le bloc immédiatement après `resolve_hit_meta_seed` pour savoir si `empty_inliers` est supprimé puis remplacé par une recherche trop longue, ou si une rechute ultérieure se produit dans `post_resolve / verify`.

## 2026-06-04 13:35 — S6 `resolve_hit` global-point carry validé jusqu'à l'entrée validation
- Le premier patch `pair-pool carry` était encore incomplet : les indices `resolve_hit_src/dst` appartenaient au référentiel global `image_positions/tile_positions`, alors que le screen aval recevait déjà un petit pool de `8` paires. Le clipping des indices contre ce petit pool empêchait de porter correctement les correspondances.
- Correctif appliqué :
  - au moment de créer `transform_origin_meta`, Ze stocke maintenant aussi `resolve_hit_src_points` et `resolve_hit_dst_points` depuis le pool global réellement utilisé par `resolve_hit`;
  - le fallback aval consomme ces coordonnées appariées directement, puis remappe les indices locaux en `0..N-1`.
- Probe borné ajouté :
  - `blind_s6_meta_seed_probe_dump_path`;
  - option d'audit `--probe-break-on-seed` pour atteindre rapidement le screen aval du seed causal sans changer le défaut produit.
- Artefact utile :
  - `reports/r47i_s6_astrometry_seed_residual_incoherence_audit_20260604_133138_sentinel232102_d502725_reassignsourceprobe_breakseed/meta_seed_probe.json`
- Résultat durable sur `hinted/S/nominal/first_pass` :
  - avant carry : `pairs=8`, `inlier_count=0`, `resolve_hit_src_count=38`, `residual_px_med≈187`;
  - après carry : `carried_pairs=38`, `inlier_count=38`, `model_scale_arcsec≈12.11577`, `residual_px_med≈4.93`;
  - entrée pré-validation : `img_in_pairs=38`, `tile_in_pairs=38`, `world_in_pairs=38`, `reassign_source=resolve_hit`, `reassign_path_tag=resolve_hit_direct`.
- Conclusion durable :
  - la rupture de référentiel `resolve_hit` -> petit pool indexé est corrigée jusqu'à l'entrée validation ;
  - le résiduel vivant descend maintenant vers la validation/native-verify de ces 38 paires, pas vers la génération du seed ni le transport des correspondances.

## 2026-06-05 01:05 — S6 `native_verify/validate`: les 38 paires sont cohérentes mais oracle-fausses
- Le carry des coordonnées monde a été complété : `resolve_hit_dst_world_points` transporte maintenant les RA/Dec absolues, tandis que `tile_points` reste en plan tangent. Cela supprime le faux RMS géant observé quand `validate_solution` recevait des coordonnées tangent-plane comme si elles étaient des coordonnées monde.
- Probe défaut sur `d50_2725` :
  - `reports/r47i_s6_astrometry_seed_residual_incoherence_audit_20260605_005506_sentinel232102_d502725_worlddump_default/validation_pairs.json`
  - `pairs=38`, `inliers=38`, `rms_px≈6.166`, `model_scale_arcsec≈12.123`
  - rejet par `validate_base` : `rms_ok=0`, `inliers_ok=1`, `scale_ok=1`, seuil `rms_thr=2.000`
  - résidu WCS sur les 38 RA/Dec reçues : médiane ≈`55.9"` / p90 ≈`87.7"` / max ≈`119.8"`.
- Probe audit-only avec `quality_rms=8` :
  - `reports/r47i_s6_astrometry_seed_residual_incoherence_audit_20260605_005013_sentinel232102_d502725_validatepass_rms8/`
  - `validate_base` passe, puis l'acceptation arrive avec `prob_matches=0`, `prob_distractors=1`, `prob_theta_match_total=0`, `prob_theta_distractor_total=18`, log-odds final ≈`-1.386`.
  - comparaison oracle ASTAP : `false_positive`, centre décalé ≈`7680"`, `corner_max≈18551"`, scale ratio ≈`5.107`.
- Lecture durable :
  - les 38 paires portées par `resolve_hit` sont auto-cohérentes dans le mauvais repère/hypothèse, mais ne valident pas le vrai WCS du FITS ;
  - le seuil RMS produit à `2px` protège ici contre une fausse solution ; le relâcher à `8px` serait dangereux sans autre garde-fou ;
  - le `probabilistic verify` confirme le manque de support : le premier préfixe est déjà un distracteur (`theta=-1`, `prob_matches=0`), mais reste accepté en audit-only parce que `tosolve=1e-3` abaisse `tokeep` à ≈`-6.907` alors que le meilleur distracteur vaut ≈`-1.386`;
  - le front suivant n'est pas de retuner `quality_rms`, mais d'aligner la logique Astrometry pour rejeter ce faux `resolve_hit` plus tôt ou construire un vrai support MatchObj quand le candidat est authentique.

## 2026-06-05 01:10 — Garde-fou ZeNear pendant la suite S6 ZeBlind
- Risque clarifié avec Tristan :
  - les travaux S6 en cours sont ZeBlind-only en accès direct (`blind_astrometry_*`, `resolve_hit`, `native_verify`, `solve_blind`) et ne doivent pas modifier le solve Near pur ;
  - la zone à risque est indirecte : un Near qui échoue puis bascule en fallback Blind peut changer de comportement si le contrat `validate/native_verify` de ZeBlind est durci ou réparé.
- Règle durable avant promotion d'un correctif S6 :
  - conserver le fallback blind immédiat de ZeNear ;
  - ne pas toucher aux modules/configs Near pour résoudre ce front S6 ;
  - relancer explicitement un protocole `testzenear`/Near+fallback avec comptage séparé `Near success`, `Blind fallback success`, `fail` ;
  - si régression ZeNear observée après un ZeBlind fonctionnel, chercher d'abord dans le comportement fallback Blind et non dans le solve Near pur.

## 2026-06-05 01:20 — S6 audit summary remis au niveau du diagnostic courant
- L'outil `tools/r47i_s6_astrometry_seed_residual_incoherence_audit.py` ne doit plus présenter l'ancienne cause `global img_points/tile_points pool` comme diagnostic actif quand les dumps montrent le nouvel état.
- Mise à jour appliquée :
  - lecture de `validation_pairs.json` dans la synthèse ;
  - classification défaut : `validate_rejects_unsupported_resolve_hit` quand les 38 paires atteignent validation, mais échouent `rms_ok=0` et ont zéro support probabiliste ;
  - classification probe relâché : `unsupported_accept_under_relaxed_probe_threshold` quand `quality_rms=8` fait accepter un préfixe distracteur sans `prob_matches`.
- Vérification de classification sur artefacts existants :
  - défaut `worldcarry_validateprobe_breakseed` => `validate_rejects_unsupported_resolve_hit`;
  - `validatepass_rms8` => `unsupported_accept_under_relaxed_probe_threshold`.

## 2026-06-05 01:25 — S6 garde-fou callback: plus de succès sans support verify positif
- Contrat Astrometry clarifié :
  - `verify_hit()` peut calculer/remplir le support MatchObj quand `K >= logaccept`;
  - le statut "solution" reste contrôlé plus tard par `toprint/tokeep/tosolve` dans `solver_handle_hit` / `onefield`.
- Correctif/garde-fou Ze appliqué :
  - nouveau flag config `blind_astrometry_require_positive_verify_match_for_solve=True`;
  - en mode Astrometry strict, `_record_match_callback_runtime()` rejette un candidat si `prob_matches=0` et `prob_theta_match_total=0`, même si un probe a abaissé `tosolve`;
  - terminal explicite : `reject_callback_no_positive_verify_match`;
  - dump d'audit ajouté : `blind_record_match_callback_dump_path`.
- Probe RMS8 de contrôle :
  - `reports/r47i_s6_astrometry_seed_residual_incoherence_audit_20260605_012235_sentinel232102_d502725_rms8_callbackdump/record_match_callback.json`
  - le faux `d50_2725` reste `validate=GOOD` sous RMS8 (`38` paires, `rms≈6.166`) mais le callback le rejette :
    - `solve_pass=False`
    - `record_match_callback_reason=no_positive_verify_match`
    - `logodds≈-1.386`, `tosolve≈-6.907`
    - `prob_matches=0`, `prob_theta_match_total=0`
- Lecture durable :
  - les probes relâchés peuvent encore servir à atteindre une zone de code, mais ne peuvent plus transformer un préfixe distracteur en faux succès ;
  - prochain front S6 : après neutralisation du faux `resolve_hit`, vérifier quel candidat vient ensuite, ou remonter l'écart vers génération/ordre/support des hypothèses.

## 2026-06-05 01:35 — S6 après neutralisation: réémission du même faux `resolve_hit`
- Probe de suite :
  - `reports/r47i_s6_astrometry_seed_residual_incoherence_audit_20260605_012539_sentinel232102_d502725_after_false_resolvehit/`
- Observation :
  - après le premier rejet `reject_callback_no_positive_verify_match`, le solveur réémet le même faux nominal (`hinted/S/nominal`, `resolve_hit`, `38` paires, `rms≈6.166`, `model_scale≈12.123`, `logodds≈-1.386`, `prob_matches=0`);
  - la branche miroir atteint aussi validation mais échoue massivement (`11` paires, `rms≈2252.8`, `scale_ok=0`);
  - aucune signature de candidat authentique distinct n'émerge dans cette fenêtre.
- Garde-fou ajouté :
  - registre runtime des signatures déjà rejetées sans support positif ;
  - les réémissions identiques sont maintenant rejetées avant nouveau callback comme `reject_duplicate_no_positive_verify_match`;
  - probe `...013022...duplicate_guard` confirme une première entrée `no_positive_verify_match`, puis une entrée `duplicate_no_positive_verify_match`.
- Lecture durable :
  - le front S6 descend maintenant vers l'orchestration/réentrée du même faux `resolve_hit` entre passages/phases ;
  - prochaine étape utile : empêcher ou classifier plus tôt cette re-entry de signature unsupported, puis vérifier si un vrai candidat apparaît derrière.

## 2026-06-05 08:45 — S6 recheck: le doublon faux atteint encore `validate` avant le garde-fou aval
- Petit refactor de sûreté/testabilité appliqué :
  - extraction module-level de `_unsupported_verify_reject_signature()` et `_should_reject_duplicate_no_positive_verify_match()`;
  - couverture unitaire ajoutée pour la stabilité de signature et la logique `seen_signature -> reject`.
- Vérifications courtes :
  - `python3 -m py_compile zeblindsolver/zeblindsolver.py tests/test_zeblindsolver.py` OK ;
  - `pytest -q tests/test_zeblindsolver.py` passe à `23 passed`.
- Rerun sentinelle de contrôle :
  - `reports/r47i_s6_astrometry_seed_residual_incoherence_audit_20260605_083646_sentinel232102_d502725_duplicate_guard_recheck/`
- Lecture causale utile tirée des artefacts partiels déjà écrits :
  - `validation_pairs.json` contient encore **deux** entrées identiques pour le faux nominal `hinted/S/nominal/d50_2725` (`38` paires, `rms≈6.166`, `reason=validation_metrics_only[...]`) ;
  - donc le garde-fou `reject_duplicate_no_positive_verify_match` coupe bien l'acceptation tardive, mais ne supprime pas encore la **réémission amont vers validate**.
- Conclusion durable :
  - le prochain delta causal ne doit plus viser le callback final ;
  - il doit classifier/couper plus tôt la réentrée unsupported, idéalement avant une deuxième validation identique, puis vérifier si un candidat authentique émerge derrière.

## 2026-06-05 09:10 — S6 pre-validate guard: le doublon `d50_2725` ne repaie plus une 2e validation identique
- Correctif appliqué :
  - ajout d'une signature `prevalidate` dérivée du candidat (`tile/level/parity/source`, échelle, erreur médiane, head des points image/tile) ;
  - stockage de cette signature quand un candidat est rejeté au callback pour `no_positive_verify_match` ;
  - nouveau court-circuit amont avant `_validate_solution_traced("validate_base", ...)` si la même signature unsupported réapparaît.
- Vérifications :
  - `python3 -m py_compile zeblindsolver/zeblindsolver.py tests/test_zeblindsolver.py` OK ;
  - `pytest -q tests/test_zeblindsolver.py` passe à `25 passed`.
- Probe sentinelle :
  - `reports/r47i_s6_astrometry_seed_residual_incoherence_audit_20260605_090434_sentinel232102_d502725_prevalidate_guard_v2/`
- Lecture causale utile :
  - `validation_pairs.json` ne contient plus qu'une seule entrée pour le faux `hinted/S/nominal/d50_2725` (`38` paires, `rms≈6.166`) ;
  - avant le patch, le rerun de contrôle en montrait deux ;
  - le cran “couper la 2e validation identique” est donc refermé.
- Conclusion durable :
  - le prochain front S6 n'est plus la duplication au niveau `validate` ;
  - il faut maintenant auditer ce qui arrive après cette première occurrence rejetée : soit un nouveau candidat utile émerge, soit la boucle amont reste enfermée sur `d50_2725` sans repasser par `validate`.

## 2026-06-05 10:38 — S6 dédup validation confirmée, mais la boucle amont reste enfermée sur `d50_2725`
- Renforcement appliqué :
  - nouvelle règle de dédup fondée sur “signature `prevalidate` déjà vue une fois en validation” ;
  - ce garde-fou agit avant une 2e `validate`, indépendamment du callback final ;
  - test unitaire ajouté pour `_should_reject_duplicate_prevalidate_validation()`.
- Vérifications :
  - `python3 -m py_compile zeblindsolver/zeblindsolver.py tests/test_zeblindsolver.py` OK ;
  - `pytest -q tests/test_zeblindsolver.py` passe à `26 passed`.
- Probe sentinelle :
  - `reports/r47i_s6_astrometry_seed_residual_incoherence_audit_20260605_103412_sentinel232102_d502725_prevalidate_validation_dedup/`
- Lecture utile :
  - `meta_seed_probe.json` ne montre plus que `first_pass/nominal` pour `d50_2725` ;
  - `validation_pairs.json` n'a qu'une seule validation `d50_2725` ;
  - mais `astrometry_exact_trace.raw.json` reste très majoritairement bloqué sur `d50_2725` (fenêtre lue dominée par `hinted/S/nominal`, queue en `reject_not_better_than_best`).
- Conclusion durable :
  - la duplication de validation est refermée ;
  - la nouvelle première divergence causale vivante est une **boucle amont de réélection sur la même tuile** (`accept_new_best / reject_not_better_than_best`) plutôt qu'un problème de `validate` ou de callback.

## 2026-06-05 10:50 — Réévaluation de trajectoire S6: la comparaison amont reste dominée par un `config_envelope_gap`
- Audit relancé :
  - `reports/r47i_s6_m106_first_divergence_audit_20260605_104603_loop_reaudit_20260605/`
- Résultat utile :
  - `probe` et `product` échouent tous deux, mais la première divergence reste classée `config_envelope_gap` ;
  - sur le même tuple initial `hinted/S/nominal/d50_2725`, les têtes diffèrent déjà fortement en enveloppe runtime :
    - `newpoint_source`: `phase_local_rank` vs `single_pass_trace_cursor`
    - `code_lookup_hits`: `33646` vs `10930`
    - `pairs_collected_initial`: `19` vs `3`
  - le diff de config frontière est encore trop large pour lire proprement la boucle `d50_2725` comme un gap solver pur (`downsample`, `max_stars`, `max_quads`, `pixel_tolerance`, `prob_verify`, `native_verify semantics`, seuils anodds, gardes d'échelle, `single_pass_newpoint`, etc.).
- Conclusion durable :
  - avant de continuer l'autopsie locale sur `accept_new_best / reject_not_better_than_best`, il faut produire un audit **matched-envelope** ;
  - règle de méthode S6 mise à jour : si la première divergence remonte `config_envelope_gap`, on n'interprète pas la boucle amont comme cause racine tant qu'on n'a pas aligné l'enveloppe runtime minimale entre `product` et `probe`.

## 2026-06-05 12:15 — S6 requalifié plus finement en `preperm_wiring_gap` sur `d50_2725`
- Outillage d'audit enrichi :
  - `tools/r47i_s6_m106_first_divergence_audit.py` publie maintenant `resolve_hit_preresolve` par variante ;
  - la tête upstream exporte aussi `perm_null_reason`.
- Probes :
  - `reports/r47i_s6_m106_first_divergence_audit_20260605_120547_product_preresolve_probe_20260605/`
  - `reports/r47i_s6_m106_first_divergence_audit_20260605_120814_product_preperm_reason_probe_20260605/`
  - synthèse : `.../hinted_d2725_preperm_gap.json`
- Lecture utile :
  - `hinted / nominal / d50_2725` garde bien un `collect scalar_final` avec `19` paires et `35` votes ;
  - mais il n'atteint ensuite ni `pre_resolve_hit`, ni `exact_trace` ;
  - la ligne upstream correspondante reste bloquée en :
    - `perm_callsite_stage=pre_perm`
    - `perm_null_reason=no_quad_survived_preperm_gates`
    - `seed_obs_slices_initial=0`
    - `seed_final_obs_slices=0`
- Conclusion durable :
  - le résiduel vivant S6 n'est plus simplement “entre collect et resolve_hit” ;
  - il est maintenant localisé comme un **`preperm_wiring_gap`** :
    - le candidat survit au collect,
    - puis ne construit aucun seed slice / aucune permutation exploitable côté `product`,
    - avant toute entrée dans `resolve_hit`.

## 2026-06-05 12:23 — S6 encore resserré: `d50_2725` meurt avant astrometry sur `pairset_scale_gate`
- Instrumentation supplémentaire ajoutée pour la tête upstream :
  - `astrometry_lookup_ready`
  - `level_slices_count`
  - compteurs `preperm_*`
  - `seed_strict_recover_*`
- Le protocole S6 exporte maintenant aussi :
  - `pairset_scale_precheck`
  - `transform_scale_precheck`
  - `scale_hard_reject`
- Probes :
  - `reports/r47i_s6_m106_first_divergence_audit_20260605_121838_product_preperm_counters_probe_20260605/`
  - `reports/r47i_s6_m106_first_divergence_audit_20260605_122127_product_preastrometry_gate_probe_20260605/`
  - synthèse : `.../hinted_d2725_preastrometry_gate_gap.json`
- Lecture utile :
  - le tuple `hinted / nominal / d50_2725` garde bien `collect scalar_final` (`19` paires, `35` votes) ;
  - mais la tête upstream montre ensuite :
    - `astrometry_lookup_ready=true`
    - `level_slices_count=7762`
    - `preperm_quad_iterations=0`
  - donc il ne faut plus chercher la première mort dans `pre_perm` / `obs_slices` ;
  - la coupure vivante est plus amont dans `product_pairset_scale_precheck.json` :
    - deux rejets `reject_pairset_scale_gate` sur `hinted / nominal / d50_2725`
    - `pairs=10` au moment du garde
    - `span_implied_scale_arcsec≈14.287` > `gate_hi≈11.963`
- Conclusion durable :
  - le résiduel vivant S6 est requalifié de `preperm_wiring_gap` vers **`preastrometry_scale_gate_gap`** ;
  - le corridor `d50_2725` est coupé par le `pairset_scale_gate` avant toute itération astrometry.

## 2026-06-05 16:00 — S6 remonte encore d'un cran: le vrai delta vivant est le `pair_scale_prefilter` produit
- Audit pair-flow borné :
  - `reports/r47i_s6_m106_first_divergence_audit_20260605_155735_product_pairflow_probe_v2_20260605/`
  - `reports/r47i_s6_m106_first_divergence_audit_20260605_155735_matched_probe_pairflow_probe_v2_20260605/`
  - synthèse : `.../pairflow_prefilter_vs_native_gap.json`
- Lecture utile :
  - sur `hinted / nominal / d50_2725`, `product` passe de `19` paires collectées à `10` après `pair_scale_prefilter` (`9` paires retirées), puis échoue au `pairset_scale_gate` avec ancre `header_scale_arcsec` ;
  - le bloc `blind_astrometry_add_stars_inbox` ne peut même plus s'exécuter ensuite, car le corridor tombe sous `min_pairs=12` ;
  - sur le même tuple, `matched_probe/native` garde `18` paires, utilise l'ancre locale `candidate_pairset_local` et ne subit pas de `pairset_scale_gate` (`gate_bounds_arcsec=null`).
- Conclusion durable :
  - le résiduel vivant S6 n'est plus à lire d'abord comme un simple `pairset_scale_gate` ;
  - le premier delta causal exploitable est désormais un **`product_pair_scale_prefilter_gap`** en amont immédiat du gate.

## 2026-06-05 16:30 — Le front S6 `d50_2725` bascule déjà avec le seul flag natif
- Micro-probe dédié :
  - `reports/r47i_s6_m106_first_divergence_audit_20260605_162433_native_flag_only_micro_20260605/`
  - synthèse : `.../native_flag_only_bridge_verdict.json`
- Lecture utile :
  - variante testée = `product` + seul `blind_astrometry_native_verify_semantics_enabled=True` ;
  - sur le même tuple `hinted / nominal / d50_2725`, ce seul flag suffit déjà à reproduire le comportement pairset natif :
    - `pairs=18`
    - `scale_anchor_source=candidate_pairset_local`
    - `decision=keep_pairset`
    - `gate_bounds_arcsec=null`
- Conclusion durable :
  - les autres relaxations de `matched_probe` ne sont pas nécessaires pour fermer ce front pré-astrometry ;
  - le résiduel vivant est donc mieux classé comme **gap de politique/mode porté par `blind_astrometry_native_verify_semantics_enabled`**, avant d'être un bug math indépendant.

## 2026-06-05 16:55 — Split des gardes S6: `pair_scale_prefilter` et `pairset_scale_gate` suffisent chacun séparément à réouvrir l'aval
- Micro-probes dédiés :
  - `reports/r47i_s6_m106_first_divergence_audit_20260605_164842_guard_split_micro_20260605/` (`no_pair_prefilter`)
  - `reports/r47i_s6_m106_first_divergence_audit_20260605_165024_no_pairset_gate_micro_20260605/`
  - `reports/r47i_s6_m106_first_divergence_audit_20260605_165024_no_pair_prefilter_no_gate_micro_20260605/`
  - synthèse : `.../guard_split_verdict.json`
- Lecture utile :
  - `pair_scale_prefilter OFF` seul :
    - le `pairset_scale_precheck` ne rejette plus `d50_2725` ; il passe en `recenter_keep` (`18 -> 10`) ;
    - le tuple atteint ensuite `pre_resolve_hit`, `exact_trace`, puis la même validation fausse aval (`18` paires, `16` inliers, `rms≈4.254`).
  - `pairset_scale_gate OFF` seul :
    - le tuple passe en `keep_pairset` avec le pool produit de `10` paires ;
    - il atteint lui aussi `pre_resolve_hit`, `exact_trace`, puis la même validation fausse aval.
  - `prefilter+gate OFF` :
    - restaure `18` paires avant astrometry, comme `native_flag_only`, mais sans ancre locale requise pour franchir ce front précis.
- Conclusion durable :
  - le blocage initial était bien un **bundle de gardes pré-astrometry**, pas un manque de permutations ou un bug math amont ;
  - dès qu'on retire l'un des deux gardes, le front vivant redescend sur le faux candidat aval `d50_2725` déjà connu.

## 2026-06-05 19:05 — Priorité produit minimale S6: `pairset_scale_gate` passe devant `pair_scale_prefilter`
- Tri produit figé via :
  - `reports/r47i_s6_m106_first_divergence_audit_20260605_165024_no_pairset_gate_micro_20260605/minimal_lever_priority_verdict.json`
- Lecture utile :
  - `pairset_scale_gate OFF` seul déverrouille `d50_2725` sans recourir au `recenter_keep` ni modifier la construction du pair-pool ;
  - `pair_scale_prefilter OFF` seul déverrouille aussi, mais au prix d'un passage `recenter_keep` (`18 -> 10`) plus bruité ;
  - le bundle complet (`prefilter+gate OFF` ou mode natif) n'est utile que pour restaurer le pool natif de `18` paires avant astrometry, pas pour réexposer le front aval.
- Conclusion durable :
  - pour les prochains probes contrôlés orientés correction produit minimale, le **levier prioritaire** à surveiller est `pairset_scale_gate` ;
  - le front vivant principal redevient le faux `resolve_hit`/validation `d50_2725` (`18` paires, `16` inliers, `rms≈4.254`).

## 2026-06-05 19:15 — Le vrai front vivant suivant est un carry `resolve_hit_meta_seed` malgré prescreen nul
- Synthèse dédiée :
  - `reports/r47i_s6_m106_first_divergence_audit_20260605_165024_no_pairset_gate_micro_20260605/resolve_hit_direct_false_carry_gap.json`
- Lecture utile :
  - une fois le bundle pré-astrometry levé, les variantes `no_pairset_gate` et `no_pair_prefilter_no_gate` convergent vers le même aval ;
  - dans les deux cas, `astrometry_origin_prescreen` est déjà mauvais :
    - `inlier_count=0`
    - résiduels très élevés
    - seulement `8` ou `10` paires au prescreen ;
  - malgré cela, le bloc `resolve_hit_meta_seed_carried` promeut `18` correspondances originales, puis `resolve_hit_direct` mène à la même fausse validation (`18` paires, `16` inliers, `rms≈4.254`).
- Conclusion durable :
  - le front vivant principal n'est plus le bundle de gardes d'échelle lui-même ;
  - il est maintenant mieux classé comme **`resolve_hit_meta_seed_carry_gap`** : carry trop permissif d'un direct-path malgré un prescreen astrometry déjà non plausible.

## 2026-06-05 19:30 — Un garde de plausibilité coupe bien le faux carry `d50_2725` et réexpose `d50_2822`
- Implémentation d'audit étroite ajoutée dans `zeblindsolver.py` :
  - `blind_astrometry_meta_seed_carry_plausibility_guard_enabled=False` par défaut
  - seuils configurables :
    - `blind_astrometry_meta_seed_carry_require_origin_inliers`
    - `blind_astrometry_meta_seed_carry_max_origin_prescreen_residual_px`
    - `blind_astrometry_meta_seed_carry_max_origin_prescreen_residual_max_px`
- Variante probe-only ajoutée dans `tools/r47i_s6_m106_first_divergence_audit.py` :
  - `no_pairset_gate_meta_seed_guard`
- Artefact de synthèse :
  - `reports/r47i_s6_m106_first_divergence_audit_20260605_192637_meta_seed_guard_only_20260605/meta_seed_guard_verdict.json`
- Résultat durable :
  - sur `hinted / nominal / d50_2725`, le même `astrometry_origin_prescreen` reste à `pairs=10`, `inlier_count=0`, `residual_px_med≈1274.69`, `residual_px_max≈1971.82` ;
  - avec le garde activé, ce tuple est maintenant publié comme `resolve_hit_meta_seed_rejected` au lieu d'être promu en `resolve_hit_meta_seed_carried` ;
  - la fausse branche aval `resolve_hit_direct -> validation d50_2725` ne réapparaît plus dans ce probe ;
  - le front suivant réexposé devient une validation `hinted / nominal / d50_2822` qui échoue très bruyamment (`pairs=10`, `inliers=7`, `rms≈250.75`).
- Conclusion durable :
  - c'est la preuve la plus nette à ce stade que le faux aval `d50_2725` dépendait bien du carry `meta_seed` permissif ;
  - le prochain front utile à lire n'est plus `d50_2725`, mais `d50_2822` après coupure du carry.

## 2026-06-05 21:25 — Le dedup-all d'audit coupe le leak `d50_2822` et réexpose `d50_2823`
- Nouveau flag d'audit ajouté dans `zeblindsolver.py` :
  - `blind_prevalidate_duplicate_reject_all_sources_enabled=False` par défaut
- Nouvelle variante d'audit ajoutée dans `tools/r47i_s6_m106_first_divergence_audit.py` :
  - `no_pairset_gate_meta_seed_guard_dedup_all`
- Artefact de synthèse :
  - `reports/r47i_s6_m106_first_divergence_audit_20260605_212412_meta_seed_guard_dedup_all_only_20260605/meta_seed_guard_dedup_all_verdict.json`
- Résultat durable :
  - sous `meta_seed_guard`, `d50_2822` était déjà réexposé comme loud-fail `inlier_mask`, mais rejoué de façon identique et inutile ;
  - le dedup pré-validation standard ne l'arrêtait pas, car il ne s'appliquait qu'aux sources `resolve_hit` ;
  - avec `dedup_all`, `d50_2725` et `d50_2822` restent bien rejetés côté `meta_seed`, `d50_2822` ne consomme plus qu'une seule validation, et le budget libéré réexpose `d50_2823`.
- Lecture utile sur le nouveau front :
  - `phase_handoff` montre maintenant `candidate_try` jusqu'à `d50_2823` en `hinted / S / nominal / first_pass` ;
  - `resolve_hit_preresolve` sur `d50_2823` publie un gros volume d'événements `pre_resolve_hit/post_resolve_hit` source `astrometry_quad` ;
  - `astrometry_exact_trace.raw.json` sur `d50_2823` devient le nouveau gros foyer actif, sans encore atteindre `validate` au budget `1/2`.
- Conclusion durable :
  - le prochain front vivant utile n'est plus `d50_2822` mais **`d50_2823` dans le corridor `astrometry_quad -> pre_resolve_hit`** ;
  - `dedup_all` reste pour l'instant un levier d'audit, pas une décision produit.

## 2026-06-05 21:40 — Le gap de portage plausible se resserre sur les invariants de `verify_hit`
- Comparatif source local utile :
  - `astrometry-net-main/solver/solver.c`
  - `astrometry-net-main/solver/verify.c`
  - `astrometry-net-main/include/astrometry/verify.h`
- Lecture durable :
  - Astrometry fait de `verify_hit()` l'autorité centrale de décision :
    - calcul des `logodds`
    - `bail`
    - `stoplooking`
    - acceptation seulement si `K >= logaccept`
    - rejet final encore revalidé dans `solver.c` si `mo->logodds < logratio_tokeep`
  - ZeSolver, au contraire, laisse encore plusieurs promotions latérales vivre autour de l'astrometry loop :
    - `resolve_hit_meta_seed`
    - `resolve_hit_direct`
    - `reassign_eval`
    - dédup pré-validation dépendante de la source
- Conclusion durable :
  - le problème le plus crédible du portage n'est pas un concept faux, mais un **affaiblissement des invariants de verify/accept** par rapport au corridor canonique Astrometry ;
  - la prochaine étape à fort ROI est d'inventorier précisément quels invariants Astrometry impose avant promotion et lesquels ZeSolver relâche ou court-circuite.

## 2026-06-05 21:50 — Premier inventaire concret des invariants relâchés chez ZeSolver
- Invariants source plausiblement affaiblis pendant le portage :
  - autorité unique de verify insuffisamment centralisée ;
  - matérialisation trop précoce de "quasi-correspondances" avant acceptation terminale ;
  - monotonie plus faible entre seuil d'entrée verify et seuil final de keep ;
  - recyclage excessif des branches faibles via `relaxed/bootstrap/rescue` ;
  - discipline de rejet encore partiellement dépendante de `reassign_source` ;
  - exigence de "positive verify match" pas encore appliquée comme contrat global.
- Conclusion durable :
  - la cible prioritaire n'est pas de retoucher encore les maths locales ;
  - la cible prioritaire devient la **restauration d'un contrat de verify/accept plus proche d'Astrometry**.

## 2026-06-05 22:10 — Le clamp strict de pré-promotion ne devient pas le front causal principal sur la fenêtre bornée
- Probe-only ajouté :
  - `blind_astrometry_strict_prepromotion_source_clamp_enabled=False` par défaut
  - variante `no_pairset_gate_meta_seed_guard_dedup_all_strict_prepromo`
- Artefacts utiles :
  - `reports/r47i_s6_m106_first_divergence_audit_20260605_220441_strict_prepromo_probe_20260605/`
- Résultat durable :
  - sous `trace_max=2`, le clamp strict ne remplace pas encore la tête causale ;
  - `d50_2725`, `d50_2822` et `d50_2823` restent tous rejetés côté `resolve_hit_meta_seed` ;
  - `d50_2822` reste le seul loud-fail en validation (`inlier_mask`, `10` paires, `7` inliers, `rms≈250.75`) ;
  - `d50_2823` demeure le nouveau foyer actif surtout en `astrometry_quad` / `resolve_hit_preresolve`, sans validation atteinte dans cette fenêtre.
- Conclusion durable :
  - l'hypothèse "promotions latérales non canoniques" reste pertinente structurellement, mais elle n'explique pas à elle seule le front vivant courant sous budget borné ;
  - le prochain audit utile doit descendre plus bas dans le couloir `astrometry_quad` de `d50_2823`.

## 2026-06-05 22:20 — Le `perm_hash_gate` est un garde de pruning budgétaire, pas juste un faux blocage local
- Nouvelle variante probe-only :
  - `no_pairset_gate_meta_seed_guard_dedup_all_no_permhash`
- Artefacts utiles :
  - `reports/r47i_s6_m106_first_divergence_audit_20260605_221501_no_permhash_probe_20260605/`
- Résultat durable :
  - sur le stack strict précédent, `d50_2823` était dominé dans l'`exact_trace` par `reject_perm_hash_gate` ;
  - quand on coupe ce garde, `d50_2823` disparaît de la fenêtre écrite, et le budget se reconcentre sur `d50_2725` et `d50_2822` ;
  - `phase_handoff` ne montre plus que `d50_2725` puis `d50_2822` ;
  - `resolve_hit_preresolve` explose surtout sur :
    - `d50_2822` (`pre_resolve_hit/post_resolve_hit`)
    - `d50_2725` (`pre_resolve_hit/post_resolve_hit`)
- Conclusion durable :
  - `perm_hash_gate` est ambivalent :
    - il peut sur-pruner localement ;
    - mais il empêche aussi les vieux faux fronts d'absorber tout le budget ;
  - un retrait produit brut de ce garde serait dangereux sans mécanisme de remplacement plus canonique.

## 2026-06-05 22:30 — `d50_2725` et `d50_2822` restent devant surtout par score/ranking amont et support brut
- Lecture durable du probe `no_permhash` :
  - `candidate_order_head` expose des scores bruts très défavorables à `d50_2823` :
    - `d50_2725 = 33646`
    - `d50_2822 = 17252`
    - `d50_2823 = 11602`
  - sans `perm_hash_gate`, `d50_2823` ne parvient même plus jusqu'au `pairset_scale_precheck` dans la fenêtre bornée ;
  - `d50_2725` et `d50_2822` gardent au contraire un support amont plus gros :
    - `d50_2725` atteint `pairset_scale_precheck` avec `10` paires
    - `d50_2822` avec `15` paires
  - `resolve_hit_preresolve` se concentre massivement sur eux, surtout `d50_2822`.
- Conclusion durable :
  - leur compétitivité observée n'est pas principalement une supériorité de validate aval ;
  - elle est surtout portée par un **avantage de ranking initial + volume de pair-pool/support** ;
  - la suite doit donc auditer aussi le mécanisme de ranking, pas seulement les gardes de pruning.

## 2026-06-05 22:45 — Le ranking Ze ajoute bien deux couches non canoniques au-dessus du score brut
- Observabilité renforcée dans `zeblindsolver.py` :
  - `candidate_order_effective_head`
  - `candidate_order_reranked_head`
- Résultat durable sur les probes `no_permhash` :
  - le `candidate_order_head` brut ne décrivait pas encore l'ordre réellement essayé ;
  - avec `soft_prior` ON, le premier rerank remet `d50_2725` puis `d50_2822` devant `d50_2823` ;
  - avec `soft_prior` OFF, `candidate_try` partait encore sur `d50_2725`, ce qui a permis d'isoler un second rerank caché via `blind_candidate_rerank_top_k` et `_quick_candidate_likelihood(...)`.
- Probe décisif :
  - variante ajoutée :
    - `no_pairset_gate_meta_seed_guard_dedup_all_no_permhash_no_softprior_no_rerank`
  - artefact :
    - `reports/r47i_s6_m106_first_divergence_audit_20260605_224538_no_permhash_no_softprior_no_rerank_probe_20260605/`
  - lecture durable :
    - sans `soft_prior` **et** sans `candidate_rerank`, `candidate_try` repart bien sur `d50_2823` en premier ;
    - `pairset_scale_precheck` le montre encore vivant (`4` paires, `decision=keep_pairset`) ;
    - `resolve_hit_preresolve` et `astrometry_exact_trace` se recentrent aussi sur `d50_2823` dans la même fenêtre ;
    - l'avance artificielle de `d50_2725` / `d50_2822` provenait donc du corridor de ranking Ze ajouté après le score brut, pas du score brut seul.
- Conclusion durable :
  - pour approcher un chemin **identique à Astrometry**, il faut désormais auditer ce `candidate_rerank` comme probable heuristique Ze non canonique ;
  - la question centrale n'est plus seulement "quel garde remplace `perm_hash_gate` ?", mais aussi "quel ranking intermédiaire doit disparaître ou être recâblé vers un vrai critère de verify canonique".

## 2026-06-05 23:05 — Astrometry n'expose pas d'équivalent au `candidate_rerank` Ze avant verify
- Comparatif source utile :
  - Astrometry : `solver.c` suit le corridor
    - `try_all_codes`
    - `try_permutations`
    - `resolve_matches`
    - `solver_handle_hit`
    - `verify_hit`
  - pré-filtres observés avant `verify_hit` :
    - permutations géométriques
    - `abscale`
    - bornes RA/Dec
  - pas d'équivalent trouvé à un rerank de tête basé sur une vraisemblance RANSAC locale avant verify.
- Côté Ze, l'heuristique incriminée est explicite :
  - `blind_candidate_rerank_top_k`
  - `_quick_candidate_likelihood(...)`
- Probe décisif complémentaire avec pruning conservé :
  - variante :
    - `no_pairset_gate_meta_seed_guard_dedup_all_no_softprior_no_rerank`
  - artefact :
    - `reports/r47i_s6_m106_first_divergence_audit_20260605_230557_permhash_on_no_softprior_no_rerank_probe_20260605/`
  - résultat durable :
    - en gardant `perm_hash_gate` mais en neutralisant `soft_prior` + `candidate_rerank`, `candidate_try` repart sur `d50_2823` ;
    - `pairset_scale_precheck` garde `d50_2823` vivant ;
    - `resolve_hit_preresolve` se reconcentre sur `d50_2823` ;
    - l'`exact_trace` montre ensuite clairement `reject_perm_hash_gate` sur ce même candidat.
- Conclusion durable :
  - le ranking Ze et le pruning `perm_hash_gate` sont désormais séparés proprement ;
  - dans un chemin de parité Astrometry, `candidate_rerank` doit être considéré comme heuristique Ze à neutraliser ou supprimer, puis le front suivant à traiter devient le `perm_hash_gate`.

## 2026-06-05 23:20 — Le seuil du `perm_hash_gate` agit comme un couperet quantifié très structurant
- Probe minimal ajouté :
  - variante :
    - `no_pairset_gate_meta_seed_guard_dedup_all_no_softprior_no_rerank_permhash2048`
  - artefact :
    - `reports/r47i_s6_m106_first_divergence_audit_20260605_232011_permhash2048_probe_20260605/`
- Résultat durable :
  - avec `soft_prior` et `candidate_rerank` déjà neutralisés, relever seulement `blind_astrometry_strict_perm_hash_max_qdelta` de `1536` à `2048` laisse `d50_2823` en tête ;
  - `resolve_hit_preresolve` reste centré sur `d50_2823` ;
  - les rejets `reject_perm_hash_gate` sur `d50_2823` chutent fortement (`1479 -> 471`) ;
  - le plus petit rejet restant est `2052`, ce qui montre que le relâchement a effectivement libéré tout le paquet juste sous le nouveau seuil ;
  - `d50_2822` disparaît de l'`exact_trace` écrit dans cette fenêtre.
- Conclusion durable :
  - le `perm_hash_gate` n'est pas seulement une heuristique de rejet binaire ;
  - son seuil quantifié a un effet massif sur la topologie de la fenêtre de recherche ;
  - un relâchement modéré semble libérer du signal utile sans réouvrir immédiatement les vieux faux fronts, ce qui en fait un bon candidat d'audit calibré pour la suite.

## 2026-06-13 02:35 — `mo_scale_native` diverge bien d'Astrometry, mais n'explique pas le zéro-match `verify`
- Comparatif source confirmé :
  - Astrometry remplit `mo->scale` depuis le WCS (`tan_pixel_scale`), donc en vrai `arcsec/pix`.
  - Ze reconstruisait encore un `mo_scale_native` à partir d'une médiane de distances en **pixels image**, ce qui est une divergence sémantique réelle.
- Probe-only ajouté :
  - `blind_astrometry_probe_native_mo_scale_from_model_enabled`
  - il recale `mo_scale_native` et `verify_pix2_scale_source` sur `model_scale_arcsec`.
- Résultat durable :
  - sur le replay forensic ouvert jusqu'à `verify`, la valeur utilisée par `verify_pix2` passe bien de `180.689` à `4.433`.
  - mais sur le **même pool figé** `test_xy/ref_xy`, le rejeu pur de la séquence `verify` reste inchangé :
    - `prob_theta_match_total = 0`
    - `prob_ibailed = 17`
    - `prob_steps = 18`
    - `prob_logodds_last = -24.953298500158034`
  - le seul effet mesuré est un gonflement faible de `sigma²` (~`+1.99%`) via `index_jitter / mo_scale`.
- Conclusion durable :
  - `mo_scale_native` doit rester dans l'inventaire des écarts Ze vs Astrometry ;
  - mais pour `d50_2823`, ce n'est **pas** le verrou causal principal du `no_positive_verify_match` ;
  - le prochain front utile reste le **pool verify / support géométrique positif**, pas `mo_scale`.

## 2026-06-13 20:23 — Ordre des test-stars et `testsigma²` blanchis comme cause dominante sur `d50_2823`
- Relecture source Astrometry ciblée validée :
  - le champ est trié par flux avant `verify` ;
  - `verify_get_test_stars()` ne change ensuite l’ordre que par :
    - déduplication
    - retrait du quad
    - `uniformize` éventuel
    - RoR
  - `testsigma²` canonique est bien :
    - `verify_pix2 * (1 + R² / Q²)`
- Côté Ze :
  - un tri flux-desc stable de parité Astrometry a été câblé ;
  - le rejeu `reports/r47i_s6_fluxsort_probe_20260613_202233/` reste inchangé vs baseline quad-removal ;
  - le diff séquentiel confirme qu’aucun delta `verify` n’apparaît sur ce cas.
- Conclusion durable :
  - sur le cas vivant `d50_2823`, l’ordre des premières test-stars n’est pas le verrou causal actif ;
  - le modèle `testsigma²` n’est plus non plus le doute méthodologique dominant ;
  - les résiduels `uniformize/reorder` côté source ne sont pas actifs ici car :
    - `blind_astrometry_verify_index_cutnside = 0`
    - le `verify_teststar_filter` Ze-side ne s’active pas sur le chemin natif strict courant.
- Direction utile suivante :
  - auditer en priorité la sémantique `RoR / effective_area` avant de replonger plus bas dans `theta/logodds`.

## 2026-06-13 20:34 — `RoR / effective_area` était bien un écart causal restant
- Comparatif source confirmé :
  - Astrometry `verify_apply_ror()` recalcule `effective_area` par comptage de bins, même quand `index_cutnside = 0` revient à une grille `1x1`.
  - Ze utilisait encore un fallback disque :
    - `effA = min(area, π * ror²)`
- Patch durable appliqué :
  - `_apply_verify_ror_filter()` calcule désormais la grille `uniformize` canonique pour `effective_area` même à `cutnside=0`.
  - Le filtrage des étoiles reste inchangé :
    - bins seulement si `nw>1 || nh>1`
    - sinon RoR radial direct
- Validation locale :
  - `py_compile` OK
  - `pytest -q tests/test_zeblindsolver.py tests/test_synthetic.py` => `42 passed`
  - test ajouté :
    - `test_apply_verify_ror_filter_uses_bin_area_fallback_at_cutnside_zero`
- Rejeu borné utile :
  - `reports/r47i_s6_ror_effa_probe_20260613_203138/`
  - diff :
    - `reports/r47i_s6_verify_sequence_diff_20260613_203300_r47i_s6_fluxsort_probe_20260613_202233_vs_r47i_s6_ror_effa_probe_20260613_203138/summary.json`
- Résultat durable :
  - le préfixe `verify` s’améliore réellement :
    - `prob_logodds` : `-1.38629 -> -0.98933`
    - `gate_pass_count` : `11 -> 19`
    - `theta_match_count` : `5 -> 8`
  - le step `0` bascule de `distractor` à `match` avec `nsig2` nettement meilleur.
- Conclusion durable :
  - `RoR / effective_area` était bien un faux écart Ze-vs-Astrometry encore causal ;
  - une fois refermé, il devient plus crédible de redescendre ensuite sur les derniers résiduels `verify_pix2` / `theta-logodds` que de revenir soupçonner `testperm`.

## 2026-06-13 20:42 — `verify_pix2` blanchi sur le baseline courant ; front vivant = `theta/logodds`
- Observabilité ajoutée dans `verify_debug_sets` pour tracer explicitement :
  - `verify_model_scale_arcsec_input`
  - `verify_pix_scale_arcsec_input`
  - `verify_mo_scale_arcsec_px_for_pix2`
  - `mo_model_scale_arcsec_input`
  - `mo_pix_scale_arcsec_input`
- Probe utile :
  - `reports/r47i_s6_pix2_obs_probe_20260613_204225/`
- Résultat durable :
  - sur le baseline courant, `verify_pix2` consomme bien :
    - `model_scale_arcsec = 6.077758799729469`
    - `pix_scale_arcsec = 6.077758799729469`
    - `verify_scale_arcsec_px_for_pix2 = 6.077758799729469`
  - `mo_scale_native` suit la même valeur :
    - `mo_scale = 6.077758799729469`
    - `mo_scale_source = probe_model_scale_arcsec`
- Conclusion durable :
  - le doute sur `verify_pix2` n’est plus le front vivant du baseline courant ;
  - l’ancienne valeur `4.433...` appartient à un artefact plus ancien/obsolète pour cette branche ;
  - la prochaine comparaison utile doit maintenant porter directement sur la séquence `real_verify_star_lists()` vs `_astrometry_verify_sequence_logodds()`.

## 2026-06-13 20:48 — Observabilité `theta/logodds` enrichie ; aucune branche `conflict` active sur le baseline courant
- Relecture du préfixe utile courant :
  - sur `r47i_s6_ror_effa_probe_20260613_203138`, la séquence active n’utilise pas la branche `conflict` ;
  - le front vivant reste donc un préfixe `match` / `distractor`, pas une réaffectation de match déjà pris.
- Patch durable d’observabilité :
  - `oldj_dbg` et `keepfg_dbg` sont désormais correctement renseignés si un conflit survient ;
  - `verify_step_dump.json` expose aussi les résumés utiles au niveau entrée :
    - `prob_best_i`
    - `prob_ibailed`
    - `prob_istopped`
    - `prob_matches`
    - `prob_conflicts`
    - `prob_distractors`
    - `prob_steps`
    - `prob_theta_match_total`
    - `prob_theta_conflict_total`
    - `prob_theta_distractor_total`
- Validation locale :
  - `py_compile` OK
  - `pytest -q tests/test_zeblindsolver.py tests/test_synthetic.py` => `42 passed`
- Conclusion durable :
  - cette itération ne change pas la logique de score ;
  - elle prépare une comparaison beaucoup plus propre avec le source Astrometry si un doute mathématique ressurgit sur `theta/logodds`.

## 2026-06-13 20:55 — Le cœur `theta/logodds` est stable ; le front vivant repart aux entrées/support
- Probe utile :
  - `reports/r47i_s6_theta_core_probe_20260613_205329/`
  - diff strict :
    - `reports/r47i_s6_verify_sequence_diff_20260613_205432_r47i_s6_ror_effa_probe_20260613_203138_vs_r47i_s6_theta_core_probe_20260613_205329/summary.json`
- Résultat durable :
  - avec les mêmes entrées (`verify_pix2`, `test_n`, `ref_n`), la séquence `theta/logodds` est strictement identique :
    - `prob_logodds = -0.989333722091672`
    - `gate_pass_count = 19`
    - `theta_match_count = 8`
    - aucun `step_delta`
  - résumé utile :
    - `prob_best_i = 0`
    - `prob_ibailed = 25`
    - `prob_conflicts = 0`
- Conclusion durable :
  - sur le baseline courant, le miroir Python du cœur `verify` n’est pas le verrou causal actif ;
  - la suite utile doit remonter aux **entrées/support** qui déterminent le premier préfixe utile, en particulier le voisin NN et la géométrie locale du step `0`.

## 2026-06-13 21:02 — `followup.md` a été volontairement reset en plan top-down court
- Décision de pilotage validée :
  - arrêter d'utiliser `followup.md` comme journal semi-historique ;
  - conserver l'historique accompli dans `memory.md` ;
  - transformer `followup.md` en tableau de bord court orienté :
    - premier point de divergence
    - plan top-down
    - sortie livrable
- Lecture durable :
  - le sprint n'est plus piloté comme une suite de coupes dichotomiques locales ;
  - la méthode officielle devient :
    - cas sentinelle fixe
    - checkpoints amont -> aval
    - arrêt sur le premier écart causal
    - un seul delta puis rejeu de la colonne complète.

## 2026-06-13 21:13 — Premier checkpoint ouvert top-down désormais matérialisé
- Nouvel outil ajouté :
  - `tools/r47i_s6_first_divergence_topdown_audit.py`
- Validation locale :
  - `python3 -m py_compile tools/r47i_s6_first_divergence_topdown_audit.py` OK
- Audit utile publié :
  - `reports/r47i_s6_first_divergence_topdown_audit_20260613_211325_r47i_s6_theta_core_probe_20260613_205329/summary.json`
- Résultat durable :
  - checkpoints déjà blanchis / fermés sur le baseline courant :
    - `input_stars_order`
    - `quad_geometry`
    - `verify_pix2_scale`
    - `verify_sequence_core`
  - **premier checkpoint encore ouvert** :
    - `verify_support_pre_step0`
  - classification actuelle du premier front ouvert :
    - divergence de **support `verify`**
    - pas une divergence du cœur `theta/logodds`
- Lecture durable :
  - le premier point à investiguer maintenant n'est plus un résiduel diffus ;
  - c'est explicitement le **support exact `test/ref` et le voisin NN du step 0** avant le cœur séquentiel.

## 2026-06-13 21:52 — Astrometry dispose maintenant d'un harness direct `verify_hit()` ; le vrai trou restant est l'injection de l'hypothèse Ze
- Outil de lecture ajouté :
  - `tools/r47i_s6_astrometry_engine_verify_audit.py`
- Premier constat durable obtenu sur le runtime Astrometry blind du même FITS :
  - le `testperm` canonique peut être **non monotone**
  - `verify_apply_ror()` peut réordonner via `uniformize`
  - donc un ordre Ze monotone n'est pas, à lui seul, un argument de non-parité
- Un essai `solve-field --verify` avec une WCS candidate reconstruite depuis Ze a été lancé :
  - artefact :
    - `reports/r47i_s6_astrometry_verify_from_ze_wcs_20260613_2140/`
  - lecture durable :
    - ce chemin ne fournit pas encore un checkpoint homologue Ze ;
    - quand la WCS candidate ne passe pas, Astrometry retombe sur une résolution blind puis produit un `matchfile` canonique Astrometry.
- Un vrai harness Astrometry direct sur `verify_hit()` a été remis en état :
  - `astrometry-net-main/solver/verify-paths.c` modernisé en mode utile `matchfile + index`
  - `astrometry-net-main/solver/Makefile` complété avec une cible `verify-paths`
  - compile réussie de `astrometry-net-main/solver/verify-paths`
- Rejeu autonome validé :
  - commande `verify-paths -i index-4107.fits -m zeverify.match -f zeverify.axy`
  - log :
    - `reports/r47i_s6_astrometry_verify_from_ze_wcs_20260613_2140/verify_paths.log`
  - audit :
    - `reports/r47i_s6_astrometry_engine_verify_audit_20260613_215213_r47i_s6_astrometry_verify_from_ze_wcs_20260613_2140/summary.json`
- Résultat durable du harness direct :
  - checkpoint Astrometry canonique proprement figé :
    - `NT=123`
    - `NR=19`
    - `NTall=127`
    - `NRall=44`
    - `quad_field_head=[4,0,3,5]`
    - `testperm_head=[1,2,7,12,11,6,8,9,16,19,13,17,10,14,23,21]`
    - `refperm_head=[13,42,26,17,41,12,10,8,39,43,28,31,29,36,0,7]`
    - `verify: logodds 128.438`, `19 matches`, `0 conflicts`, `54 distractors`, `besti=72`
- Lecture durable :
  - on ne manque plus d'un outil Astrometry au bon étage ;
  - le verrou méthodologique restant est maintenant beaucoup plus précis :
    - **produire/injecter l'équivalent Ze du `matchfile` / `MatchObj`**
    - puis comparer `verify_support_pre_step0` à type d'entrée strictement identique.

## 2026-06-14 00:57 — Premier écart Ze vs Astrometry matérialisé au même étage : support `verify_hit()`
- Instrumentation ajoutée dans `zeblindsolver.py` pour exporter les données nécessaires à un `MatchObj` Ze homologué :
  - `field_xy_all_native` / `field_xy_all_native_n`
  - `ref_world_all_native` / `ref_world_all_native_n`
- Outil ajouté :
  - `tools/r47i_s6_build_ze_matchfile.py`
  - rôle : reconstruire un `.axy`, un `.match` et une WCS TAN Astrometry à partir du `verify_debug_sets.json` Ze.
- Probe Ze utile :
  - `reports/r47i_s6_ze_matchobj_worlddump_probe_20260614_005416/`
- Matchfile Ze reconstruit :
  - `reports/r47i_s6_ze_matchfile_20260614_005704_r47i_s6_ze_matchobj_worlddump_probe_20260614_005416/`
  - fit TAN depuis les paires Ze projetées :
    - `ref_fit_n=417`
    - `fit_rms_px≈0.185`
    - `fit_median_px≈0.123`
    - `fit_max_px≈0.656`
    - `fit_scale_arcsec_px≈6.073`
- Rejeu Astrometry direct sur le `MatchObj` Ze :
  - log :
    - `reports/r47i_s6_ze_matchfile_20260614_005704_r47i_s6_ze_matchobj_worlddump_probe_20260614_005416/verify_paths_ze.log`
  - audit :
    - `reports/r47i_s6_astrometry_engine_verify_audit_20260614_005744_r47i_s6_ze_matchfile_20260614_005704_r47i_s6_ze_matchobj_worlddump_probe_20260614_005416/summary.json`
- Résultat durable :
  - Astrometry sur le `MatchObj` Ze produit :
    - `quad_field_head=[1,68,152,244]`
    - `NTall=439`
    - `NRall=247`
    - `NT=434`
    - `NR=134`
    - `testperm_head=[0,16,30,65,28,4,6,92,55,61,241,15,23,17,46,42]`
    - `refperm_head=[222,154,245,164,95,234,175,110,44,46,176,183,58,53,178,160]`
    - `logodds=-1.38629`, `0 matches`, `0 conflicts`, `1 distractor`
  - Le miroir Ze courant sur la même hypothèse consomme au contraire :
    - `prob_verify_nt=48`
    - `prob_verify_nr=395`
    - `prob_logodds≈-0.98933`
- Conclusion durable :
  - le premier écart causal n'est plus une incertitude méthodologique ni le cœur `theta/logodds` ;
  - il est maintenant matérialisé au checkpoint `verify_support_pre_step0` ;
  - Ze construit un support de vérification différent d'Astrometry :
    - test side trop réduit car il part d'un sous-ensemble pré-filtré au lieu du champ image complet ;
    - ref side trop large car il part du pool tile/world Ze au lieu du pool index projeté par champ/rayon comme `verify_hit()`.
- Prochain delta unique :
  - réaligner la construction du support Ze sur Astrometry avant toute nouvelle analyse aval.

## 2026-06-14 01:26 — Premier delta d'alignement support appliqué ; le front se resserre sur le ref-pool
- Patch appliqué dans `zeblindsolver.py` :
  - en mode `astrometry_native_verify_semantics`, le test-pool part maintenant du champ image complet au lieu d'être capé par `blind_astrometry_verify_teststar_max_keep`
  - le ref-scope par champ/rayon WCS est activé automatiquement en mode native, avec suppression du padding Ze `*1.10` sur le rayon
  - export ajouté :
    - `ref_world_px`
    - `ref_world_px_n`
  - cet export permet de reconstruire un `MatchObj` propre avec les RA/Dec réellement alignés sur `ref_xy_px`
- Outil mis à jour :
  - `tools/r47i_s6_build_ze_matchfile.py` consomme `ref_world_px` en priorité
  - fallback ajouté pour les IDs globaux Ze encodés `(tile_index << 32) + local_star_id`
- Validation locale :
  - `python3 -m py_compile zeblindsolver/zeblindsolver.py tools/r47i_s6_build_ze_matchfile.py` OK
- Probe principal après patch :
  - `reports/r47i_s6_native_support_refworld_probe_20260614_012409/`
  - Ze support effectif :
    - `field_xy_all_native_n=439`
    - `test_xy_px=435`
    - `ref_xy_px=123`
    - `prob_verify_nt=435`
    - `prob_verify_nr=123`
    - `prob_logodds=-1.3862943611198908`
- Matchfile propre reconstruit :
  - `reports/r47i_s6_ze_matchfile_20260614_012525_r47i_s6_native_support_refworld_probe_20260614_012409/summary.json`
  - fit WCS :
    - `ref_fit_n=123`
    - `ref_id_mode=direct_ref_world_px`
    - `fit_rms_px≈0.076`
    - `fit_median_px≈0.059`
    - `fit_scale_arcsec_px≈6.073`
- Oracle Astrometry `verify-paths` sur ce nouveau `MatchObj` Ze :
  - log :
    - `reports/r47i_s6_ze_matchfile_20260614_012525_r47i_s6_native_support_refworld_probe_20260614_012409/verify_paths_ze.log`
  - audit :
    - `reports/r47i_s6_astrometry_engine_verify_audit_20260614_012557_r47i_s6_ze_matchfile_20260614_012525_r47i_s6_native_support_refworld_probe_20260614_012409/summary.json`
  - Astrometry reconstruit :
    - `NTall=439`
    - `NRall=248`
    - `NT=434`
    - `NR=134`
    - `quad_field_head=[1,68,152,244]`
    - `logodds=-1.38629`
- Conclusion durable :
  - le delta a refermé la grosse divergence test-pool (`48 -> 435`, proche du `434` Astrometry)
  - le checkpoint `verify_support_pre_step0` n'est pas encore fermé
  - le premier sous-écart restant est désormais le **ref-pool/index source** :
    - Ze native actuel : `NR=123`
    - Astrometry oracle depuis le même `MatchObj` : `NR=134`
  - il reste aussi un micro-écart test/RoR :
    - Ze `NT=435`
    - Astrometry `NT=434`
- Prochain delta logique :
  - brancher ou reproduire une source ref-pool strictement Astrometry (`startree_search_for` / index 4107), puis seulement ensuite traiter le `435 -> 434`.

## 2026-06-14 01:42 — Support C forcé dans Ze : le cœur séquentiel est blanchi, le front restant est amont
- Instrumentation Astrometry ajoutée :
  - `astrometry-net-main/solver/verify.c` peut écrire un dump complet du support consommé par `verify_hit()` via `C_VERIFY_SUPPORT_DUMP=/path/support.json`
  - le dump contient :
    - `testperm`
    - `refperm`
    - `test_xy`
    - `ref_xy`
    - `testsigma2`
    - `refstarid`
    - `NT/NR/NTall/NRall`
- Validation compilation :
  - `make verify-paths` OK après modification de `verify.c`
- Oracle C extrait sur le `MatchObj` Ze propre :
  - `reports/r47i_s6_ze_matchfile_20260614_012525_r47i_s6_native_support_refworld_probe_20260614_012409/c_verify_support.json`
  - valeurs :
    - `NT=434`
    - `NR=134`
    - `NTall=439`
    - `NRall=248`
    - `testperm_head=[0,16,30,65,28,4,6,92,55,61,241,15,23,17,46,42]`
    - `refperm_head=[223,155,246,165,93,235,176,108,45,47,177,184,59,54,179,161]`
- Injection/replay Ze ajouté :
  - `zeblindsolver.py` accepte maintenant `testsigma2` dans le JSON `blind_astrometry_verify_forced_teststars_json_path`
  - en mode `blind_astrometry_verify_forced_inputs_always_enabled`, ce profil `testsigma2` est utilisé tel quel pour le replay C
- Probe de contrôle sans `testsigma2` forcé :
  - `reports/r47i_s6_forced_c_support_probe_20260614_013814/`
  - support fermé (`NT=434`, `NR=134`) mais step 0 divergeait :
    - Ze classait le premier voisin en `match`
    - Astrometry le classait en `distractor`
  - cause : variance Ze trop large sur ce step
- Probe de contrôle avec support + `testsigma2` C forcés :
  - `reports/r47i_s6_forced_c_support_sigma_probe_20260614_014048/`
  - résultat :
    - `prob_verify_nt=434`
    - `prob_verify_nr=134`
    - `prob_logodds=-1.3862943611198908`
    - `prob_matches=0`
    - `prob_conflicts=0`
    - `prob_distractors=1`
    - step `0` = `distractor`
- Conclusion durable :
  - à support `test/ref` et profil `testsigma²` identiques, le miroir Python Ze reproduit le préfixe Astrometry observé
  - le cœur séquentiel `theta/logodds` est donc blanchi pour ce checkpoint
  - le front causal restant est strictement amont :
    - produire le ref-pool Astrometry `startree_search_for` en Python
    - reproduire le profil `testsigma²` Astrometry sans injection forcée

## 2026-06-14 02:03 — Delta amont fermé : `verify_pix2` Astrometry + `testperm` uniformize
- Divergence math identifiée :
  - Astrometry `verify-paths` initialise `pix2=1.0`, puis ajoute `(index_jitter / mo.scale)^2`
  - Ze réutilisait `blind_prob_sigma_px=1.6`, donc `verify_pix2_base=2.56`
  - effet : `testsigma²` trop large et step 0 pouvait être classé `match` au lieu de `distractor`
- Patch Ze appliqué :
  - nouveau champ config `blind_astrometry_verify_pix_base_px=1.0`
  - en mode `astrometry_native_verify_semantics` / replay forcé, `verify_pix2_base` utilise ce défaut Astrometry au lieu de `blind_prob_sigma_px`
- Divergence d'ordre test-stars identifiée :
  - Astrometry appelle `verify_uniformize_field()` dans `verify_apply_ror()`
  - cette fonction réordonne `testperm` en balayant les bins row-major par couches, elle ne filtre pas seulement par RoR
  - Ze filtrait les bins mais ne reproduisait pas cette permutation
  - Ze perdait aussi `_testperm0` en remplaçant les IDs par `arange(...)`
- Patch Ze appliqué :
  - port de la permutation `verify_uniformize_field()` dans `_apply_verify_ror_filter`
  - conservation du vrai `_testperm0` après dédup/retrait du quad
  - résolution automatique du `CUTNSIDE` par niveau si la config vaut `0` :
    - `S=156`
    - `M=110`
    - `L=78`
- Probe de validation :
  - `reports/r47i_s6_auto_cutnside_pixbase_probe_20260614_020139/`
  - runtime original conservé (`blind_astrometry_verify_index_cutnside=0`) ; l'auto-résolution donne le comportement attendu
  - `verify_pix2_input=1.0270715472637384`
  - `testperm_head` Ze = Astrometry :
    - `[0,16,30,65,28,4,6,92,55,61,241,15,23,17,46,42,...]`
  - `testsigma²` de tête collé au C à environ `0.002` près
  - step 0 :
    - `theta_label=distractor`
    - `cum_logodds=-1.3862943611198908`
- Front restant après ce delta :
  - test/RoR queue : Ze garde un extra test `295` (`NT=435` vs Astrometry `434`)
  - front principal : refpool/index source toujours divergent
    - Ze `NR=123`
    - Astrometry `NR=134`
    - refstarid Ze et C n'ont pas encore le même espace d'identifiants
- Prochaine étape logique :
  - reproduire ou exporter strictement la source `startree_search_for()` / sweep-sort Astrometry pour fermer `NR 123 -> 134`
  - seulement ensuite traiter le micro-résidu `NT 435 -> 434`

## 2026-06-14 07:20 — Delta test-stars fermé : dédup Astrometry `verify_deduplicate_field_stars()`
- Résidu du checkpoint `verify_support_pre_step0` isolé :
  - Ze gardait une étoile test supplémentaire, original id `295`, à `x≈6.659`, `y≈1902.537`
  - Astrometry ne la garde pas dans `c_verify_support.json`
  - comparaison canonique :
    - Ze avant patch : `NT=435`
    - Astrometry : `NT=434`
- Cause source confirmée dans `astrometry-net-main/solver/verify.c` :
  - `verify_get_test_stars()` appelle `verify_deduplicate_field_stars(v, vf, 1.0)` avant retrait des étoiles du quad
  - ce filtre retire les étoiles tardives dans l'ordre d'entrée si elles sont dans le rayon `testsigma` d'une étoile plus précoce
  - pour le sentry, `295` est proche de `75` :
    - `d²≈166.873`
    - `testsigma²(75)≈179.633`
    - donc `295` doit être supprimée
- Patch Ze appliqué :
  - ajout du helper `_astrometry_verify_dedup_teststar_indices()`
  - utilisation en mode `astrometry_native_verify_semantics` avant retrait du quad
  - replay forcé laissé intact pour ne pas redédupliquer les payloads C déjà figés
- Validation :
  - check isolé sur le payload `r47i_s6_auto_cutnside_pixbase_probe_20260614_020139` retire exactement `[295]`
  - probe runtime `reports/r47i_s6_cdedup_probe_20260614_071553/` :
    - `NT=434`
    - `295` absent de `teststarid_use`
  - `python3 -m py_compile zeblindsolver/zeblindsolver.py tools/r47i_s6_build_ze_matchfile.py`
  - `pytest -q tests/test_zeblindsolver.py tests/test_synthetic.py` => `43 passed`
- Lecture durable :
  - le sous-écart test-stars/RoR est fermé
  - le front actif du checkpoint reste strictement le ref-pool/index :
    - cible Astrometry canonique : `NR=134`, `NRall=248`, `refperm_head=[223,155,246,165,...]`
    - Ze doit reproduire la source `startree_search_for()` / ordre ref avant de redescendre dans `verify`

## 2026-06-24 — Source ref-pool Astrometry reproduite en Python ; checkpoint support fermé

- Une implémentation Python pure du star-tree FITS Astrometry a été ajoutée :
  - lecture des chunks `kdtree_lr_stars`, `kdtree_split_stars`, `kdtree_range_stars`, `kdtree_data_stars` et `sweep`
  - décodage natif `u32`
  - parcours KD-tree équivalent à `startree_search_for(..., KD_OPTIONS_SMALL_RADIUS)`
  - retour des IDs natifs dans l'ordre du parcours
- Le chemin a été branché explicitement dans `astrometry_native_verify_semantics` via :
  - `blind_astrometry_startree_index_path`
- Un écart amont supplémentaire a été identifié et corrigé :
  - Ze réduisait encore le rayon du champ avec `approx_scale_deg * 1.15`
  - ce clamp n'existe pas dans le chemin `verify_hit()` Astrometry
  - il est maintenant désactivé en mode natif
- Validation sentinelle `M106 / d50_2823` :
  - avant correction rayon : `NRall=51`, `NR=48`
  - après correction : Ze `NT=434`, `NR=134`, `NRall=247`
  - oracle C homologue : `NT=434`, `NR=134`, `NRall=247`
  - `testperm` identique
  - même ensemble de références utiles
  - même premier résultat `logodds=-1.38629436`
- Les seules différences de permutation restantes sont à l'intérieur de groupes de même `sweep` :
  - Astrometry utilise `qsort_r`, non stable sur les égalités
  - Ze garde un tri stable/déterministe
  - ce résidu n'est pas une divergence sémantique contractuelle
- Artefact :
  - `reports/r47i_s7_startree_parity_closure_20260624/summary.md`
- Validation locale :
  - `python3 -m py_compile` OK
  - `pytest -q tests/test_zeblindsolver.py tests/test_synthetic.py` => `44 passed`
- Le prochain front autorisé est désormais :
  - smoke multicase ZeBlind court
  - puis garde-fou complet `testzenear`

## 2026-06-24 — Smoke multicase startree et garde-fou ZeNear court

- Le smoke multicase historique à 3 FITS a été rejoué avec et sans la nouvelle source startree :
  - contrôle courant : `2/3`
  - startree natif : `1/3`
  - échec commun : `233520`
  - seul delta : `232350`
- L'écart `232350` a été audité :
  - le contrôle acceptait une hypothèse à seulement `4` inliers et RMS `≈27.16 px`
  - son support probabiliste positif venait de `2` matches faibles dans le pool tile-world
  - le pool startree canonique donne `0` match positif et empêche la promotion
  - conclusion : rejet d'un faux positif probable, pas régression d'une solution canonique
- Garde-fou ZeNear court sur 5 FITS :
  - `4/5` succès
  - débit `≈18.58 images/min`
  - échec unique `234013` : `near solver could not estimate a similarity transform`
  - cet échec appartient à une classe déjà observée avant le patch blind-only
- L'outil `bench_zenear_from_list.py` a été réparé :
  - suppression de l'ancien chemin figé `/home/tristan/ZeSolver`
  - import désormais résolu depuis la racine réelle du dépôt
- Verdict :
  - aucune régression ZeNear attribuable au startree
  - le tuning produit blind-only peut être rouvert prudemment
  - le `testzenear` complet reste à exécuter
- Artefact :
  - `reports/r47i_s7_multicase_and_zenear_guard_20260624/summary.md`

## 2026-06-24 — Garde-fou testzenear complet et double fallback Blind corrigé

- Le lot canonique propre de `30` FITS a été retrouvé dans :
  - `reports/eq_ircut_cleanbench_20260518_230249/data`
- Garde-fou ZeNear isolé, paramètres CPU courants :
  - `20/30`
  - `141.97 s`
  - `12.68 images/min`
  - les `10` échecs sont tous `near solver could not estimate a similarity transform`
- Garde-fou batch produit sur copies explicitement nettoyées de tout WCS :
  - `580` cartes supprimées
  - `0` fichier avec WCS avant lancement
  - `22/30` résolus par Near, soit la même cardinalité Near que la baseline historique
  - `8` non-résolus : `232945`, `233027`, `233048`, `233130`, `233211`, `233232`, `233828`, `234013`
- Le run global a été borné après environ `42 min 09 s` :
  - `0` succès Blind
  - `8` premiers échecs Blind terminés
  - seconde phase Blind encore active
  - RSS observé jusqu'à `≈3.4 GiB`
  - recherches globales répétées avec environ `1240–1269` candidats de niveau S
- Cause orchestration isolée :
  - la phase Near process appelle `solve_path(..., allow_blind_fallback=False)`
  - `_run_index_near_solver()` ignorait ce contrat tant que `near_defer_blind_fallback=false`
  - chaque échec Near lançait donc un Blind interne, puis repartait dans la phase Blind batch
- Correctif ciblé dans `zesolver.py` :
  - `allow_blind_fallback=False` force désormais le différé vers la phase Blind batch
  - le comportement séquentiel `allow_blind_fallback=True` conserve le fallback Blind immédiat
- Tests ajoutés :
  - batch : appels Near `[False, False]`, aucun Blind interne
  - séquentiel : appels `[False, False, True]`, fallback immédiat conservé
  - `2 passed`
  - garde-fous ZeBlind existants : `44 passed`
- Artefact :
  - `reports/r47i_s7_testzenear_full_product_clean_20260624/summary.md`
- Prochain pas unique :
  - smoke produit borné sur les `8` non-résolus pour mesurer un seul passage Blind par image avant tout nouveau benchmark complet.

## 2026-06-24 — Smoke produit Blind borné et concurrence RAM-adaptative

- Le smoke batch produit a été rejoué sur les `8` non-résolus canoniques :
  - `8/8` Near correctement différés vers la phase Blind
  - seulement `6` démarrages Blind, correspondant aux `6` workers disponibles
  - les `2` derniers sont restés en file
  - aucun second passage Blind observé
- Le run à `6` workers a été borné à `1200 s` :
  - succès : `0`
  - fichiers terminés : `0`
  - RSS maximal : `≈3.2 GiB`
  - chaque worker est descendu jusqu'aux recherches globales coûteuses
  - causes candidates dominantes : validations à `4–7` inliers mais RMS très élevé
- Cause runtime isolée :
  - la phase Blind héritait directement du nombre de workers Near
  - sur la machine à `7.52 GiB`, cela lançait `6` recherches lourdes concurrentes
- Correctif produit dans `zesolver.py` :
  - dimensionnement Blind séparé et borné par la RAM
  - caps : `2/3/4/6/8` workers selon les classes `8/16/32/64/>64 GiB`
  - override manuel via `ZE_BLIND_WORKERS`
- Probe de validation à `2` workers, borné à `240 s` :
  - stratégie loggée : `workers=2`
  - démarrages Blind : `2`
  - RSS maximal : `≈2.0 GiB`, soit `-37.5 %` face au run à 6 workers
  - aucun doublon
- Validation :
  - `48 passed`
  - `py_compile` OK
  - `git diff --check` OK
- Artefact :
  - `reports/r47i_s7_product_blind8_bounded_20260624/summary.md`
- Prochain pas unique :
  - smoke multicase avec oracle WCS qualité pour séparer vrais succès, faux positifs et coût des échecs avant tout tuning de budgets.
