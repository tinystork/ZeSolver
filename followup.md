# Follow-up ZeSolver / ZeBlind

## Mission courante - ZN3.10B jeu hybride GUI fallback 4D

- [x] Construire un dataset hybride separe depuis
  `/home/tristan/near_bench_cmp30/thread4/`, sans modifier les originaux:
  `/home/tristan/near_bench_cmp30/zn310b_gui_fallback4d_20260716_153931/`.
- [x] Selectionner `8` sources couvertes par les index 4D installes:
  `3` M31, `3` M106, `2` NGC6888. Aucun NGC3628 n'est present dans
  `thread4`.
- [x] Creer les variantes `control_clean`, `no_hints`, `wrong_hints` et le lot
  `gui_mixed` (`3` CONTROL, `3` NOHINT, `2` BADHINT), avec pixels strictement
  identiques entre variantes.
- [x] Retirer tous les hints consommes par Near dans NOHINT:
  `RA/DEC`, `OBJCTRA/OBJCTDEC`, `OBJRA/OBJDEC`, `OBJ_RA/OBJ_DEC`,
  `CRVAL1/CRVAL2`, plus `TELRA/TELDEC`, `CENTRA/CENTDEC` et `OBJECT` par
  prudence; noms fichiers neutralises.
- [x] Ajouter les outils:
  `tools/build_zn310b_gui_dataset.py`,
  `tools/diagnose_zn310b_gui_fallback.py`.
- [x] Nettoyer l'integration `SEED_SCALE`:
  plus d'ecriture FITS longue, metadata conservee en stats runtime avec lecture
  legacy compatible.
- [x] Nettoyer les textes/configs GUI:
  chaine normale `ZeNear -> ZeBlind 4D -> Astrometry.net optionnel`, backend
  historique limite au diagnostic expert/dev, gate en `diagnostic`.
- [x] Ajouter logs legers ZN3.10B pour Near/4D/web/historique/backend final.
- [x] Rapports crees:
  `reports/zenear_zn310b_*`, rapport principal
  `reports/zenear_zn310b_summary.md`.
- [x] Tests:
  suite cible ZN2->ZN3.10B `95 passed`; `py_compile` OK.
- [ ] Etape manuelle restante:
  lancer le vrai GUI sur
  `/home/tristan/near_bench_cmp30/zn310b_gui_fallback4d_20260716_153931/gui_mixed`
  avec Astrometry.net desactive, puis analyser le log via
  `tools/diagnose_zn310b_gui_fallback.py`.
- [ ] Verdict:
  `PRE_GUI_READY`; le fallback GUI 4D reel ne doit pas etre declare valide tant
  que le log GUI n'a pas confirme CONTROL->Near, NOHINT/BADHINT->4D exactement
  une fois, sans historique ni web.

## Mission courante - ZN3.8-IMG parite interne bin_and_find_stars

- [x] Geler le verrou causal reel:
  `solve_near` strict appelait bien `astap_adaptive_image_detection`, mais lui
  passait l'image normalisee `0..1` issue de `to_luminance_for_solve`.
- [x] Prouver la premiere divergence:
  les pixels binned Python natifs ADU sont identiques aux pixels binned ASTAP,
  tandis que les pixels normalises divergent immediatement (`PIXEL_TYPE_DIVERGENCE`).
- [x] Prouver l'effet de retry:
  sur `232102`, l'image native retourne `58` etoiles via la passe locale; l'image
  normalisee retourne `0`, puis les fallbacks CPU generiques produisaient les
  `1713` etoiles diluees.
- [x] Correctif unique applique, borne image stricte:
  `astap_iso_image_for_solve()` fournit les ADU natifs en strict ASTAP-ISO et
  les fallbacks generiques sont desactives en strict.
- [x] Catalogue non modifie:
  avec `nrstars_image=58`, `232102` recalcule naturellement `248` etoiles
  catalogue vs `249` ASTAP et resout au stage `initial`.
- [x] Temoins:
  `233459` reste resolu (`54` image, `249` catalogue) et `230409` reste resolu
  (`252` image, `448` catalogue).
- [x] Performance:
  `232102` passe d'environ `73.6s` pre-fix a `1.30s` post-fix sur le run ZN3.8.
- [x] Rapports ZN3.8 crees:
  `reports/zenear_zn38_*`, rapport principal
  `reports/zenear_zn38_summary.md`.
- [x] Outil et tests crees:
  `tools/diagnose_zn38_image_selector.py`,
  `tests/test_zn38_astap_image_selector.py`.
- [x] Validation:
  suite cible ZN2->ZN3.8 `67 passed`; `py_compile` OK.
- [ ] Suite conseillee:
  relancer le corpus et remplacer le contrat fallback 4D par une autre fixture
  Near-failure reelle, puisque `232102` est maintenant resolu par Near.

## Mission courante - ZN3.4 hardening / promotion gate strict ASTAP-ISO

- [x] Retirer l'artifice de flux synthetique decroissant ZN3.3:
  le strict ASTAP-ISO preserve maintenant l'ordre ASTAP explicitement via
  `img_ranks = np.arange(stars.size)`, tandis que `flux` redevient le flux HFD
  mesure. Hors strict, le tri historique par flux mesure reste inchange.
- [x] Geler le temoin M31 runtime canonique:
  `8/8`, WCS offline `WCS_CONFORMANT` sur les huit, RMS
  `0.520/0.573/0.707/0.659/0.828/0.765/0.669/2.863 px`.
- [x] Expliquer `231915`:
  RMS eleve classe `BENIGN_METRIC_EFFECT`; le WCS runtime reste conforme a
  l'oracle offline, donc pas une degradation WCS sur le temoin gele.
- [x] Inventorier et executer le corpus elargi sans filtrage opportuniste:
  manifeste `164` entrees (`M31=84`, `M106=61`, `NGC6888=17`, `S50=2`),
  doublons conserves et identifies par checksum.
- [x] Controles negatifs:
  mauvais centre, image blanche, catalogue vide => `0` faux positif.
- [x] Performance:
  warm M31 mediane ~`1.408s`, p95 ~`1.487s`; cold subprocess median wall
  ~`4.723s`; cout dominant observe = detection HFD Python/import/cache froid.
- [x] Rapports ZN3.4 crees:
  `reports/zenear_zn34_*`, rapport principal
  `reports/zenear_zn34_promotion_gate.md`.
- [x] Outils ZN3.4 crees:
  `tools/diagnose_zn34_promotion_gate.py`,
  `tools/diagnose_zn34_corpus.py`,
  `tools/diagnose_zn34_wcs.py`,
  `tools/diagnose_zn34_performance.py`,
  `tools/diagnose_zn34_negative_controls.py`.
- [x] Tests:
  `pytest -q tests/test_zn2_astap_tools.py tests/test_zn3_input_lists.py tests/test_zn31_astap_oracles.py tests/test_zn32_catalog_reader.py tests/test_zn33_image_detector.py tests/test_zn34_hardening.py`
  => `39 passed`.
- [x] Py compile:
  `python -m py_compile zeblindsolver/metadata_solver.py tools/diagnose_zn34_corpus.py tools/diagnose_zn34_wcs.py tools/diagnose_zn34_performance.py tools/diagnose_zn34_promotion_gate.py`
  => OK.
- [x] Canari synthetique historique:
  `pytest -q tests/test_metadata_solver.py::test_metadata_solver_solves_synthetic_frame`
  => `1 failed`, support strict insuffisant (`4` etoiles image/catalogue),
  pas de relachement de seuil.
- [ ] Verdict:
  `E - Faux positif ou WCS incorrect`; promotion globale interdite car le
  corpus elargi contient `48` WCS incorrects/degrades (M106/NGC6888/M31 elargi),
  malgre le temoin M31 runtime `8/8`.
- [ ] Prochaine etape unique:
  autopsier les WCS elargis M106/NGC6888 (oracle FITS vs solution ZeNear,
  hints/FOV/binning/projection) avant toute promotion produit.

## Mission courante - ZN3.3-IMG parite detection image ASTAP

- [x] Cartographier le detecteur ASTAP CLI:
  `get_background` histogramme, retries `star_level/star_level2/30*sigma`,
  puis sections locales `7*sigma`, HFD, marquage `img_sa`.
- [x] Implementer en Python la route stricte:
  `estimate_astap_global_background`,
  `_astap_sigma_clipped_mean_from_histogram`,
  `astap_section_grid`,
  `_astap_hfd_measure`,
  `astap_adaptive_image_detection`.
- [x] Garder le changement borne au strict ASTAP-ISO:
  hors strict, le detecteur historique reste le chemin utilise.
- [x] Preserver l'ordre ASTAP de scan/section:
  la route Python emet un flux synthetique decroissant pour que le tri existant
  de `solve_near` conserve cet ordre.
- [x] Parite `230409`:
  fond `1202`, bruit `83.0948`, global `131`, local `252`,
  recouvrement centroide `252/252` a `<=0.25 px`.
- [x] Porte image:
  `IMG-P` image Python + catalogue Python corrige ZN3.2-CAT = `8/8`.
- [x] Vrai solveur:
  `solve_near` strict natif = `8/8` sur le corpus M31 runtime.
- [x] Rapports crees:
  `reports/zenear_zn33img_baseline.json`,
  `reports/zenear_zn33img_astap_algorithm_map.md`,
  `reports/zenear_zn33img_astap_equivalence.json`,
  `reports/zenear_zn33img_background_noise_parity.json`,
  `reports/zenear_zn33img_global_pass_parity.json`,
  `reports/zenear_zn33img_local_section_parity.json`,
  `reports/zenear_zn33img_merge_dedup_parity.json`,
  `reports/zenear_zn33img_centroid_parity.json`,
  `reports/zenear_zn33img_ranking_parity.json`,
  `reports/zenear_zn33img_gate.json`,
  `reports/zenear_zn33img_failure_classification.json`,
  `reports/zenear_zn33img_native_solver_results.json`,
  `reports/zenear_zn33img_image_parity.md`.
- [x] Outils crees:
  `tools/diagnose_zn33_image_detector.py`,
  `tools/diagnose_zn33_image_sections.py`,
  `tools/diagnose_zn33_image_gate.py`.
- [x] Tests:
  `pytest -q tests/test_zn2_astap_tools.py tests/test_zn3_input_lists.py tests/test_zn31_astap_oracles.py tests/test_zn32_catalog_reader.py tests/test_zn33_image_detector.py`
  => `29 passed`.
- [x] Py compile:
  `python -m py_compile zeblindsolver/metadata_solver.py tools/diagnose_zn33_image_detector.py tools/diagnose_zn33_image_gate.py`
  => OK.
- [ ] Limite restante:
  `tests/test_metadata_solver.py::test_metadata_solver_solves_synthetic_frame`
  reste en echec avec seulement 4 etoiles image/catalogue; a traiter comme
  canari secondaire separe, sans relacher de seuil.

## Mission courante - ZN3 parité listes image/catalogue ASTAP-ISO

- [x] Reprendre la baseline ZN2 sans toucher au coeur géométrique:
  C00 `0/8`, C10 `0/8`, C01 faible, C11 `8/8`.
- [x] Ajouter les briques Python testables:
  `astap_compatible_mean_bin_image`,
  `astap_binned_to_full_coords`,
  `astap_full_to_binned_coords`,
  `choose_astap_compatible_bin_factor`,
  `astap_background_noise_stats`,
  `astap_sqrt_snr_selection_mask`,
  `astap_compatible_image_detection`.
- [x] Corriger le mapping strict binned existant:
  appliquer `(binfactor - 1) * 0.5 + binfactor * coord`, donc `0.5 + 2*x`
  pour M31 x2, au lieu de `2*x`.
- [x] Ajouter l'injection catalogue diagnostic-only
  `diagnostic_catalog_stars_csv`, desactivee par defaut.
- [x] Creer le probe `tools/diagnose_zn3_input_list_parity.py`.
- [x] Produire les rapports ZN3:
  `reports/zenear_zn3_baseline.json`,
  `reports/zenear_zn3_binned_image_parity.json`,
  `reports/zenear_zn3_image_detection_parity.json`,
  `reports/zenear_zn3_image_ranking_parity.json`,
  `reports/zenear_zn3_catalog_radec_parity.json`,
  `reports/zenear_zn3_catalog_projection_parity.json`,
  `reports/zenear_zn3_catalog_ranking_parity.json`,
  `reports/zenear_zn3_progressive_replacement_matrix.json`,
  `reports/zenear_zn3_native_solver_results.json`,
  `reports/zenear_zn3_failure_classification.json`,
  `reports/zenear_zn3_input_list_parity.md`.
- [x] Resultat probe ZN3 courant:
  P11 `0/8`, O11 `8/8`, natif strict `0/3` sur `230409`,
  `230650`, `231844`; arret conforme, pas d'extension native aux huit.
- [x] Classification d'echec stable dans le probe:
  `NO_SIGNATURE_MATCHES` pour P00/P10/P01/P11 sur les premiers cas.
- [x] Tests:
  `pytest -q tests/test_zn3_input_lists.py tests/test_zn2_astap_tools.py`
  => `11 passed`; `py_compile` OK.
- [x] Test suite demandee:
  `pytest -q tests/test_metadata_solver.py tests/test_zn2_astap_tools.py tests/test_zn3_input_lists.py`
  => `1 failed, 15 passed`; echec existant/attendu
  `test_metadata_solver_solves_synthetic_frame` (support strict trop faible,
  4 etoiles image/catalogue, sans relacher les seuils).
- [ ] Blocage principal:
  les dumps catalogue ASTAP ZN2 ne contiennent pas RA/Dec/magnitude/tile;
  impossible de trancher proprement selection vs repere catalogue en ZN3.
- [ ] Prochaine etape unique:
  etendre l'instrumentation ASTAP opt-in pour dumper RA/Dec/magnitude/tile
  et, si possible, l'image binned avant `find_stars`; puis relancer ZN3
  catalogue RA/Dec et binned pixel-parity avant toute promotion produit.

## Mission courante - ZN2 ASTAP interne vs ZeNear

- [x] Reprendre le constat ZN1: ASTAP `8/8`, ZeNear `0/8`, echec avant
  transformation (`iso_refs=0`, `matches_raw=0`), `ASTAP -extract` non
  suffisant comme verite interne.
- [x] Creer le gate d'equivalence:
  `tools/astap_zn2_build_and_compare.py`.
- [x] Creer le probe d'import/rejeu des dumps internes:
  `tools/diagnose_zn2_astap_internal_parity.py`.
- [x] Enregistrer baseline et provenance:
  `reports/zenear_zn2_baseline.json`.
- [x] Rejouer B0 systeme sur les 8 runtime M31:
  `/usr/local/bin/astap -> /opt/astap/astap`, sha256
  `582e4b0672e3b62222b01c27d3a61525d3876bde814669470e59a21d4299b7af`,
  resultat `8/8`.
- [x] Verrou d'equivalence source/binaire leve:
  compilation locale via `lazbuild -B astap_command_line_linux.lpi`, puis
  B1 local `8/8` avec argument explicite `-z 2`.
- [x] Preuve importante:
  sans `-z 2`, le binaire local resout aussi mais en regime different
  (`1x1`, compteurs etoiles/quads plus hauts). Avec `-z 2`, les compteurs
  B0/B1 sont identiques sur les huit images.
- [x] Cartographier le chemin source ASTAP CLI:
  `Tastap.DoRun`, `solve_image`, `bin_and_find_stars`, `find_quads`,
  `find_fit_using_hash`, `find_offset_and_rotation`.
- [x] Documenter le suspect binning x2:
  conversion pleine resolution `(binfactor - 1) * 0.5 + binfactor * coord`.
- [x] Instrumentation ASTAP locale opt-in:
  variable `ASTAP_ZN2_DUMP_DIR`, inactive par defaut, dump etoiles image,
  etoiles catalogue standard, quads image/catalogue, matches hash et vecteurs
  de solution.
- [x] Validation instrumentation:
  B2 instrumente dump off `8/8`; B3 instrumente dump on `8/8`; compteurs
  structurels conserves, hausse de temps due a l'I/O acceptee.
- [x] Dumps internes `solve_image` generes:
  `reports/zn2_astap_internal_dumps/`.
- [x] Matrice C00/C10/C01/C11:
  C00 `0/8`, C10 `0/8`, C01 `1/8` faible (4 refs), C11 `8/8`.
- [x] Chaine Q sur listes ASTAP:
  le coeur ZeNear regenere les memes quads et retrouve les memes fits
  (`matrix/offset` delta ~1e-12), donc quads/signatures/lookup/transform ne
  sont pas la cause avec les listes ASTAP.
- [x] Verdict ZN2:
  `H - cause mixte ordonnee`; premiere divergence dans la liste image interne
  ASTAP (binning x2 + detection/classement), selection catalogue ZeNear aussi
  divergente et necessaire pour restaurer l'hypothese.
- [x] Tests outillage ZN2:
  `pytest -q tests/test_zn2_astap_tools.py` => `3 passed`;
  `py_compile` des probes et `metadata_solver.py` OK.
- [ ] Prochaine etape unique ZN3:
  aligner d'abord le chemin strict ASTAP-ISO de ZeNear sur la construction des
  listes d'entree ASTAP, en commencant par le binning/detection/classement
  image x2 et en validant avec la selection catalogue ASTAP dans le meme probe.

## Mission courante - P2.24 app/direct, budgets 4D, concurrence

- [x] Creer le probe `tools/diagnose_p223_m31_intermittent_4d.py`.
- [x] Enregistrer baseline/env/config effective dans `reports/zeblind_p223_baseline.json`.
- [x] Preparer les 8 FITS M31 bornes avec copies runtime sans WCS/hints dans `reports/zeblind_p223_corpus.json`.
- [x] Ajouter une telemetry neutre 4D: rangs candidats par index, stop `hard_budget_exceeded`, cout par sous-etape, flags diagnostiques inactifs par defaut.
- [x] Reproduire deux passes directes `solve_blind`, sequentielles, sans ZeNear/fallback: `8/8` puis `8/8`, offline OK.
- [x] Classer la divergence etape 2: les cinq echecs GUI ne sont pas reproductibles en direct; ne pas lancer/promouvoir les matrices C/D/E avant isolation app-path.
- [x] P2.24 probe app-path headless :
  `tools/diagnose_p224_app_path_budget_concurrency.py`.
- [x] P2.24 baseline/timeline/matrices/parite :
  `reports/zeblind_p224_baseline.json`,
  `reports/zeblind_p224_timeline.json`,
  `reports/zeblind_p224_concurrency_matrix.json`,
  `reports/zeblind_p224_budget_matrix.json`,
  `reports/zeblind_p224_app_direct_parity.json`.
- [x] Budget 4D separe :
  `blind_astrometry_4d_search_budget_s=45.0` demarre a l'entree de la route
  Astrometry 4D ; `blind_global_hard_budget_s=0.0` dans le profil
  `zeblind_4d_experimental`.
- [x] Stop reasons separes :
  `user_cancelled`, `blind_attempt_budget_exceeded`,
  `astrometry_4d_search_budget_exceeded`, `max_hypotheses`, `max_accepts`,
  `candidate_exhausted`, `confident_accept`, `best_within_budget`.
- [x] Policy workers :
  profil 4D experimental => 1 worker blind par defaut, override explicite
  `ZE_BLIND_WORKERS` preserve ; backend historique inchange.
- [x] App-path headless M31 P2.24 :
  1 worker `8/8`, repetition `8/8`, 2 workers `8/8` deux fois, faux positifs
  offline `0`.
- [x] Diagnostic concurrence :
  2 workers ameliore le batch (`~202s -> ~165s`) mais degrade le cout image
  (`route mediane ~17s -> ~30.5s`, total max ~49.6s), causal avec l'ancien
  budget global 45 incluant le pre-route.
- [x] Tests P2.24 :
  cible principale `43 passed`, tests voisins `54 passed`, `py_compile` OK.
- [ ] Limite restante : GUI reel non reclique pendant cette passe ; validation
  utilisateur GUI `8/8` a refaire avant P2.25.
- [ ] Prochaine etape unique :
  P2.25 - reprendre l'optimisation bornee du parcours/cout interne apres
  validation GUI utilisateur.

## Mission courante

Realigner prudemment ZeBlind avec le coeur conceptuel Astrometry.net sur le bloc
amont :

1. generation du code de quad ;
2. canonicalisation AB/C/D ;
3. recherche de candidats en code-space ;
4. construction d'hypothese ;
5. verification finale.

Verdict d'audit courant : ZeBlind n'est pas encore un portage du coeur
Astrometry.net. Le point de divergence prioritaire est le couple `quad code /
candidate lookup` :

- Astrometry.net : code continu 4D AB/C/D dans le repere du backbone AB, puis
  range search KD-tree tolerante.
- ZeBlind actuel : `opposite_edge_ratio_8bit_v1`, hash 3D quantifie de ratios
  d'aretes opposees, vote par tuile, puis recherche/validation aval.

Nuance a garder : ZeBlind a deja une recherche approchee, mais elle opere sur le
mauvais code-space et apres selection de candidat. Elle ne restaure donc pas
l'equivalence Astrometry.net.

## Non-objectifs immediats

- Ne pas toucher a ZeNear.
- Ne pas toucher a la GUI.
- Ne pas toucher au packaging sauf import bloquant.
- Ne pas changer les seuils au hasard.
- Ne pas ajouter de rescue, fallback, retry, scoring ou reranking produit avant
  d'avoir teste le coeur AB/C/D + range search.
- Ne pas supprimer l'ancien backend `opposite_edge_ratio_8bit_v1`.
- Ne pas presenter ZeBlind comme un port Astrometry.net tant que le code-space
  AB/C/D et la range search compatible ne sont pas implementes et verifies.

## Plan actif

### P1. Diagnostic offline, sans changer l'algorithme

- [x] Ajouter des fonctions pures de diagnostic :
  - `astrometry_ab_code_4d(points, order)`
  - canonicalisation compatible Astrometry : inversion AB, `mean(x)<=0.5`,
    ordre C/D.
- [x] Produire un dump comparatif borne pour quelques quads observes/catalogue :
  - hash ZeBlind actuel ;
  - code AB/C/D ;
  - ordre AB/C/D ;
  - distance de code theorique ;
  - bucket exact trouve/non trouve.
- [x] Ajouter un probe offline sur un FITS oracle :
  - comparer lookup exact ZeBlind vs range search AB/C/D ;
  - mesurer a quel etage les bons quads disparaissent.
- [x] Rapport court publie :
  `reports/zeblind_quad_code_diagnostic.md`
- [x] Verdict experimental sur deux cas oracle :
  divergence confirmee entre `opposite_edge_ratio_8bit_v1` et range search
  AB/C/D 4D dans cette enveloppe.

### P2.0 Mini-backend 4D offline en memoire

- [x] Construire un mini-index 4D en memoire, non branche au runtime :
  - codes AB/C/D float ;
  - ids de quads ;
  - `tile_key` ;
  - recherche `cKDTree` avec `code_tol=0.015`.
- [x] Generer les quads image avec le meme code AB/C/D, sans hash ratio dans
  le chemin de recherche.
- [x] Construire des hypotheses WCS directement depuis les hits 4D et mesurer :
  - inliers ;
  - RMS ;
  - scale ratio ;
  - separation centre/corners contre WCS oracle.
- [x] Cas obligatoires traites :
  - `232350 / d50_2823` ;
  - `232102 / d50_2823`.
- [x] Rapport court publie :
  `reports/zeblind_p20_memory_4d_backend_probe.md`
- [x] Verdict P2.0 :
  positif. Sur `232350 / d50_2823`, un hit range 4D perdu par le hash exact
  produit une hypothese WCS plausible au rang `4`, avant le premier hit
  hash-exact plausible au rang `8`. `232102 / d50_2823` reste non-regresse
  dans le probe offline.

### P2. Backend experimental derriere flag

- [x] Introduire un schema explicite, sans supprimer l'ancien :
  `quad_hash_schema="astrometry_ab_code_4d_v1"`.
- [x] Construire un index experimental focalise avec :
  - codes 4D float ;
  - ids de quads ;
  - tile/source ;
  - range search 4D tolerante.
- [x] Ajouter un loader/searcher experimental offline :
  - `cKDTree` ;
  - `code_tol=0.015` explicite ;
  - hits tries par distance ;
  - deduplication image-quad/catalog-record ;
  - caps de cout par hit et par quad image.
- [x] P2.1 valide sur index disque focalise :
  - rapport : `reports/zeblind_p21_disk_4d_index_probe.md` ;
  - index : `reports/p21_astrometry_ab_code_4d_v1_d50_2823_S_stars2000_q40000.npz` ;
  - `232350 / d50_2823` : gain concret reproduit, premier hit plausible
    perdu par hash au rang `90`, avant le premier hash-exact au rang `93` ;
  - `232102 / d50_2823` : non-regresse, premier hit plausible au rang `1`.
- [x] Ajouter une route runtime experimentale dans le solveur principal :
  - observed quads en ordre Astrometry ;
  - range search 4D ;
  - construction WCS directe depuis le hit ;
  - pas de vote tuile avant hypothese dans ce mode.
- [x] Interdire les rescues/fallbacks implicites dans ce mode, sauf flag de
  test explicite.
- [x] Valider P2.2 sur les deux cas oracle avec l'index focalise :
  - rapport : `reports/zeblind_p22_runtime_4d_route.md` ;
  - `232350 / d50_2823` : solution runtime 4D, `171` hits, `93` hypotheses
    testees, premier accepte au rang `93`, `40` inliers, RMS `0.593 px` ;
  - `232102 / d50_2823` : non-regresse, `173` hits, premier accepte au rang
    `1`, `53` inliers, RMS `0.416 px`.
- [x] Conserver l'ancien backend comme defaut flag OFF :
  - `quad_hash_schema` reste `opposite_edge_ratio_8bit_v1` ;
  - aucun chargement 4D n'est tente si le flag/schema 4D n'est pas actif ;
  - tests courts backend historique passes.
- [x] P2.3 isoler le contrat source-list backend 4D :
  - rapport : `reports/zeblind_p23_4d_source_list_contract.md` ;
  - audit A/B/C produit pour `232350 / d50_2823` et `232102 / d50_2823` ;
  - cause fermee : le filtre standard retire des etoiles critiques utilisees
    par les bons quads 4D diagnostic ;
  - `232350` : A accepte au rang `93` avec raw_quad `[9, 15, 7, 24]`,
    B retire le rang brut `24`, hits `171 -> 100`, aucun accepte ;
  - `232102` : A accepte au rang `1` avec raw_quad `[0, 15, 26, 8]`,
    B retire le rang brut `26`, hits `173 -> 86`, aucun accepte ;
  - candidat C `astrometry_like_candidate` audite seulement, non active produit.
- [x] Politique source-list 4D explicite :
  - ajout de `blind_astrometry_4d_source_policy`, defaut `standard_runtime` ;
  - `diagnostic_unfiltered` bypass le filtre source uniquement quand la route
    `astrometry_ab_code_4d_v1` est demandee ;
  - `blind_star_quality_filter` reste `True` dans la validation P2.3 ;
  - ancien backend et flag 4D OFF inchanges.
- [x] Validation P2.3 :
  - rapport : `reports/zeblind_p23_4d_policy_validation.md` ;
  - `232350` resout : `171` hits, premier accepte `93`, `40` inliers,
    RMS `0.593 px` ;
  - `232102` resout : `173` hits, premier accepte `1`, `53` inliers,
    RMS `0.416 px` ;
  - tests courts : `74 passed`.
- [x] P2.4 bake-off politiques source-list 4D :
  - rapport : `reports/zeblind_p24_4d_source_policy_bakeoff.md` ;
  - politiques comparees : `standard_runtime`, `diagnostic_unfiltered`,
    `astrometry_like_candidate` ;
  - cas obligatoires : `232350 / d50_2823`, `232102 / d50_2823` ;
  - optionnels pertinents avec l'index focalise `d50_2823` :
    `232144`, `232205`, `232247` ;
  - `standard_runtime` echoue sur les deux cas obligatoires, et aussi sur
    `232247` ;
  - `diagnostic_unfiltered` resout tous les cas inclus, cout simple et stable ;
  - `astrometry_like_candidate` resout aussi tous les cas inclus, mais cout
    median total environ `1.44x` vs `diagnostic_unfiltered` sur ce bake-off ;
  - decision : utiliser `diagnostic_unfiltered` comme baseline experimentale
    P2.5, garder `astrometry_like_candidate` comme candidate P2.5 bis sans
    promotion produit ;
  - extension ciblee `d50_2822` possible ensuite avec politique source-list
    explicitement fixee ; ne pas lancer all30.
- [x] P2.5 integration produit experimentale ciblee :
  - rapport : `reports/zeblind_p25_4d_experimental_product_slice.md` ;
  - documentation : `docs/zeblind_astrometry_4d_experimental.md` ;
  - probe produit : `tools/diagnose_p25_4d_experimental_product_slice.py` ;
  - activation clarifiee : le backend 4D est selectionne par
    `quad_hash_schema="astrometry_ab_code_4d_v1"` ; le flag index seul ne
    suffit plus et conserve le backend historique ;
  - en mode schema 4D strict, absence de flag/index 4D explicite = echec 4D
    explicite, pas de fallback silencieux vers l'ancien backend ;
  - source policy baseline : `diagnostic_unfiltered`, limitee au backend 4D ;
  - `d50_2823` valide sans regression sur les deux cas obligatoires :
    `232350` accepte rang `93`, `40` inliers, RMS `0.593 px` ;
    `232102` accepte rang `1`, `53` inliers, RMS `0.416 px` ;
  - index cible `d50_2822` construit uniquement avec les memes parametres que
    `d50_2823` : `reports/p25_astrometry_ab_code_4d_v1_d50_2822_S_stars2000_q40000.npz`,
    `40000` entrees, `2000` etoiles ;
  - mini-validation `d50_2822` sur `232144`, `232205`, `232247`, `232329`,
    `232431` : `0/5` succes, donc couverture utile non demontree dans ce
    perimetre ; plusieurs cas ont des supports partiels (`19` a `31` inliers)
    mais restent sous le seuil produit `40` sans changement de seuil ;
  - decision : mode 4D utilisable proprement pour P2.5, ancien backend isole,
    mais ne pas conclure que `d50_2822` apporte une couverture produit avant
    un mini-corpus cible/diagnostic supplementaire. Toujours pas de all30.
- [x] P2.6 diagnostic cible couverture/routage 4D avec oracle Astrometry.net :
  - rapport : `reports/zeblind_p26_4d_oracle_tile_routing.md` ;
  - script : `tools/diagnose_p26_4d_oracle_tile_routing.py` ;
  - cadrage respecte : WCS Astrometry.net utilise seulement pour lire centre,
    coins, footprint approximative et tuiles intersectees ; runtime blind 4D
    lance sur FITS sans WCS, avec index explicite ;
  - oracle coverage : les 7 cas M106 testes ont tous `d50_2823` comme tuile
    principale probable ; `d50_2822` est secondaire/partielle ;
  - `d50_2822` n'est pas une bonne tuile principale pour les 5 echecs P2.5 :
    `1/5` seulement depasse `50%` de footprint (`232431`, ambigu) ;
  - `d50_2823` recupere `3/5` echecs P2.5 :
    `232144` (`52` inliers, RMS `0.695`), `232205` (`52`, `0.608`),
    `232247` (`45`, `0.596`) ;
  - restants :
    `232329 / d50_2823` echoue proche seuil avec meilleur rejet `37` inliers,
    RMS `0.259`; `232431` plafonne a `31` inliers sur `d50_2823` et
    `d50_2822` ;
  - `d50_2821` et `d50_2824` non construits/testes car oracle footprint `0%`
    sur ce perimetre ; pas d'elargissement ;
  - decision : probleme de routage/couverture explique une partie nette
    (`3/5`) mais pas tout ; prochain bloc causal = couverture/catalogue ou
    validation pour `232329`/`232431`, sans tuning ni all30.
- [x] P2.7 diagnostic densite/couverture catalogue sur `d50_2823` :
  - rapport : `reports/zeblind_p27_4d_d50_2823_density_probe.md` ;
  - script : `tools/diagnose_p27_4d_d50_2823_density_probe.py` ;
  - variantes ciblees construites/testees uniquement pour `d50_2823` :
    baseline `2000/40000`, A `3000/80000`, B `4000/120000`,
    schema `astrometry_ab_code_4d_v1`, sampler `catalog_ring_coverage`,
    `code_tol=0.015`, dtype `float32` ;
  - constat catalogue important : la source `d50_2823` disponible ne contient
    que `2000` etoiles reelles, donc A/B augmentent les quads (`80000` /
    `120000`) mais pas la densite etoiles ;
  - `232329` ne passe pas naturellement : meilleur support reste `37` inliers,
    RMS `0.259`, seuil produit `40` conserve ;
  - `232431` ne progresse pas : meilleur support reste `31` inliers, RMS
    `0.333` ;
  - les quads utiles et hits augmentent avec A/B, mais les inliers max ne
    montent pas ; le meilleur compromis cout/succes reste la baseline
    `2000/40000` sur ce probe ;
  - non-regression courte OK : `232144`, `232205`, `232247`, `232350`,
    `232102` resolvent sur les variantes testees (`15/15`) ;
  - decision : ne pas augmenter aveuglement les quads ; prochain front causal =
    vraie profondeur catalogue/source index, union/multi-index bornee ou
    validation, toujours sans baisse de seuil ni all30.
- [x] P2.8 audit validation par support catalogue reel :
  - rapport : `reports/zeblind_p28_4d_validation_support_audit.md` ;
  - script : `tools/diagnose_p28_4d_validation_support_audit.py` ;
  - cadrage respecte : WCS Astrometry.net utilise seulement comme oracle offline
    de tuilage/support, jamais comme entree du solveur blind ;
  - aucun changement ZeNear/GUI/default/seuil/backend-core, aucun tuning, aucun
    all30 ;
  - `232431 / d50_2823` : support mono-tuile insuffisant pour le seuil produit
    `40` (`31` etoiles catalogue strictement dans le champ, `32` avec marge,
    meilleur rejet `31` inliers, RMS `0.333`) ; classer comme solution
    geometrique faible support dans une future validation experimentale, pas
    comme solution produit acceptee ;
  - `232431 / d50_2822` donne aussi `31` inliers ; l'union support-only
    `d50_2823+d50_2822` expose `63` etoiles champ / `61` matchables, donc une
    strategie multi-index bornee est plausible a auditer ;
  - `232329 / d50_2823` : near-miss geometrique (`37` inliers, RMS `0.259`,
    marge `-3`) ; le champ contient `41` etoiles catalogue, mais seulement `39`
    matchables dans les detections brutes et `37` dans la source-list 4D
    gardee ; deux inliers manquants sont dans les detections brutes mais hors
    source-list 4D, le dernier n'est pas un appariement oracle brut au rayon
    `3 px` ;
  - conclusion : `quality_inliers=40` reste le seuil produit strict, mais il est
    trop absolu comme diagnostic des champs pauvres ; distinguer a l'avenir
    `GEOMETRIC_OK_LOW_SUPPORT` / `VALIDATION_NEAR_MISS_LOW_CATALOG_SUPPORT`
    d'une acceptation produit.
- [x] P2.9 audit runtime multi-index borne avec validation union-catalogue :
  - rapport : `reports/zeblind_p29_4d_bounded_multi_index_union_validation.md` ;
  - script : `tools/diagnose_p29_4d_bounded_multi_index_union_validation.py` ;
  - cadrage respecte : deux index explicitement charges (`d50_2823`,
    `d50_2822`), aucune decouverte/construction de `d50_2821`/`d50_2824`,
    WCS Astrometry.net utilise seulement pour l'ordre/footprint diagnostic ;
  - aucun changement ZeNear/GUI/default/seuil/backend-core, aucun rescue global,
    aucun all30, aucune promotion produit ;
  - validation union-catalogue dedupliquee `d50_2823+d50_2822`, rayon matching
    conserve `3 px`, seuils conserves `quality_inliers=40`,
    `quality_rms=1.2` ;
  - `232431` passe naturellement en union : mono `d50_2823` plafonne a
    `31` inliers / RMS `0.333`, mono `d50_2822` a `31` / `0.190`, mais union
    accepte au rang `1` avec `62` inliers / RMS `0.273` ;
  - `232329` passe naturellement en union : mono `d50_2823` reste a
    `37` inliers / RMS `0.259`, mono `d50_2822` a `20` / `0.306`, premiere
    acceptation union au rang `3` avec `57` inliers / RMS `0.959`, meilleur
    union `57` / `0.300` ;
  - non-regression courte : les cinq cas de comparaison ont aussi une validation
    union `ACCEPTED_PRODUCT_THRESHOLD` ;
  - conclusion causale : les deux echecs restants etaient dus a la validation
    mono-tuile / support catalogue fragmente, pas a un echec geometrique du
    backend 4D ; prochaine etape possible = mini-corpus M106 multi-index borne,
    toujours sans all30 et sans activation produit par defaut.
- [x] P2.10 mini-corpus M106 multi-index borne et politique d'acceptation :
  - rapport : `reports/zeblind_p210_4d_m106_bounded_multi_index_corpus.md` ;
  - script : `tools/diagnose_p210_4d_m106_bounded_multi_index_corpus.py` ;
  - interface experimentale preparee, OFF par defaut :
    `blind_astrometry_4d_index_paths`, `blind_astrometry_4d_validation_catalog_policy`,
    `blind_astrometry_4d_accept_policy`, `blind_astrometry_4d_max_accepts` ;
  - runtime probe sans WCS oracle : liste explicite d'index `d50_2823`,
    `d50_2822`, validation union-catalogue dedupliquee, seuils conserves
    `quality_inliers=40`, `quality_rms=1.2`, `match_radius_px=3.0` ;
  - mini-corpus teste : 7 cas P2.9 + extras bornes `232513`, `232534`,
    `232658` (`10/30`, pas de all30) ;
  - `first_accept` resout `10/10`, `best_within_budget` resout `10/10` ;
  - `232329` reste recupere : first accept `57` inliers / RMS `0.959`, best
    within budget `57` / `0.300` ;
  - `232431` reste recupere : `62` inliers / RMS `0.273` des le rang `1` ;
  - `best_within_budget` apporte un gain RMS sur `9/10` cas, donc utile comme
    politique de qualite, tandis que `first_accept` suffit pour le succes brut ;
  - controle mauvaise liste `d50_2822` seule sur 5 cas a footprint <= `40%` :
    `0/5` acceptation, pas de faux positif evident dans ce controle ;
  - cout median mesure : environ `11.2s` total, `4.3s` validation ;
  - decision : strategie propre comme option experimentale utilisateur, mais ne
    pas promouvoir produit avant corpus borne plus large, controles faux
    positifs plus durs et contrat de validation separe.
- [x] P2.11 hardening runtime 4D multi-index dans le solveur principal :
  - rapport : `reports/zeblind_p211_4d_experimental_runtime_hardening.md` ;
  - script : `tools/diagnose_p211_4d_experimental_runtime_hardening.py` ;
  - `solve_blind` consomme maintenant explicitement
    `blind_astrometry_4d_index_paths`,
    `blind_astrometry_4d_validation_catalog_policy`,
    `blind_astrometry_4d_accept_policy` et
    `blind_astrometry_4d_max_accepts` uniquement quand
    `quad_hash_schema="astrometry_ab_code_4d_v1"` ;
  - validation union-catalogue `union_candidate_tiles` limitee au backend 4D,
    avec dedup catalogue et seuils conserves (`quality_inliers=40`,
    `quality_rms=1.2`, `match_radius_px=3.0`) ;
  - policies supportees : `first_accept` et `best_within_budget`, ce dernier
    borné par `max_hypotheses`, `blind_global_hard_budget_s` et
    `blind_astrometry_4d_max_accepts` ;
  - replay P2.10 via solveur principal : `10/10` sur M106 borne, dont
    `232329` a `57` inliers / RMS `0.300` et `232431` a `62` / `0.273` ;
  - controle mauvaise liste `d50_2822` seule : `0/5` acceptation ;
  - tests courts : `79 passed` ;
  - decision : mode pret comme option experimentale utilisateur, toujours OFF
    par defaut, sans promotion produit ni fallback silencieux.
- [x] P2.12 validation elargie M106 bornee, faux positifs renforces et budget :
  - rapport : `reports/zeblind_p212_4d_m106_30_bounded_validation.md` ;
  - script : `tools/diagnose_p212_4d_m106_30_bounded_validation.py` ;
  - runtime `solve_blind`, FITS copies sans WCS, liste explicite
    `d50_2823+d50_2822`, aucun WCS Astrometry.net comme entree runtime ;
  - aucun changement ZeNear/GUI/default/seuil, aucun all-sky, aucun rebuild,
    aucune decouverte/construction d'index runtime ;
  - corpus M106 local elargi : `30` images ;
  - `first_accept` : `28/30` ;
  - `best_within_budget` : `28/30` ;
  - les 10 cas P2.10/P2.11 restent valides (`10/10`) ;
  - `best_within_budget` ameliore le RMS sur `27/30` cas, sans degradation RMS ;
  - echecs restants :
    - `233828` : meilleur rejet `37` inliers / RMS `0.942`, scale OK,
      seuil inliers `40` non atteint ;
    - `234013` : meilleur rejet `28` inliers / RMS `0.910`, scale OK,
      seuil inliers `40` non atteint ;
  - controles faux positifs :
    - mauvaise liste `d50_2822` seule sur low-footprint : `0/10`
      acceptation ;
    - ordre inverse `[d50_2822, d50_2823]` sur controles prioritaires :
      `10/10`, aucune degradation RMS > `0.25 px` ;
    - listes partielles : `d50_2823` seule `5/10`, `d50_2822` seule `1/10`
      (le succes `232658` est coherent avec la couverture, pas un controle
      low-footprint) ;
    - index absent/incompatible : erreurs explicites, pas de fallback.
  - cout mesure `best_within_budget` : median `19.220s`, p95 `23.617s`,
    validation mediane `7.621s`, aucun hit `max_wall_s`, `12` cas touchent
    `max_accepts` ;
  - decision : option experimentale utilisateur credible mais P2.12 reste
    partiel ; avant promotion/elargissement, diagnostiquer `233828` et
    `234013` (support/source-list/couverture), sans baisser les seuils.
- [x] P2.13 autopsie ciblee des deux echecs M106 :
  - rapport : `reports/zeblind_p213_4d_m106_failure_autopsy.md` ;
  - script : `tools/diagnose_p213_4d_m106_failure_autopsy.py` ;
  - diagnostic uniquement : aucun seuil change, aucun all-sky, aucune tuile
    construite, aucun changement ZeNear/GUI/default/backend ;
  - WCS Astrometry.net utilise seulement comme oracle offline ; runtime blind
    toujours sur copies FITS sans WCS et index explicites ;
  - `233828` :
    - meilleur rejet union : `37` inliers / RMS `0.942`, scale OK ;
    - footprint oracle : `d50_2823` `87.1%`, `d50_2822` `29.0%` ;
    - support union : `55` etoiles brutes matchables mais `37` seulement dans
      la source-list 4D gardee ;
    - conclusion : near-miss, bloc causal `source_list_selection_drops_matchable_stars`.
  - `234013` :
    - meilleur rejet union : `28` inliers / RMS `0.910`, scale OK ;
    - footprint oracle : `d50_2822` `100%`, `d50_2725` seulement `3.2%`,
      `d50_2823` marginal ;
    - support union : `42` etoiles brutes matchables mais `28` seulement dans
      la source-list 4D gardee ;
    - conclusion : pas mauvais routage principal ; bloc causal
      `source_list_selection_drops_matchable_stars`.
  - direction suivante unique : auditer/rendre plus stable le ranking
    source-list 4D pour conserver davantage d'etoiles oracle matchables dans
    les `120` sources, sans augmenter les quads ni baisser
    `quality_inliers=40`.
- [x] P2.14 bake-off source-list 4D :
  - rapport : `reports/zeblind_p214_4d_source_policy_bakeoff.md` ;
  - script : `tools/diagnose_p214_4d_source_policy_bakeoff.py` ;
  - diagnostic uniquement : aucun seuil change, aucun all-sky, aucun rebuild,
    aucun changement ZeNear/GUI/default ;
  - WCS Astrometry.net utilise seulement comme oracle offline de retention ;
    runtime `solve_blind` sur copies FITS sans WCS avec index explicites ;
  - politiques testees sur les deux echecs avant elargissement :
    baseline cap 120, caps simples 160/200/250, grilles 4x4 et 6x4 caps
    160/200, stratification flux/qscore caps 160/200, astrometry-like cap 200 ;
  - `233828` est recuperable sans baisse de seuil :
    `head_cap200` atteint `40` inliers / RMS `0.939`, `head_cap250` `43`,
    `grid4x4_cap160` `41`, `grid6x4_cap160` `42`,
    `grid6x4_cap200` `44`, `astrometry_like_cap200` `42` ;
  - `234013` ne passe avec aucune politique raisonnable testee :
    head 160/200/250 et astrometry-like 200 restent a `28` inliers ;
    grilles/stratification font pareil ou pire ;
  - retention oracle `234013` bloquee a `28/42` matchables pour head
    120/160/200/250 et astrometry-like 200 : le probleme n'est pas un simple
    cap <=250 ;
  - stop criterion active : pas de run 30/30 ni controles faux positifs
    complets apres le focus negatif sur `234013` ;
  - decision : aucune source policy promouvable P2.14 ; meilleure politique
    partielle seulement `grid4x4_cap160` ;
  - prochaine direction unique : auditer les matchables perdues de `234013`
    au-dela du top 250 / source-list superieure a 250 en diagnostic seulement,
    ou revoir le ranking detecteur plus profond, sans promotion produit.
- [x] P2.15 separation `quad_sources` / `verification_sources` 4D :
  - rapport : `reports/zeblind_p215_4d_split_quad_verify_sources.md` ;
  - script : `tools/diagnose_p215_4d_split_quad_verify_sources.py` ;
  - patch runtime borne : la route 4D accepte une liste de validation separee
    uniquement via `prep_cache`; aucun changement ZeNear/GUI/default ;
  - configs testees sur `234013` + controles `233828`, `233705`, `233644`,
    `233602`, `233520` : `quad_cap=120/200`, verification `250/500/1000/full` ;
  - `234013` ne passe pas :
    - coherent max `q120_v500` = `30` inliers / RMS `0.974` ;
    - full verification contient les `42` etoiles oracle-matchables, mais les
      compteurs hauts (`q120_vfull` `215`) sont des rejets hors echelle/RMS
      invalide, pas des solutions ;
  - `233828` passe grace a la separation des listes :
    `q120_v250` = `43` inliers / RMS `1.102` ;
  - controles sains : les cinq autres cas passent sur toutes les configs, sans
    faux positif observe ; pas de controle mauvaise tuile post-passage car
    `234013` ne passe jamais ;
  - cout median : validation `5.590s`, total solveur `7.331s` ;
  - decision : la separation reproduit mieux Astrometry.net et corrige certains
    plafonds, mais le gap principal `234013` n'est pas resolu par les caps ;
    prochaine etape = auditer centroïdes/matching/residus des etoiles restantes
    avant comparaison detecteur `image2xy`.
- [x] P2.16 audit residuel et raffinement iteratif du WCS candidat 4D :
  - rapport : `reports/zeblind_p216_4d_candidate_refine_residual_audit.md` ;
  - script : `tools/diagnose_p216_4d_candidate_refine_residual_audit.py` ;
  - JSON : `reports/zeblind_p216_4d_candidate_refine_residual_audit.json` ;
  - diagnostic uniquement : aucun changement ZeNear/GUI/default, aucun seuil
    change, aucun all-sky, aucun WCS oracle runtime, aucun changement du coeur
    AB/C/D ni ranking source-list ;
  - candidat `234013` isole exactement :
    origine `d50_2822`, rang replay `95`, image quad `[5,3,4,0]`,
    catalogue quad `[196,54,175,7]`, echelle `2.373471"/px` ;
  - resultat cle : la premisse "12 etoiles au-dela de 3 px" est infirmee sur
    la liste complete ; les `42/42` etoiles oracle-matchables sont deja sous
    `3 px` avec le WCS quad initial (`42` inliers / RMS `1.102`) ;
  - le refit TAN strict ameliore seulement la qualite (`43` inliers /
    RMS `0.992`) ; rayon de collecte temporaire `4.5/5 px` non necessaire ;
  - matching glouton, mutual nearest-neighbour et biparti donnent le meme
    support initial (`42`) ; pas de perte par conflit d'unicite ;
  - SIP ordre 2 non necessaire, car TAN/initial passe deja les seuils ;
  - controles runtime bornes `q120_v500` :
    `233828=47/1.147`, `233705=58/0.656`, `233644=60/0.739`,
    `233602=58/0.464`, `233520=61/0.420` ;
  - controle mauvaise tuile : `234013` avec `d50_2823` seule ne passe pas
    malgre `121` matches bruts, rejet hors echelle (`scale=26.056`) ;
  - conclusion causale : le gap restant n'est pas l'absence de tweak/refine,
    mais l'absence d'isolation/telemetrie du meilleur candidat coherent avec
    la liste complete, masque par des rejets hors echelle dans P2.15.
- [x] P2.17 audit parite runtime/replay du candidat 4D :
  - rapport : `reports/zeblind_p217_4d_runtime_replay_parity.md` ;
  - script : `tools/diagnose_p217_4d_runtime_replay_parity.py` ;
  - JSON : `reports/zeblind_p217_4d_runtime_replay_parity.json` ;
  - matrice controles : `reports/zeblind_p217_runtime_validation_matrix.json` ;
  - cause racine fermee : les entrees du rang `95` sont identiques entre
    runtime et replay (`same_candidate_identity`, `same_wcs`,
    `same_verification_source_hash`, `same_catalog_hash`, `same_match_pairs`
    tous vrais) ;
  - la divergence P2.15 venait de la metrique de validation runtime 4D :
    le replay validait le RMS pixel direct `catalogue world2pix -> image`,
    tandis que la route runtime repassait les memes paires dans
    `validate_solution` (`image pix2world -> catalogue`) ;
  - sur les memes `42` paires du candidat `234013` : metrique directe
    `42` inliers / RMS `1.102`, ancienne metrique inverse `41` / RMS `1.203`,
    donc rejet de justesse a seuil inchange ;
  - patch borne route 4D : validation finale sur les distances pixel deja
    collectees par le matching 4D, avec l'ancienne validation inverse conservee
    en telemetrie `legacy_inverse_*` ;
  - `solve_blind` recupere maintenant naturellement `234013` en `q120_vfull`,
    rang selectionne `95`, `42` inliers / RMS `1.102`, sans refit ni rescue ;
  - controles `q120_vfull` passes : `233828=55/0.934`,
    `233705=64/0.593`, `233644=64/0.664`, `233602=66/0.534`,
    `233520=61/0.379` ;
  - mauvaise tuile `d50_2823` seule reste rejetee :
    `121` matches, RMS `2.102`, echelle invalide `26.051"/px` ;
  - prochaine etape possible : rejouer le mode sur les 30 M106, maintenant que
    la parite locale du candidat rang `95` est prouvee.
- [x] P2.18 cloture M106 all30 + hardening contrat metrique directe :
  - rapport : `reports/zeblind_p218_4d_m106_all30_direct_metric_closure.md` ;
  - script : `tools/diagnose_p218_4d_m106_all30_direct_metric_closure.py` ;
  - JSON : `reports/zeblind_p218_4d_m106_all30_direct_metric_closure.json` ;
  - tests de contrat ajoutes dans `tests/test_quad_code_diagnostic.py` :
    metrique finale basee sur distances pixel directes, pas de rematch inverse,
    invariance a l'ordre des paires, `legacy_inverse_*` sans effet sur sorting
    d'acceptation, cas frontiere `234013` direct accepte / legacy rejete ;
  - validation tests : `pytest -q tests/test_quad_code_diagnostic.py
    tests/test_zeblindsolver.py` => `72 passed` ;
  - all30 M106 via `solve_blind`, `q120_vfull`, index
    `[d50_2823,d50_2822]`, `union_candidate_tiles`,
    `best_within_budget`, seuils conserves : `30/30` ;
  - `234013` passe naturellement : rang `95`, `42` inliers /
    RMS direct `1.102`, legacy inverse `41` / `1.203` ;
  - controles negatifs : `0/8` acceptations, incluant `d50_2822` seule sur
    cinq low-footprint, `d50_2823` seule sur `234013`, index absent,
    catalogue incompatible/shuffle ;
  - ordre inverse `[d50_2822,d50_2823]` : `30/30` ;
  - garde policy `first_accept` sur `234013` : direct accepte malgre
    `legacy_inverse_quality=FAIL` ;
  - stats direct/legacy all30 : decision differente uniquement sur `234013`,
    delta RMS inverse-direct median `0.085`, p95 `0.195`, max `0.235`,
    aucun cas ou direct est nettement pire (> `0.05 px`) ;
  - cout `q120_vfull` mesure : median total `7.559s`, p95 `11.415s`,
    median validation `5.390s`, p95 `9.208s`, median hypotheses `184`,
    p95 `301`, max `verification_sources=5000`, aucun `max_wall_s` ;
  - statut : backend 4D experimental release candidate sur M106, backend
    historique toujours inchange/OFF par defaut ;
  - prochaine limite produit : elargir la non-regression hors champ M106 et
    indices voisins avant promotion produit plus large.
- [x] P2.19 validation multi-champs bornee :
  - rapport : `reports/zeblind_p219_4d_multifield_validation.md` ;
  - inventaire : `reports/zeblind_p219_local_field_inventory.json` ;
  - corpus selectionne :
    `reports/zeblind_p219_selected_multifield_corpus.json` ;
  - JSON complet :
    `reports/zeblind_p219_4d_multifield_validation.json` ;
  - script : `tools/diagnose_p219_4d_multifield_validation.py` ;
  - contexte local : seul champ FITS hors M106 exploitable dans les chemins
    bornes = `NGC6888` (`10` images dans
    `/home/tristan/zemosaic/example/astap solved`) ; les champs
    `NGC3628`, `NGC6823`, `IC1848` sont seulement references par d'anciens
    rapports mais les FITS `/home/tristan/zemosaic/example/fresh` sont absents ;
  - deux index compacts P2.19 generes a partir du manifest local existant,
    bornes aux tuiles `d50_2644` et `d50_2645` ;
  - baseline `solve_blind`, `q120_vfull`, `union_candidate_tiles`,
    `best_within_budget`, seuils inchanges : `NGC6888=10/10` ;
  - stats NGC6888 : inliers `59-63`, RMS max `0.492`, median total
    `3.224s`, p95 total `3.393s`, median validation `1.443s`, aucun
    `max_wall_s` ;
  - WCS offline ASTAP : `0` faux positif, toutes les solutions acceptees sont
    coherentes avec le WCS de reference ;
  - controles : mauvais index M106 rejete, index absent explicite, format index
    incompatible rejete, image M106 avec index NGC6888 rejetee, ordre inverse
    `[d50_2645,d50_2644]` = `10/10` ;
  - direct vs legacy hors M106 : aucune decision differente, delta RMS
    inverse-direct median `0.037`, p95 `0.067`; un cas ou direct est pire de
    plus de `0.05 px` mais reste tres sous seuil (`RMS direct 0.453`) ;
  - comparaison historique informative sur 3 images : historique passe aussi,
    mais plus lent (~`11s`) et avec moins d'inliers (`14-19`) que le 4D
    (`59-61`, ~`3s`) ;
  - tests : `pytest -q tests/test_quad_code_diagnostic.py
    tests/test_zeblindsolver.py` => `76 passed` ; `py_compile` OK ;
  - verdict : generalisation partielle positive, pas RC multi-champs general
    car seulement `1` champ hors M106 disponible localement ;
  - prochaine direction unique : fournir/selectionner `3-5` champs FITS
    supplementaires avec WCS offline, construire `1-2` index compacts bornes
    par champ, puis relancer exactement le probe P2.19 avant routage produit.
- [x] P2.19b extension incrementale multi-instruments / multi-montures :
  - [x] nouveau corpus borne a
    `/home/tristan/zemosaic/example/various_fresh` identifie (`15` FITS) ;
  - [x] script dedie ajoute :
    `tools/diagnose_p219b_incremental_multiregime.py` ;
  - [x] plages d'echelle figees avant run :
    `s50=1.90..2.85"/px`, `s30=3.19..4.79"/px`,
    `c11=0.16..0.27"/px` ;
  - [x] temoins historiques reduits M106/NGC6888 executes en premier :
    `4/4` (`234013`, `232205`, NGC6888 rang faible, NGC6888 rang eleve) ;
  - [x] baseline nouvelle M31 S50/S30 avec vrais index 4D compacts :
    `9/9`, `0` faux positif offline, S30 `3/3` ;
  - [x] controles negatifs cibles : mauvais champs rejetes, index absent et
    format incompatible explicites, plage S30 avec fenetre S50 rejetee,
    ordre inverse accepte, index secondaire seul accepte pour le cadrage nord ;
  - [x] livrables :
    `reports/zeblind_p219b_new_corpus_inventory.json`,
    `reports/zeblind_p219b_selected_corpus.json`,
    `reports/zeblind_p219b_incremental_validation.json`,
    `reports/zeblind_p219b_incremental_validation.md` ;
  - verdict : `A - Extension positive` sur les regimes exploitables
    (M31 S50/S30 + temoins M106/NGC6888) ;
  - limite : C11/ASI462 inventorie mais non valide, faute de WCS offline
    fiable dans cette iteration ;
  - hygiene runtime corrigee et verifiee : copies de travail sans WCS ni
    champs `RA/DEC/OBJCTRA/...` avant `solve_blind` ;
  - contrainte : ne pas toucher a ZeNear, GUI, backend par defaut, coeur AB/C/D
    ou seuils qualite.
- [x] P2.20 manifest d'index 4D et pool celeste mixte :
  - [x] baseline figee sur commit `60676f7af6bbb16b96063086b129fcf707df48d3` ;
  - [x] manifest experimental versionne cree :
    `reports/zeblind_p220_mixed_index_manifest.json` ;
  - [x] chargeur strict ajoute dans
    `tools/diagnose_p220_4d_mixed_index_pool.py` :
    schema/version, chemins, SHA-256, metadata NPZ, doublons, entrees
    disabled, erreurs explicites sans fallback ;
  - [x] corpus borne P2.20 :
    `reports/zeblind_p220_selected_mixed_pool_corpus.json` (`15` images :
    M106 `4`, NGC6888 `4`, M31 `7`, S50/S30, Alt-Az/EQ) ;
  - [x] hygiene runtime : copies de travail sans WCS, sans hints RA/Dec et
    sans champs d'identite de cible avant `solve_blind` ;
  - [x] baseline specialisee : `15/15` ;
  - [x] pool mixte commun de six index pour toutes les images, ordre A :
    `15/15`, `0` faux positif offline, `0` mauvaise region acceptee ;
  - [x] ordres deterministes A/B/C/D : decisions stables, tous `15/15`,
    aucun `max_wall_s`, aucun `max_hypotheses` limitant ;
  - [x] cas frontiere M106 `234013` conserve en pool mixte :
    `42` inliers / RMS `1.102`, tuile `d50_2822`, rang `95` en ordre A
    et succes conserve dans les quatre ordres ;
  - [x] controles negatifs manifest : absences, JSON invalide, version,
    checksum, schema, metadata contradictoire, doublons et disabled OK ;
  - [x] controles negatifs celestes : mauvais pools et mauvaise plage S30/S50
    rejetes sans faux accept ;
  - [x] cout mesure : ralentissement median pool/specialise `2.91x`,
    p95 `3.55x`, max `3.57x`; validation full reste dominante ;
  - [x] budget parasite principal ordre A :
    `d50_2823_S_q40000` (`1294` hypotheses testees) ;
  - [x] `max_accepts=64` atteint sur quelques cas, mais sans effet de
    decision ; `max_hypotheses=2000` et `max_wall_s=45` non atteints ;
  - [x] tests :
    `pytest -q tests/test_quad_code_diagnostic.py tests/test_zeblindsolver.py
    tests/test_p220_manifest_loader.py` => `84 passed, 1 warning` ;
    `python -m py_compile zeblindsolver/zeblindsolver.py
    tools/diagnose_p220_4d_mixed_index_pool.py` OK ;
  - [x] livrables :
    `reports/zeblind_p220_baseline.json`,
    `reports/zeblind_p220_mixed_index_manifest.json`,
    `reports/zeblind_p220_selected_mixed_pool_corpus.json`,
    `reports/zeblind_p220_mixed_pool_validation.json`,
    `reports/zeblind_p220_mixed_pool_validation.md`,
    `tools/diagnose_p220_4d_mixed_index_pool.py` ;
  - verdict : `A - Pool mixte valide` ;
  - prochaine etape unique : creer le preset `zeblind_4d_experimental` et
    effectuer un test in situ via le chemin applicatif reel.
- [x] P2.21 promotion applicative du backend ZeBlind 4D experimental :
  - [x] baseline figee sur commit `60676f7af6bbb16b96063086b129fcf707df48d3`
    avec statut Git et SHA-256 des six index dans
    `reports/zeblind_p221_baseline.json` ;
  - [x] chargeur strict integre au package :
    `zeblindsolver/index_manifest_4d.py` (`schema`, version, chemins
    relatifs, SHA-256, metadata NPZ, doublons, disabled, erreurs typees) ;
  - [x] manifest runtime portable cree :
    `config/zeblind_4d_experimental_manifest.json`, chemins relatifs vers
    `indexes/astrometry_4d/*.npz`, sans dependance `/home/tristan/...` ;
  - [x] profil centralise ajoute :
    `zeblindsolver/profiles.py`, `zeblind_4d_experimental`, contrat P2.20
    exact (`quad_sources=120`, `validation=full`, `union_candidate_tiles`,
    `best_within_budget`, seuils `40/1.2/3.0`, budgets `2500/2000/64/45s`) ;
  - [x] settings persistants ajoutes avec defaut historique :
    `blind_backend_profile="historical"`,
    `blind_4d_manifest_path=None`; ancienne config => historique ;
  - [x] CLI applicatif reel cable :
    `--blind-profile historical|zeblind_4d_experimental`,
    `--blind-4d-manifest`, `--blind-only`; manifest obligatoire en 4D,
    aucun fallback historique en cas d'erreur manifest ;
  - [x] helper centralise `build_blind_solve_config` utilise pour appliquer le
    profil 4D et transmettre les six index explicitement a `solve_blind` ;
  - [x] garde interne ajoutee dans `solve_blind` : le schema 4D avec chemins
    explicites ne requiert plus l'ancien `index_root/manifest.json`, et les
    RA/Dec du header ne filtrent pas le pool 4D ;
  - [x] presets corriges : S50 `~250 mm`, `1.90..2.85"/px`; S30 `~150 mm`,
    `3.19..4.79"/px` ;
  - [x] outil in situ ajoute :
    `tools/diagnose_p221_app_integration.py` ;
  - [x] validation applicative reelle :
    `7/7` sur M106/NGC6888/M31, S50/S30, Alt-Az/EQ, copies FITS sans WCS ni
    hints RA/Dec/OBJECT, WCS ecrit et relisible, `0` faux positif offline,
    `0` echec d'ecriture WCS ;
  - [x] cas frontiere `234013` conserve en chemin CLI/applicatif :
    tuile `d50_2822`, `47` inliers, RMS direct `1.106` ;
  - [x] controles : manifest absent explicite sans fallback, backend
    historique preserve et toujours par defaut ;
  - [x] livrables :
    `reports/zeblind_p221_baseline.json`,
    `reports/zeblind_p221_app_integration_validation.json`,
    `reports/zeblind_p221_app_integration_validation.md`,
    `docs/zeblind_astrometry_4d_experimental.md` ;
  - [x] tests :
    diagnostic P2.21 => verdict `A`, `7/7` ;
    `pytest -q tests/test_p221_app_integration.py
    tests/test_p220_manifest_loader.py tests/test_presets.py
    tests/test_settings_persistence.py tests/test_quad_code_diagnostic.py
    tests/test_zeblindsolver.py` => `91 passed, 1 skipped, 1 warning` ;
    `python -m py_compile zesolver.py zeblindsolver/zeblindsolver.py
    zeblindsolver/index_manifest_4d.py tools/diagnose_p221_app_integration.py`
    OK ;
  - limite test : `pytest -q` global non exploitable avec le Python systeme
    courant car `astroalign` manque ; le venv applicatif a `astroalign` mais
    pas `pytest` ;
  - verdict : `A - Integration applicative validee` ;
  - prochaine etape unique : integration GUI minimale et test utilisateur in
    situ.
- [x] P2.22 integration GUI minimale ZeBlind 4D experimental :
  - [x] Easy/Wizard : case indentee `Utiliser ZeBlind 4D experimental` sous
    `Activer le blind solver`, controle desactive quand le blind global est
    decoche, note de couverture limitee ;
  - [x] Expert : combo `Profil ZeBlind` (`Historique` /
    `ZeBlind 4D experimental`), champ `Manifest 4D`, boutons
    parcourir/verifier, statut de validation ;
  - [x] meme etat persistant sous-jacent pour toutes les vues :
    `blind_backend_profile`, `blind_4d_manifest_path`, `blind_enabled`,
    `interface_mode`; ancienne config => `historical` ;
  - [x] resolver central du manifest par defaut ajoute :
    `zesolver/blind4d_runtime.py`, sans scan de dossier ni dependance
    `/home/tristan/...` ;
  - [x] preflight GUI avant batch : manifest 4D charge une fois avant le
    premier FITS, erreur explicite sans fallback historique ;
  - [x] chainage affiche/logue :
    `ZeNear -> ZeBlind historique -> Astrometry*` ou
    `ZeNear -> ZeBlind 4D experimental -> Astrometry*` ;
  - [x] traductions FR/EN ajoutees pour les controles 4D, statuts manifest et
    messages d'erreur ;
  - [x] test in situ par chemin applicatif/GUI headless :
    `4/4` sur `234013`, NGC6888, M31 S50, M31 S30 ; WCS ecrit/relu `4/4`,
    faux positifs offline `0`, echecs WCS `0` ;
  - [x] `234013` conserve : tuile `d50_2822`, `47` inliers, RMS `1.106 px` ;
  - [x] controles : manifest absent bloque avant solve, backend historique
    reste par defaut et ne charge pas le manifest, mode simple ne remplace pas
    le profil 4D ;
  - [x] Stop : route `request_cancel` -> `cancel_check` preservee jusque dans
    le backend 4D ; clic long interactif a repeter pendant la campagne
    utilisateur ;
  - [x] livrables :
    `reports/zeblind_p222_gui_baseline.json`,
    `reports/zeblind_p222_gui_integration_validation.json`,
    `reports/zeblind_p222_gui_integration_validation.md`,
    doc utilisateur 4D mise a jour ;
  - [x] tests :
    system Python cible => `96 passed, 2 skipped, 1 warning` ;
    venv + Qt offscreen cible => `98 passed, 1 warning` ;
    `py_compile` OK ;
    global venv => `128 passed, 1 skipped, 4 failed` sur tests
    historiques/environnementaux non lies (`database` absente, fixture
    download SHA, synthetic Near/CUDA) ;
  - verdict : `A - GUI experimental valide` ;
  - prochaine etape unique : campagne utilisateur grandeur nature et collecte
    structuree des echecs.

### P3. Refactor seulement si P2 confirme

- [ ] Separer clairement deux backends :
  - `ratio_bucket_v1` ;
  - `astrometry_ab_code_4d_v1`.
- [ ] Faire du backend Astrometry-like le chemin canonique du mode strict
  seulement apres preuve oracle.
- [ ] Rebrancher scoring/rerank/rescue uniquement apres :
  - oracle positif ;
  - oracle negatif ;
  - non-regression ZeNear ;
  - mesure runtime/memoire.

## Tests obligatoires avant patch produit

- [ ] Unit : `astrometry_ab_code_4d` reproduit les equations Astrometry sur une
  geometrie simple.
- [ ] Unit : inversion AB transforme le code en `1-code` et conserve les
  invariants apres canonicalisation.
- [ ] Unit : permutation C/D respecte `cx<=dx` et `mean(x)<=0.5`.
- [ ] Unit : parite normale/flipped reproduit `quad_flip_parity()` sans passer
  par `hash ^ 1`.
- [ ] Unit : deux codes proches mais dans des buckets differents sont rates par
  lookup exact et retrouves par range search.
- [x] Tests diagnostic ajoutes et passes :
  `python -m pytest -q tests/test_quad_code_diagnostic.py`
- [ ] Script court : mini-index AB/C/D en memoire, interrogation d'un quad
  observe bruite, voisin catalogue retrouve.
- [ ] Oracle FITS : cas echouant ZeBlind mais passant Astrometry.net, avec dump
  des codes AB/C/D, voisins range, hit WCS et verification minimale.
- [ ] Regression : l'ancien backend `opposite_edge_ratio_8bit_v1` reste
  chargeable et testable quand le nouveau flag est inactif.
- [x] Regression P2.5 : route 4D OFF par defaut et active uniquement par schema
  `astrometry_ab_code_4d_v1`; le flag/index 4D seul ne change pas le backend
  historique.
- [x] Probe P2.5 : `python tools/diagnose_p25_4d_experimental_product_slice.py --rebuild-2822`

## Commandes de validation courtes

- Changement hash / quads / index :
  `python -m pytest -q tests/test_zeblindsolver.py tests/test_synthetic.py tests/test_quad_storage.py`
- Changement Near / metadata solver :
  `python -m pytest -q tests/test_metadata_solver.py`
- Validation lourde :
  reserver `test_real_s50.py`, `testzenear`, probes oracle et runs longs aux
  jalons ou aux changements touchant directement leur surface.

## 2026-07-15 — ZN3.1 oracle ASTAP et portes image/catalogue

- [x] Instrumentation ASTAP opt-in enrichie dans le CLI local, toujours pilotee
  par `ASTAP_ZN2_DUMP_DIR` et compilee par
  `lazbuild -B ASTAP-main/command-line_version/astap_command_line_linux.lpi`.
- [x] Nouveaux dumps oracle:
  `_astap_binned.fits`, `_astap_binned_metadata.json`,
  `_astap_background.json`, `_astap_detection_passes.csv`,
  `_astap_raw_image_candidates.csv`, `_astap_ranked_image_stars.csv`,
  `_astap_catalog_raw.csv`, `_astap_catalog_projected.csv`,
  `_astap_catalog_ranked.csv`.
- [x] Baseline preservee: ASTAP instrumente `-z 2` reste `8/8`; O11 listes
  ASTAP reste `8/8`; ZeNear actuel reste `0/8`.
- [x] Parite binned M31 prouvee: image binned Python brute == ASTAP pixel a
  pixel (`MAE=0`, `max=0`, `corr=1.0` sur `230409`).
- [x] Porte image isolee:
  - IMG-O0/O11 succes ;
  - IMG-B1 et IMG-B2 identiques cote binning/detection Python (`63` etoiles
    sur `230409`, `NO_SIGNATURE_MATCHES`) ;
  - IMG-D1 avec candidats ASTAP produit le fit oracle (`53` matches raw,
    `50` refs sur `230409`) ;
  - IMG-P Python complet reste `NO_SIGNATURE_MATCHES`.
- [x] Porte catalogue isolee:
  - CAT-O0/CAT-R1 succes oracle ;
  - CAT-P catalogue ZeNear actuel reste `NO_SIGNATURE_MATCHES` avec environ
    `249` etoiles contre `448` ASTAP sur `230409`.
- [x] Diagnostic catalogue RA/Dec: ASTAP dump maintenant RA/Dec/mag/tile ; le
  dump ZeNear ZN1 montre une RA brute incompatible (`~159 deg`) et meme avec
  normalisation diagnostique `/15`, aucun recouvrement a `2"` sur `230409`.
- [x] Rapports principaux:
  `reports/zenear_zn31_oracle_completion.md`,
  `reports/zenear_zn31_*`.
- [x] Outils ajoutes:
  `tools/diagnose_zn31_astap_oracles.py`,
  `tools/diagnose_zn31_image_gate.py`,
  `tools/diagnose_zn31_catalog_gate.py`.
- [x] Tests: `pytest -q tests/test_zn2_astap_tools.py tests/test_zn3_input_lists.py tests/test_zn31_astap_oracles.py`
  => `17 passed`; `py_compile` OK.
- Verdict: `D - causes precises identifiees, builders encore incomplets`.
- Prochaine etape unique: reproduire la detection image ASTAP adaptative
  (`30*sigma` global puis `7*sigma` local en sections) avant toute reprise P11 ;
  en parallele, ajouter un dump ZeNear tile/order RA normalisee pour isoler la
  selection catalogue D50.

## 2026-07-15 — ZN3.2-CAT audit causal catalogue D50

- [x] Mission gardee strictement catalogue : image ASTAP oracle injectee, pas de
  changement builder image/binning/detection, pas de changement quads/signatures/
  lookup/transformation/rescue/seuils.
- [x] Baseline reproduite :
  - CAT-O0 image ASTAP + catalogue ASTAP : `8/8` ;
  - CAT-P0 image ASTAP + catalogue ZeNear avant correction : `0/8` ;
  - CAT-P apres correction : `8/8`.
- [x] Cause racine identifiee : le facteur 15 apparait avant lecture catalogue,
  dans le parsing du centre FITS strict Near. La cle numerique `RA=10.6125`
  etait traitee comme heures par `parse_angle(..., is_ra=True)` et devenait
  `159.1875 deg`.
- [x] Correctif unique applique : en strict ASTAP-ISO, `_extract_near_center_angle`
  conserve la cle numerique `RA` en degres ; les formes textuelles type
  `OBJCTRA="hh:mm:ss"` restent parsees comme hourangle. Hors strict, le
  comportement historique reste inchange.
- [x] Audit D50 : le champ RA brut est un `uint24` full-circle converti en
  degres/radians, pas en heures. Le facteur 15 n'est pas dans le record D50.
- [x] Identite physique : ASTAP `area_id=1188/1240` mappe vers les fichiers
  physiques `d50_2602.1476`/`d50_2702.1476`; apres correction, `230409`
  donne `448/448` lignes communes ASTAP/ZeNear.
- [x] Count waterfall : le pseudo-cap `~249` n'etait pas causal ; apres
  correction les comptes catalogue suivent ASTAP (`448,459,457,462,484,457,491,443`).
- [x] Outils ajoutes :
  `tools/diagnose_zn32_catalog_reader.py`,
  `tools/diagnose_zn32_catalog_records.py`,
  `tools/diagnose_zn32_catalog_gate.py`.
- [x] Rapports principaux :
  `reports/zenear_zn32cat_catalog_parity.md`,
  `reports/zenear_zn32cat_*`,
  dumps `reports/zn32cat_*`.
- [x] Tests :
  `pytest -q tests/test_zn2_astap_tools.py tests/test_zn3_input_lists.py tests/test_zn31_astap_oracles.py tests/test_zn32_catalog_reader.py`
  => `23 passed`; `py_compile` OK.
- Verdict : `A - porte catalogue franchie`.
- Prochaine mission recommandee : retourner exclusivement a la detection image
  adaptative ASTAP (`30*sigma` global puis `7*sigma` local en sections).

## 2026-07-15 — ZN3.5 oracle/gate/chaine 4D-only

- [x] Chaine cible durcie: `ZeNear strict -> gate -> ZeBlind 4D`; le backend
  historique ne doit plus etre appele automatiquement et retourne maintenant
  `BLIND4D_CONFIGURATION_REQUIRED` dans la chaine produit.
- [x] Gate runtime-only ajoutee dans `metadata_solver.py` avant l'ecriture WCS:
  CD fini/conditionne, echelle, centre, matches uniques, residus robustes,
  support spatial et holdout approximatif.
- [x] Probe ZN3.5 ajoute: branches `clean_base/astap_branch/zenear_branch/
  chain_4d_branch`, checksums pixels, manifeste sentinelle, preflight 4D,
  couverture 4D, reclassification ZN3.4 conservatrice.
- [x] Preflight 4D: manifeste local valide (`BLIND4D_READY`), 6 index actifs,
  aucun index manquant, aucun appel historique.
- [x] Tests de contrat: gate accepte un WCS coherent, rejette mauvais centre/CD/
  support degenere; wrapper Near et orchestrateur app ne retombent plus sur
  l'historique.
- [ ] Replay ASTAP propre complet non lance pendant cette passe; les conflits
  ZN3.4 restent `UNRESOLVED` tant que les oracles ASTAP propres ne sont pas
  reconstruits.
- [ ] Autopsie M106 reste ouverte: aucun correctif FOV/search/listes applique,
  premier checkpoint divergent non etabli.
- Verdict courant: `C/G partiel` — methode oracle/gate et chaine 4D-only
  securisees, mais pas de promotion tant que le replay ASTAP propre et M106 ne
  sont pas termines.

## 2026-07-15 — ZN3.5B replay borne triplet

- [x] Securite gate: `strict_acceptance_mode` repasse en `diagnostic` par
  defaut; `enforce` reste disponible explicitement.
- [x] Probe reprenable ajoute:
  `tools/diagnose_zn35_replay.py` avec `--case`, `--stage`, `--resume`,
  `--force`, `--timeout-per-stage` et statuts persistants par etape.
- [x] Triplet execute sur branches propres:
  - `230409`: ASTAP propre valide, Near strict succes, gate diagnostic
    `ACCEPT`, WCS Near vs ASTAP `WCS_CONFIRMED` (centre ~`0.23"`).
  - `233459`: ASTAP propre valide, Near strict succes, gate diagnostic
    `ACCEPT`, WCS Near vs ASTAP `WCS_CONFIRMED` (centre ~`0.89"`).
    Le WCS original FITS est pollue (`WRONG_FIELD`, centre ~`1071"`).
  - `232102`: ASTAP propre valide, Near strict echoue sans WCS
    (`near solver could not estimate a similarity transform`), premier ecart
    classe `QUAD_MATCH_DIVERGENCE`.
- [x] ZeBlind 4D reel lance sur `232102`: preflight OK, champ couvert,
  `blind4d_call_count=1`, backend historique `false`, WCS 4D confirme vs ASTAP
  (`WCS_CONFIRMED`, `59` inliers, RMS ~`0.231 px`).
- [x] Livrables ZN3.5B:
  `reports/zenear_zn35b_execution_summary.md`,
  `reports/zenear_zn35b_triplet_chain.json`,
  `reports/zenear_zn35b_first_divergence.json`,
  `reports/zn35b_runs/<case>/stages/*.json`.
- [x] Tests: suite cible ZN2->ZN3.5B `48 passed`; `py_compile` OK.
- Verdict: `D - Near echoue, ZeBlind 4D reussit dans la couverture installee`.
  Prochaine mission corrective ciblee: exposer les attempts internes Near puis
  comparer image/catalogue/quads ASTAP vs Near sur `232102`, sans changer les
  seuils ni les algorithmes avant preuve.

## 2026-07-15 — ZN3.6 parite listes/quads/hits M106

- [x] Instrumentation Near opt-in ajoutee: `diagnostic_iso_trace=False` par
  defaut; quand activee, `_astap_iso_hypothesis` dumpe les listes finales
  réellement envoyees a `_astap_iso_find_quads`, les quads et les hits par
  stage (`initial`, `autofov_*`, `spiral_*`, `recenter_second_pass`).
- [x] ASTAP relance avec `ASTAP_ZN2_DUMP_DIR` sur `230409`, `233459`,
  `232102`; dumps internes image/catalogue/quads/matches disponibles sous
  `reports/zn36_runs/<case>/astap/dumps/`.
- [x] Near relance avec trace ISO sur le triplet; `233459` utilise le stage
  gagnant `autofov_1_win_1.595116`, `232102` le meilleur stage observe
  `spiral_9`.
- [x] Matrice croisee 232102:
  `H00` (image ASTAP + catalogue ASTAP dans le moteur Near) reussit;
  `H10`, `H01`, `H11` echouent. Le coeur Near resout donc le cas avec les
  listes ASTAP finales.
- [x] Les anciens dumps Near sont clarifies: les `1713` lignes image de
  `232102` sont bien la liste initiale/finale-for-quads du stage initial;
  au meilleur stage trace, l'image reste `1713` et le catalogue monte a
  `3045` final-for-quads.
- [x] Tests ZN2->ZN3.6: `53 passed`; `py_compile` OK.
- Verdict: `B - cause exacte identifiee, correctif non applique`. Premiere
  divergence prouvee a la frontiere des listes finales d'entree (image et
  catalogue) avant les quads. Aucun correctif algorithmique applique.
- Prochaine etape ciblee: isoler pourquoi le chemin strict M106 Near conserve
  trop d'etoiles finales (`1713/3045`) alors qu'ASTAP transmet `58/249`, sans
  modifier quads/signatures/lookup/seuils.

## 2026-07-15 — ZN3.7 selection ASTAP / listes M106

- [x] Probe cree: `tools/diagnose_zn37_selection.py`; tests:
  `tests/test_zn37_astap_list_selection.py`.
- [x] ASTAP selection path documente: `bin_and_find_stars` fournit directement
  la liste image finale consommee par `find_quads`; le catalogue utilise
  `nrstars_required=round(nrstars_image*height/width)` puis oversize et quota
  `nrstars_required2`.
- [x] Tentatives Near `232102` toutes tracees: `initial`, `autofov_*`,
  `spiral_*`; `spiral_9` etait retenu par ZN3.6 car meilleur stage en evidence
  de hits, mais aucun stage Near ne resout en S11.
- [x] Image `232102`: `57/58` etoiles ASTAP sont presentes dans Near a 2 px,
  rang Near median `40`, max `233`; l'intersection image Near + catalogue ASTAP
  reussit, donc les etoiles utiles existent mais la selection/classement Near
  dilue les quads.
- [x] Catalogue `232102`: l'identite physique stricte reste indisponible dans
  la trace Near actuelle (`zewcs290` n'expose pas row_index/byte_offset), mais
  le surrogate RA/Dec/mag ne retrouve que `5/249` records ASTAP dans le
  catalogue Near du stage choisi; les prefixes Near catalogue echouent.
- [x] Comparaison positive `233459`: Near reussit avec listes plus petites et
  etoiles utiles plus tot dans l'ordre; les hits utiles survivent.
- Verdict: `F - plusieurs causes independantes`. Premiere divergence dans
  l'ordre du pipeline = selection/classement image strict M106 ne reproduisant
  pas ASTAP; divergence catalogue dependante ensuite via quota/fenetre.
  Aucun correctif applique, aucun seuil/quad/signature/lookup/transform/gate/4D
  modifie. Prochaine mission: porter uniquement la regle image ASTAP exacte.

## 2026-07-16 — ZN3.9 qualification corpus ADU

- [x] Probe cree: `tools/diagnose_zn39_qualification.py`; tests:
  `tests/test_zn39_qualification.py`.
- [x] Audit ADU: strict conserve les ADU natifs et ne laisse pas passer la
  normalisation `0..1`; les entrees 2D et 3D channel-first sont qualifiees,
  les cubes channel-last sont explicitement marques non qualifies par la
  sonde (pas de fallback silencieux).
- [x] Corpus deduplique SHA256: `142` images uniques =
  M106 `60` (61 disponibles dont 1 doublon SHA), M31 canonique `8`,
  M31 etendu `45`, NGC6888 `27`, NGC3628 `2`.
- [x] ASTAP oracles propres: `142/142` valides.
- [x] Near strict replay avec `/home/tristan/zesolver_index`: `142/142`
  succes, `142/142` WCS confirmes vs oracle ASTAP propre par comparaison de
  centre projete; M31 canonique `8/8`, M106 `60/60`, NGC6888 `27/27`,
  NGC3628 `2/2`.
- [x] Invariants stricts observes: `generic_fallback_called=0`,
  `historical_blind_called=0`, entree detecteur strict en ADU natif partout.
- [ ] Non promu: validation stellaire independante complete, fixture
  Near-failure -> 4D, chain 4D reelle, gate holdout et points d'entree
  CLI/app/GUI restent a qualifier.
- Verdict: `I - qualification incomplete` au sens promotion produit, mais
  correctif ADU fortement qualifie sur le corpus positif disponible. Tests:
  suite cible ZN2->ZN3.9 `79 passed`; `py_compile` OK.
