# Follow-up ZeSolver / ZeBlind

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
