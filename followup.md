# Follow-up ZeSolver / ZeBlind

## Cap actuel
Obtenir un chemin **ZeBlind + index Astrometry** réellement **livrable** en procédant **de l'amont vers l'aval**, avec une seule question à chaque itération :

> **Quel est le premier point de divergence causal entre Ze et Astrometry sur le cas sentinelle ?**

Le cas sentinelle de travail reste :
- `M106`
- tuile / candidat : `d50_2823`
- même image
- même enveloppe de replay borné / diffable

## Règles de conduite
- [X] Travailler en **top-down strict** : ne plus chasser des résiduels aval tant que le premier écart amont n'est pas identifié.
- [X] Conserver **un seul front causal actif** à la fois.
- [X] Toute conclusion durable validée va dans `memory.md`, pas ici.
- [X] `followup.md` reste un **plan d'exécution court**, pas un journal.
- [X] Toute comparaison doit être faite sur le **même cas**, au **même étage**, avec un artefact diffable.

## État utile déjà acquis
- [x] Plusieurs faux fronts aval ont été fermés ou blanchis :
  - `ref pool cap`
  - `mo_scale_native / verify_pix2`
  - `quad_center / Q2`
  - fuite des quad stars
  - `RoR / effective_area`
  - ordre initial des test-stars
  - `testsigma²`
  - cœur séquentiel `theta / logodds` sur le baseline courant
- [x] Le diagnostic méthodologique a changé :
  - on ne pilote plus le sprint comme une suite de coupes dichotomiques locales ;
  - on repart maintenant en **recherche du premier point de divergence global**.

## Plan actif

### P1. Fixer la colonne de checkpoints top-down
- [X] Publier une grille unique de checkpoints pour `d50_2823` :
  - image stars d'entrée
  - ordre / filtrage image
  - quad candidat retenu
  - transform estimée
  - support `test/ref` avant `verify`
  - premier step utile de `verify`
  - décision finale
- [X] Pour chaque checkpoint, définir :
  - artefact Ze de référence
  - source Astrometry de référence
  - scalaires / cardinalités / ordres à comparer

### P2. Isoler le premier écart causal Ze vs Astrometry
- [X] Rejouer le cas sentinelle avec instrumentation bornée compatible diff.
- [X] Comparer les checkpoints dans l'ordre amont -> aval.
- [X] S'arrêter au **premier checkpoint divergent**, sans descendre plus bas tant qu'il n'est pas compris.
- [X] Classer ce checkpoint :
  - divergence de données d'entrée
  - divergence d'ordre / filtrage
  - divergence géométrique / transform
  - divergence de support `verify`

### P3. Corriger un seul écart, puis revalider la colonne
- [X] Appliquer **un seul delta causal**.
- [X] Rejouer exactement le même protocole top-down.
- [X] Vérifier si :
  - le premier écart est fermé
  - ou si le front s'est déplacé au checkpoint suivant

### P4. Quand la colonne sentinelle devient propre, rouvrir la perspective livrable
- [X] Fermer la colonne sentinelle jusqu'à `verify_support_pre_step0`
- [X] Smoke multicase court
- [X] Garde-fou ZeNear court `5` FITS
- [X] Garde-fou ZeNear `testzenear` complet :
  - runner isolé : `20/30` en `141.97 s`
  - batch produit : `22/30` résolus par Near, identique à la baseline historique `22 Near`
- [X] Vérifier qu'aucune fermeture de divergence n'a cassé ZeNear
- [X] Réouvrir prudemment le tuning produit blind-only

## Prochain front produit
- [X] Rejouer le lot canonique `testzenear` sur copies sans WCS
- [X] Isoler le double fallback Blind du batch :
  - `allow_blind_fallback=False` était ignoré dans `_run_index_near_solver()`
  - les `8` non-résolus repartaient donc dans une seconde phase Blind
- [X] Corriger le contrat :
  - phase Near batch = aucun Blind interne
  - phase Blind batch = un seul passage par non-résolu
- [X] Rejouer d'abord un smoke produit borné sur les `8` non-résolus après ce correctif
- [X] Séparer le dimensionnement de la phase Blind de la phase Near :
  - machine `7.52 GiB` : `6 -> 2` workers
  - pic RSS observé : `3.2 -> 2.0 GiB`
  - override explicite : `ZE_BLIND_WORKERS`
- [ ] Construire un smoke blind multicase avec oracle qualité, sans seuils probe permissifs
- [ ] Mesurer séparément :
  - vrais succès
  - faux positifs rejetés par le pool startree
  - coût runtime/mémoire du chargement des index 4107/4108/4109

## Prochain cran exact
- [X] Construire l'audit **`first_divergence_topdown`** sur `d50_2823`
- [X] Formaliser le contrat source-vs-artefact des checkpoints :
  - ordre d'entrée test-stars
  - géométrie quad
  - `verify_pix2`
  - support `verify` avant step `0`
  - cœur séquentiel
- [X] Auditer le checkpoint désormais identifié comme premier point ouvert avec un pendant Astrometry homologué :
  - `verify_support_pre_step0`
- [X] Commencer par l'amont immédiat du baseline utile courant :
  - support exact du step `0`
  - voisin NN retenu
  - géométrie locale du premier match utile
- [X] Écarter le faux soupçon méthodologique :
  - un step `0` non porté par le `carry support` n'est pas, à lui seul, une divergence
  - Astrometry conserve l'ordre d'entrée des test-stars après `dedup` / retrait du quad / `RoR`
- [X] Obtenir un premier pendant Astrometry blind runtime sur le même FITS :
  - `.axy` local propre
  - `astrometry-engine` instrumenté
  - parseur de blocs `C_VERIFY_ENTRY` / `C_VERIFY_TERM`
- [X] Raffiner la lecture méthodologique avec ce runtime Astrometry :
  - le `testperm` canonique peut être **non monotone**
  - `verify_apply_ror()` peut réordonner via `uniformize`
- [X] Obtenir un harness Astrometry direct sur `verify_hit()` à partir d'un `matchfile` :
  - `verify-paths` remis en état en mode utile `matchfile + index`
  - rejeu autonome validé sur `zeverify.match`
  - checkpoint canonique figé : `NT=123`, `NR=19`, `quad_field_head=4,0,3,5`
- [X] Établir que le chemin `solve-field --verify` avec WCS Ze n'est **pas** encore homologue au sentry Ze :
  - il retombe sur une résolution blind
  - le `matchfile` obtenu est canonique Astrometry, pas encore l'hypothèse Ze injectée
- [X] Comparer maintenant l'état Ze sentinelle avec un pendant Astrometry **homologué au même étage** :
  - produire ou injecter un `matchfile`/`MatchObj` **équivalent Ze**
  - puis rejouer `verify-paths` dessus
  - et comparer enfin `verify_support_pre_step0` au même type d'entrée
- [X] Le faire dans le cadre de la colonne complète, pour éviter de retomber dans une chasse locale aval
- [X] Appliquer le premier delta d'alignement support :
  - test-stars depuis le champ image complet, plus depuis le sous-ensemble `test_xy_px`
  - ref-scope forcé par champ/rayon WCS en mode `astrometry_native_verify_semantics`
  - export `ref_world_px` aligné avec `ref_xy_px` pour reconstruire un `MatchObj` diffable
- [X] Fermer le sous-écart restant du même checkpoint `verify_support_pre_step0` :
  - Ze avant dédup C-like : `NT=435`, `NR=123`
  - Astrometry sur `MatchObj` Ze propre : `NT=434`, `NR=134`
  - support C forcé dans Ze : `NT=434`, `NR=134`, step 0 `distractor`, `logodds=-1.38629436`
  - [X] aligner `verify_pix2` sur Astrometry : `DEFAULT_VERIFY_PIX=1.0` + jitter index (`verify_pix2_input=1.027071547...`)
  - [X] aligner l'ordre des test-stars amont : port de `verify_uniformize_field()` + résolution automatique `CUTNSIDE` par niveau (`S=156`, `M=110`, `L=78`)
  - résultat probe auto : `testperm_head=[0,16,30,65,28,4,6,92,...]`, step 0 `distractor`, `logodds=-1.38629436`
  - [X] fermer le résidu test/RoR de queue :
    - cause confirmée : `verify_deduplicate_field_stars()` C retire l'étoile tardive `295`, voisine de `75` sous `testsigma`
    - patch Ze : helper `_astrometry_verify_dedup_teststar_indices()`
    - validation : `pytest -q tests/test_zeblindsolver.py tests/test_synthetic.py` => `43 passed`
  - [X] reproduire en Python la source ref-pool/index Astrometry stricte :
    - lecteur Python pur du KD-tree FITS `stars`
    - parcours `startree_search_for()` reproduit
    - tri par `sweep` branché dans le chemin native
    - clamp de rayon Ze retiré en mode natif
    - oracle homologue courant : Ze/C `NT=434`, `NR=134`, `NRall=247`
    - résidu d'ordre limité aux égalités `sweep` de `qsort_r` non stable

## Définition du livrable
Le sprint devient réellement livrable quand les conditions suivantes sont vraies :
- [X] le premier point de divergence Ze vs Astrometry est identifié puis fermé sur le cas sentinelle
- [X] le chemin sentinelle devient explicable de bout en bout sans zone grise méthodologique
- [X] les garde-fous Near ne régressent pas
- [X] on peut alors reprendre un tuning blind-only produit sur une base propre
