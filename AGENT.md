# AGENT.md — Mission active de ZeSolver

**Projet :** ZeSolver  
**Écosystème :** ZeMosaic / ZeSeestarStacker  
**Auteur principal :** Tinystork — Tristan Nauleau  
**Mise à jour :** 23 juillet 2026  
**Phase active :** Stabilisation post-P3B-1D du runtime batch et du GUI  
**Statut :** P3B-1D terminé ; P3B-1E suspendu jusqu’à fermeture des quatre correctifs ci-dessous

---

## 1. Portée

Ce fichier s’applique à tout le dépôt, sauf instruction plus spécifique dans un
sous-répertoire.

Il remplace la mission active précédente. Les travaux déjà terminés restent
conservés dans `docs/stabilization/` et `docs/architecture/` et ne doivent pas
être recommencés sans régression reproduite.

La prochaine intégration de distribution officielle des Bibliothèques ZeSolver
(P3B-1E) est temporairement suspendue. Avant de poursuivre la simplification et
la distribution, quatre problèmes observés lors d’un run réel doivent être
traités dans l’ordre imposé par ce fichier.

---

## 2. État actuel

| Chantier | État |
|---|---|
| P0 — baseline et non-régression | Terminé, à préserver |
| P1 — `CatalogLibrary` | Intégré |
| P1D — bibliothèque ASTAP unique | Terminé, sous réserve du correctif de vue Blind 4D |
| P2 — réglages, profils et cœur | Stabilisés |
| P3A — GUI/pipeline | Terminé |
| P3A-V1/V2/V3 — Stop, terminaison, progression | Terminé, à préserver |
| P3B-1A — retrait Benchmark | Terminé |
| P3B-1B — audit Développement | Terminé |
| P3B-1C — réorganisation Développement | Terminé |
| P3B-1D — gestionnaire de bibliothèques | Terminé |
| **P3B-S1 — batch, mémoire et réutilisation runtime** | **Mission active prioritaire** |
| P3B-S2 — validité de la vue Blind 4D | Bloqué par S1 |
| P3B-S3 — nettoyage WCS asynchrone | Bloqué par S2 |
| P3B-S4 — familles ASTAP absentes | Bloqué par S3 |
| P3B-1E — distribution officielle | Suspendu jusqu’à fermeture S1–S4 |
| P4 — packaging/publication | Non prêt |

Le projet est actuellement :

```text
READY_FOR_P3B_RUNTIME_STABILIZATION
```

Il n’est pas encore :

```text
READY_FOR_P3B1E_LIBRARY_DISTRIBUTION_INTEGRATION
READY_FOR_P4_PACKAGING
READY_FOR_PUBLIC_RELEASE
```

---

## 3. Incident réel à reproduire

Le run utilisateur du 23 juillet 2026 constitue la référence initiale.

Configuration et symptômes observés :

- une Bibliothèque ZeSolver fondée uniquement sur D50 ;
- D50 détectée avec 1 476 tuiles et couverture annoncée complète ;
- 100 fichiers FITS ;
- 6 threads ;
- nettoyage WCS réussi sur 100 fichiers, avec 1 776 cartes retirées ;
- interface figée pendant le nettoyage ;
- préflight catalogue total d’environ 74,5 secondes ;
- résolution Blind 4D d’environ 34,5 secondes puis échec
  `BLIND4D_LIBRARY_VIEW_INVALID` ;
- résolution Near réussie sur le corpus ;
- démarrages des solves par vagues, avec de longues pauses entre les vagues ;
- consommation mémoire observée jusqu’à environ 4,3 Go ;
- débit final très inférieur aux précédents essais sur base partielle.

Diagnostic de travail à confirmer, et non à accepter aveuglément :

1. `BatchSolverPipeline` construit actuellement un nouveau `SolverPipeline`
   dans chaque tâche/fichier ;
2. chaque nouveau pipeline peut répéter la résolution des ressources catalogue,
   la création du provider Near, la création de caches et la validation Blind ;
3. `SolverPipeline.solve()` résout actuellement le runtime Blind 4D avant de
   tenter Near, y compris lorsque Near réussit ;
4. le cache de runtime Blind appartient à une instance de pipeline jetée après
   un seul fichier et perd donc son bénéfice ;
5. les résultats Near ne sont pas nécessairement transmis au GUI au moment où
   chaque future se termine ;
6. la condition de cohérence de couverture dans
   `zesolver/catalog_library/blind4d_view.py` paraît suspecte lorsqu’une
   couverture FULL/all-sky est considérée comme une erreur ;
7. le nettoyage WCS intégré exécute la boucle FITS dans le thread Qt principal ;
8. la création de bibliothèque doit considérer l’absence de familles ASTAP non
   installées comme un état normal, pas comme un échec final.

Avant toute correction, reproduire et instrumenter suffisamment pour confirmer
ou infirmer chacun de ces points.

---

## 4. Invariants absolus

### 4.1 Résultats astrométriques

Ne pas modifier sans preuve et mission explicite :

- les algorithmes ZeNear ;
- les algorithmes ZeBlind 4D ;
- les seuils d’acceptation ;
- les profils ;
- les règles WCS ;
- l’ordre logique Near → Blind 4D → Astrometry.net ;
- les formats de résultats ;
- les pixels.

Une optimisation de cycle de vie ne doit pas changer le résultat astrométrique.
Comparer les résultats avant/après sur les corpus existants.

### 4.2 Fichiers et WCS

- Ne jamais modifier les pixels.
- Ne pas écraser silencieusement un WCS existant.
- Préserver les HDU et les métadonnées non concernées.
- Les rasters utilisent un sidecar ; le raster source reste inchangé.
- Pour un FITS, le statut principal reflète le WCS du HDU `PRIMARY`.
- Une interruption ne doit jamais laisser un fichier annoncé comme résolu avec
  un en-tête incomplet.
- Toute modification du chemin d’écriture impose une relecture WCS et, si les
  données FITS sont réécrites, un contrôle d’intégrité des pixels.

### 4.3 Routage

Le routage `AUTO / PIPELINE / LEGACY` reste explicite :

- FITS compatible → pipeline autorisé ;
- raster → legacy ;
- fallback web incompatible avec pipeline → legacy ;
- moteur forcé incompatible → erreur claire ;
- aucun fallback silencieux ;
- la route legacy reste disponible pendant cette stabilisation.

### 4.4 Cycle GUI

Pour chaque `run_id` :

```text
1 événement terminal
1 message terminal
1 tentative de copie du log
1 transition finale vers IDLE
```

Pour chaque fichier et chaque run :

```text
au maximum 1 résultat terminal GUI
```

Conserver :

- progression au fil du traitement ;
- compteur traité / total / restant ;
- mise à jour des lignes en temps réel ;
- Stop réactif ;
- relance après Stop ;
- fermeture propre ;
- rejet des callbacks tardifs ;
- widgets modifiés uniquement dans le thread Qt principal ;
- aucune fausse progression à 100 % après annulation.

### 4.5 Architecture

Le cœur ne doit importer aucun module GUI.

Chaîne cible :

```text
GUI
→ ProductSettings / SolveRequest
→ GuiSolveController
→ PIPELINE ou LEGACY
→ cœur
→ résultat adapté
→ GUI
```

Le GUI ne doit pas appeler directement les algorithmes internes Near ou Blind.

---

## 5. Règles de séquencement

Les quatre missions suivantes sont obligatoirement séquentielles.

```text
S1 → S2 → S3 → S4 → reprise éventuelle de P3B-1E
```

Ne pas mélanger les quatre corrections dans un seul gros patch.

Pour chaque mission :

1. observer l’état Git ;
2. reproduire le problème ;
3. écrire ou renforcer les tests qui échouent avant correction ;
4. effectuer le plus petit changement sûr ;
5. exécuter les tests ciblés ;
6. exécuter les barrières générales ;
7. produire un rapport dédié ;
8. conclure par le gate attendu ;
9. ne commencer l’étape suivante que si le gate précédent est positif.

Ne pas pousser sur le dépôt distant sans autorisation explicite.

---

# 6. P3B-S1 — Stabiliser le batch, la mémoire et les runtimes

## 6.1 Objectif

Supprimer les coûts répétés par fichier et empêcher la mémoire de croître ou de
provoquer du thrashing pendant les gros batchs.

La correction doit préserver strictement les résultats, le routage, Stop, la
progression et la terminaison exactement une fois.

## 6.2 Fichiers à inspecter en priorité

```text
zesolver/core/batch/runner.py
zesolver/core/pipeline.py
zesolver/core/preflight.py
zesolver/catalog_resources.py
zesolver/gui_pipeline/pipeline_runner.py
zesolver/gui_pipeline/progress_adapter.py
zesolver/gui_pipeline/lifecycle.py
```

Inspecter aussi les factories de pipeline et les adaptateurs de réglages qui
peuvent recréer indirectement les ressources.

Ne modifier `zeblindsolver/` que si une preuve démontre que le problème ne peut
pas être corrigé proprement dans le cycle de vie du pipeline.

## 6.3 Travail obligatoire

### A. Mesurer le cycle actuel

Ajouter des tests ou compteurs déterministes permettant de connaître :

- le nombre de créations de `SolverPipeline` par phase ;
- le nombre de résolutions de `SolverCatalogResources` ;
- le nombre de résolutions Near runtime/provider ;
- le nombre de résolutions ou matérialisations Blind 4D ;
- le nombre de créations de caches catalogue ;
- l’ordre et le moment d’émission des résultats ;
- le comportement avec 1, 2 et plusieurs workers.

L’instrumentation produit doit rester légère. Ne pas ajouter une dépendance
obligatoire uniquement pour mesurer la mémoire.

### B. Réutiliser les pipelines au niveau worker

Le pipeline ne doit plus être recréé pour chaque fichier.

Implémentation acceptable, à choisir après audit :

- un pipeline par thread worker via stockage thread-local ;
- ou un initialiseur de `ThreadPoolExecutor` ;
- ou une petite pool explicite de pipelines ;
- ou une instance partagée uniquement si tous ses composants mutables sont
  démontrés thread-safe.

Préférer :

```text
ressources immuables partagées au niveau du batch
+ état mutable et cache possédés par chaque worker
```

Ne pas partager aveuglément un provider, un cache ou un objet Astropy/NumPy
mutable entre threads.

### C. Résoudre les ressources une seule fois par run

La sélection de `CatalogLibrary`, les chemins, familles et métadonnées de
couverture doivent être résolus au niveau du run, puis injectés dans les
pipelines worker.

L’ouverture ou la validation complète de la bibliothèque ne doit pas être
répétée pour chaque fichier.

Les changements de configuration entre deux runs doivent néanmoins produire de
nouvelles ressources ; aucun cache global permanent ne doit masquer un
changement utilisateur.

### D. Rendre Blind 4D paresseux

Pour un fichier qui réussit avec Near :

- ne pas charger les gros index Blind 4D ;
- ne pas matérialiser leur vue stricte par fichier ;
- ne pas répéter leur checksum ou validation coûteuse ;
- conserver uniquement la télémétrie catalogue légère nécessaire.

Le runtime Blind 4D doit être résolu :

- immédiatement pour un run `blind_only` ;
- avant la phase Blind si le batch est explicitement en deux phases ;
- ou au premier véritable fallback Blind ;
- au maximum une fois par run ou une fois par worker si une contrainte de
  thread-safety l’impose et est documentée.

Une indisponibilité Blind ne doit pas empêcher Near de réussir, sauf mode qui
exige explicitement Blind.

### E. Émettre les résultats au fil des futures

Lors de la phase Near :

- un succès Near final doit pouvoir être transmis dès sa fin ;
- un échec Near destiné à la phase Blind ne doit pas être présenté comme un
  échec terminal ;
- aucun fichier ne doit recevoir deux résultats terminaux ;
- `preserve_order=True` doit continuer à ordonner le résultat final du batch,
  sans bloquer la progression temps réel ;
- Stop et `stop_on_error` doivent conserver leur sémantique.

### F. Borner les caches et libérer les références

Vérifier :

- que la taille de cache configurée est réellement appliquée ;
- que les tableaux lourds temporaires ne restent pas référencés après le solve ;
- que la liste des futures terminées n’entretient pas inutilement les résultats
  et exceptions lourds ;
- que les pipelines worker sont détruits à la fin du run ;
- qu’un second run ne réutilise pas un état périmé du premier.

Ne pas utiliser `gc.collect()` comme correction principale. Il peut seulement
servir de garde ponctuelle si une raison mesurée le justifie.

## 6.4 Tests ciblés obligatoires

Créer ou renforcer des tests couvrant au minimum :

1. 100 requêtes Near réussies avec 6 workers ;
2. nombre de créations de pipeline borné par le nombre de workers, et non par le
   nombre de fichiers ;
3. ressources catalogue résolues une seule fois par batch lorsque possible ;
4. aucun chargement Blind lourd pour un batch dont tous les fichiers réussissent
   en Near ;
5. un échec Near déclenche bien la phase Blind ;
6. `blind_only` charge Blind sans tenter Near ;
7. une ressource Blind invalide ne bloque pas un succès Near en mode normal ;
8. résultats de progression émis avant la fin de toute la phase Near ;
9. aucun doublon terminal ;
10. ordre final conservé lorsque demandé ;
11. Stop pendant Near ;
12. Stop entre Near et Blind ;
13. Stop pendant Blind ;
14. deuxième run après Stop ;
15. changement de bibliothèque entre deux runs ;
16. aucune régression sur les tests pipeline, batch, GUI et WCS existants.

Tests à examiner notamment :

```text
tests/test_batch_pipeline_concurrency.py
tests/test_batch_pipeline_scheduling.py
tests/test_batch_pipeline_cancellation.py
tests/test_batch_pipeline_routing.py
tests/test_batch_blind_fallback.py
tests/test_solver_pipeline_preflight.py
tests/test_solver_pipeline_routing.py
tests/test_catalog_library_pipeline_integration.py
tests/test_p224_budget_concurrency.py
```

## 6.5 Validation réelle obligatoire

Sur la même machine et le même corpus de 100 FITS :

- 6 workers ;
- même bibliothèque D50 ;
- mêmes réglages ;
- même politique WCS ;
- relever temps total, temps avant premier solve, rythme des démarrages, mémoire
  maximale observée et statut final ;
- vérifier l’absence des longues pauses périodiques entre groupes ;
- vérifier que la mémoire atteint un plateau et ne croît pas avec le nombre de
  fichiers ;
- vérifier que le pic est inférieur à la référence observée d’environ 4,3 Go ou
  expliquer précisément la mémoire résiduelle restante ;
- lancer ensuite un second batch dans la même session ;
- tester Stop puis relance.

Le gate ne doit pas être accordé sur un micro-test uniquement.

## 6.6 Critères de sortie

S1 est terminée seulement si :

- les pipelines ne sont plus créés par fichier ;
- les ressources ne sont plus validées lourdement par fichier ;
- Blind est paresseux pour les succès Near ;
- la progression est effectivement temps réel ;
- le pic mémoire ne croît pas avec la longueur du batch ;
- le run de 100 FITS ne présente plus les vagues séparées par de longues pauses ;
- résultats et WCS restent équivalents ;
- Stop, relance et terminaison exactement une fois restent valides ;
- tests ciblés et barrières générales sont verts.

Gate attendu :

```text
READY_FOR_P3B_S2_BLIND4D_VIEW_FIX
```

Sinon :

```text
NOT_READY_FOR_P3B_S2_BLIND4D_VIEW_FIX
```

---

# 7. P3B-S2 — Corriger la vue Blind 4D de la bibliothèque

## 7.1 Objectif

Une bibliothèque annoncée `READY_FULL`, contenant un index Blind 4D valide et
1 476/1 476 tuiles, doit produire un runtime Blind 4D disponible.

Le cas observé suivant est incohérent et doit être reproduit :

```text
CatalogLibrary status: READY_FULL
blind4d_index_count: 1
blind4d_all_sky: true

puis

BLIND4D_LIBRARY_VIEW_INVALID
blind4d_index_count: 0
```

## 7.2 Fichiers à inspecter

```text
zesolver/catalog_library/blind4d_view.py
zesolver/catalog_library/coverage.py
zesolver/catalog_library/validation.py
zesolver/catalog_library/manifest.py
zesolver/catalog_resources.py
zesolver/blind4d_runtime.py
```

## 7.3 Travail obligatoire

### A. Confirmer l’invariant de couverture

Examiner la condition qui ajoute
`BLIND4D_VIEW_COVERAGE_INCONSISTENT` lorsque :

```python
coverage.all_sky or coverage.status is CoverageStatus.FULL
```

Cette condition paraît inversée, mais ne pas la remplacer mécaniquement sans
vérifier la définition de `CatalogCoverage`, `FULL`, `PARTIAL` et `all_sky`.

Définir explicitement les invariants attendus :

- `FULL` et `all_sky=True` peuvent être valides ;
- une couverture partielle peut être valide et doit rester annoncée partielle ;
- `all_sky=True` avec un ensemble incomplet doit être invalide ;
- des doublons de tuiles, tuiles manquantes, checksums incorrects, ordre runtime
  incohérent ou fichier absent restent invalides.

### B. Ne pas masquer les vraies erreurs

La correction ne doit pas transformer toutes les vues en valides.

Préserver les erreurs concernant :

- absence d’index ;
- index absent ou corrompu ;
- checksum incorrect ;
- schéma non supporté ;
- ordre runtime manquant, dupliqué ou incohérent ;
- tuiles dupliquées entre index ;
- couverture déclarée différente du contenu réel ;
- chemin invalide.

### C. Valider le runtime complet

Après correction :

1. ouvrir la bibliothèque réelle D50 ;
2. construire sa vue Blind 4D ;
3. vérifier `view.valid` ;
4. matérialiser le manifeste strict ;
5. le recharger avec le loader 4D ;
6. appeler `resolve_blind4d_runtime()` ;
7. vérifier `available=True`, `index_count=1` et couverture 1 476/1 476 ;
8. exécuter un vrai solve Blind sur une image dont Near ne fournit pas la
   solution ou en mode `blind_only`.

## 7.4 Tests ciblés obligatoires

Ajouter au minimum :

- vue FULL/all-sky valide ;
- vue PARTIAL valide mais explicitement partielle ;
- incohérence all-sky/tiles invalide ;
- absence d’index invalide ;
- ordre runtime manquant invalide ;
- doublon de tuiles invalide ;
- checksum incorrect invalide ;
- bibliothèque `READY_FULL` → runtime Blind disponible ;
- bibliothèque Near-only → code d’indisponibilité stable ;
- solve Blind réel ou fixture équivalente avec la vue library-owned.

Tests à examiner notamment :

```text
tests/test_catalog_library_blind4d_integration.py
tests/test_catalog_library_validation.py
tests/test_catalog_library_status.py
tests/test_catalog_resource_resolution.py
tests/test_solver_pipeline_blind_production.py
tests/test_regression_blind4d.py
```

## 7.5 Critères de sortie

S2 est terminée seulement si :

- la bibliothèque réelle D50 n’est plus rejetée à tort ;
- le runtime Blind 4D est réellement disponible ;
- les cas corrompus restent rejetés ;
- la couverture partielle n’est jamais présentée comme complète ;
- un solve Blind réel fonctionne ;
- les résultats Near ne sont pas modifiés ;
- les tests ciblés et barrières générales sont verts.

Gate attendu :

```text
READY_FOR_P3B_S3_ASYNC_WCS_CLEANER
```

Sinon :

```text
NOT_READY_FOR_P3B_S3_ASYNC_WCS_CLEANER
```

---

# 8. P3B-S3 — Rendre le nettoyage WCS intégré asynchrone

## 8.1 Objectif

Le nettoyage WCS doit continuer à utiliser la logique fiable de
`zewcscleaner.py`, mais ne doit plus bloquer le thread Qt principal.

Le nettoyage de 100 FITS doit :

- laisser la fenêtre réactive ;
- afficher une progression réelle ;
- pouvoir être annulé proprement ;
- rafraîchir l’état WCS des fichiers après traitement ;
- ne jamais modifier les pixels.

## 8.2 Fichiers à inspecter

```text
zesolver.py
zewcscleaner.py
zesolver/gui_pipeline/lifecycle.py
zesolver/cancellation.py
```

Réutiliser de préférence le patron de worker Qt déjà employé par le Gestionnaire
de Bibliothèques, plutôt que créer une seconde architecture incompatible.

## 8.3 Travail obligatoire

### A. Conserver une fonction non-GUI testable

`process_fits()` ou une façade équivalente doit rester indépendante de Qt et
utilisable par le CLI/tests.

Ne pas déplacer la logique FITS dans un widget.

### B. Worker Qt dédié

Exécuter la boucle de fichiers dans :

- un `QObject` déplacé dans un `QThread` ;
- ou une infrastructure worker existante offrant les mêmes garanties.

Le worker émet des signaux contenant uniquement des données simples :

- fichier courant ;
- index courant ;
- total ;
- cartes supprimées ;
- HDU modifiés ;
- erreur éventuelle ;
- statut terminal.

Aucun widget ne doit être lu ou modifié depuis le worker.

### C. Annulation coopérative

Vérifier l’annulation entre deux fichiers.

Ne pas interrompre brutalement une écriture FITS en cours. Laisser le fichier
courant se terminer proprement, puis arrêter avant le suivant.

Le résultat final doit distinguer :

```text
terminé
annulé
échoué
```

### D. Progression et rafraîchissement

- progression mise à jour à chaque fichier ;
- compteur courant/total ;
- journal non bloquant ;
- bouton Résoudre désactivé tant que le nettoyage préalable est actif ;
- bouton Stop ou Annuler correctement routé ;
- après nettoyage, rescanner l’état WCS et mettre à jour les lignes ;
- ne pas lancer le solve si le nettoyage préalable a échoué ou a été annulé,
  sauf choix utilisateur explicite déjà prévu par le produit.

### E. Fermeture

Fermer l’application pendant le nettoyage doit :

- demander ou déclencher l’annulation ;
- attendre proprement la fin de l’écriture FITS en cours ;
- ne pas laisser de thread Qt vivant ;
- ne pas produire de callback tardif vers une fenêtre détruite.

## 8.4 Tests ciblés obligatoires

Couvrir au minimum :

1. nettoyage de plusieurs fichiers sans bloquer l’event loop ;
2. progression 1…N ;
3. succès total ;
4. fichier sans WCS ;
5. fichier invalide ;
6. erreur partielle avec résumé ;
7. annulation entre deux fichiers ;
8. fermeture pendant nettoyage ;
9. absence de double événement terminal ;
10. rafraîchissement des statuts WCS ;
11. pixels strictement identiques avant/après ;
12. préservation des HDU ;
13. lancement du solve après nettoyage réussi ;
14. aucune régression du cleaner autonome.

## 8.5 Validation graphique réelle

Avec environ 100 FITS :

- déplacer/redimensionner la fenêtre pendant le nettoyage ;
- changer d’onglet ;
- vérifier la progression ;
- tester Annuler ;
- relancer le nettoyage ;
- lancer ensuite un solve ;
- fermer l’application pendant une opération ;
- vérifier les fichiers et le journal après relance.

Un test offscreen ne remplace pas cette validation.

## 8.6 Critères de sortie

S3 est terminée seulement si :

- le GUI ne freeze plus ;
- la progression est réelle ;
- l’annulation et la fermeture sont propres ;
- les statuts WCS sont rafraîchis ;
- les pixels et HDU sont préservés ;
- le cleaner autonome continue de fonctionner ;
- les tests ciblés et barrières générales sont verts.

Gate attendu :

```text
READY_FOR_P3B_S4_ASTAP_FAMILY_HANDLING
```

Sinon :

```text
NOT_READY_FOR_P3B_S4_ASTAP_FAMILY_HANDLING
```

---

# 9. P3B-S4 — Accepter proprement une seule famille ASTAP

## 9.1 Objectif

La création d’une Bibliothèque ZeSolver depuis un dossier contenant uniquement
D50 doit se terminer comme un succès complet pour D50.

L’absence de D05, D20, D80, V50, G05 ou d’autres familles supportées ne doit pas
être une erreur si elles n’ont pas été sélectionnées et ne sont pas présentes.

## 9.2 Fichiers à inspecter

```text
zesolver/catalog_library/management.py
zesolver/catalog_library/adoption.py
zesolver/catalog_library/validation.py
zesolver.py
zewcs290/catalog290.py
zeblindsolver/astap_db_reader.py
```

Inspecter séparément :

- la détection ;
- la sélection GUI ;
- le calcul du nombre total d’étapes ;
- la construction ;
- la validation finale ;
- les messages utilisateur.

## 9.3 Travail obligatoire

### A. Détection réelle

La liste des familles à construire doit provenir des fichiers réellement
détectés sous la racine choisie.

Cas attendu pour une racine ne contenant que D50 :

```text
families = ("d50",)
```

Ne pas initialiser une attente sur toutes les entrées de `FAMILY_SPECS`.

### B. Sélection explicite

Si le GUI permet une sélection personnalisée :

- afficher uniquement les familles détectées ;
- une famille demandée mais absente produit une erreur claire avant le build ;
- une famille non demandée et absente ne produit aucune erreur ;
- le mode Standard sélectionne toutes les familles détectées, pas toutes les
  familles théoriquement supportées.

### C. Progression exacte

Le dénominateur de progression doit être fondé sur le nombre de familles
sélectionnées et les étapes réellement exécutées.

Pour D50 seule :

- aucune étape fantôme pour D05/D20/D80/V50/G05 ;
- pas d’attente terminale après la fin de D50 ;
- pas de progression bloquée sous 100 % ;
- pas d’erreur finale après publication réussie.

### D. Statut et message final

Une bibliothèque D50 valide doit produire :

- résultat `LibraryOperationResult` réussi ;
- `catalog.json` cohérent ;
- source Near D50 disponible ;
- index Blind D50 publié ;
- couverture annoncée selon le contenu réel ;
- sélection automatique de la bibliothèque ;
- message utilisateur de succès.

Les familles absentes peuvent apparaître en log DEBUG/INFO, mais pas sous forme
d’erreur utilisateur.

## 9.4 Tests ciblés obligatoires

Couvrir au minimum :

- D50 seule ;
- deux familles présentes ;
- racine contenant les familles directement ;
- racine contenant des sous-dossiers D50/D20 ;
- sélection personnalisée d’une famille présente ;
- demande explicite d’une famille absente ;
- aucune famille détectée ;
- annulation pendant une famille ;
- progression correcte ;
- publication et validation finales ;
- réparation d’une bibliothèque Near-only D50 ;
- chemins avec espaces et caractères non ASCII.

## 9.5 Validation graphique réelle

Créer une nouvelle bibliothèque depuis une installation ASTAP ne contenant que
D50 et vérifier :

- détection immédiate de D50 ;
- aucune attente pour d’autres familles ;
- progression cohérente ;
- aucune erreur terminale ;
- bibliothèque automatiquement sélectionnée ;
- vérification `READY_FULL` ou `READY_PARTIAL` cohérente ;
- run Near ;
- run Blind après S2.

## 9.6 Critères de sortie

S4 est terminée seulement si :

- D50 seule est un parcours pleinement supporté ;
- aucune famille absente non demandée ne provoque d’erreur ;
- la progression reflète les familles réellement traitées ;
- le message final correspond au résultat réel ;
- la bibliothèque produite fonctionne avec Near et Blind ;
- les tests ciblés et barrières générales sont verts.

Gate attendu :

```text
READY_FOR_P3B1E_LIBRARY_DISTRIBUTION_INTEGRATION
```

Sinon :

```text
NOT_READY_FOR_P3B1E_LIBRARY_DISTRIBUTION_INTEGRATION
```

---

## 10. Barrières générales obligatoires

Après chaque mission S1, S2, S3 et S4 :

```bash
.venv/bin/python tools/check_core_boundaries.py
.venv/bin/python tools/run_regression_suite.py --hermetic
.venv/bin/python -m pytest -q
```

Puis :

```bash
.venv/bin/python -m compileall -q \
  zeblindsolver \
  zewcs290 \
  zesolver \
  tools \
  tests \
  zesolver.py \
  zewcscleaner.py \
  zeindexcheck.py

git diff --check
git status --short --branch
```

Pour les surfaces GUI :

```bash
QT_QPA_PLATFORM=offscreen \
.venv/bin/python -m pytest tests/test_gui_* -q
```

Tout skip externe doit être explicite dans le rapport.

Ne jamais présenter :

- un test non exécuté comme réussi ;
- un test offscreen comme une validation graphique réelle ;
- un micro-benchmark comme preuve suffisante pour un batch de 100 fichiers ;
- une bibliothèque validée Near comme preuve que Blind fonctionne.

---

## 11. Méthode Git

Avant toute modification :

```bash
git status --short --branch
git diff --check
```

Règles :

- ne pas restaurer les changements utilisateur ;
- ne pas supprimer les travaux P3B-1A à P3B-1D ;
- séparer S1, S2, S3 et S4 en changements distincts ;
- ne pas pousser sans autorisation explicite ;
- ne pas ajouter les catalogues, index, FITS de test volumineux, logs, venv,
  caches ou ZIP au dépôt ;
- préférer des fixtures synthétiques/hermétiques pour les tests automatisés ;
- conserver un rollback clair.

---

## 12. Rapport attendu après chaque mission

Créer un rapport dédié dans `docs/stabilization/` contenant :

1. objectif ;
2. état Git initial ;
3. reproduction avant correction ;
4. diagnostic confirmé ou infirmé ;
5. architecture choisie et raison ;
6. fichiers modifiés ;
7. comportement avant/après ;
8. tests ajoutés ;
9. tests ciblés exécutés ;
10. barrières générales ;
11. validation manuelle réelle ;
12. mesures de temps et mémoire lorsque pertinent ;
13. résultats WCS avant/après ;
14. warnings et limites ;
15. tests non exécutés et raison ;
16. état Git final ;
17. une seule prochaine étape ;
18. gate exact.

Nommage recommandé :

```text
docs/stabilization/p3b_s1_batch_runtime_memory_report_20260723.md
docs/stabilization/p3b_s2_blind4d_library_view_report_20260723.md
docs/stabilization/p3b_s3_async_wcs_cleaner_report_20260723.md
docs/stabilization/p3b_s4_astap_family_handling_report_20260723.md
```

---

## 13. Reprise de P3B-1E

P3B-1E ne peut reprendre qu’après obtention du gate :

```text
READY_FOR_P3B1E_LIBRARY_DISTRIBUTION_INTEGRATION
```

À ce moment seulement, mettre à jour ce fichier avec la mission de distribution
officielle des bibliothèques.

Ne pas profiter de S1–S4 pour ajouter :

- téléchargement officiel ;
- URLs de release ;
- packaging ;
- installateur ;
- nouvelle projection ;
- nouveau profil astrométrique ;
- optimisation GPU non liée au défaut reproduit ;
- refonte graphique supplémentaire.

---

## 14. Principe final

La priorité immédiate n’est pas d’ajouter des fonctions.

Elle est de garantir que :

- un gros batch reste rapide et borné en mémoire ;
- Near ne paie pas le coût de Blind lorsqu’il réussit ;
- une bibliothèque valide est réellement utilisable par Blind ;
- le nettoyage WCS ne bloque pas l’interface ;
- une installation ASTAP D50 seule est un parcours normal ;
- les résultats WCS et les fichiers restent dignes de confiance.

**Mesurer. Corriger une cause à la fois. Préserver les résultats. Ne reprendre la
simplification et la distribution qu’après démonstration.**
