> [!NOTE]
> **Document historique — remplacé le 18 juillet 2026**
>
> Cette feuille de route décrit la mission initiale définie le 16 juillet 2026.
> Elle est conservée comme référence et ne constitue plus l’instruction active
> des agents.
>
> État lors de son archivage :
>
> - P0 baseline/non-régression : terminé ;
> - P0 sûreté FITS : validé pour P3B, clôture publication encore requise ;
> - P1 CatalogLibrary : cœur terminé, intégration produit encore requise ;
> - P1 couverture 4D : partielle et explicitement limitée ;
> - P2 façade/extraction : suffisamment stabilisées pour P3B ;
> - P3 : phase active ;
> - P4 : non commencé ;
> - P5 : différé.
>
> Les instructions actives se trouvent dans `/AGENT.md`.

**Projet :** ZeSolver  
**Écosystème :** ZeMosaic / ZeSeestarStacker  
**Auteur principal :** Tinystork — Tristan Nauleau  
**Document de mission :** 16 juillet 2026  
**Statut du projet :** solveur fonctionnel, phase de stabilisation avant publication

---

## 1. Portée de ce document

Ce fichier s'applique à l'ensemble du dépôt ZeSolver, sauf instruction plus spécifique placée dans un sous-répertoire.

Il définit la mission des agents de développement intervenant sur le projet, l'ordre obligatoire des priorités et les règles destinées à préserver les résultats astrométriques déjà obtenus.

Le projet a franchi la phase de preuve de concept : les moteurs ZeNear et ZeBlind 4D produisent maintenant des solutions exploitables. La mission n'est donc plus de multiplier les expérimentations, mais de transformer ce résultat en un logiciel fiable, compréhensible, installable et publiable.

---

## 2. Constat de départ

ZeSolver est fonctionnel, mais son code conserve les traces normales d'une longue phase de recherche et de développement :

- plusieurs modules sont devenus très volumineux et concentrent trop de responsabilités ;
- la logique métier, l'orchestration, l'interface graphique, les réglages expérimentaux et les outils de diagnostic sont encore fortement liés ;
- de nombreux paramètres internes du solveur sont exposés ou persistés alors qu'ils ne devraient pas être manipulés par un utilisateur normal ;
- l'interface contient encore des éléments utiles au développement, au benchmark et à la construction des index ;
- la couverture du ciel par les index ZeBlind 4D doit être complétée ou clairement bornée ;
- les résultats actuels doivent être transformés en tests automatisés avant toute refactorisation importante ;
- l'écriture des solutions WCS dans les fichiers FITS doit être rendue explicitement sûre ;
- la gestion des catalogues doit être unifiée autour d'une bibliothèque ASTAP unique du point de vue de l'utilisateur ;
- le packaging et l'installation sur une machine vierge ne sont pas encore au niveau d'une version publique stable.

La réussite actuelle doit être considérée comme une **baseline fonctionnelle à protéger**.

---

## 3. Mission générale

Transformer ZeSolver en un produit publiable sans dégrader les capacités de résolution acquises.

L'ordre de travail imposé est :

1. figer et mesurer le comportement actuel ;
2. protéger les fichiers et empêcher les faux WCS ;
3. unifier l'architecture des catalogues ;
4. stabiliser les API internes ;
5. découper progressivement les monolithes ;
6. reconstruire une interface utilisateur simple ;
7. fiabiliser le packaging et l'installation ;
8. seulement ensuite poursuivre les optimisations avancées.

Une refactorisation élégante qui réduit le taux de résolution, augmente les faux positifs ou rend les résultats moins reproductibles est une régression et doit être refusée.

---

## 4. Ordre final des priorités

| Priorité | Chantier | Bloquant pour publication |
|---|---|---:|
| **P0** | Baseline, corpus et non-régression | Oui |
| **P0** | Protection des FITS et absence de faux WCS | Oui |
| **P1** | Bibliothèque ASTAP unique et gestion automatique | Oui |
| **P1** | Couverture 4D complète ou clairement limitée | Oui |
| **P2** | Façade stable et réduction des configurations | Oui |
| **P2** | Découpage progressif des monolithes | Oui |
| **P3** | Nouvelle interface utilisateur | Oui |
| **P4** | Packaging, documentation et tests sur machine vierge | Oui |
| **P5** | Optimisations GPU et fonctionnalités avancées | Non |

### Règle de séquencement

- Les travaux P0 précèdent toute refactorisation structurelle importante.
- La conception P1 doit être arrêtée avant l'implémentation finale de la nouvelle interface P3.
- P2 peut commencer par de petites extractions uniquement lorsque les tests P0 protègent le comportement concerné.
- P5 ne doit jamais retarder ou fragiliser P0 à P4.

---

# MISSION P0 — Baseline, corpus et non-régression

## 5. Objectif

Créer une référence mesurable et immuable du comportement actuel de ZeSolver afin que chaque modification ultérieure puisse être comparée objectivement.

## 5.1 Baseline fonctionnelle

Créer et documenter une référence de type :

```text
v0.9.0-functional-baseline
```

Cette baseline doit identifier au minimum :

- le commit exact ;
- la version Python ;
- les versions des dépendances principales ;
- les paramètres et profils de solve utilisés ;
- le ou les manifestes ZeBlind 4D ;
- les index et catalogues utilisés, avec version ou empreinte ;
- le corpus de test ;
- les rapports de réussite, d'échec, de temps d'exécution et de qualité.

Ne jamais modifier rétroactivement une baseline publiée. Toute évolution crée une nouvelle baseline.

## 5.2 Corpus de référence

Le corpus doit couvrir progressivement :

- champs larges, moyens et étroits ;
- régions stellaires denses et pauvres ;
- images avec et sans métadonnées fiables ;
- métadonnées volontairement erronées ;
- images bruitées, sous-exposées, saturées ou légèrement filées ;
- différents capteurs, focales, binning et échantillonnages ;
- FITS avec WCS valide, invalide, partiel ou absent ;
- cas historiquement difficiles et cas déjà résolus durant le développement 4D ;
- cas négatifs qui doivent échouer proprement.

Le dépôt ne doit pas nécessairement contenir toutes les images lourdes. Un manifeste de corpus peut référencer leur emplacement, leur empreinte SHA-256 et les résultats attendus.

## 5.3 Oracle de résultat

Pour chaque image de référence, enregistrer lorsque pertinent :

- succès, échec ou résultat attendu ;
- backend attendu ou autorisé ;
- centre RA/Dec ;
- échelle en secondes d'arc par pixel ;
- rotation et parité ;
- nombre d'inliers ;
- RMS en pixels et/ou secondes d'arc ;
- durée indicative ;
- index ou famille utilisée ;
- tolérances d'acceptation ;
- raison attendue en cas d'échec.

Les tests ne doivent pas imposer une égalité flottante exacte. Utiliser des tolérances astronomiquement justifiées.

## 5.4 Niveaux de tests attendus

Mettre en place :

1. **tests unitaires** des fonctions mathématiques, catalogues, parsing et validation ;
2. **tests de caractérisation** protégeant le comportement actuel ;
3. **tests d'intégration** ZeNear, ZeBlind 4D et chaîne complète ;
4. **tests négatifs** contre les faux positifs ;
5. **tests batch** incluant annulation, erreurs partielles et reprise ;
6. **tests de performance** informatifs avec comparaison à la baseline.

## 5.5 Critères de sortie P0 — baseline

P0-baseline est terminé lorsque :

- une commande unique lance la suite de non-régression ;
- les cas de référence sont décrits par un manifeste versionné ;
- les seuils de succès sont explicites ;
- les faux positifs sont testés ;
- les rapports sont lisibles par un humain et exploitables en CI ;
- une modification qui dégrade les résultats bloque automatiquement l'intégration.

---

# MISSION P0 — Protection des FITS et absence de faux WCS

## 6. Principe de sûreté

Un échec propre est préférable à une solution WCS fausse.

Le solveur ne doit jamais déclarer une image résolue sur le seul fait qu'une transformation mathématique a pu être calculée. Toute solution doit être validée par des critères indépendants suffisants.

## 6.1 Validation obligatoire d'une solution

Une solution acceptée doit notamment vérifier, selon le backend :

- un nombre minimum d'inliers ;
- un RMS sous le seuil du profil ;
- une distribution spatiale raisonnable des correspondances ;
- une échelle et une transformation physiquement cohérentes ;
- une projection WCS valide sur l'image ;
- l'absence de valeurs non finies ;
- une cohérence avec les indices utilisateur lorsqu'ils sont explicitement contraignants ;
- une validation finale utilisant la liste d'étoiles suffisante, et pas uniquement le quad candidat.

Les seuils appartiennent aux profils internes versionnés. Ils ne doivent pas être dispersés dans le GUI.

## 6.2 Règles d'écriture FITS

Par défaut :

- ne jamais modifier les données pixel ;
- privilégier une copie de sortie ou une sauvegarde vérifiable ;
- ne pas écraser silencieusement un WCS existant ;
- écrire le WCS de manière atomique ou avec stratégie de restauration ;
- relire et valider le fichier après écriture ;
- conserver les métadonnées non WCS sauf raison documentée ;
- signaler clairement toute modification partielle ;
- garantir qu'une interruption ne laisse pas un fichier annoncé comme résolu avec un en-tête incomplet.

Tout test d'écriture doit comparer avant/après :

- empreinte ou égalité des pixels ;
- structure des HDU ;
- cartes d'en-tête attendues ;
- ouverture du fichier par Astropy ;
- construction et utilisation du WCS final.

## 6.3 Critères de sortie P0 — FITS

Ce chantier est terminé lorsque :

- les pixels sont garantis inchangés par tests automatisés ;
- les écritures interrompues ou invalides sont récupérables ;
- les faux WCS du corpus négatif sont rejetés ;
- toute solution écrite est relue et revalidée ;
- les modes écrasement, copie et sauvegarde sont documentés et testés.

---

# MISSION P1 — Bibliothèque ASTAP unique

## 7. Décision d'architecture

ASTAP/HNSKY devient la source locale de données stellaires de référence pour ZeSolver.

Cela signifie une source de vérité unique du point de vue du produit, mais pas nécessairement un fichier physique unique.

- ZeNear lit les tuiles stellaires ASTAP nécessaires à ses requêtes.
- ZeBlind 4D utilise des index de quads dérivés de ces mêmes tuiles.
- Les index sont des artefacts générés, versionnés et vérifiables ; ils ne constituent pas une seconde source astronomique indépendante.
- Astrometry.net peut rester un fallback web facultatif, clairement distinct du fonctionnement local.

## 7.1 Architecture cible indicative

```text
ZeSolverCatalog/
├── catalog.json
├── astap/
│   └── raw/
│       ├── d50_....1476
│       ├── d80_....1476
│       └── ...
└── indexes/
    └── blind4d/
        ├── manifest.json
        ├── index_....npz
        └── ...
```

Le nom et le format exacts pourront évoluer, mais l'utilisateur doit sélectionner ou installer une seule **Bibliothèque de catalogue ZeSolver**.

## 7.2 Composant `CatalogLibrary`

Créer une façade interne unique responsable de :

- découvrir ou ouvrir la bibliothèque ;
- lire son manifeste ;
- vérifier les versions et empreintes ;
- exposer les tuiles ASTAP à ZeNear ;
- exposer les index compatibles à ZeBlind 4D ;
- indiquer la couverture céleste et les échelles disponibles ;
- détecter les éléments manquants ou incompatibles ;
- proposer une réparation ou reconstruction explicite ;
- fournir un état simple au GUI : prêt, incomplet, incompatible ou absent.

Le GUI ne doit pas manipuler directement les chemins de tuiles, familles, hashes ou manifestes 4D.

## 7.3 Choix des familles

Ne pas imposer arbitrairement D50 seul avant validation sur le corpus.

D50 est le candidat principal, mais la sélection finale doit être fondée sur :

- le taux de réussite ;
- les très grands et très petits champs ;
- les régions stellaires pauvres et denses ;
- la profondeur des images ;
- la taille totale de la bibliothèque ;
- les temps de requête et de génération des index.

Plusieurs familles ASTAP peuvent être utilisées tout en restant une bibliothèque unique pour l'utilisateur.

## 7.4 Critères de sortie P1 — bibliothèque

Ce chantier est terminé lorsque :

- ZeNear et ZeBlind 4D consomment la même bibliothèque via une API commune ;
- les données brutes et les index dérivés sont reliés par des empreintes et versions ;
- l'utilisateur ne choisit plus séparément une base et un manifeste 4D ;
- les erreurs d'installation ou de compatibilité sont explicites ;
- le fonctionnement avec bibliothèque partielle est défini et testé.

---

# MISSION P1 — Couverture ZeBlind 4D

## 8. Objectif

Garantir une couverture 4D adaptée à la promesse publique du logiciel.

Deux formes de publication sont acceptables :

1. couverture complète et validée pour les plages de champ annoncées ;
2. version beta à couverture limitée, avec limites détectables et clairement documentées.

Une couverture limitée ne doit jamais être présentée comme générale.

## 8.1 Travaux attendus

- inventorier les index installés et leur couverture ;
- définir les zones célestes, densités, échelles et familles couvertes ;
- générer les index manquants avec processus reproductible ;
- valider chaque nouvel index sur le corpus correspondant ;
- tester l'ordre et l'union des index ;
- mesurer consommation mémoire, taille disque et temps de chargement ;
- détecter automatiquement lorsqu'une requête sort de la couverture ;
- produire une erreur claire ou déclencher le fallback autorisé.

## 8.2 Critères de sortie P1 — couverture

- manifeste de couverture versionné ;
- aucun index incompatible chargé silencieusement ;
- couverture complète ou limites publiques explicites ;
- tests de bord et de zones non couvertes ;
- procédure reproductible de génération et de vérification.

---

# MISSION P2 — Façade stable et configurations réduites

## 9. Objectif

Séparer l'API produit des détails expérimentaux du solveur.

## 9.1 API cible

Le GUI, le CLI et les traitements batch doivent converger vers une façade comparable à :

```python
result = SolverPipeline.solve(request)
```

Objets publics recommandés :

```text
SolveRequest
SolveResult
CatalogLibrary
SolverPipeline
ProductSettings
```

Les noms définitifs peuvent changer, mais les responsabilités doivent rester simples et stables.

## 9.2 Séparer deux catégories de réglages

### Paramètres utilisateur

Exemples :

- fichiers ou dossier d'entrée ;
- dossier de sortie ;
- bibliothèque de catalogue ;
- conserver ou écraser un WCS existant ;
- nombre de tâches ou mode Auto ;
- fallback web autorisé ou non ;
- indices instrumentaux facultatifs.

### Profils internes versionnés

Exemples :

```text
zenear-v1
zeblind4d-v1
```

Ils contiennent :

- seuils de détection ;
- limites de candidats ;
- nombre de quads ;
- stratégies de profondeur ;
- tolérances de validation ;
- règles de fallback ;
- caps de buckets ;
- options expérimentales validées.

Ces détails ne doivent pas être exposés dans l'interface normale ni multipliés dans plusieurs structures de configuration.

## 9.3 Compatibilité

Lors de l'introduction de la façade :

- préserver temporairement les anciennes entrées si nécessaire ;
- ajouter des adaptateurs explicites ;
- émettre des avertissements de dépréciation ;
- ne supprimer les anciens chemins qu'après migration des tests, du GUI et du CLI.

## 9.4 Critères de sortie P2 — façade

- GUI et CLI utilisent la même orchestration ;
- une seule source de valeurs par défaut produit existe ;
- les profils internes sont versionnés ;
- le nombre de paramètres exposés à l'utilisateur est fortement réduit ;
- les anciens réglages inutiles sont migrés ou ignorés explicitement ;
- tous les tests P0 restent conformes.

---

# MISSION P2 — Découpage progressif des monolithes

## 10. Principe

Ne jamais réécrire les grands solveurs en une seule opération.

Le découpage doit être progressif, couvert par les tests et sans changement fonctionnel involontaire.

## 10.1 Architecture cible indicative

```text
zesolver/
├── core/
│   ├── models.py
│   ├── pipeline.py
│   └── errors.py
├── catalog/
│   ├── library.py
│   ├── astap.py
│   └── integrity.py
├── solver/
│   ├── near.py
│   ├── blind4d.py
│   ├── validation.py
│   └── profiles.py
├── io/
│   ├── images.py
│   ├── fits.py
│   └── wcs.py
├── runtime/
│   ├── batch.py
│   ├── workers.py
│   └── resources.py
├── gui/
└── cli/
```

Cette arborescence est une direction, pas une obligation de renommage immédiat.

## 10.2 Ordre d'extraction conseillé

1. modèles et types de résultats ;
2. erreurs métier typées ;
3. lecture d'images et métadonnées ;
4. écriture et validation WCS ;
5. accès catalogue ;
6. fonctions mathématiques pures ;
7. validation des candidats ;
8. télémétrie et rapports ;
9. orchestration Near/Blind ;
10. interface et batch.

Après chaque extraction :

- lancer les tests ciblés ;
- lancer la non-régression complète ;
- comparer taux de réussite, RMS et temps ;
- documenter tout changement de comportement volontaire.

## 10.3 Exceptions et erreurs silencieuses

Tout `except Exception` doit être classé :

- protection légitime d'une fonction optionnelle ;
- erreur journalisée en debug ;
- erreur métier convertie en type explicite ;
- défaut à corriger ;
- bloc à supprimer.

Aucune erreur affectant le catalogue, le calcul WCS, l'écriture FITS ou la validation d'une solution ne doit être ignorée silencieusement.

## 10.4 Critères de sortie P2 — découpage

- modules à responsabilité identifiable ;
- dépendances orientées du produit vers le cœur, pas l'inverse ;
- fonctions critiques testables sans GUI ;
- aucune importation du GUI par le cœur ;
- aucun changement de résultat non documenté ;
- taille et complexité des fonctions critiques réduites progressivement.

---

# MISSION P3 — Nouvelle interface utilisateur

## 11. Objectif

Permettre à un utilisateur ne connaissant ni ASTAP, ni les quads, ni les manifestes de résoudre ses images sans comprendre l'architecture interne.

## 11.1 Parcours principal cible

```text
1. Choisir les images
2. Vérifier la bibliothèque
3. Résoudre
4. Consulter les résultats
```

L'écran principal doit montrer au maximum :

- fichiers ou dossier sélectionnés ;
- nombre d'images ;
- état de la bibliothèque ;
- dossier de sortie ;
- règle concernant les WCS existants ;
- bouton Résoudre ;
- progression ;
- statut par fichier ;
- résumé final et accès au journal.

## 11.2 Réglages normaux

Limiter les réglages normaux à :

- bibliothèque de catalogue ;
- dossier de sortie ;
- traitement parallèle Auto ou manuel ;
- copie, sauvegarde ou écrasement ;
- fallback web facultatif ;
- langue ;
- niveau de journal.

## 11.3 Outils avancés séparés

Sortir du parcours principal :

- construction ou réparation d'index ;
- benchmark ;
- reconstruction des hashes ;
- index checker ;
- WCS cleaner ;
- téléchargement manuel des familles ;
- paramètres de quads et de validation ;
- profils expérimentaux ;
- diagnostics développeur.

Ces fonctions peuvent vivre dans un menu **Outils avancés**, une application distincte ou le CLI.

## 11.4 Règles d'architecture GUI

- le GUI appelle uniquement la façade produit ;
- aucune logique astrométrique critique dans les callbacks ;
- aucun paramètre expérimental directement lié à un widget du mode normal ;
- tâches longues hors du thread d'interface ;
- annulation propre et testée ;
- messages d'erreur orientés action ;
- état de l'application déterministe et reproductible.

## 11.5 Critères de sortie P3

Un nouvel utilisateur doit pouvoir :

1. installer ou sélectionner la bibliothèque ;
2. choisir des images ;
3. lancer la résolution ;
4. comprendre les succès et les échecs ;
5. retrouver les fichiers produits ;

sans ouvrir la documentation technique des index.

---

# MISSION P4 — Packaging, documentation et machine vierge

## 12. Objectif

Produire une installation reproductible et cohérente avec le nom du produit ZeSolver.

## 12.1 Packaging Python

À traiter :

- nom de distribution et version cohérents ;
- section `[build-system]` explicite ;
- découverte correcte des packages ;
- inclusion des ressources, layouts, traductions et icônes ;
- points d'entrée GUI et CLI ;
- extras séparés : `gui`, `gpu`, `dev` ;
- licence au format SPDX moderne ;
- exclusion des logs, caches, catalogues locaux et corpus lourds ;
- construction wheel et source distribution ;
- installation editable et installation wheel testées.

## 12.2 Installation sur machine vierge

Tester au minimum :

- environnement Python propre ;
- absence du dépôt dans `PYTHONPATH` ;
- premier lancement sans réglages existants ;
- bibliothèque absente, partielle puis complète ;
- chemins contenant espaces et caractères non ASCII ;
- lancement GUI et CLI ;
- traitement d'un petit corpus ;
- désinstallation ou mise à jour.

Tester les plateformes effectivement annoncées. Ne pas promettre une plateforme non validée.

## 12.3 Documentation minimale de publication

- README orienté utilisateur ;
- guide d'installation ;
- guide de bibliothèque/catalogue ;
- guide rapide de résolution ;
- description des limites connues ;
- politique de sauvegarde FITS ;
- FAQ de diagnostic ;
- guide contributeur ;
- crédits et licences des projets et données amont ;
- changelog ;
- notes de version.

## 12.4 Version de publication

Tant que la couverture générale, l'installation et les tests multi-machines ne sont pas démontrés, préférer une version beta telle que :

```text
0.9.0b1
```

Ne déclarer une version `1.0.0` qu'après satisfaction des critères P0 à P4.

## 12.5 Critères de sortie P4

- installation réussie depuis un artefact publié sur machine vierge ;
- premier lancement guidé ;
- catalogue installable et vérifiable ;
- corpus minimal résolu ;
- documentation suffisante pour un utilisateur externe ;
- licences et crédits vérifiés ;
- aucun fichier de développement local embarqué par erreur.

---

# MISSION P5 — Optimisations GPU et fonctions avancées

## 13. Statut

P5 n'est pas bloquant pour la première publication.

Exemples de travaux P5 :

- optimisation GPU supplémentaire ;
- parallélisme avancé ;
- nouveaux profils d'instruments ;
- amélioration automatique des paramètres de détection ;
- nouvelles projections ou modèles de distorsion ;
- optimisation de taille et chargement des index ;
- nouvelles fonctions de benchmark ou d'analyse.

## 13.1 Conditions d'acceptation

Une optimisation doit :

- conserver les résultats P0 ;
- disposer d'un chemin CPU fonctionnel ;
- démontrer son gain sur benchmark reproductible ;
- ne pas augmenter les faux positifs ;
- ne pas rendre l'installation normale dépendante du GPU ;
- rester désactivable ou revenir proprement au CPU.

---

## 14. Règles générales de développement

### 14.1 Avant toute modification

L'agent doit :

1. lire ce fichier ;
2. identifier la priorité concernée ;
3. inspecter les modules et tests existants ;
4. décrire le comportement actuel ;
5. définir les critères mesurables de réussite ;
6. vérifier qu'une baseline protège la zone modifiée.

### 14.2 Taille des changements

Préférer :

- petites modifications cohérentes ;
- commits séparant refactorisation et changement fonctionnel ;
- extraction sans modification de comportement ;
- migration progressive avec compatibilité temporaire.

Éviter :

- réécriture complète d'un solveur ;
- renommage massif mêlé à une évolution algorithmique ;
- suppression de paramètres avant migration ;
- changement de seuil sans preuve sur le corpus ;
- optimisation fondée sur un seul fichier test.

### 14.3 Tests obligatoires

Pour chaque changement :

- exécuter les tests ciblés ;
- exécuter la compilation ou analyse syntaxique ;
- exécuter la non-régression pertinente ;
- comparer les métriques avant/après ;
- signaler précisément les tests non exécutés et pourquoi.

Ne jamais annoncer qu'un changement est fiable sans indiquer les validations réellement effectuées.

### 14.4 Gestion des performances

Les performances sont importantes, mais viennent après la justesse.

Mesurer séparément :

- chargement et détection ;
- recherche de candidats ;
- validation ;
- écriture ;
- mémoire maximale ;
- temps médian et percentile 95 ;
- taux de succès et de faux positifs.

Une accélération qui réduit la robustesse ne doit pas devenir le comportement par défaut.

### 14.5 Compatibilité et dépendances

- éviter les dépendances lourdes sans bénéfice démontré ;
- conserver un fonctionnement CPU ;
- isoler les dépendances GUI et GPU dans des extras ;
- documenter toute nouvelle dépendance ;
- ne pas utiliser d'API privée d'une bibliothèque sans test et justification ;
- vérifier la compatibilité avec les versions Python annoncées.

### 14.6 Données et fichiers volumineux

Ne pas ajouter au dépôt sans décision explicite :

- catalogues stellaires complets ;
- index 4D volumineux ;
- logs ;
- environnements virtuels ;
- caches Python ;
- résultats de benchmark temporaires ;
- corpus lourd non filtré.

Utiliser des manifestes, checksums, scripts de téléchargement ou artefacts de release.

---

## 15. Format attendu pour chaque mission confiée à un agent

Chaque intervention importante doit produire un compte rendu comprenant :

### Objectif

Ce qui devait être obtenu.

### État initial

Comportement observé, fichiers concernés et risques.

### Modifications

Liste concise des changements réalisés.

### Validation

Commandes exécutées, corpus utilisé et métriques obtenues.

### Comparaison baseline

Succès, RMS, performances et éventuelles différences.

### Limites ou risques restants

Ce qui n'est pas démontré ou doit encore être fait.

### Prochaine étape recommandée

Une seule étape prioritaire, rattachée à P0–P5.

---

## 16. Première séquence de travail recommandée

### Étape A — Geler la baseline

- créer le tag ou commit de référence ;
- archiver les paramètres et manifestes ;
- établir un premier manifeste de corpus ;
- enregistrer les métriques actuelles.

### Étape B — Construire le harnais de non-régression

- automatiser l'exécution du corpus ;
- produire JSON et résumé lisible ;
- définir les tolérances ;
- intégrer les cas négatifs.

### Étape C — Sécuriser les FITS

- vérifier l'intégrité des pixels ;
- ajouter les stratégies copie/sauvegarde ;
- relire et valider tout WCS écrit ;
- tester les interruptions et fichiers invalides.

### Étape D — Spécifier `CatalogLibrary`

- documenter le format de bibliothèque ;
- définir le manifeste commun ;
- définir les API ZeNear et ZeBlind ;
- inventorier la couverture 4D actuelle.

### Étape E — Introduire la façade produit

- créer les modèles de requête et résultat ;
- adapter le CLI puis le GUI ;
- migrer progressivement les réglages ;
- conserver les anciens chemins jusqu'à validation.

La refonte visuelle complète ne commence qu'après stabilisation des étapes A à E.

---

## 17. Définition de « prêt à publier »

ZeSolver est prêt pour une publication publique lorsque :

- la baseline et le corpus sont versionnés ;
- la non-régression est automatisée ;
- les faux WCS sont activement testés ;
- les FITS sont protégés ;
- la bibliothèque ASTAP est unifiée et vérifiable ;
- la couverture 4D est complète ou honnêtement limitée ;
- le GUI et le CLI passent par une façade commune ;
- l'interface principale ne présente plus les outils de développement ;
- l'installation fonctionne depuis un artefact sur machine vierge ;
- la documentation et les licences sont prêtes ;
- les limites connues sont publiées ;
- aucun blocage P0 à P4 ne reste ouvert.

---

## 18. Principe final

La priorité absolue de ZeSolver est la confiance :

- confiance dans la solution WCS ;
- confiance dans l'intégrité du fichier ;
- confiance dans la reproductibilité ;
- confiance dans l'installation ;
- confiance dans les limites annoncées.

**Mesurer avant de modifier. Protéger avant de simplifier. Simplifier avant de publier. Optimiser après avoir fiabilisé.**
