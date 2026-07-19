# AGENT.md — Mission active de ZeSolver

**Projet :** ZeSolver
**Écosystème :** ZeMosaic / ZeSeestarStacker
**Auteur principal :** Tinystork — Tristan Nauleau
**Mise à jour :** 19 juillet 2026
**Phase active :** P1D — Achèvement de la bibliothèque ASTAP unique
**Statut :** P3B en pause, cœur stabilisé, P1D-4 à préparer

---

## 1. Portée

Ce fichier s’applique à tout le dépôt, sauf instruction plus spécifique dans un
sous-répertoire.

Il contient uniquement :

- l’état actuel ;
- les invariants à préserver ;
- la mission active ;
- les validations obligatoires ;
- les travaux encore ouverts.

Les étapes déjà accomplies sont documentées dans `docs/stabilization/` et
`docs/architecture/`. Elles ne doivent pas être recommencées sans régression
reproduite ou demande explicite.

L’ancienne mission générale du 16 juillet 2026 est archivée dans :

```text
docs/stabilization/original_stabilization_roadmap_20260716.md
```

---

## 2. État actuel

| Chantier | État |
|---|---|
| P0 — baseline et non-régression | Terminé, à préserver |
| P0 — pixels et WCS | Validé pour poursuivre ; audit final avant publication |
| P1 — `CatalogLibrary` | Cœur et adaptateurs intégrés ; fermeture ASTAP unique active |
| P1 — couverture ZeBlind 4D | Partielle et explicitement détectée |
| P1D — bibliothèque ASTAP unique | **Phase active : P1D-4 prochaine** |
| P1D-1A — provider ZeNear ASTAP-native | Terminé |
| P1D-1B — basculement produit ZeNear | Terminé |
| P1D-2A — provenance et plan d'adoption | Terminé |
| P1D-2B — adoption atomique CatalogLibrary | Terminé |
| P1D-3A — builder Blind 4D direct ASTAP | Terminé |
| P1D-3B — validation runtime Blind 4D directe | Terminé |
| P1D-4 — manifeste 4D possédé par la bibliothèque | Prochaine |
| P2 — réglages, profils et façade | Stabilisés |
| P2 — extraction du cœur | Suffisante pour P3B |
| P3A — GUI/pipeline | Terminé |
| P3A-V1 — Stop et relance | Terminé |
| P3A-V2 — fin exactement une fois | Terminé |
| P3A-V3 — progression et état WCS temps réel | Terminé |
| P3B — GUI simplifié | En pause, non abandonné |
| P4 — packaging et publication | Ouvert |
| P5 — optimisations avancées | Différé |

Le projet est :

```text
READY_FOR_P1D4_LIBRARY_OWNED_BLIND4D_MANIFEST
```

Il n’est pas encore :

```text
READY_FOR_PUBLIC_RELEASE
```

---

## 3. Invariants absolus

### 3.1 Résolution

Pendant P1D et lors de la reprise P3B, ne pas modifier sans mission dédiée :

- ZeNear ;
- ZeBlind 4D ;
- les seuils ;
- les profils ;
- les catalogues ;
- les index ;
- les règles d’acceptation WCS ;
- l’ordre Near → Blind 4D → Astrometry.net ;
- les formats de résultats.

Toute modification simultanée des algorithmes, seuils, profils ou règles
d’acceptation est interdite pendant P1D. Une refonte GUI qui modifie un
résultat astrométrique est une régression.

### 3.2 Fichiers et WCS

- Ne jamais modifier les pixels.
- Ne pas écraser silencieusement un WCS existant.
- Préserver les HDU et les métadonnées non concernées.
- Les rasters utilisent un sidecar WCS ; le raster source reste inchangé.
- Pour un FITS, le statut principal reflète le WCS du HDU `PRIMARY`.
- Une interruption ne doit pas laisser un fichier annoncé comme résolu avec un
  en-tête incomplet.
- Toute modification du chemin d’écriture impose une comparaison avant/après et
  une relecture WCS.

### 3.3 Routage

Le routage `AUTO / PIPELINE / LEGACY` reste explicite :

- FITS compatible → pipeline autorisé ;
- raster → legacy ;
- fallback Astrometry.net incompatible avec pipeline → legacy ;
- moteur forcé incompatible → erreur claire ;
- aucun fallback silencieux ;
- route legacy conservée jusqu’à validation finale du nouveau GUI.

### 3.4 Cycle de vie GUI

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

- progression en temps réel ;
- compteur traité / total / restant ;
- mise à jour des lignes au fil du run ;
- rafraîchissement après nettoyage WCS ;
- Stop réactif ;
- relance après Stop ;
- fermeture propre ;
- rejet des callbacks tardifs ;
- widgets modifiés uniquement dans le thread Qt principal ;
- aucune fausse progression à 100 % après annulation.

### 3.5 Architecture

Le cœur ne doit importer aucun module GUI.

La chaîne cible est :

```text
GUI
→ ProductSettings / SolveRequest
→ GuiSolveController
→ PIPELINE ou LEGACY
→ cœur
→ résultat adapté
→ GUI
```

Le nouveau GUI ne doit pas appeler directement `solve_near()` ou
`solve_blind()`.

---

## 4. Mission active — P1D

### 4.1 Objectif

Fermer l’architecture de bibliothèque ASTAP unique avant de reprendre la
simplification GUI.

Objectif produit :

```text
une Bibliothèque ZeSolver sélectionnée par l’utilisateur
→ ZeNear lit ASTAP/HNSKY depuis cette bibliothèque
→ ZeBlind 4D utilise des index dérivés de ces mêmes sources
→ provenance, versions, paramètres et empreintes reliés
→ plus de choix normal séparé db_root / index_root / familles / manifeste 4D
```

Astrometry.net reste un fallback web facultatif et distinct.

### 4.2 Invariants P1D

Pendant P1D :

- ne pas modifier les algorithmes Near ;
- ne pas modifier les algorithmes Blind ;
- ne pas modifier les seuils ;
- ne pas modifier les profils ;
- ne pas modifier les catalogues ou index existants ;
- ne pas modifier les FITS ou les pixels ;
- ne pas modifier le comportement GUI hors branchement strictement nécessaire ;
- conserver les invariants P0, P2 et P3A ;
- conserver un rollback legacy explicite ;
- ne jamais masquer la couverture partielle Blind 4D.

### 4.3 Séquence P1D retenue

Ordre de travail :

1. **P1D-0 — Audit de fermeture**
   - cartographier les dépendances historiques restantes ;
   - recommander la stratégie ;
   - aucune modification de solveur.
2. **P1D-1A — Provider ZeNear ASTAP-native** — terminé
   - fournisseur de tuiles ASTAP natif ajouté ;
   - provider historique conservé comme rollback/oracle ;
   - parité provider démontrée.
3. **P1D-1B — Basculement produit ZeNear** — terminé
   - `CatalogLibrary` valide sélectionne ASTAP-native par défaut ;
   - routes PIPELINE et LEGACY partagent la même politique ;
   - rollback `legacy-index` explicite conservé ;
   - aucun fallback silencieux vers legacy en mode natif.
4. **P1D-2 — Manifest/provenance/réparation** — terminé
   - enrichir `catalog.json` pour reconstruction déterministe ;
   - ajouter adoption `REFERENCE_EXISTING` non destructive.
5. **P1D-3 — Builder Blind 4D depuis ASTAP**
   - **P1D-3A terminé** : builder direct ASTAP, coeur partagé, déterminisme
     et comparaison exacte avec le chemin historique actuel ;
   - **P1D-3B terminé** : validation runtime baseline produit vs index directs
     sur M106 all30, mini-corpus intégré, cas difficiles et contrôles négatifs.
6. **P1D-4 — Manifeste 4D possédé par la bibliothèque** — prochaine
   - générer/valider la vue stricte 4D depuis `catalog.json`.
7. **P1D-5 — Surface produit**
   - masquer les chemins `db_root`, `index_root`, familles et manifestes 4D du
     mode normal ;
   - conserver les overrides en outils avancés/diagnostic.

### 4.4 Gate P1D-0

P1D-0 est fermé lorsque :

- le flux actuel ZeNear est documenté ;
- le flux actuel ZeBlind 4D est documenté ;
- les dépendances historiques restantes sont inventoriées ;
- les stratégies compatibilité et ASTAP-native sont comparées ;
- une prochaine étape unique est choisie ;
- `AGENT.md` marque P3B comme en pause et P1D comme phase active ;
- seuls les documents sont modifiés.

Gate attendu :

```text
READY_FOR_P1D1_ASTAP_RUNTIME_UNIFICATION
```

---

## 5. Mission en pause — P3B

P3B est mise en pause, pas abandonnée. Elle reprend après fermeture suffisante
de P1D pour éviter de simplifier le GUI autour de surfaces catalogue encore
historiques.

### 5.1 Objectif P3B

Créer une interface principale simple permettant à un utilisateur non expert de :

```text
1. choisir les images
2. vérifier la bibliothèque
3. choisir la politique WCS
4. résoudre
5. suivre la progression
6. consulter les résultats
```

L’utilisateur normal ne doit pas avoir besoin de comprendre les familles ASTAP,
les manifestes 4D, les quads, les buckets ou les seuils internes.

### 5.2 Écran principal attendu

L’écran principal doit montrer au maximum :

- dossier ou fichiers d’entrée ;
- nombre et état initial des images ;
- état de la bibliothèque ZeSolver ;
- dossier de sortie ou politique d’écriture ;
- règle concernant les WCS existants ;
- fallback web facultatif ;
- bouton **Résoudre** ;
- bouton **Stop** ;
- barre de progression ;
- compteur traité / total / restant ;
- statut par fichier ;
- résumé final ;
- accès au journal.

### 5.3 Outils avancés

Sortir du parcours principal :

- benchmark ;
- construction ou réparation d’index ;
- reconstruction des hashes ;
- index checker ;
- explorateur de catalogues ;
- WCS Cleaner autonome ;
- téléchargements manuels ;
- paramètres Near/Blind ;
- paramètres de quads ;
- profils expérimentaux ;
- diagnostics développeur.

Ces fonctions peuvent rester dans :

- un menu **Outils avancés** ;
- des fenêtres distinctes ;
- des utilitaires séparés ;
- le CLI.

Ne supprimer aucune capacité utile uniquement pour alléger l’écran principal.

### 5.4 `CatalogLibrary`

Le mode normal doit présenter un seul concept :

```text
Bibliothèque ZeSolver
```

Il ne doit plus demander séparément :

- la base ASTAP ;
- le dossier d’index ;
- le manifeste 4D ;
- les familles.

Les chemins historiques peuvent rester dans une zone de compatibilité pendant la
migration.

La couverture partielle doit rester visible. Ne jamais la présenter comme
all-sky.

### 5.5 Non-objectifs

P3B ne traite pas :

- la génération de nouveaux index ;
- l’extension de couverture 4D ;
- l’optimisation GPU ;
- le packaging ;
- les installateurs ;
- la publication ;
- les nouvelles fonctions astrométriques.

---

## 6. Stratégie de migration P3B

Ne pas remplacer le GUI en une seule opération.

Ordre recommandé :

1. figer une baseline locale P3A-V3 ;
2. spécifier le parcours, les états et les erreurs ;
3. créer une coque GUI importable et testable ;
4. brancher entrées et scan ;
5. brancher `CatalogLibrary` ;
6. brancher politique WCS et réglages produit ;
7. brancher `GuiSolveController` ;
8. brancher progression, Stop, résultats et journal ;
9. déplacer les outils avancés ;
10. comparer ancien et nouveau GUI ;
11. retirer l’ancien GUI uniquement après parité.

Pendant la migration :

- conserver un rollback clair ;
- séparer refactorisation et changement fonctionnel ;
- préférer de petits commits locaux ;
- ne pas déplacer les solveurs pour des raisons purement visuelles ;
- ne pas multiplier les nouvelles abstractions sans usage réel.

---

## 7. Méthode de travail

Avant toute modification :

1. lire ce fichier ;
2. identifier la sous-phase active P1D ou P3B ;
3. lire seulement les rapports pertinents ;
4. inspecter le code et les tests ;
5. reproduire le comportement actuel ;
6. définir des critères mesurables ;
7. vérifier `git status` ;
8. ne pas toucher aux changements utilisateur sans demande.

Aucune erreur affectant le routage, le catalogue, le WCS, l’écriture, Stop, la
progression ou la terminaison ne doit être ignorée silencieusement.

Les erreurs utilisateur doivent indiquer une action possible.

---

## 8. Validation obligatoire

### 8.1 Tests ciblés

Exécuter les tests directement liés aux fichiers modifiés.

Pour le GUI, couvrir au minimum :

- sélection moteur ;
- pipeline ;
- legacy ;
- progression temps réel ;
- absence de doublons ;
- nettoyage WCS ;
- Stop ;
- relance ;
- callback tardif ;
- terminaison exactement une fois ;
- copie du log ;
- fermeture.

Commande indicative :

```bash
QT_QPA_PLATFORM=offscreen \
.venv/bin/python -m pytest tests/test_gui_* -q
```

### 8.2 Barrières générales

Après une étape P1D ou P3B significative :

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

Tout skip externe doit être explicite et mentionné dans le rapport.

Aucune nouvelle catégorie de warning sans justification.

### 8.3 Validation graphique réelle

Avant de fermer une sous-phase GUI, tester dans une session graphique réelle :

1. FITS forcé en pipeline ;
2. AUTO avec route réellement sélectionnée ;
3. raster et sidecar ;
4. nettoyage WCS intégré ;
5. progression en temps réel ;
6. Stop puis relance ;
7. fermeture pendant un run ;
8. journal et résumé ;
9. langue, si la zone est traduite.

Ne jamais présenter un test offscreen comme un test graphique réel.

Toute modification d’écriture impose aussi un contrôle d’intégrité des pixels.

---

## 9. Travaux restant après P1D/P3B

### 9.1 Couverture 4D

Avant une promesse générale :

- compléter la couverture ;
- ou publier une bêta explicitement limitée ;
- versionner le manifeste de couverture ;
- tester les zones hors couverture ;
- documenter le fallback.

### 9.2 P4 — Packaging

P4 doit traiter :

- nom de distribution cohérent avec ZeSolver ;
- version de publication cohérente ;
- section `[build-system]` ;
- wheel et sdist ;
- points d’entrée GUI et CLI ;
- ressources embarquées ;
- extras `gui`, `gpu` et `dev` ;
- dépendances GPU réellement facultatives ;
- fonctionnement CPU sans CUDA ;
- installation editable et depuis wheel ;
- machine vierge ;
- premier lancement ;
- chemins avec espaces et caractères non ASCII ;
- mise à jour et désinstallation.

La version inscrite dans `pyproject.toml` ne constitue pas une preuve de maturité.

### 9.3 Documentation

Avant publication :

- README utilisateur ;
- guide d’installation ;
- guide de bibliothèque ;
- guide rapide ;
- limites connues ;
- politique WCS et sauvegardes ;
- FAQ ;
- guide contributeur ;
- changelog ;
- crédits et licences.

### 9.4 P5

Les optimisations GPU, nouvelles projections et nouveaux profils restent différés
jusqu’à la fermeture de P4.

---

## 10. Dépendances

- Conserver un chemin CPU fonctionnel.
- PySide6 reste une dépendance GUI isolable.
- Les dépendances GPU doivent être facultatives.
- Ne pas imposer CUDA à un utilisateur CPU.
- Documenter toute nouvelle dépendance.
- Éviter les API privées non testées.
- Ne pas annoncer une plateforme non validée.
- Ne pas ajouter une dépendance lourde pour un simple effet visuel.

---

## 11. Git et fichiers volumineux

Avant et après chaque mission :

```bash
git status --short --branch
git diff --check
```

Règles :

- aucun push sans autorisation explicite ;
- ne pas restaurer un changement utilisateur sans demande ;
- commits locaux seulement lorsque la mission le prévoit ;
- rollback clair pendant P1D/P3B ;
- ZIP de transmission non suivis par Git ;
- ne pas ajouter catalogues, index, corpus, logs, venv ou caches ;
- utiliser manifestes, checksums, téléchargements ou artefacts de release.

---

## 12. Rapport attendu

Chaque mission importante doit produire :

1. objectif ;
2. état initial observé ;
3. fichiers modifiés ;
4. comportement avant/après ;
5. tests ciblés ;
6. non-régression ;
7. tests manuels réels ;
8. warnings ;
9. tests non exécutés et raison ;
10. état Git ;
11. limites ;
12. une seule prochaine étape ;
13. décision de gate.

Ne jamais prétendre qu’un test a été exécuté s’il ne l’a pas été.

---

## 13. Gates

### Sortie P1D-0

Conclure :

```text
READY_FOR_P1D1_ASTAP_RUNTIME_UNIFICATION
```

uniquement si l’audit localise les dépendances historiques, compare les
stratégies A/B, choisit une seule prochaine étape et ne modifie que la
documentation.

Sinon :

```text
NOT_READY_FOR_P1D1_ASTAP_RUNTIME_UNIFICATION
```

### Sortie P1D-1B

Conclure :

```text
READY_FOR_P1D2_CATALOG_PROVENANCE_AND_ADOPTION
```

uniquement si :

- une `CatalogLibrary` valide suffit à lancer ZeNear sans ancien index ;
- ASTAP-native est le provider effectif avec une bibliothèque ;
- PIPELINE et LEGACY utilisent la même politique centrale ;
- aucun manifeste historique ou `tiles/*.npz` n'est consulté en mode natif ;
- aucun fallback silencieux vers le provider legacy n'existe ;
- le rollback `legacy-index` explicite fonctionne ;
- la télémétrie annonce le provider réel et les fallbacks réels ;
- la comparaison externe ne montre aucune régression inexpliquée ;
- l'intégrité FITS et catalogue est préservée ;
- les barrières automatisées sont vertes.

Sinon :

```text
NOT_READY_FOR_P1D2_CATALOG_PROVENANCE_AND_ADOPTION
```

### Sortie P3B

Conclure :

```text
READY_FOR_P4_PACKAGING
```

uniquement si :

- le parcours principal est simple ;
- `CatalogLibrary` est la ressource produit visible ;
- les outils avancés sont séparés ;
- FITS et rasters fonctionnent ;
- progression, Stop, relance et fermeture fonctionnent ;
- l’ancien GUI peut être retiré sans perte essentielle ;
- les tests automatisés sont verts ;
- la validation graphique réelle est positive ;
- les limites de couverture sont visibles.

Sinon :

```text
NOT_READY_FOR_P4_PACKAGING
```

### Publication

ZeSolver ne peut être déclaré publiable qu’après P4 :

```text
READY_FOR_PUBLIC_BETA
```

---

## 14. Principe final

La priorité de ZeSolver est la confiance :

- confiance dans le WCS ;
- confiance dans les fichiers ;
- confiance dans le cycle GUI ;
- confiance dans la reproductibilité ;
- confiance dans les limites annoncées ;
- confiance dans l’installation.

**Protéger les résultats acquis. Simplifier le produit. Publier seulement ce qui
est démontré.**
