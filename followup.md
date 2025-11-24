# Feuille de Route pour l'Évolution de ZeBlind Solver

## 1. Contexte et Diagnostic

L'analyse actuelle de `zeblind` a révélé deux goulots d'étranglement majeurs qui expliquent à la fois sa lenteur et son faible taux de succès par rapport à des solutions comme ASTAP :

1.  **Index Monolithique et Chargement Intégral (`Eager Loading`) :** `zeblind` construit un index unique et massif pour tout le ciel (`.npz`) et le charge entièrement en mémoire au démarrage.
    *   **Conséquence :** Temps de chargement extrêmement long et consommation mémoire rédhibitoire.

2.  **Génération Fragile des Motifs-Étoiles (Quads) :** La sélection des étoiles pour former les "empreintes digitales" (quads) est basée sur des heuristiques trop restrictives (ex: les N étoiles les plus brillantes).
    *   **Conséquence :** Le solveur ne génère pas les bons motifs depuis l'image à analyser, et ne trouve donc aucune correspondance dans l'index. C'est la cause principale de l'échec de résolution.

## 2. Plan d'Action Détaillé

L'objectif est de restructurer `zeblind` pour adopter les concepts qui font la force d'ASTAP : une structure d'index modulaire et une génération de motifs robuste.

---

### Phase 1 : Remplacer l'Index Monolithique par un Index "en Tuiles"

**Objectif :** Éliminer le temps de chargement et réduire l'empreinte mémoire à quasi zéro, en imitant l'approche "à la demande" d'ASTAP.

- [x] **Étape 1.1 : Définir un format d'index "en seaux" (Bucketed).**
    *   Abandonner le fichier `.npz` unique. L'index sera découpé en un grand nombre de petits fichiers binaires (par exemple, 4096 fichiers).
    *   Chaque fichier (ou "seau") correspondra à une plage de hashes. Par exemple, `index_000.bin` contiendra tous les quads dont le hash commence par `0x000...`, `index_001.bin` pour ceux commençant par `0x001...`, etc. (en se basant sur les 12 premiers bits du hash).
    *   Chaque fichier seau contiendra une liste de hashes (triés) et les données associées (coordonnées des étoiles du quad).

- [x] **Étape 1.2 : Mettre à jour le script de construction de l'index (`tools/build_blind_index.py`).**
    *   Modifier le script pour qu'au lieu de tout stocker en mémoire, il écrive chaque quad généré dans le fichier "seau" approprié en fonction du préfixe de son hash.
    *   **Point crucial :** À la fin du processus, chaque fichier seau doit être trié par valeur de hash. Cela permettra une recherche par dichotomie (binaire) ultra-rapide.

- [x] **Étape 1.3 : Réécrire la logique de chargement et de recherche (`zesolver/blindindex.py`).**
    *   La méthode `BlindIndex.load` ne doit plus rien charger. Elle doit simplement vérifier la présence du répertoire contenant les fichiers de l'index.
    *   La méthode `BlindIndex.query` doit être complètement réécrite :
        1.  Pour chaque hash généré depuis l'image, déterminer le fichier seau pertinent (ex: `hash >> 52` pour un préfixe de 12 bits sur un hash de 64 bits).
        2.  Utiliser le **memory-mapping** (`mmap` en Python) pour ouvrir ce petit fichier seau. C'est quasi-instantané et n'utilise presque pas de RAM.
        3.  Effectuer une **recherche binaire** (`numpy.searchsorted` ou `bisect`) sur les hashes dans le fichier mappé en mémoire pour trouver des correspondances.

---

### Phase 2 : Rendre la Génération de Quads Robuste

**Objectif :** Augmenter drastiquement le taux de succès en générant les mêmes motifs stables et variés qu'ASTAP, indépendamment des variations de magnitude des étoiles.

- [x] **Étape 2.1 : Remplacer l'algorithme de sélection des quads.**
    *   Cette modification doit être appliquée à la fois dans le **constructeur d'index** et dans le **solveur** (`zeblindsolver/asterisms.py`).
    *   Abandonner la logique "prendre les N étoiles les plus brillantes".
    *   Adopter l'approche d'astrometry.net :
        1.  Itérer sur une grande partie des étoiles de l'image (ou de la tuile catalogue). Appelons l'étoile de départ `A`.
        2.  Pour chaque `A`, trouver une voisine `B` à une certaine distance. Pour être robuste au zoom, il faut tester plusieurs échelles de distance (proches, moyennes, lointaines).
        3.  Considérer le segment `A-B` comme une base. Chercher maintenant deux autres étoiles, `C` et `D`, dont les positions relatives par rapport à ce segment `A-B` sont stables.
        4.  La géométrie du quad `A,B,C,D` est alors utilisée pour calculer le hash.
    *   Cette méthode garantit la création d'une collection de motifs beaucoup plus riche et plus stable.

- [x] **Étape 2.2 : Reconstruire l'index complet.**
    *   Une fois le nouvel algorithme de génération de quads implémenté, il sera nécessaire de reconstruire entièrement la base de données de l'index. Cet index sera plus grand mais infiniment plus utile.
    *   `zebuildindex` expose désormais l'option `--quads-only` pour relancer uniquement la phase de hash (sans re-projeter tout ASTAP), ce qui facilite les reconstructions complètes à chaque changement du sampler.

---

### Phase 3 : (Optionnel) Améliorer la Détection d'Étoiles

**Objectif :** Fournir une liste d'étoiles plus propre et plus complète en entrée de la phase 2, améliorant encore les chances de succès.

- [x] **Étape 3.1 : Implémenter un seuillage adaptatif.**
    *   Dans `zeblindsolver/star_detect.py`, remplacer le calcul global de la `moyenne` et de l'`écart-type` par un calcul local. Une méthode simple est de diviser l'image en une grille (ex: 8x8) et de calculer ces valeurs pour chaque cellule, afin de s'adapter aux variations de fond de ciel.

- [ ] **Étape 3.2 : Envisager des méthodes plus avancées.**
    *   Pour aller plus loin, des techniques comme la **transformée en ondelettes** ou la **Différence de Gaussiennes (DoG)** sont la norme dans les logiciels professionnels (comme SExtractor) pour séparer les étoiles du bruit et des artéfacts à différentes échelles.
    *   Piste DoG multi-échelle : (1) estimer le bruit local (médiane ou gaussien large) ; (2) filtrer l'image avec deux gaussiennes centrées autour du FWHM estimé (ex. sigma et 1.6–2.0*sigma) puis soustraire pour obtenir une carte DoG qui maximise les sources ponctuelles ; (3) calculer une carte sigma locale (grille 8x8/16x16) sur la carte DoG et seuiller à k*sigma (k ajustable) ; (4) extraire les composantes connexes, rejeter les blobs allongés/gros (cosmiques/traînées), recentrer par barycentre, mesurer flux et FWHM locale ; (5) fusionner les détections proches issues de plusieurs paires de sigmas.
    *   Piste ondelettes (starlet) : (1) décomposition à trous sur 3–5 échelles ; (2) estimation du bruit par échelle (MAD) puis seuillage dur/soft à k*sigma ; (3) somme pondérée des cartes seuillées pour une carte de probabilité ; (4) détection des maxima locaux sur cette carte et report au plan d'origine pour une position sub-pixel ; (5) option de pondération par l'échelle dominante pour écarter galaxies/étirements.
    *   Intégration (sans code pour l'instant) : prévoir un paramètre `--detect-method {adaptive,dog,wavelet}` (et une option GUI), exposer `--dog-sigma`, `--dog-k`, `--wavelet-levels`, `--wavelet-k`, et retomber automatiquement sur la méthode adaptative si les libs requises manquent. Ajouter des tests synthétiques (PSF gaussienne + bruit blanc + gradient) pour comparer TP/FP et la stabilité sub-pixel entre méthodes.
    Pour aller plus loin, lors de l'implémentation, vous pourriez vous inspirer (ou même utiliser) des fonctions existantes dans des bibliothèques comme photutils (affiliée à Astropy), qui implémente déjà des détecteurs d'étoiles performants (par exemple DAOStarFinder et IRAFStarFinder) et des outils de segmentation.

En suivant ce plan, `zeblind` sera transformé d'un prototype conceptuel à un solveur performant, rapide et efficace, fidèle à l'inspiration d'ASTAP.
