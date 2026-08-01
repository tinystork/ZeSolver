# AGENT.md — État courant ZeSolver

**Projet :** ZeSolver  
**Écosystème :** ZeMosaic / ZeSeestarStacker  
**Auteur principal :** Tinystork — Tristan Nauleau  
**Mise à jour :** 1 août 2026
**Phase active :** Release Candidate acceptance — validation manuelle finale

---

## Portée

Ce fichier s’applique à tout le dépôt, sauf instruction plus spécifique dans un
sous-répertoire.

Les rapports de mission détaillés vivent dans `docs/stabilization/`. Ne pas
recommencer S5C/S5D sans régression reproduite.

---

## État Validé

| Chantier | État |
|---|---|
| P0 — baseline et non-régression | Terminé, à préserver |
| P1 — `CatalogLibrary` | Intégré |
| P2 — réglages, profils et cœur | Stabilisés |
| P3A — GUI/pipeline | Terminé |
| P3B-1D — gestionnaire de bibliothèques | Terminé |
| S5C — cycle batch, ressources, mémoire, preflight GUI | Terminé |
| S5D — partitionnement D50 fixed32 et recherche progressive | Terminé |
| S5D-2 — adoption CatalogLibrary, GUI, négatifs | Terminé |
| S5D-3 — budget progressif et réutilisation quads image | Terminé |
| S5E — persistance instrument et vérification CatalogLibrary | Terminé |
| P3B-1E — distribution officielle des Bibliothèques ZeSolver | Intégré |
| P3B-1F — assistant de démarrage et activation transactionnelle | Terminé |
| P3B-1G/P3B-1G1 — téléchargement parallèle, reprise, pause | Terminé |
| P3B-1H — sélecteur Système/Clair/Sombre | Terminé |

Le produit fixed32 complet est le chemin runtime normal quand disponible:

```text
/home/tristan/ZeSolverCatalog/new
blind4d_catalog_mode_effective=library-view
blind4d_index_count=47
blind4d_covered_tiles=1476
blind4d_total_tiles=1476
blind4d_all_sky=true
blind4d_external_fallback_used=false
```

Le monolithe `direct-d50` reste conservé comme compatibilité, mais ne doit pas
être sélectionné dans l’ordre runtime par défaut si les 47 shards fixed32 sont
valides.

---

## Invariants

- Ne pas modifier les algorithmes ZeNear ou ZeBlind sans mission explicite.
- Ne pas relâcher `quality_inliers`, `quality_rms`, `match_radius` ou `code_tol`
  pendant les missions de persistance/GUI.
- Ne pas reconstruire D50 pour S5E.
- Ne pas supprimer le monolithe ni les shards existants.
- Conserver le routage logique Near puis Blind lazy.
- Le GUI ne doit pas bloquer le thread Qt avec des hashes complets ou des
  chargements NPZ lourds au démarrage.

---

## État Produit Actuel

- Le wizard de démarrage couvre l'installation officielle, le paquet local, la
  réutilisation d'une Bibliothèque ZeSolver existante, et la base ASTAP Near-only.
- Les parcours Bibliothèque activent transactionnellement la bibliothèque en
  mode produit:
  `catalog_library_path=<path>`, `near_catalog_mode=auto`,
  `blind4d_catalog_mode=auto`.
- Le téléchargement officiel est parallèle, multi-source, reprend les `.part`
  compatibles, conserve le cache terminé, et expose Pause/Reprendre/Annuler.
- Le thème GUI est configurable et persistant: Système, Clair, Sombre.
- Le rollback avancé historique reste disponible hors wizard:
  `near_catalog_mode=legacy-index` et `blind4d_catalog_mode=external-manifest`.

---

## Limites Réelles Restantes

- FAST invalide conservativement si le chemin canonique change.
- FAST vérifie l'identité légère déclarée et les métadonnées de fichiers; FULL
  reste la route explicite pour les hashes complets et validations coûteuses.
- La validation manuelle Windows reste le gate final avant promotion de `test`
  vers `main` et beta publique.
- Ne pas déclarer `PRODUCTION_READY` avant le rapport de Release Candidate
  acceptance sur le paquet destiné aux utilisateurs.

---

## Prochaine Étape

```text
READY_FOR_RELEASE_CANDIDATE_ACCEPTANCE
```

Le passage à `PRODUCTION_READY_FOR_PUBLIC_BETA` nécessite encore le gate final
sur installation fraîche/profil vierge décrit dans les rapports de stabilisation.
