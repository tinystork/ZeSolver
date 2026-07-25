# AGENT.md — État courant ZeSolver

**Projet :** ZeSolver  
**Écosystème :** ZeMosaic / ZeSeestarStacker  
**Auteur principal :** Tinystork — Tristan Nauleau  
**Mise à jour :** 25 juillet 2026
**Phase active :** P3B-1E — intégration de distribution officielle des Bibliothèques ZeSolver

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

## Limites Réelles Restantes

- Le cache FAST invalide conservativement si le chemin canonique change.
- FAST vérifie l'identité légère déclarée et les métadonnées de fichiers; FULL
  reste la route explicite pour les hashes complets et validations coûteuses.
- Aucun packaging/distribution officielle des bibliothèques n'est encore intégré.

---

## Prochaine Étape

```text
READY_FOR_P3B1E_LIBRARY_DISTRIBUTION_INTEGRATION
```

La prochaine étape unique devient:

```text
P3B-1E — intégration de distribution officielle des Bibliothèques ZeSolver
```

Tant que S5E n’est pas validée, ne pas engager P3B-1E ni P4.
