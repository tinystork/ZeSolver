# ZN3.9 Summary

Verdict: `I - Qualification incomplete` unless all phases were explicitly run and validated.

1. `astap_iso_image_for_solve` supports 2D mono arrays and 3D channel-first cubes; channel-last cubes are flagged unsupported by audit.
2. BSCALE/BZERO are handled by Astropy before the function receives `hdu.data`; no double application is performed by the function.
3. Normalisation `0..1` cannot reach the strict detector in the audited `solve_near` path.
4. Generic CPU fallback cannot replace a strict list according to code guards and replay flags.
5. Images uniques par groupe: `{'M106': 60, 'M31_canonical': 8, 'M31_extended': 45, 'NGC3628': 2, 'NGC6888': 27}`.
6. Oracles ASTAP valides: `142`.
7. M31 canonique Near confirmé: `8/8`.
8. M106 Near confirmé: `60/60`.
9. NGC6888 Near confirmé: `27/27`.
10. NGC3628: `{'images_uniques': 2, 'ASTAP_valides': 2, 'Near_success_true': 2, 'Near_WCS_CONFIRMED': 2, 'Near_WCS_ACCEPTABLE': 0, 'Near_WRONG_FIELD': 0, 'Near_echec': 0, 'generic_fallback_called': 0, 'temps_median': 2.0707940370048163}`.
11. Echecs Near restants: `0`.
12. Etages d'echec: `Counter()`.
13. WCS Near incorrect accepté: `0`.
14. Validation WCS indépendante stellaire complète: non, comparaison WCS ASTAP propre utilisée comme validation minimale.
15. Fixture Near-failure naturelle: `None`.
16. Fixture contrôlée: `NOT_CREATED_IN_THIS_RUN`.
17. Fallback 4D appelé exactement une fois: non testé dans cette passe.
18. Profil 4D testé produit: `zeblind_4d_experimental` / status `BLIND4D_READY`.
19. WCS 4D incorrect accepté: 0 observé, mais chain non exécutée.
20. Couverture 4D distinguée: preflight oui, exécution non.
21. Gate holdout parfaitement séparé: non.
22. Gate enforce par défaut: non.
23. Rejet Near empêche écriture WCS: garanti par tests antérieurs, non rejoué ici.
24. Rejet déclenche 4D: non rejoué ici.
25. Performance 232102 proche 1.3s: voir performance JSON si run Near exécuté.
26. Hausses 230409/233459: non qualifiées sans matrice froide/chaude complète.
27. Points d'entrée: non qualifiés dans cette passe.
28. ASTAP absent runtime: oui pour Near, utilisé seulement oracle offline.
29. Backend historique inactif: aucun appel observé.
30. Promotion stricte: non.
31. Hors qualification: full stellar holdout, chain 4D réelle, entrypoints CLI/app/GUI, performance cold/warm complète.
