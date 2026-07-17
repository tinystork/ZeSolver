# ZN3.4 promotion gate

Verdict: E - Faux positif ou WCS incorrect

1. M31 gele runtime reste 8/8: 8/8; matrice M31 elargie: 81/84.
2. M106 trouves/valides: 61 / 15.
3. NGC6888 trouves/valides: 17 / 16.
4. Autres groupes couverts: M106, M31, NGC6888, S50.
5. `star_level`/`star_level2`: code couvert par tests synthetiques; le corpus large observe surtout 30 sigma + local.
6. Binning 1/2: fonctions testees; le corpus principal 1080x1920 utilise bin 2.
7. Faux positifs controles negatifs: 0; WCS incorrects/degrades dans corpus positif: 48.
8. WCS verifies independamment quand un WCS FITS oracle existe.
9. RMS 231915: WCS runtime conforme; temoin gele WCS={'WCS_CONFORMANT': 8}.
10. Le RMS eleve 231915 est classe BENIGN_METRIC_EFFECT, mais reste a surveiller.
11. Premier run plus lent: cout froid/import/cache et detection HFD Python.
12. Temps froids/chauds: voir rapports performance.
13. Flux synthetique supprime.
14. L ordre strict est explicite via `img_ranks`, pas via flux physique.
15. Canari 4 etoiles: requalifie insuffisant pour le contrat strict.
16. Fixture synthetique resoluble: contrat documente, integration test limitee dans ce passage.
17. Hints numeriques/textuels: tests dedies.
18. Hors strict inchange: tests dedies et non-regression matrix.
19. Direct teste; CLI/app/GUI restent partiellement couverts seulement.
20. Packaging: audit statique, pas d ASTAP/Lazarus requis par `solve_near` strict.
21. Dumps desactives par defaut.
22. ZeBlind inchange.
23. Cas non couverts: GUI reel, oracles independants pour certains doublons, chemins star_level/star_level2 sur vraies images.
24. Promouvable: non en promotion globale tant que 48 WCS elargis sont incorrects/degrades.
