# P1D-3B — Validation runtime des index Blind 4D directs

## 1. Corrections préalables

Deux corrections ont été faites avant la validation corpus.

- `tools/compare_blind4d_builders.py` utilise maintenant le défaut moteur de `Astap4DBuildConfig` pour `mag_cap` et le rapport expose la metadata écrite par l'index direct. Un rapport ne peut plus annoncer une valeur différente de celle réellement transmise au builder.
- `build_4d_index_from_payload_tiles()` référence maintenant la position réellement émise dans `tile_keys` lorsqu'une tuile précédente ne produit aucun record. Les cas `vide -> valide`, `valide -> vide` et `valide -> vide -> valide` sont couverts, avec contrôle de `Quad4DIndex.search_records()`.

Le harnais P1D-3B a aussi reçu deux corrections de rapport uniquement :

- fallback de l'échelle WCS depuis `wcs_metrics` lorsque `stats["pix_scale_arcsec"]` est absent ;
- priorité donnée à l'équivalence runtime baseline/direct lorsque le WCS source FITS n'est pas un oracle canonique fiable.

## 2. Configuration exacte

Configuration de génération directe figée :

```json
{
  "code_tol_recommended": 0.015,
  "dtype": "float32",
  "family": "d50",
  "level": "S",
  "mag_cap": 15.0,
  "max_quads_per_tile": 40000,
  "max_stars_per_tile": 2000,
  "quad_schema": "astrometry_ab_code_4d_v1",
  "sampler_tag": "catalog_ring_coverage",
  "source_max_stars": 2000,
  "source_star_truncation_mode": "native_prefix"
}
```

Configuration runtime commune A/B :

- `quad_hash_schema=astrometry_ab_code_4d_v1`
- `validation_catalog_policy=union_candidate_tiles`
- `source_policy=diagnostic_unfiltered`
- `code_tol=0.015`
- `max_stars=120`
- `max_quads=2500`
- `max_hits=2000`
- `max_hits_per_image_quad=8`
- `max_hypotheses=2000`
- `max_accepts=64`
- `match_radius_px=3.0`
- `quality_inliers=40`
- `quality_rms=1.2`
- `pixel_scale_range=1.79..2.99 arcsec/px`
- `blind_astrometry_4d_search_budget_s=45`

ZeNear et les fallbacks hors Blind 4D ne sont pas utilisés ; `solve_blind()` est appelé sur des copies FITS sans WCS avec liste explicite d'index 4D.

## 3. Empreintes des index directs

Les six index ont été construits deux fois dans `/tmp/p1d3b_direct_runtime_m9e0xbxr`. Les empreintes scientifiques A/B sont identiques.

| tuile | empreinte scientifique | SHA256 fichier A | build A/B |
|---|---|---|---:|
| `d50_2602` | `46ef4056caf08dc75ab07bdfb6c1e63eeca303efe7a0602b0fc2ad1c9364128c` | `2dd56343ac1fdcb0599fc04bed4d06b3fe7f3d0a3b91fbd94c7f4c5188221b03` | 50.51s / 45.71s |
| `d50_2644` | `9332097036a5f83bb02187cf9d6bfafd892af23d4226d5842124762f9e36314f` | `af1f4e7701d8af41aaa5b1ed953a05d3f9c04eb6f79a1c08f3112b255881b628` | 48.58s / 46.70s |
| `d50_2645` | `0162724c7b8c003bb21e8d5ea7f10f7f26fe492bdfc053e8085ad9d08d3ef3e8` | `231b1a4eadc3180bbf46f438b9deee4103e558ebbb12af9434c48c34e1812fa3` | 45.82s / 47.86s |
| `d50_2702` | `8bcb95174fd9636ba56e8c317719862c563b858bb0db28f3e804510fc4b4c3ef` | `9db86ac49754840dc37b1a09d7557a60c98e41a8acb251179369161f879343a2` | 52.96s / 37.65s |
| `d50_2822` | `e486a9db9b0f5069cbeb66d842c829ce01353a1b27229f396a733acbe5392686` | `6e431a2fd97be8858f41d5fe95b6b42619d1a8b48b5ff433580a33ae76080825` | 46.64s / 45.67s |
| `d50_2823` | `17b50d06e6e91244113b8a9310e9eed609431096d81cc104fe1460ba1c2056ae` | `2f9722e5ea5a844bda66415d977ec73303b61b404f47b08ece04cf190d673ac8` | 50.82s / 48.05s |

Chaque index direct contient `2000` étoiles et `40000` records.

## 4. Manifestes de comparaison

Deux manifestes stricts temporaires ont été générés et validés par `load_4d_index_manifest()` :

- baseline : `/tmp/p1d3b_direct_runtime_z8s_quji/baseline_manifest.json`
- direct : `/tmp/p1d3b_direct_runtime_z8s_quji/direct_manifest.json`

Ils conservent le même ordre logique :

```text
d50_2823, d50_2822, d50_2644, d50_2645, d50_2602, d50_2702
```

Seuls les chemins NPZ diffèrent. Le manifeste produit `config/zeblind_4d_experimental_manifest.json` et les six NPZ produit ne sont pas modifiés.

## 5. Corpus

Corpus exécuté :

- M106 all30 : `30/30` FITS disponibles sous `reports/eq_ircut_cleanbench_20260518_230249/data`.
- mini-corpus P2.10/P2.11 : inclus dans all30 (`232102`, `232144`, `232205`, `232247`, `232329`, `232350`, `232431`, `232513`, `232534`, `232658`).
- cas difficiles : `233828` et `234013` inclus.
- contrôles négatifs P2 bornés : `d50_2822` seule sur cinq images faible-footprint, puis `d50_2823` seule sur `234013`.

Les corpus externes additionnels configurables par variables projet n'ont pas été trouvés dans ce checkout ; aucun résultat n'a été inventé.

## 6. Politiques

Politiques exécutées sur all30 :

- `first_accept`
- `best_within_budget`

Même ordre d'index, même runtime, mêmes seuils, mêmes budgets, mêmes copies FITS séparées pour baseline et direct.

## 7. Tableau par FITS

Colonnes : politique, label, classification, tuile baseline/direct, inliers/RMS baseline, inliers/RMS direct, écart WCS direct vs baseline centre/corners en arcsec, hits directs, rang direct, temps direct.

| politique | FITS | classification | tile A | tile B | A inliers/RMS | B inliers/RMS | B vs A | hits B | rang B | temps B |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| `first_accept` | `232102` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 60/0.547 | 60/0.547 | 0.000/0.000 | 503 | 1 | 3.00 |
| `first_accept` | `232144` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 61/0.248 | 61/0.248 | 0.000/0.000 | 742 | 1 | 2.94 |
| `first_accept` | `232205` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 59/0.714 | 59/0.714 | 0.000/0.000 | 604 | 1 | 2.87 |
| `first_accept` | `232247` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 66/0.875 | 66/0.875 | 0.000/0.000 | 648 | 1 | 3.00 |
| `first_accept` | `232329` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 60/0.693 | 60/0.303 | 0.681/3.409 | 565 | 1 | 2.93 |
| `first_accept` | `232350` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 62/1.067 | 62/1.067 | 0.000/0.000 | 576 | 1 | 2.89 |
| `first_accept` | `232431` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 64/0.210 | 64/0.210 | 0.000/0.000 | 524 | 1 | 2.89 |
| `first_accept` | `232513` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 64/0.918 | 64/0.918 | 0.000/0.000 | 711 | 1 | 2.89 |
| `first_accept` | `232534` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 61/0.843 | 61/0.843 | 0.000/0.000 | 673 | 1 | 2.98 |
| `first_accept` | `232658` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2822 | d50_2822 | 66/0.606 | 66/0.606 | 0.000/0.000 | 603 | 58 | 7.15 |
| `first_accept` | `232739` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2822 | d50_2822 | 66/0.711 | 66/0.711 | 0.000/0.000 | 609 | 42 | 5.05 |
| `first_accept` | `232821` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2822 | d50_2822 | 67/0.572 | 67/0.572 | 0.000/0.000 | 647 | 85 | 7.37 |
| `first_accept` | `232842` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2822 | d50_2822 | 67/0.812 | 67/0.812 | 0.000/0.000 | 698 | 100 | 8.09 |
| `first_accept` | `232924` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2822 | d50_2822 | 64/0.606 | 64/0.606 | 0.000/0.000 | 616 | 88 | 8.46 |
| `first_accept` | `232945` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2822 | d50_2822 | 65/0.684 | 65/0.684 | 0.000/0.000 | 542 | 76 | 6.54 |
| `first_accept` | `233027` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2822 | d50_2822 | 56/0.507 | 56/0.507 | 0.000/0.000 | 663 | 78 | 6.73 |
| `first_accept` | `233048` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2822 | d50_2822 | 53/0.356 | 53/0.356 | 0.000/0.000 | 546 | 63 | 6.26 |
| `first_accept` | `233130` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2822 | d50_2822 | 60/0.863 | 60/0.863 | 0.000/0.000 | 549 | 71 | 6.50 |
| `first_accept` | `233211` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 56/0.991 | 56/0.991 | 0.000/0.000 | 428 | 1 | 2.89 |
| `first_accept` | `233232` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2822 | d50_2822 | 57/0.960 | 57/0.960 | 0.000/0.000 | 426 | 56 | 5.75 |
| `first_accept` | `233314` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 56/0.620 | 56/0.620 | 0.000/0.000 | 497 | 1 | 2.95 |
| `first_accept` | `233356` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 62/0.848 | 62/0.513 | 1.931/2.831 | 488 | 1 | 3.03 |
| `first_accept` | `233417` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 62/0.613 | 62/0.613 | 0.000/0.000 | 494 | 1 | 2.95 |
| `first_accept` | `233459` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 61/0.926 | 61/0.926 | 0.000/0.000 | 851 | 1 | 2.96 |
| `first_accept` | `233520` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 61/0.843 | 61/0.843 | 0.000/0.000 | 422 | 1 | 2.97 |
| `first_accept` | `233602` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 66/0.678 | 66/0.678 | 0.000/0.000 | 596 | 1 | 2.95 |
| `first_accept` | `233644` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 65/0.844 | 65/0.844 | 0.000/0.000 | 737 | 3 | 3.10 |
| `first_accept` | `233705` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 65/0.686 | 65/0.758 | 0.218/1.235 | 548 | 1 | 3.12 |
| `first_accept` | `233828` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 56/1.088 | 56/1.088 | 0.000/0.000 | 813 | 1 | 3.15 |
| `first_accept` | `234013` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2822 | d50_2822 | 49/1.126 | 49/1.126 | 0.000/0.000 | 522 | 81 | 7.61 |
| `best_within_budget` | `232102` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 59/0.231 | 59/0.231 | 0.000/0.000 | 503 | 17 | 6.02 |
| `best_within_budget` | `232144` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 61/0.174 | 61/0.174 | 0.000/0.000 | 742 | 53 | 6.11 |
| `best_within_budget` | `232205` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 59/0.226 | 59/0.199 | 0.059/0.705 | 604 | 11 | 6.23 |
| `best_within_budget` | `232247` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 66/0.278 | 66/0.278 | 0.000/0.000 | 648 | 70 | 6.23 |
| `best_within_budget` | `232329` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 60/0.260 | 60/0.260 | 0.000/0.000 | 565 | 42 | 33.26 |
| `best_within_budget` | `232350` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 63/0.572 | 63/0.572 | 0.000/0.000 | 576 | 16 | 31.50 |
| `best_within_budget` | `232431` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 64/0.210 | 64/0.210 | 0.000/0.000 | 524 | 1 | 9.66 |
| `best_within_budget` | `232513` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2822 | d50_2822 | 63/0.300 | 63/0.300 | 0.000/0.000 | 711 | 165 | 11.69 |
| `best_within_budget` | `232534` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 61/0.303 | 61/0.303 | 0.000/0.000 | 673 | 7 | 11.16 |
| `best_within_budget` | `232658` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2822 | d50_2822 | 66/0.231 | 66/0.231 | 0.000/0.000 | 603 | 109 | 9.11 |
| `best_within_budget` | `232739` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2822 | d50_2822 | 66/0.203 | 66/0.203 | 0.000/0.000 | 609 | 133 | 8.82 |
| `best_within_budget` | `232821` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2822 | d50_2822 | 67/0.239 | 67/0.239 | 0.000/0.000 | 647 | 147 | 11.10 |
| `best_within_budget` | `232842` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2822 | d50_2822 | 67/0.218 | 67/0.218 | 0.000/0.000 | 698 | 114 | 11.62 |
| `best_within_budget` | `232924` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2822 | d50_2822 | 64/0.423 | 64/0.423 | 0.000/0.000 | 616 | 113 | 10.32 |
| `best_within_budget` | `232945` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2822 | d50_2822 | 65/0.491 | 65/0.491 | 0.000/0.000 | 542 | 125 | 9.84 |
| `best_within_budget` | `233027` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2822 | d50_2822 | 56/0.470 | 56/0.470 | 0.000/0.000 | 663 | 150 | 10.46 |
| `best_within_budget` | `233048` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2822 | d50_2822 | 53/0.302 | 53/0.302 | 0.000/0.000 | 546 | 86 | 9.58 |
| `best_within_budget` | `233130` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2822 | d50_2822 | 59/0.566 | 59/0.566 | 0.000/0.000 | 549 | 149 | 31.03 |
| `best_within_budget` | `233211` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2822 | d50_2822 | 57/0.694 | 57/0.694 | 0.000/0.000 | 428 | 98 | 23.78 |
| `best_within_budget` | `233232` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2822 | d50_2822 | 57/0.372 | 57/0.372 | 0.000/0.000 | 426 | 94 | 22.72 |
| `best_within_budget` | `233314` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 56/0.482 | 56/0.482 | 0.000/0.000 | 497 | 49 | 27.15 |
| `best_within_budget` | `233356` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 62/0.357 | 62/0.357 | 0.000/0.000 | 488 | 23 | 27.18 |
| `best_within_budget` | `233417` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 62/0.461 | 62/0.461 | 0.000/0.000 | 494 | 26 | 27.41 |
| `best_within_budget` | `233459` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 60/0.298 | 60/0.300 | 0.081/0.847 | 851 | 15 | 5.96 |
| `best_within_budget` | `233520` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 61/0.379 | 61/0.371 | 0.145/0.843 | 422 | 16 | 6.58 |
| `best_within_budget` | `233602` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 66/0.463 | 66/0.463 | 0.000/0.000 | 596 | 27 | 32.46 |
| `best_within_budget` | `233644` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 65/0.709 | 65/0.709 | 0.000/0.000 | 737 | 20 | 34.48 |
| `best_within_budget` | `233705` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 64/0.592 | 64/0.592 | 0.000/0.000 | 548 | 7 | 21.60 |
| `best_within_budget` | `233828` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2823 | d50_2823 | 56/0.965 | 56/0.950 | 0.092/0.440 | 813 | 24 | 43.63 |
| `best_within_budget` | `234013` | `SAME_SUCCESS_EQUIVALENT_WCS` | d50_2822 | d50_2822 | 47/1.106 | 47/1.106 | 0.000/0.000 | 522 | 94 | 29.90 |

## 8. Succès, gains et pertes

Résultat all30 :

```text
baseline produit: 60/60 succès
direct ASTAP:     60/60 succès
DIRECT_LOSS:       0
INVALID_DIRECT:    0
DIRECT_GAIN:       0
classification:    60 SAME_SUCCESS_EQUIVALENT_WCS
```

Les cas difficiles sont conservés :

- `233828`: succès équivalent sous `first_accept` et `best_within_budget`.
- `234013`: succès équivalent sous `first_accept` et `best_within_budget`, tuile `d50_2822`, `47/1.106` en best.

## 9. WCS et oracles

Le WCS source contenu dans plusieurs FITS M106 n'est pas un oracle pixel-à-pixel fiable pour toutes les copies évaluées : des ratios d'échelle incompatibles ont été détectés. Le rapport marque donc ces références comme non canoniques lorsque nécessaire.

Le juge utilisé pour la décision P1D-3B est :

- succès runtime sous mêmes seuils ;
- inliers/RMS/échelle dans les garde-fous ;
- comparaison WCS baseline produit vs direct sur centre et coins ;
- rejet des contrôles négatifs.

Pour les 60 couples all30, le WCS direct est équivalent à la baseline produit. Les écarts non nuls restent faibles, maximum observé dans le tableau : `1.931"` centre et `3.409"` coins en `first_accept`, `0.145"` centre et `0.847"` coins en `best_within_budget` hors 233828 (`0.092"`/`0.440"`).

## 10. Télémétrie des candidats

Les lignes du tableau contiennent hits 4D, rang candidat sélectionné, inliers, RMS et temps direct. Les deux cas difficiles restent dans le budget :

- `233828 best`: hits directs `813`, rang `24`, `56` inliers, RMS `0.950`, temps `43.63s`.
- `234013 best`: hits directs `522`, rang `94`, `47` inliers, RMS `1.106`, temps `29.90s`.

## 11. Analyse des divergences

Les vieux NPZ produit restent la baseline comportementale. Leur provenance historique incomplète n'est pas traitée comme une preuve contre le builder direct.

Les divergences payload observées en P1D-3A ne se traduisent ici par :

- aucune perte de succès ;
- aucun faux positif ;
- aucune modification WCS validée au-delà du seuil d'équivalence ;
- aucune modification de garde-fou.

Les différences mineures de RMS/rang sur quelques FITS sont acceptées car les WCS restent équivalents et validés par le runtime.

## 12. Performance

Résumé runtime all30 :

| politique/mode | succès | médiane total | p95 total | max total | médiane recherche | médiane validation |
|---|---:|---:|---:|---:|---:|---:|
| `first_accept:baseline` | 30 | 3.08s | 7.47s | 8.23s | 0.224s | 0.076s |
| `first_accept:direct` | 30 | 3.01s | 8.09s | 8.46s | 0.229s | 0.050s |
| `best_within_budget:baseline` | 30 | 22.23s | 39.71s | 45.01s | 0.235s | 19.69s |
| `best_within_budget:direct` | 30 | 11.39s | 34.48s | 43.63s | 0.223s | 8.38s |

Le coût direct est comparable en `first_accept` et meilleur sur cette mesure `best_within_budget`. La recherche KD reste faible ; la validation domine les temps longs.

## 13. Intégrité

Snapshots avant/après inchangés pour :

- `/opt/astap`
- `/home/tristan/zesolver_index`
- `indexes/astrometry_4d`
- `config/zeblind_4d_experimental_manifest.json`

Les SHA256 des six NPZ produit restent inchangés. Les index directs et copies FITS sont restés en `/tmp`.

Contrôles négatifs :

```text
d50_2822_only_low_footprint: 0 faux positif sur 10 probes baseline/direct
d50_2823_only_234013:        0 faux positif sur 2 probes baseline/direct
total:                       0/12 faux positif
```

## 14. Tests

Tests ciblés exécutés pendant la mission :

```bash
.venv/bin/python -m pytest \
  tests/test_astap_4d_direct_builder.py \
  tests/test_astap_4d_builder_cli.py \
  tests/test_astap_4d_runtime_validation.py \
  -q
```

Résultat final : `20 passed`.

Lot ciblé élargi :

```bash
.venv/bin/python -m pytest \
  tests/test_astap_4d_tile_materialization.py \
  tests/test_astap_4d_direct_builder.py \
  tests/test_astap_4d_builder_determinism.py \
  tests/test_astap_4d_builder_parity.py \
  tests/test_astap_4d_builder_boundaries.py \
  tests/test_astap_4d_builder_cli.py \
  tests/test_astap_4d_runtime_validation.py \
  tests/test_quad_storage.py \
  tests/test_quad_code_diagnostic.py \
  tests/test_regression_blind4d.py \
  tests/test_catalog_library_provenance.py \
  tests/test_catalog_library_blind4d_integration.py \
  -q
```

Résultat : `72 passed, 2 skipped, 2 warnings`.

Tests préalables de correction :

```bash
.venv/bin/python -m pytest tests/test_astap_4d_runtime_validation.py -q
```

Résultat : `7 passed`.

## 15. Barrières générales

État initial :

```text
HEAD de1bc8e064d813037750e11ab0bbb6ce9a3a673f
git diff --check OK
branche test...origin/test
```

Barrières exécutées :

```bash
.venv/bin/python tools/check_core_boundaries.py
.venv/bin/python tools/run_regression_suite.py --hermetic
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests zesolver.py zewcscleaner.py zeindexcheck.py
git diff --check
git status --short --branch
```

Résultats :

- `tools/check_core_boundaries.py` : OK.
- `tools/run_regression_suite.py --hermetic` : PASS, `524 passed, 1 skipped, 9 deselected, 58 warnings`.
- `.venv/bin/python -m pytest -q` : `524 passed, 10 skipped, 58 warnings`.
- `compileall` final sans `zedatabase.py` : OK.
- `git diff --check` final : OK.

Warnings observés :

- `datetime.utcnow()` déprécié dans `zeblindsolver/db_convert.py` lors des tests de parité builder ;
- warnings multiprocessing `fork()` multi-threadés dans les tests historiques ;
- warning Astropy FITS card trop longue dans `test_p220_manifest_loader`.

Ces warnings existaient dans les surfaces de test générales et ne constituent pas une nouvelle catégorie liée au builder direct.

## 16. Limites

- Le WCS source des FITS M106 n'est pas toujours un oracle pixel-à-pixel canonique ; le rapport ne le présente pas comme tel.
- Les corpus externes supplémentaires par variables projet n'étaient pas configurés dans ce checkout.
- P1D-3B ne remplace pas les NPZ produit ni le manifeste strict produit.
- P1D-3B ne génère pas encore le manifeste 4D depuis `CatalogLibrary`; c'est P1D-4.

## 17. État Git final

```text
## test...origin/test
 M .gitignore
 M AGENT.md
 M docs/architecture/catalog_library_implementation.md
 M docs/architecture/catalog_manifest_example.json
 M docs/architecture/catalog_manifest_schema.json
 M zeblindsolver/__init__.py
 M zeblindsolver/quad_index_4d.py
 M zesolver/catalog_library/__init__.py
 M zesolver/catalog_library/coverage.py
 M zesolver/catalog_library/manifest.py
 M zesolver/catalog_library/models.py
 M zesolver/catalog_library/validation.py
 M zesolver/catalog_resources.py
?? docs/architecture/catalog_atomic_adoption.md
?? docs/architecture/catalog_provenance_and_adoption.md
?? docs/architecture/direct_astap_blind4d_builder.md
?? docs/stabilization/p1d2a_catalog_provenance_plan_report.md
?? docs/stabilization/p1d2b_atomic_catalog_adoption_report.md
?? docs/stabilization/p1d3a_direct_astap_blind4d_builder_report.md
?? docs/stabilization/p1d3b_direct_blind4d_runtime_validation_report.md
?? tests/test_astap_4d_builder_boundaries.py
?? tests/test_astap_4d_builder_cli.py
?? tests/test_astap_4d_builder_determinism.py
?? tests/test_astap_4d_builder_parity.py
?? tests/test_astap_4d_direct_builder.py
?? tests/test_astap_4d_runtime_validation.py
?? tests/test_astap_4d_tile_materialization.py
?? tests/test_catalog_library_adopted_runtime.py
?? tests/test_catalog_library_adoption_cli.py
?? tests/test_catalog_library_adoption_conflicts.py
?? tests/test_catalog_library_adoption_idempotence.py
?? tests/test_catalog_library_adoption_plan.py
?? tests/test_catalog_library_adoption_rollback.py
?? tests/test_catalog_library_atomic_writer.py
?? tests/test_catalog_library_fingerprints.py
?? tests/test_catalog_library_paths.py
?? tests/test_catalog_library_provenance.py
?? tests/test_catalog_library_repair_actions.py
?? tools/adopt_catalog_library.py
?? tools/compare_blind4d_builders.py
?? tools/validate_direct_blind4d_runtime.py
?? zeblindsolver/astap_4d_builder.py
?? zesolver/catalog_library/adoption.py
?? zesolver/catalog_library/atomic_adoption.py
```

Les fichiers déjà modifiés/non suivis de P1D-2A/P1D-2B/P1D-3A sont conservés ; aucun commit ni push n'a été effectué.

## 18. Prochaine étape unique

P1D-4 : générer et valider la vue stricte Blind 4D depuis `CatalogLibrary`, sans bascule implicite.

## 19. Décision de gate

```text
READY_FOR_P1D4_LIBRARY_OWNED_BLIND4D_MANIFEST
```
