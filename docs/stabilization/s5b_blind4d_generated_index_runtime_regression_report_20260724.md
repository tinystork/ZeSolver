# S5B - Blind 4D generated index runtime regression - 2026-07-24

## 1. Objectif

Diagnostiquer pourquoi la Bibliotheque ZeSolver D50 complete publiee charge
correctement en `library-view` mais ne resout plus deux FITS positifs avec faux
hints, puis corriger le chemin de construction standard sans modifier ZeNear,
les seuils d'acceptation, les budgets runtime ou le fallback externe.

## 2. Etat Git initial

```text
## test...origin/test
 M tests/test_catalog_blind4d_manifest_view.py
 M zesolver/catalog_library/blind4d_view.py
?? docs/stabilization/s5_blind4d_library_view_validation_report_20260723.md
```

`git diff --check` initial: OK.

Ces fichiers S5 etaient preexistants et ont ete conserves.

## 3. Reproduction utilisateur

FITS verifies sous `/home/tristan/near_bench_cmp30/testzeblind/`:

- `001_Light_mosaic_M 106_20.0s_IRCUT_20250518-233459_FAKE_HINT.fit`
- `002_Light_M 31_11_30.0s_IRCUT_20250922-230409_FAKE_HINT.fit`

Controle avant solve:

- pixels lus sans modification;
- WCS celeste absent;
- faux hints presents: `RA=0.0`, `DEC=-60.0`, `OBJCTRA=00:00:00.000`,
  `OBJCTDEC=-60:00:00.000`;
- optique conservee: `FOCALLEN=250`, `XPIXSZ/YPIXSZ=2.9`, binning `1/1`;
- index utilise: `/home/tristan/ZeSolverCatalog/new/indexes/blind4d/d50_4d.npz`.

## 4. Preuve de chargement library-view

La bibliotheque S5 reste chargeable:

```text
status=READY_FULL
family=d50
index=direct-d50
tiles=1476/1476
coverage=FULL
all_sky=true
runtime=library-view
external_fallback=false
```

L'echec S5B est donc different d'un echec de chargement: le runtime demarre,
charge l'index et execute la recherche, mais ne valide aucune hypothese.

## 5. Hypotheses initiales

- densite du build gestionnaire inferieure aux index P1D-3B qualifies;
- topologie all-sky monolithique pouvant saturer `max_hits=2000`;
- troncature ou classement de candidats supprimant le bon candidat;
- defaut payload/offsets.

## 6. Matrice A/B/C/D/E

E n'a pas ete execute: reconstruire un all-sky D50 dense `2000/40000` est
couteux et n'etait pas necessaire pour demontrer la premiere cause. Une
estimation simple depuis D donne environ `1476/2 * 2.1 MiB`, soit autour de
`1.5 GiB` compresse, avec un temps de build potentiellement tres long.

| variant | topology | stars/tile | quads/tile | total stars | total quads | file size | result |
|---|---|---:|---:|---:|---:|---:|---|
| A | six targeted P1D-3B repo indexes | 2000 | 40000 | 12000 | 240000 | 6.3 MB | SOLVED 2/2 |
| B | current manager all-sky monolith | 400 | 8000 | 590400 | 11808000 | 260.1 MB | UNSOLVED 0/2 |
| C | targeted current manager density | 400 | 8000 | 800 | 16000 | 0.37 MB | UNSOLVED 0/2 |
| D | targeted P1D-3B density | 2000 | 40000 | 4000 | 80000 | 2.1 MB | SOLVED 2/2 |

Runtime results:

| variant | FITS | hits | tested | accepted | first accepted | selected tile | runtime | result |
|---|---|---:|---:|---:|---:|---|---:|---|
| A | M106 233459 | 836 | 67 | 64 | 1 | d50_2823 | 4.26s | SOLVED |
| A | M31 230409 | 595 | 583 | 48 | 359 | d50_2602 | 20.55s | SOLVED |
| B | M106 233459 | 2000 | 31 | 0 | - | - | 45.44s | UNSOLVED |
| B | M31 230409 | 2000 | 32 | 0 | - | - | 45.88s | UNSOLVED |
| C | M106 233459 | 21 | 21 | 0 | - | - | 1.99s | UNSOLVED |
| C | M31 230409 | 7 | 7 | 0 | - | - | 1.69s | UNSOLVED |
| D | M106 233459 | 335 | 67 | 64 | 1 | d50_2823 | 3.18s | SOLVED |
| D | M31 230409 | 227 | 222 | 48 | 1 | d50_2602 | 6.55s | SOLVED |

## 7. Metadonnees comparees

Index B courant:

```text
schema=astrometry_ab_code_4d_v1
family=d50
tile_count=1476
max_stars_per_tile=400
max_quads_per_tile=8000
source_max_stars=2000
mag_cap=15.5
sampler_tag=catalog_ring_coverage
code_tol=0.015
builder_version=astap_direct_4d_v1
```

Index D cible qualifie:

```text
schema=astrometry_ab_code_4d_v1
family=d50
tile_count=2
max_stars_per_tile=2000
max_quads_per_tile=40000
source_max_stars=2000
mag_cap=15.0
sampler_tag=catalog_ring_coverage
code_tol=0.015
builder_version=astap_direct_4d_v1
```

La divergence n'est donc pas seulement structurelle: le payload B est
scientifiquement plus pauvre par tuile et plus permissif en magnitude que la
configuration qualifiee.

## 8. Analyse des 2000 hits

B atteint exactement `hits=2000` sur les deux FITS, avec un KD lookup court
(`~0.04s`) et une validation dominante (`~29-30s`). Le budget est consomme en
validant des hypotheses non pertinentes avant qu'une solution ne soit trouvee.

Mais C demontre que la saturation all-sky n'est pas la cause minimale: avec les
deux bonnes tuiles seulement, la densite gestionnaire actuelle produit trop peu
de hits utiles (`21` et `7`) et aucun candidat valide. D, avec les memes tuiles
mais la densite P1D-3B, resout immediatement.

## 9. Localisation du candidat oracle

- M106 `233459`: oracle A/D sur `d50_2823`.
- M31 `230409`: oracle A/D sur `d50_2602`.

Le candidat oracle existe dans les index denses P1D-3B et manque ou devient
non exploitable dans les index construits en `400/8000`.

## 10. Cause racine

`CatalogLibraryManagementService._build_blind4d_indexes()` utilisait les
defauts de `Astap4DBuildConfig()`. Ces defauts etaient:

```text
mag_cap=15.5
source_max_stars=2000
max_stars_per_tile=400
max_quads_per_tile=8000
```

Ils ne correspondaient pas a la configuration P1D-3B qualifiee:

```text
mag_cap=15.0
source_max_stars=2000
max_stars_per_tile=2000
max_quads_per_tile=40000
```

La regression fonctionnelle est donc d'abord une regression de configuration de
build. La topologie monolithique all-sky reste un risque non cloture pour un
futur all-sky dense.

## 11. Correction choisie

`Astap4DBuildConfig` et `AstapTileMaterializationConfig` utilisent maintenant
les constantes qualifiees:

```text
QUALIFIED_MAG_CAP=15.0
QUALIFIED_SOURCE_MAX_STARS=2000
QUALIFIED_MAX_STARS_PER_TILE=2000
QUALIFIED_MAX_QUADS_PER_TILE=40000
```

Le gestionnaire standard, qui propage les defauts du builder quand l'utilisateur
ne surcharge rien, construit donc maintenant avec la densite qualifiee.

## 12. Solutions rejetees

- augmenter `max_hits`: masque la saturation et augmente le cout sans restaurer
  la densite scientifique;
- augmenter le budget 45s: masque le symptome;
- relacher `quality_inliers`, `quality_rms` ou `code_tol`: interdit et non
  necessaire;
- forcer le fallback externe: contraire a S5;
- accepter le meilleur rejet: faux positif potentiel;
- corriger ZeNear: hors perimetre et non implique.

## 13. Fichiers modifies

S5B:

- `zeblindsolver/astap_4d_builder.py`
- `tests/test_astap_4d_runtime_validation.py`
- `tests/test_astap_4d_builder_cli.py`
- `tests/test_catalog_library_management_service.py`
- `tools/diagnose_s5b_blind4d_generated_index_regression.py`
- `docs/stabilization/s5b_blind4d_generated_index_runtime_regression_report_20260724.md`

Preexistants S5 conserves:

- `zesolver/catalog_library/blind4d_view.py`
- `tests/test_catalog_blind4d_manifest_view.py`
- `docs/stabilization/s5_blind4d_library_view_validation_report_20260723.md`

## 14. Avant / apres

Avant:

```text
manager standard -> d50_4d.npz 400/8000 mag_cap=15.5
library-view -> hits=2000 accepted=0 -> UNSOLVED
```

Apres code:

```text
manager standard -> Astap4DBuildConfig defaults 2000/40000 mag_cap=15.0
targeted D rebuild -> both reference FITS SOLVED via Blind 4D
```

La bibliotheque publiee `/home/tristan/ZeSolverCatalog/new` n'a pas ete
reconstruite pendant S5B; elle reste l'ancien artefact `400/8000`.

## 15. Configuration finale de build

Configuration standard finale du builder direct ASTAP:

```json
{
  "family": "d50",
  "level": "S",
  "mag_cap": 15.0,
  "source_max_stars": 2000,
  "source_star_truncation_mode": "native_prefix",
  "max_stars_per_tile": 2000,
  "max_quads_per_tile": 40000,
  "sampler_tag": "catalog_ring_coverage",
  "code_tol_recommended": 0.015,
  "dtype": "float32"
}
```

## 16. Structure finale des index

Le patch ne change pas encore la topologie: le gestionnaire produit toujours un
index direct par famille (`direct-d50`, fichier `d50_4d.npz`). Seule la densite
standard est corrigee.

## 17. Publication via le gestionnaire

Validation complete via une nouvelle destination vide non executee pendant ce
tour, car une reconstruction D50 complete dense est couteuse. Un test unitaire
intercepte cependant le service normal et verifie que la configuration standard
propage bien `2000/40000` et `mag_cap=15.0`.

## 18. Tests positifs

Tests reels sur copies temporaires:

- M106 `233459`: A SOLVED, D SOLVED.
- M31 `230409`: A SOLVED, D SOLVED.

Cas difficiles P1D-3B `233828`/`234013` non relances pendant S5B; leur parite
P1D-3B precedente reste documentee dans le rapport P1D-3B.

## 19. Controles negatifs

Controles negatifs P1D-3B non relances pendant S5B. Aucun seuil d'acceptation
n'a ete modifie; le risque de faux positif n'est pas augmente par le patch de
configuration, mais la validation negative complete reste requise apres rebuild
final.

## 20. Parite P1D-3B

La parite fonctionnelle ciblee est restauree sur les deux FITS S5B via la
configuration P1D-3B dense. La parite corpus `60/60` n'a pas ete relancee sur
une bibliotheque generee complete.

## 21. Temps, disque et memoire

Mesures S5B:

- B all-sky courant: `260.1 MB`, `590400` etoiles, `11808000` quads;
- C cible courant: `0.37 MB`, `800` etoiles, `16000` quads;
- D cible dense: `2.1 MB`, `4000` etoiles, `80000` quads;
- D solve: M106 `3.18s`, M31 `6.55s`;
- B solve: deux echecs au budget `~45s`.

RSS peak non mesure proprement. L'audit pre-troncature complet sur l'index
all-sky a ete interrompu car il rechargeait le KD tree lourd et monopolisait CPU
et memoire.

## 22. Suite hermetique

```text
.venv/bin/python tools/check_core_boundaries.py
core boundary check: OK

.venv/bin/python tools/run_regression_suite.py --hermetic
PASS
617 passed, 1 skipped, 9 deselected, 59 warnings
```

## 23. Suite complete

```text
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
617 passed, 10 skipped, 59 warnings

.venv/bin/python -m compileall -q zeblindsolver zewcs290 zesolver tools tests zesolver.py zewcscleaner.py zeindexcheck.py
OK

git diff --check
OK
```

## 24. Validation reelle avec faux hints

Validation reelle sur la bibliotheque existante:

```text
B/current library-view: UNSOLVED 0/2
blind4d_catalog_source=catalog_library_view equivalent via explicit library index path
external fallback not used
```

Validation ciblee sur indexes D reconstruits depuis `/opt/astap`:

```text
M106 233459 -> SOLVED d50_2823
M31 230409 -> SOLVED d50_2602
```

## 25. Warnings

Warnings connus des suites:

- `datetime.utcnow()` dans `db_convert.py` / `zewcscleaner.py`;
- warning multiprocessing `fork()` dans tests legacy;
- warning Astropy FITS card longue.

Anomalies hors perimetre non corrigees: preflight, batch runtime, GUI progress,
familles absentes, warning GUI partiel, packaging.

## 26. Tests non executes

- E all-sky dense `2000/40000`;
- reconstruction complete d'une nouvelle bibliotheque D50 dense via GUI/service
  vers une destination vide;
- corpus P1D-3B `60/60` sur bibliotheque generee complete;
- controles negatifs P1D-3B sur bibliotheque generee complete;
- mesure RSS peak fiable;
- audit complet du candidat oracle apres rang 2000 dans l'all-sky courant.

## 27. Etat Git final

```text
## test...origin/test
 M tests/test_astap_4d_builder_cli.py
 M tests/test_astap_4d_runtime_validation.py
 M tests/test_catalog_blind4d_manifest_view.py
 M tests/test_catalog_library_management_service.py
 M zeblindsolver/astap_4d_builder.py
 M zesolver/catalog_library/blind4d_view.py
?? docs/stabilization/s5_blind4d_library_view_validation_report_20260723.md
?? docs/stabilization/s5b_blind4d_generated_index_runtime_regression_report_20260724.md
?? tools/diagnose_s5b_blind4d_generated_index_regression.py
```

## 28. Limites

La cause densite est demontree. La topologie all-sky monolithique dense n'est
pas encore qualifiee. Il reste possible qu'un all-sky dense monolithique exige
ensuite une diversification des hits ou un partitionnement deterministe.

## 29. Une seule prochaine etape

Construire une nouvelle bibliotheque D50 complete dans une destination vide via
le service/gestionnaire standard corrige, puis relancer S5B A/B sur les deux
FITS positifs, `233828`, `234013`, le mini-corpus M106 et les controles
negatifs P1D-3B.

## 30. Decision de gate

La correction fonctionnelle de densite est appliquee et testee, mais les
criteres complets S5B ne sont pas tous demontres sur une bibliotheque D50
complete reconstruite par le gestionnaire.

```text
NOT_READY_FOR_S6_BATCH_RUNTIME_AND_MEMORY_STABILIZATION
```
