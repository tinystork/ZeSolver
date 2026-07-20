# Builder Blind 4D direct depuis ASTAP

## Objectif

P1D-3A ajoute un chemin moteur capable de construire un payload Blind 4D depuis
les shards ASTAP/HNSKY bruts, sans passer par l'ancien `index_root/manifest.json`,
les `tiles/*.npz` historiques ni les `hash_tables`.

Le produit n'est pas bascule vers ces nouveaux index dans P1D-3A. Les profils,
seuils, solveurs, règles d'acceptation et manifestes produit existants restent
inchangés.

## Chaîne historique

La chaîne historique reste disponible comme oracle:

```text
ASTAP/HNSKY
-> zeblindsolver.db_convert.build_index_from_astap()
-> index_root/manifest.json
-> index_root/tiles/*.npz
-> zeblindsolver.quad_index_4d.build_experimental_4d_index()
-> NPZ Blind 4D runtime
```

La matérialisation historique applique, dans cet ordre:

1. lecture ASTAP native;
2. centre TAN par centre cartésien des étoiles;
3. fallback au centre de layout si aucune projection TAN n'est finie;
4. projection TAN;
5. filtre des projections non finies;
6. `mag_cap`;
7. limite source `max_stars`;
8. politique `star_truncation_mode` (`native_prefix` ou `brightest_mag`);
9. écriture des tableaux `ra_deg`, `dec_deg`, `mag`, `x_deg`, `y_deg`,
   `sweep_rank`;
10. tri stable par magnitude dans le builder 4D;
11. limite `max_stars_per_tile`;
12. `sample_quads()`;
13. `build_astrometry_quad_records()`;
14. assemblage du payload runtime.

## Chaîne directe

La nouvelle API moteur est:

```python
from zeblindsolver.astap_4d_builder import Astap4DBuildConfig, build_4d_index_from_astap

build_4d_index_from_astap(db_root, out_path, config=config)
```

Elle lit uniquement `db_root`, écrit uniquement `out_path`, refuse d'écraser un
fichier existant par défaut et ne dépend ni de `zesolver`, ni de `CatalogLibrary`,
ni du GUI.

Le chemin direct est:

```text
ASTAP/HNSKY
-> materialize_astap_tile_for_4d()
-> Quad4DPayloadTile
-> build_4d_index_from_payload_tiles()
-> NPZ Blind 4D runtime
```

## Contrat de matérialisation

`Quad4DSourceTile` expose:

- `tile_key`, `family`, `tile_code`;
- `center_ra_deg`, `center_dec_deg`;
- `ra_deg`, `dec_deg`, `mag`;
- `x_deg`, `y_deg`;
- `source_star_indices`;
- compteurs avant/après filtres;
- `tan_center_policy`;
- indicateur `used_layout_fallback`.

La configuration associée est `AstapTileMaterializationConfig`:

- `mag_cap`;
- `source_max_stars`;
- `source_star_truncation_mode`;
- `tan_center_policy`;
- `max_stars_per_tile`.

Le contrat reproduit l'ordre historique des opérations jusqu'à la limite source.
Le tri stable par magnitude et la limite 4D sont appliqués par le coeur partagé
afin que le chemin historique et le chemin direct utilisent exactement la même
génération de quads et le même assemblage.

## Configuration du builder

`Astap4DBuildConfig` est immuable et contient:

- `family`;
- `tile_keys`;
- `level`;
- `mag_cap`;
- `source_max_stars`;
- `source_star_truncation_mode`;
- `tan_center_policy`;
- `max_stars_per_tile`;
- `max_quads_per_tile`;
- `sampler_tag`;
- `code_tol_recommended`;
- `dtype`;
- `projection_implementation`;
- `projection_version`;
- `quad_schema`;
- `quad_version`;
- `builder_version`.

Les valeurs historiques inconnues ne sont pas inventées dans les comparaisons:
elles doivent venir du manifeste historique, des métadonnées 4D ou d'un paramètre
manuel explicite.

## Coeur partagé

`zeblindsolver.quad_index_4d.build_4d_index_from_payload_tiles()` assemble le
payload Blind 4D depuis des tuiles déjà matérialisées. Le wrapper historique
`build_experimental_4d_index()` charge les `tiles/*.npz` puis appelle ce même
coeur. Le builder direct appelle aussi ce coeur après lecture ASTAP.

Les primitives scientifiques restent inchangées:

- `sample_quads()`;
- `build_astrometry_quad_records()`;
- schéma AB 4D `astrometry_ab_code_4d_v1`.

## Payload

Le NPZ produit conserve le schéma runtime existant:

- `codes_4d`;
- `quad_star_indices`;
- `source_quad_indices`;
- `tile_key_indices`;
- `ratio_hashes`;
- `tile_keys`;
- `catalog_ra_dec`;
- `catalog_xy`;
- `metadata`.

La metadata directe déclare:

```text
source_catalog = astap_raw
```

Elle ajoute, quand disponibles:

- `source_family`;
- `source_tiles`;
- `source_fingerprint`;
- `build_parameters`;
- `provenance_fingerprint`;
- `builder_version`.

## Empreinte scientifique

`scientific_payload_fingerprint()` calcule une empreinte déterministe sur:

- les noms des tableaux;
- leurs shapes;
- leurs dtypes;
- leurs bytes contigus;
- les métadonnées canoniques.

Sont exclus de l'empreinte:

- `generated_at`;
- chemins machine locaux comme `source_index_root` et `source_db_root`;
- chemins de sortie temporaires.

La canonicalisation JSON utilise `sort_keys=True` et des séparateurs compacts.
L'ordre accidentel d'un dictionnaire et les métadonnées ZIP du `.npz` ne
participent pas à l'empreinte.

## Comparaison

`compare_4d_indexes()` compare les tableaux runtime, les tile keys et les
empreintes scientifiques. Pour chaque tableau divergent, il rapporte:

- shapes;
- dtypes;
- premier index divergent;
- valeurs gauche/droite;
- écart absolu et relatif maximal pour les flottants.

L'égalité des tableaux peut être vraie même si les empreintes de payload complet
diffèrent, par exemple entre `source_catalog=tile_npz` et
`source_catalog=astap_raw`.

## CLI avancé

`tools/compare_blind4d_builders.py` construit et compare dans un répertoire de
sortie explicite:

```bash
.venv/bin/python tools/compare_blind4d_builders.py \
  --astap-root /opt/astap \
  --legacy-index-root /home/tristan/zesolver_index \
  --existing-4d-index indexes/astrometry_4d/d50_2822_S_q40000.npz \
  --tile-key d50_2822 \
  --out-dir /tmp/p1d3a-d50_2822 \
  --report-json /tmp/p1d3a-d50_2822.json
```

Par défaut, `--out-dir` doit être vide ou absent. L'outil ne modifie ni
`catalog.json`, ni le manifeste strict existant, ni les sources ASTAP, ni les
index produit.

## Limites P1D-3A

P1D-3A ne construit pas de manifeste produit durable et ne branche aucun runtime
produit sur les index directs. La validation corpus solveur appartient à P1D-3B.

Les six NPZ stricts existants ont une provenance partielle et ne sont pas
reproductibles exactement par le builder historique actuel, même depuis leur
`source_index_root` déclaré. Cette limite est documentée dans le rapport
P1D-3A.
