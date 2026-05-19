# Follow-up ZeSolver / ZeBlind

## Mission active
Rendre **ZeBlind seul** réellement exploitable en rapprochant le chemin natif Python de la **sémantique Astrometry verify/accept**, sans retomber dans un hybride Ze/Astrometry.

## État consolidé utile pour reprendre vite
- [x] Le repro de référence est propre et figé: `reports/forensic_case055_repro_v1` + `reports/forensic_m106_reference_v1`.
- [x] L’audit de référence Astrometry existe: `reports/astrometry_math_audit_20260519_0114.md`.
- [x] Le meilleur chemin natif réel actuel est d’environ `accept_logodds=+9.274`.
- [x] Le seuil forensic observé à battre reste `toprint≈12`.
- [x] Le résiduel principal identifié n’est plus un simple seuil, mais une **géométrie/projection locale native encore imparfaite**.

## Prochain cran — piloté par l’audit Astrometry

### P0 — Figer la cible Astrometry exacte à reproduire
- [x] Rédiger un mini-contrat de parité exécutable pour le prochain patch, à partir de `astrometry_math_audit_20260519_0114.md`.
  - inclure uniquement les invariants P0 du prochain cran:
    1. entrées verify (`full field stars` vs `full projected index stars`)
    2. ordre/support ref local
    3. géométrie locale projetée / recentrage
    4. `testsigma²` réellement consommé
    5. accept path réellement comparé
  - livrable publié: `reports/forensic_case055_repro_v1/next_cran_parity_contract.md`

### P1 — Instrumenter le prochain écart utile, pas plus
- [x] Ajouter l’instrumentation minimale qui manque encore pour le cran “projection locale native”.
  - cible: rendre visibles dans `verify_hit_trace` ou stats équivalentes:
    - shift/recentrage local effectivement appliqué
    - ancre test/ref utilisée
    - `d2min` avant/après si possible
    - cardinalités finales `test/ref`
  - contrainte: instrumentation seule, pas de tuning mêlé
  - gate: `python3 -m py_compile zeblindsolver/zeblindsolver.py`
  - validation publiée: `reports/forensic_case055_repro_v1/native_local_ref_recenter_instrumentation_check.json`
  - lecture clé: le trace expose maintenant bien le recentrage local effectif (`shift_xy`, ancre test/ref, `d2min_before=52.3967`, `d2min_after=0.0`, `prob_verify_nt=8`, `prob_verify_nr=2`)

### P2 — Tester un seul delta causal amont sur la géométrie/projection native
- [x] Prototyper un patch unique qui améliore la **projection locale native** sans JSON forcé.
  - direction recommandée: transformer le recentrage local actuellement prouvé en comportement plus fidèle à la cible Astrometry, sans bricolage externe
  - contrainte: un seul delta causal, pas de retuning de sigma dans le même patch
  - patch testé: nouveau mode `blind_astrometry_native_local_ref_recenter_mode='local_centroid'`

- [x] Rerun le probe `case055` ON avec ce seul delta et publier l’A/B.
  - baseline de comparaison: meilleur point natif actuel (`native_local_ref_recenter_proto_ab.json`)
  - succès minimal attendu:
    - `accept_logodds > 9.274`
    - ou baisse claire de `d2min/nsig2`
    - ou meilleure stabilité du premier préfixe utile
  - artefact publié: `reports/forensic_case055_repro_v1/native_local_ref_recenter_mode_ab.json`

- [x] Si le delta échoue, documenter précisément **pourquoi** et identifier le prochain écart unique restant.
  - livrable attendu: une synthèse courte, pas une todo diffuse
  - lecture clé: le mode `local_centroid` détruit le recentrage utile (`d2min_after: 0.0 -> 467.18`, `accept_logodds: +9.274 -> -1.386`)
  - conclusion: le prochain écart unique restant n’est pas “un meilleur shift global”, mais la **qualité du choix/ancrage local** servant au recentrage natif

### Tâche ajoutée après échec du recentrage centroid
- [x] Tester un delta causal centré sur le **choix de l’ancre locale** du recentrage natif, sans changer la famille de patch.
  - direction recommandée: conserver le recentrage pairwise, mais vérifier si l’ancre locale doit être choisie sur un support plus informé que le simple `argmin d2`
  - patch testé: nouveau mode `blind_astrometry_native_local_ref_recenter_mode='head_objective'`
  - artefact publié: `reports/forensic_case055_repro_v1/native_local_ref_recenter_head_objective_ab.json`
  - lecture clé: ce mode dégrade le préfixe utile (`accept_logodds: +9.274 -> +6.579`) en déplaçant le premier match utile de `i=0` à `i=2`
  - conclusion: l’ancrage `anchor_pair` reste le meilleur point local observé dans cette famille

### P3 — Vérifier si le résiduel est encore local ou devient global
- [x] Quantifier le gap restant après le patch amont.
  - si le score passe dans une zone proche de `12`, rester sur un dernier levier local
  - sinon, conclure explicitement qu’on quitte le cran “géométrie locale” pour un problème plus global de parité
  - artefact publié: `reports/forensic_case055_repro_v1/current_native_geometry_family_gap_summary.json`
  - conclusion: dans la famille de patchs locale actuelle, `anchor_pair` reste le meilleur mode à `+9.274`, soit encore ~`2.73` sous `toprint≈12`
  - lecture de reprise: le prochain cran utile sort probablement de la micro-famille “recentrage local” et doit viser une correction plus globale de projection/géométrie native

### P4 — Seulement après convergence suffisante
- [blocked] Publier la conclusion de parité d’entrée.
  - blocage réel: non autorisé tant que le meilleur flux natif reste nettement sous `toprint≈12`

- [blocked] Lancer une validation plus large ZeBlind seul sur un lot utile.
  - blocage réel: trop tôt tant que le cran case055 n’a pas convergé proprement

- [blocked] Décider des activations produit par défaut.
  - blocage réel: aucune décision produit saine avant convergence verify/accept plus crédible

## Discipline
- [x] Un seul delta causal par itération.
- [x] Toute conclusion durable va dans `memory.md`.
- [x] `followup.md` doit rester courte, actionnable, et facilement reprenable après interruption.
