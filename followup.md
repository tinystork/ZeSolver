# Follow-up ZeSolver — base de travail propre (blind-only)

Source de vérité pour l’historique et l’accompli: `memory.md` (section `2026-04-22 -> 2026-04-23`).

## Backlog actif uniquement

### P7 — Verrou parité (quand l’info est fiable)

- [ ] Ajouter un mode parity-locked ZeBlind (nominal-only / mirror-only).
- [ ] N’activer le lock que si la confiance parité est suffisante.
- [ ] Fallback automatique dual-parity si confiance insuffisante.
- [ ] Bench A/B focus4 + lot complet (temps + solve-rate).

### P8 — Uniformisation / dédup en vérification

- [ ] Uniformiser spatialement les étoiles de verify (downsample non aléatoire).
- [ ] Dédupliquer paires/correspondances redondantes avant scoring lourd.
- [ ] Bench impact direct sur coût verify + robustesse.

## Gate de promotion patch (checklist minimale)

- [ ] Comparatif signé avant/après (temps total, solve count, fichiers en échec).
- [ ] Même protocole d’entrée/sortie que baseline (éviter biais WCS pré-existant).
- [ ] Trace explicite des raisons d’arrêt (fail-fast/budget/log-odds) en logs.
