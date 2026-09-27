# Release Candidate Acceptance — generic field robustness (2026-09-27)

Release: **ZeSolver 1.2.2**
Source mission: `ZS-GFR-20260927` (architect Junior, implementation Coco, review Nono)
Full evidence: `.a2a-reports/zs-generic-field-reconstruction-20260927.final.md`
(promotion-relevant summary below; this note stays in the `test` branch only — the
public `main` projection excludes `docs/stabilization`).

## Scope promoted

- Orientation-generic catalogue star quota (surface law) + single application of
  `oversize`.
- Deterministic hint resolution and real preset propagation to Near, with per-field
  observability and a GUI indicator derived from actual consumption.
- Per-run hint telemetry (parallel-safe).
- Non-2D auxiliary HDU handling during WCS probing (`b813854`, previously on `beta`
  only and included by this promotion).

## Evidence actually measured (not inferred)

| check | result |
| --- | --- |
| `tests/test_zn310b_gui_fallback_dataset.py` | 2 failures, **identical on base `b813854`** (missing external fixtures) — pre-existing, unrelated |
| ZeSolver full suite (once) | 1041 passed, 2 failed (above), 38 skipped |
| Targeted suites (quota/hints/metadata_solver) | 64 passed |
| HDU regressions (`-k hdu`) | 11 passed |
| Corpus Seestar F1–F3 | SOLVED, bit-identical to pre-change reference (F1: ra 323.84492693939256, dec 57.41210350478987, scale 2.374388578654938, inliers 203, rms 0.3934089442276982) |
| Corpus ASI294 F4–F6 | UNSOLVED → SOLVED, WCS vs ASTAP oracle Δcenter ≤ 1.6e-4°, Δscale ≤ 0.026 %, no wrong-field |
| Hint-offset qualification | 8/8 directions at 0.25 / 0.50 / 0.75 / 1.00 × FOV, portrait and landscape |
| Negative controls | header +10°, header +2°, ×2.0 FOV, missing metadata: all refused cleanly |
| Performance (N=3 medians) | F1 1.50 s → 1.60 s (run-to-run noise); F4 4.52 s (0/3 solved) → 4.22 s (3/3 solved) |
| Public API v1 | import OK, surface unchanged (`API_VERSION` 1.2) |
| ZSSS interop | `zeseestarstacker` unmodified; ZeSolver WCS recognised (`solve_status_skipped_existing_wcs`) and stacked |

## Not verified here (explicit limitations)

- Generality beyond 3 ASI raw frames from a single night/target: the argument is
  geometric (no tuned constant), not demonstrated on a wider corpus.
- The `Nimg <= 140` regime (where the `oversize` defect lived) is covered by unit
  tests only — no real image in that regime was available.
- macOS packaging/runtime audit was **not** performed for this release.
- Blind 4D behaviour is unchanged; its failures on these frames are out of scope.

## Owner gate

The owner declared ZeSolver validated for this promotion and explicitly authorised
the version bump and the `test` → `main` publication on 2026-09-27. Product-level
status wording in `AGENT.md` is intentionally left untouched by this note.
