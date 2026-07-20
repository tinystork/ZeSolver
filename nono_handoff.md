# Nono Handoff — ZeBlind ↔ Astrometry (strict mirror)

## Scope
- Repository: /home/tristan/ZeSolver
- Target: ZeBlind only
- Forbidden: any ZeNear-impacting change

## Mandatory loop (autonomous)
1. Read /home/tristan/ZeSolver/followup.md (section "Mission amendée" + "Handoff mission").
2. Pick exactly one causal delta from checklist.
3. Implement patch in ZeBlind path only.
4. Run case055 evidence.
5. Write report: /home/tristan/ZeSolver/reports/nono_iter_<N>/summary.json
6. Append concise status line to /home/tristan/ZeSolver/nono_status.md
7. Repeat.

## Evidence required each iteration
- run id
- NT/NR
- step0: theta, d2min, nsig2, gate
- prob_verify_pool_source
- explicit C function/line being mirrored
- verdict: improved / neutral / regress

## Definition of done
- case055 without forcing JSON:
  - step0 theta != -1
  - gate 5σ pass
- then step-level Ze↔C alignment 0..20
- mini-lot control non-regression (001/027/028/055)
