#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zeblindsolver.quad_index_4d import ASTROMETRY_AB_CODE_4D_SCHEMA, Quad4DIndex

import tools.diagnose_p23_4d_source_list_contract as p23


DEFAULT_REPORT = ROOT / "reports/zeblind_p24_4d_source_policy_bakeoff.md"
DEFAULT_JSON = ROOT / "reports/zeblind_p24_4d_source_policy_bakeoff.json"
MANDATORY_CASES = {
    "232350": ("Light_mosaic_M 106_20.0s_IRCUT_20250518-232350.fit", "d50_2823"),
    "232102": ("Light_mosaic_M 106_20.0s_IRCUT_20250518-232102.fit", "d50_2823"),
}
OPTIONAL_CANDIDATES = (
    "232144",
    "232205",
    "232247",
    "232329",
    "232431",
    "232513",
    "232534",
)
CRITICAL_RANKS = {"232350": 24, "232102": 26}


def _filename(label: str) -> str:
    return f"Light_mosaic_M 106_20.0s_IRCUT_20250518-{label}.fit"


def _policy_accepts(policy: dict[str, Any]) -> bool:
    return bool(policy.get("first_accepted"))


def _case_accepted(case: dict[str, Any]) -> bool:
    return any(_policy_accepts(policy) for policy in (case.get("policies") or {}).values())


def _critical_presence(case: dict[str, Any]) -> dict[str, bool] | None:
    label = str(case.get("label"))
    if label not in CRITICAL_RANKS:
        return None
    critical = int(CRITICAL_RANKS[label])
    out: dict[str, bool] = {}
    for name, policy in (case.get("policies") or {}).items():
        ranks = policy.get("source", {}).get("raw_ranks") or []
        out[str(name)] = critical in {int(v) for v in ranks if v is not None}
    return out


def _run_case(label: str, filename: str, tile_key: str, disk_index: Quad4DIndex, args: argparse.Namespace, *, mandatory: bool) -> dict[str, Any]:
    source = args.data_dir.expanduser().resolve() / filename
    raw, image_shape, detect_meta = p23._detect_runtime_stars(source, args)
    lists, list_build_stats = p23._make_lists(raw, image_shape, args)
    policies: dict[str, Any] = {}
    for name, stars in lists.items():
        policies[name] = p23._evaluate_policy(name, stars, raw, image_shape, disk_index, args)
    case = {
        "label": label,
        "filename": filename,
        "tile_key": tile_key,
        "mandatory": bool(mandatory),
        "source_fits": str(source),
        "image_shape": list(image_shape),
        "detect_meta": detect_meta,
        "list_build_stats": list_build_stats,
        "policies": policies,
        "critical_rank_presence": None,
        "d50_2823_pertinent": False,
    }
    case["critical_rank_presence"] = _critical_presence(case)
    case["d50_2823_pertinent"] = _case_accepted(case)
    return case


def _fmt_num(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _accepted_summary(policy: dict[str, Any]) -> tuple[str, str, str]:
    hit = policy.get("first_accepted")
    if not hit:
        return "", "", ""
    return str(hit.get("rank", "")), str(hit.get("inliers", "")), _fmt_num(hit.get("rms_px"), 3)


def _cost_ratios(case: dict[str, Any]) -> dict[str, float | None]:
    policies = case.get("policies") or {}
    diag = policies.get("diagnostic_unfiltered", {})
    ast = policies.get("astrometry_like_candidate", {})
    diag_total = float(diag.get("timing", {}).get("total_s", 0.0) or 0.0)
    ast_total = float(ast.get("timing", {}).get("total_s", 0.0) or 0.0)
    diag_quad = float(diag.get("timing", {}).get("quad_build_s", 0.0) or 0.0)
    ast_quad = float(ast.get("timing", {}).get("quad_build_s", 0.0) or 0.0)
    return {
        "astrometry_like_total_over_diagnostic": (ast_total / diag_total) if diag_total > 0.0 else None,
        "astrometry_like_quad_over_diagnostic": (ast_quad / diag_quad) if diag_quad > 0.0 else None,
        "astrometry_like_extra_total_s": (ast_total - diag_total) if diag_total > 0.0 else None,
    }


def _decision(cases: list[dict[str, Any]]) -> dict[str, Any]:
    mandatory = [case for case in cases if case.get("mandatory")]
    included = [case for case in cases if case.get("mandatory") or case.get("d50_2823_pertinent")]

    def solves(policy_name: str, case: dict[str, Any]) -> bool:
        return _policy_accepts(case["policies"][policy_name])

    diag_mandatory_ok = all(solves("diagnostic_unfiltered", case) for case in mandatory)
    ast_mandatory_ok = all(solves("astrometry_like_candidate", case) for case in mandatory)
    standard_mandatory_ok = all(solves("standard_runtime", case) for case in mandatory)
    diag_included_ok = all(solves("diagnostic_unfiltered", case) for case in included)
    ast_included_ok = all(solves("astrometry_like_candidate", case) for case in included)
    ratios = [_cost_ratios(case).get("astrometry_like_total_over_diagnostic") for case in included]
    ratios_f = [float(v) for v in ratios if v is not None]
    median_ratio = sorted(ratios_f)[len(ratios_f) // 2] if ratios_f else None

    if diag_mandatory_ok and (not standard_mandatory_ok):
        next_policy = "diagnostic_unfiltered"
        if ast_mandatory_ok:
            ast_status = "candidate_p25_bis"
        else:
            ast_status = "not_ready"
    else:
        next_policy = "undecided"
        ast_status = "undecided"

    if diag_included_ok and ast_included_ok:
        source_front = "stable_enough_for_d50_2822_extension_with_fixed_policy"
    elif diag_mandatory_ok:
        source_front = "mandatory_stable_but_optional_divergence_requires_caution"
    else:
        source_front = "source_list_not_stable"

    return {
        "standard_mandatory_ok": bool(standard_mandatory_ok),
        "diagnostic_mandatory_ok": bool(diag_mandatory_ok),
        "astrometry_like_mandatory_ok": bool(ast_mandatory_ok),
        "diagnostic_included_ok": bool(diag_included_ok),
        "astrometry_like_included_ok": bool(ast_included_ok),
        "astrometry_like_median_total_ratio_vs_diagnostic": median_ratio,
        "next_mini_corpus_policy": next_policy,
        "astrometry_like_status": ast_status,
        "source_front": source_front,
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    decision = payload["decision"]
    cases = payload["cases"]
    included = [case for case in cases if case.get("mandatory") or case.get("d50_2823_pertinent")]
    skipped = [case for case in cases if (not case.get("mandatory")) and (not case.get("d50_2823_pertinent"))]
    lines = [
        "# ZeBlind P2.4 - bake-off source-list 4D",
        "",
        "## Verdict",
        "",
        f"- Verdict global: **{payload['global_verdict']}**",
        f"- Index 4D: `{payload['params']['quad4d_index']}`",
        "- Aucun all30, aucun rebuild complet, aucun elargissement d'index, aucun changement de seuil.",
        "- Le defaut produit reste `standard_runtime`; `diagnostic_unfiltered` reste un opt-in experimental 4D.",
        "",
        "## Matrice",
        "",
        "| cas | obligatoire | politique | etoiles | critique | quads | hits/testes | accepte | inliers | RMS | total | quad | KD | validation |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in included:
        presence = case.get("critical_rank_presence") or {}
        for name in ("standard_runtime", "diagnostic_unfiltered", "astrometry_like_candidate"):
            policy = case["policies"][name]
            rank, inliers, rms = _accepted_summary(policy)
            timing = policy.get("timing") or {}
            crit = ""
            if presence:
                crit = "oui" if bool(presence.get(name)) else "non"
            lines.append(
                "| `{}` | {} | `{}` | {} | {} | {} | {}/{} | {} | {} | {} | {} | {} | {} | {} |".format(
                    case["label"],
                    "oui" if case.get("mandatory") else "non",
                    name,
                    policy["source"]["kept"],
                    crit,
                    policy["quads_4d_generated"],
                    policy["hits_4d"],
                    policy["hits_tested"],
                    rank,
                    inliers,
                    rms,
                    _fmt_num(timing.get("total_s"), 3),
                    _fmt_num(timing.get("quad_build_s"), 3),
                    _fmt_num(timing.get("lookup_s"), 3),
                    _fmt_num(timing.get("validation_s"), 3),
                )
            )
    lines.extend(["", "## Details", ""])
    for case in included:
        lines.extend([f"### {case['label']} / {case['tile_key']}", ""])
        lines.append(f"- FITS: `{case['filename']}`")
        lines.append(f"- etoiles brutes detectees: `{case['detect_meta']['raw_detected']}`")
        if case.get("critical_rank_presence"):
            critical = CRITICAL_RANKS[str(case["label"])]
            lines.append(f"- etoile critique rang brut `{critical}` presente par politique: `{case['critical_rank_presence']}`")
        for name in ("standard_runtime", "diagnostic_unfiltered", "astrometry_like_candidate"):
            policy = case["policies"][name]
            timing = policy.get("timing") or {}
            lines.extend(
                [
                    f"- `{name}` rangs bruts conserves: `{policy['source']['raw_ranks']}`",
                    f"  - rejets: `{dict(sorted(policy['reason_counts'].items(), key=lambda item: item[1], reverse=True)[:12])}`",
                    f"  - timings: total `{_fmt_num(timing.get('total_s'), 4)}s`, quad `{_fmt_num(timing.get('quad_build_s'), 4)}s`, KD `{_fmt_num(timing.get('lookup_s'), 4)}s`, validation `{_fmt_num(timing.get('validation_s'), 4)}s`",
                ]
            )
        cost = _cost_ratios(case)
        lines.append(f"- cout `astrometry_like_candidate` vs `diagnostic_unfiltered`: `{cost}`")
        lines.append("")
    if skipped:
        lines.extend(["## Cas optionnels non retenus", ""])
        for case in skipped:
            hits = {name: policy["hits_4d"] for name, policy in case["policies"].items()}
            lines.append(f"- `{case['label']}`: index `d50_2823` juge non pertinent pour ce bake-off (aucune politique acceptee), hits `{hits}`.")
        lines.append("")
    lines.extend(
        [
            "## Reponses",
            "",
            f"1. `diagnostic_unfiltered` baseline stable ou contournement local: `{payload['answers']['diagnostic_unfiltered']}`",
            f"2. `astrometry_like_candidate` robustesse potentielle: `{payload['answers']['astrometry_like_candidate']}`",
            f"3. Cout supplementaire de `astrometry_like_candidate`: `{payload['answers']['astrometry_like_cost']}`",
            f"4. Politique pour le prochain mini-corpus 4D: `{payload['answers']['next_policy']}`",
            f"5. Extension `d50_2822`: `{payload['answers']['d50_2822']}`",
            "",
            "## Decision",
            "",
            f"- `standard_runtime` obligatoire OK: `{decision['standard_mandatory_ok']}`",
            f"- `diagnostic_unfiltered` obligatoire OK: `{decision['diagnostic_mandatory_ok']}`",
            f"- `astrometry_like_candidate` obligatoire OK: `{decision['astrometry_like_mandatory_ok']}`",
            f"- prochaine baseline experimentale: `{decision['next_mini_corpus_policy']}`",
            f"- statut Astrometry-like: `{decision['astrometry_like_status']}`",
            f"- front source-list: `{decision['source_front']}`",
            "",
            "## Parametres",
            "",
            "```json",
            json.dumps(payload["params"], indent=2),
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _answers(payload: dict[str, Any]) -> dict[str, str]:
    decision = payload["decision"]
    ratio = decision.get("astrometry_like_median_total_ratio_vs_diagnostic")
    ratio_text = "non mesure" if ratio is None else f"ratio median total ~{float(ratio):.2f}x vs diagnostic_unfiltered"
    if decision["diagnostic_mandatory_ok"]:
        diag = "baseline experimentale stable sur les cas obligatoires; pas un defaut produit"
    else:
        diag = "non stable sur les cas obligatoires"
    if decision["astrometry_like_mandatory_ok"]:
        ast = "resout les cas obligatoires; a garder comme candidate P2.5 bis sans promotion"
    else:
        ast = "ne resout pas autant que diagnostic_unfiltered dans ce bake-off"
    if decision["source_front"] == "stable_enough_for_d50_2822_extension_with_fixed_policy":
        d50 = "oui, possible de passer a une extension ciblee d50_2822 avec politique source-list fixee explicitement"
    elif decision["source_front"] == "mandatory_stable_but_optional_divergence_requires_caution":
        d50 = "possible seulement prudemment; les optionnels indiquent encore un front a surveiller"
    else:
        d50 = "non, stabiliser la source-list avant extension"
    return {
        "diagnostic_unfiltered": diag,
        "astrometry_like_candidate": ast,
        "astrometry_like_cost": ratio_text,
        "next_policy": str(decision["next_mini_corpus_policy"]),
        "d50_2822": d50,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="P2.4 bake-off of source-list policies for the experimental 4D backend.")
    ap.add_argument("--data-dir", type=Path, default=p23.DEFAULT_DATA_DIR)
    ap.add_argument("--index-root", type=Path, default=p23.DEFAULT_INDEX_ROOT)
    ap.add_argument("--quad4d-index", type=Path, default=p23.DEFAULT_4D_INDEX)
    ap.add_argument("--optional-limit", type=int, default=3)
    ap.add_argument("--optional-candidates", default=",".join(OPTIONAL_CANDIDATES))
    ap.add_argument("--max-stars", type=int, default=120)
    ap.add_argument("--max-quads", type=int, default=2500)
    ap.add_argument("--quality-rms", type=float, default=1.2)
    ap.add_argument("--quality-inliers", type=int, default=40)
    ap.add_argument("--code-tol", type=float, default=0.015)
    ap.add_argument("--max-hits-4d", type=int, default=2000)
    ap.add_argument("--max-hits-per-image-quad", type=int, default=8)
    ap.add_argument("--max-hypotheses", type=int, default=2000)
    ap.add_argument("--max-useful-examples", type=int, default=16)
    ap.add_argument("--image-strategy", default="log_spaced")
    ap.add_argument("--match-radius-px", type=float, default=3.0)
    ap.add_argument("--detect-k-sigma", type=float, default=3.0)
    ap.add_argument("--detect-min-area", type=int, default=5)
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--blind-star-min-sep-px", type=float, default=0.0)
    ap.add_argument("--astrometry-like-boxes", type=int, default=10)
    ap.add_argument("--astrometry-like-min-keep-ratio", type=float, default=0.05)
    ap.add_argument("--m106-hints", action="store_true", default=True)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    args = ap.parse_args()

    disk_index = Quad4DIndex.load(args.quad4d_index.expanduser().resolve())
    cases: list[dict[str, Any]] = []
    for label, (filename, tile_key) in MANDATORY_CASES.items():
        cases.append(_run_case(label, filename, tile_key, disk_index, args, mandatory=True))

    wanted_optional = [part.strip() for part in str(args.optional_candidates or "").split(",") if part.strip()]
    kept_optional = 0
    for label in wanted_optional:
        if kept_optional >= max(0, int(args.optional_limit)):
            break
        source = args.data_dir.expanduser().resolve() / _filename(label)
        if not source.exists():
            continue
        case = _run_case(label, _filename(label), "d50_2823", disk_index, args, mandatory=False)
        cases.append(case)
        if bool(case.get("d50_2823_pertinent")):
            kept_optional += 1

    decision = _decision(cases)
    if decision["next_mini_corpus_policy"] == "diagnostic_unfiltered":
        verdict = "P2.4 positif: diagnostic_unfiltered est la baseline experimentale P2.5; astrometry_like reste candidate separee"
    else:
        verdict = "P2.4 indecis: divergence de politiques, ne pas choisir encore"
    payload: dict[str, Any] = {
        "schema": "zeblind.p24_4d_source_policy_bakeoff.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "global_verdict": verdict,
        "cases": cases,
        "decision": decision,
        "params": {
            "quad_hash_schema": ASTROMETRY_AB_CODE_4D_SCHEMA,
            "quad4d_index": str(args.quad4d_index.expanduser().resolve()),
            "index_root": str(args.index_root.expanduser().resolve()),
            "max_stars": int(args.max_stars),
            "max_quads": int(args.max_quads),
            "quality_rms": float(args.quality_rms),
            "quality_inliers": int(args.quality_inliers),
            "code_tol": float(args.code_tol),
            "max_hits_4d": int(args.max_hits_4d),
            "max_hits_per_image_quad": int(args.max_hits_per_image_quad),
            "max_hypotheses": int(args.max_hypotheses),
            "image_strategy": str(args.image_strategy),
            "match_radius_px": float(args.match_radius_px),
            "optional_candidates": wanted_optional,
            "optional_limit": int(args.optional_limit),
        },
    }
    payload["answers"] = _answers(payload)
    args.json_out.expanduser().resolve().write_text(json.dumps(payload, indent=2, default=p23._json_default), encoding="utf-8")
    _write_report(args.report.expanduser().resolve(), payload)
    print(json.dumps({"global_verdict": verdict, "report": str(args.report), "json": str(args.json_out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
