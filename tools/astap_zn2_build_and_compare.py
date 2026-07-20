#!/usr/bin/env python3
"""ZN2 ASTAP source/binary equivalence gate.

This is intentionally a gate, not a workaround.  ZN2 may only instrument
ASTAP after a locally built reference binary is proven equivalent to the
system binary used in ZN1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
except Exception:  # pragma: no cover - reported in baseline JSON.
    np = None  # type: ignore[assignment]
    fits = None  # type: ignore[assignment]
    WCS = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnose_zn1_zenear_astap_parity import (  # noqa: E402
    IMAGE_NAMES,
    hint_from_header,
    parse_astap_text,
    parse_astap_ini,
    run_cmd,
    safe_stem,
    sha256,
)


def _one_git(args: list[str]) -> str:
    cp = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return (cp.stdout or cp.stderr).strip()


def git_info() -> dict[str, Any]:
    return {
        "commit": _one_git(["rev-parse", "HEAD"]),
        "branch": _one_git(["branch", "--show-current"]),
        "status_short": _one_git(["status", "--short"]),
    }


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in ("numpy", "scipy", "astropy"):
        try:
            mod = __import__(name)
            out[name] = str(getattr(mod, "__version__", None))
        except Exception:
            out[name] = None
    return out


def which_all(names: list[str]) -> dict[str, str | None]:
    return {name: shutil.which(name) for name in names}


def source_hashes(astap_source: Path) -> dict[str, str | None]:
    rels = [
        "command-line_version/astap_command_line.lpr",
        "command-line_version/unit_command_line_solving.pas",
        "command-line_version/unit_command_line_star_database.pas",
        "command-line_version/unit_command_line_general.pas",
        "command-line_version/unit_command_line_stars_wide_field.pas",
        "unit_astrometric_solving.pas",
        "unit_star_align.pas",
    ]
    out: dict[str, str | None] = {}
    for rel in rels:
        path = astap_source / rel
        out[rel] = sha256(path) if path.exists() else None
    return out


def astap_version_probe(binary: str) -> dict[str, Any]:
    cp = run_cmd([binary, "-help"], timeout=10)
    text = (cp.get("stdout") or "") + "\n" + (cp.get("stderr") or "")
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return {
        "binary": binary,
        "returncode": cp.get("returncode"),
        "timeout": cp.get("timeout", False),
        "first_line": first,
        "elapsed_s": cp.get("elapsed_s"),
    }


def wcs_summary(path: Path) -> dict[str, Any]:
    if fits is None or WCS is None or np is None:
        return {"exists": path.exists(), "error": "astropy/numpy unavailable"}
    if not path.exists():
        return {"exists": False}
    try:
        h = fits.getheader(path)
        w = WCS(h)
        width = int(h.get("NAXIS1", 0) or 0)
        height = int(h.get("NAXIS2", 0) or 0)
        out: dict[str, Any] = {
            "exists": True,
            "has_celestial": bool(w.has_celestial),
            "width": width,
            "height": height,
        }
        if w.has_celestial and width and height:
            ra, dec = w.pixel_to_world_values(width / 2.0, height / 2.0)
            out["center_ra_deg"] = float(ra)
            out["center_dec_deg"] = float(dec)
            cd = np.asarray(w.pixel_scale_matrix, dtype=float)
            out["cd"] = cd.tolist()
            out["pixel_scale_arcsec"] = float(np.sqrt(abs(np.linalg.det(cd))) * 3600.0)
            out["cd_det"] = float(np.linalg.det(cd))
        return out
    except Exception as exc:
        return {"exists": path.exists(), "error": str(exc)}


def run_astap(
    binary: str,
    runtime: Path,
    out_base: Path,
    astap_db: Path,
    family: str,
    extra_args: list[str],
) -> dict[str, Any]:
    for ext in (".ini", ".wcs", ".log"):
        try:
            out_base.with_suffix(ext).unlink()
        except FileNotFoundError:
            pass
    hint = hint_from_header(runtime)
    cmd = [
        binary,
        "-f",
        str(runtime),
        "-ra",
        f"{float(hint['ra_hours']):.10f}",
        "-spd",
        f"{float(hint['spd_deg']):.10f}",
        "-fov",
        f"{float(hint['fov_deg']):.10f}",
        "-r",
        "3",
        "-d",
        str(astap_db),
        "-D",
        family,
        *extra_args,
        "-wcs",
        "-log",
        "-o",
        str(out_base),
    ]
    cp = run_cmd(cmd, timeout=180)
    text = (cp.get("stdout") or "") + "\n" + (cp.get("stderr") or "")
    log_path = out_base.with_suffix(".log")
    if log_path.exists():
        text += "\n" + log_path.read_text(errors="ignore")
    ini = parse_astap_ini(out_base.with_suffix(".ini"))
    return {
        "cmd": cmd,
        "returncode": cp.get("returncode"),
        "timeout": cp.get("timeout", False),
        "elapsed_s": cp.get("elapsed_s"),
        "success": bool(ini.get("success", False)),
        "ini": ini,
        "log_metrics": parse_astap_text(text),
        "wcs": wcs_summary(out_base.with_suffix(".wcs")),
        "stdout_excerpt": (cp.get("stdout") or "")[:1000],
        "stderr_excerpt": (cp.get("stderr") or "")[:1000],
    }


def try_build_reference(astap_source: Path, build_dir: Path) -> dict[str, Any]:
    build_dir.mkdir(parents=True, exist_ok=True)
    tools = which_all(["lazbuild", "fpc", "ppcx64"])
    result: dict[str, Any] = {
        "status": "not_attempted",
        "tools": tools,
        "source_dir": str(astap_source),
        "build_dir": str(build_dir),
        "output_binary": str(build_dir / "astap_zn2_reference"),
    }
    if not tools.get("lazbuild"):
        result.update({
            "status": "blocked",
            "reason": "lazbuild not found in PATH; local ASTAP source/binary equivalence cannot be proven",
        })
        return result

    project = astap_source / "command-line_version" / "astap_command_line_linux.lpi"
    if not project.exists():
        result.update({"status": "blocked", "reason": f"missing Lazarus project: {project}"})
        return result

    cmd = [tools["lazbuild"] or "lazbuild", "-B", str(project)]
    cp = run_cmd(cmd, cwd=project.parent, timeout=600)
    produced = project.parent / "astap_cli"
    result.update({
        "status": "built" if produced.exists() and cp.get("returncode") == 0 else "failed",
        "cmd": cmd,
        "returncode": cp.get("returncode"),
        "stdout_excerpt": (cp.get("stdout") or "")[-4000:],
        "stderr_excerpt": (cp.get("stderr") or "")[-4000:],
        "elapsed_s": cp.get("elapsed_s"),
    })
    if produced.exists():
        dst = build_dir / "astap_zn2_reference"
        shutil.copy2(produced, dst)
        result["output_sha256"] = sha256(dst)
    return result


def compare_wcs(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"comparable": False}
    if not (a.get("has_celestial") and b.get("has_celestial")):
        return out
    out["comparable"] = True
    for key in ("center_ra_deg", "center_dec_deg", "pixel_scale_arcsec", "cd_det"):
        if key in a and key in b:
            out[f"delta_{key}"] = float(abs(float(a[key]) - float(b[key])))
    return out


def write_markdown(path: Path, baseline: dict[str, Any], equivalence: dict[str, Any]) -> None:
    build = equivalence.get("build", {})
    b0 = equivalence.get("B0_system", [])
    b1 = equivalence.get("B1_local_reference", [])
    lines = [
        "# ZN2 ASTAP Binary Equivalence",
        "",
        "## Verdict",
        "",
    ]
    if build.get("status") == "blocked":
        lines += [
            "Equivalence source/binaire non prouvee.",
            "",
            f"- blocage: `{build.get('reason')}`",
            f"- `lazbuild`: `{build.get('tools', {}).get('lazbuild')}`",
            f"- `fpc`: `{build.get('tools', {}).get('fpc')}`",
            "",
            "Conformement a ZN2, aucune instrumentation ASTAP et aucune modification ZeNear ne doivent etre promues tant que ce verrou n'est pas leve.",
        ]
    else:
        lines += [
            f"- build status: `{build.get('status')}`",
            f"- B0 system: `{sum(1 for r in b0 if r.get('success'))}/{len(b0)}`",
            f"- B1 local: `{sum(1 for r in b1 if r.get('success'))}/{len(b1)}`",
            f"- ASTAP solve extra args: `{equivalence.get('astap_extra_args')}`",
        ]
    lines += [
        "",
        "## Binaire systeme",
        "",
        f"- `which astap`: `{baseline.get('which_astap')}`",
        f"- sha256: `{baseline.get('system_astap_sha256')}`",
        f"- version probe: `{baseline.get('astap_version_probe')}`",
        "",
        "## Sources locales",
        "",
        f"- path: `{baseline.get('astap_source_dir')}`",
        f"- archive/source hash: `{baseline.get('astap_source_archive_sha256')}`",
        "",
        "## B0 systeme",
        "",
    ]
    for rec in b0:
        metrics = rec.get("log_metrics", {})
        lines.append(
            f"- `{rec.get('name')}`: success={rec.get('success')} "
            f"stars={metrics.get('image_stars')} image_quads={metrics.get('image_quads')} "
            f"db_stars={metrics.get('catalog_stars')} db_quads={metrics.get('catalog_quads')} "
            f"matches={metrics.get('matched_quads')} elapsed={rec.get('elapsed_s')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", type=Path, default=REPO_ROOT / "reports")
    ap.add_argument("--runtime-dir", type=Path, default=REPO_ROOT / "reports" / "zn1_runtime")
    ap.add_argument("--astap-source", type=Path, default=REPO_ROOT / "ASTAP-main")
    ap.add_argument("--astap-bin", default=shutil.which("astap") or "astap")
    ap.add_argument("--astap-db", type=Path, default=Path("/opt/astap"))
    ap.add_argument("--family", default="d50")
    ap.add_argument(
        "--astap-extra-arg",
        action="append",
        default=["-z", "2"],
        help="Extra ASTAP solve argument. Defaults to '-z 2' to make the ZN1 downsample/binning explicit.",
    )
    ap.add_argument("--skip-system-run", action="store_true")
    args = ap.parse_args()

    reports = args.reports_dir.resolve()
    build_dir = reports / "zn2_astap_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    out_run_dir = build_dir / "runs"
    out_run_dir.mkdir(parents=True, exist_ok=True)

    system_astap = Path(args.astap_bin)
    source_zip = args.astap_source.with_suffix(".zip")
    baseline = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": git_info(),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": package_versions(),
        "which_astap": shutil.which("astap"),
        "astap_bin_argument": str(args.astap_bin),
        "system_astap_path": str(system_astap),
        "system_astap_sha256": sha256(system_astap) if system_astap.exists() else None,
        "astap_version_probe": astap_version_probe(str(args.astap_bin)),
        "astap_db": str(args.astap_db),
        "astap_family": args.family,
        "astap_extra_args": list(args.astap_extra_arg),
        "astap_source_dir": str(args.astap_source),
        "astap_source_archive_sha256": sha256(source_zip) if source_zip.exists() else None,
        "source_hashes": source_hashes(args.astap_source),
        "compiler_tools": which_all(["lazbuild", "fpc", "ppcx64"]),
        "runtime_files": [],
    }
    for name in IMAGE_NAMES:
        runtime = args.runtime_dir / f"{safe_stem(name)}_runtime.fit"
        baseline["runtime_files"].append({
            "name": name,
            "path": str(runtime),
            "exists": runtime.exists(),
            "sha256": sha256(runtime) if runtime.exists() else None,
        })

    build = try_build_reference(args.astap_source, build_dir)
    equivalence: dict[str, Any] = {
        "build": build,
        "astap_extra_args": list(args.astap_extra_arg),
        "B0_system": [],
        "B1_local_reference": [],
        "comparisons": [],
    }

    if not args.skip_system_run:
        for name in IMAGE_NAMES:
            runtime = args.runtime_dir / f"{safe_stem(name)}_runtime.fit"
            if not runtime.exists():
                equivalence["B0_system"].append({"name": name, "success": False, "error": "missing runtime FITS"})
                continue
            out_base = out_run_dir / f"{safe_stem(name)}_B0_system"
            rec = run_astap(str(args.astap_bin), runtime, out_base, args.astap_db, args.family, list(args.astap_extra_arg))
            rec["name"] = name
            equivalence["B0_system"].append(rec)

    local_bin = Path(build.get("output_binary", ""))
    if build.get("status") == "built" and local_bin.exists():
        for name in IMAGE_NAMES:
            runtime = args.runtime_dir / f"{safe_stem(name)}_runtime.fit"
            out_base = out_run_dir / f"{safe_stem(name)}_B1_local"
            rec = run_astap(str(local_bin), runtime, out_base, args.astap_db, args.family, list(args.astap_extra_arg))
            rec["name"] = name
            equivalence["B1_local_reference"].append(rec)
        for a, b in zip(equivalence["B0_system"], equivalence["B1_local_reference"]):
            equivalence["comparisons"].append({
                "name": a.get("name"),
                "success_equal": bool(a.get("success")) == bool(b.get("success")),
                "wcs_delta": compare_wcs(a.get("wcs", {}), b.get("wcs", {})),
                "metrics_equal": a.get("log_metrics") == b.get("log_metrics"),
            })

    reports.mkdir(parents=True, exist_ok=True)
    (reports / "zenear_zn2_baseline.json").write_text(
        json.dumps(baseline, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    (reports / "zenear_zn2_astap_binary_equivalence.json").write_text(
        json.dumps(equivalence, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(reports / "zenear_zn2_astap_binary_equivalence.md", baseline, equivalence)

    summary = {
        "baseline": str(reports / "zenear_zn2_baseline.json"),
        "equivalence": str(reports / "zenear_zn2_astap_binary_equivalence.json"),
        "build_status": build.get("status"),
        "build_reason": build.get("reason"),
        "B0_success": f"{sum(1 for r in equivalence['B0_system'] if r.get('success'))}/{len(equivalence['B0_system'])}",
        "B1_success": f"{sum(1 for r in equivalence['B1_local_reference'] if r.get('success'))}/{len(equivalence['B1_local_reference'])}",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 2 if build.get("status") == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
