#!/usr/bin/env python3
"""Brute-force benchmark harness for the ZeSolver blind pipeline."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zeblindsolver.zeblindsolver import FITS_EXTENSIONS, SolveConfig, solve_blind


@dataclass(frozen=True)
class SweepProfile:
    label: str
    overrides: dict[str, Any]
    description: str | None = None


@dataclass
class AttemptResult:
    image: Path
    sweep_index: int
    profile_label: str
    overrides: dict[str, Any]
    success: bool
    elapsed_s: float
    message: str
    tile_key: str | None
    stats: dict[str, Any]


DEFAULT_CONFIG = SolveConfig()
DEFAULT_SWEEP: list[SweepProfile] = [
    SweepProfile("baseline", {}, "CLI defaults."),
    SweepProfile(
        "dense-stars",
        {"max_stars": 900, "max_quads": 12000},
        "Loosen detection caps for crowded fields.",
    ),
    SweepProfile(
        "loose-detect",
        {"detect_k_sigma": 2.2, "detect_min_area": 4, "max_stars": 1200, "pixel_tolerance": 3.2},
        "Aggressive detection for faint/noisy frames.",
    ),
    SweepProfile(
        "wide-buckets",
        {"bucket_limit_override": 6144, "vote_percentile": 32},
        "Relax voting quotas when few quads match.",
    ),
    SweepProfile(
        "downsample-2",
        {"downsample": 2, "max_quads": 20000, "vote_percentile": 25},
        "Use 2x downsampling for very large sensors.",
    ),
]

_CONFIG_FIELD_LUT = {field.name.lower(): field.name for field in fields(SolveConfig)}
_FITS_ENDINGS = tuple(
    sorted(
        {ext.lower() for ext in FITS_EXTENSIONS}
        .union({f"{ext.lower()}.gz" for ext in FITS_EXTENSIONS})
        .union({f"{ext.lower()}.fz" for ext in FITS_EXTENSIONS})
    )
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a set of SolveConfig sweeps against one or more FITS images."
    )
    parser.add_argument("inputs", nargs="+", help="FITS paths, directories, glob patterns, or @list files.")
    parser.add_argument("--index-root", type=Path, required=True, help="Path to the blind index root.")
    parser.add_argument("--grid", type=Path, help="JSON file describing sweep entries.")
    parser.add_argument(
        "--continue-after-success",
        action="store_true",
        help="Try all sweeps even if one succeeds for a given image.",
    )
    parser.add_argument(
        "--allow-write",
        action="store_true",
        help="Allow WCS headers to be written back to the source FITS (default: copy to temp).",
    )
    parser.add_argument("--limit", type=int, help="Maximum number of inputs to run after expansion.")
    parser.add_argument("--output-json", type=Path, help="Write the detailed run log to JSON.")
    parser.add_argument("--output-csv", type=Path, help="Write a flat run log to CSV.")
    parser.add_argument("--log-level", default="INFO", help="Harness log level and base SolveConfig log level.")
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_CONFIG.max_candidates)
    parser.add_argument("--max-stars", type=int, default=DEFAULT_CONFIG.max_stars)
    parser.add_argument("--max-quads", type=int, default=DEFAULT_CONFIG.max_quads)
    parser.add_argument("--detect-k-sigma", type=float, default=DEFAULT_CONFIG.detect_k_sigma)
    parser.add_argument("--detect-min-area", type=int, default=DEFAULT_CONFIG.detect_min_area)
    parser.add_argument("--bucket-cap-s", type=int, dest="bucket_cap_S", default=DEFAULT_CONFIG.bucket_cap_S)
    parser.add_argument("--bucket-cap-m", type=int, dest="bucket_cap_M", default=DEFAULT_CONFIG.bucket_cap_M)
    parser.add_argument("--bucket-cap-l", type=int, dest="bucket_cap_L", default=DEFAULT_CONFIG.bucket_cap_L)
    parser.add_argument("--sip-order", type=int, choices=(2, 3), default=DEFAULT_CONFIG.sip_order)
    parser.add_argument("--quality-rms", type=float, default=DEFAULT_CONFIG.quality_rms)
    parser.add_argument("--quality-inliers", type=int, default=DEFAULT_CONFIG.quality_inliers)
    parser.add_argument("--pixel-tolerance", type=float, default=DEFAULT_CONFIG.pixel_tolerance)
    parser.add_argument("--downsample", type=int, choices=range(1, 5), default=DEFAULT_CONFIG.downsample)
    parser.add_argument("--vote-percentile", type=int, default=DEFAULT_CONFIG.vote_percentile)
    parser.add_argument("--bucket-limit-override", type=int, default=DEFAULT_CONFIG.bucket_limit_override or 0)
    parser.add_argument("--tile-cache-size", type=int, default=None)
    parser.add_argument("--focal-length", type=float, dest="focal_length_mm")
    parser.add_argument("--pixel-size", type=float, dest="pixel_size_um")
    parser.add_argument("--pixel-scale", type=float, dest="pixel_scale_arcsec")
    parser.add_argument("--pixel-scale-min", type=float, dest="pixel_scale_min_arcsec")
    parser.add_argument("--pixel-scale-max", type=float, dest="pixel_scale_max_arcsec")
    parser.add_argument("--ra-hint", type=float, dest="ra_hint_deg")
    parser.add_argument("--dec-hint", type=float, dest="dec_hint_deg")
    parser.add_argument("--radius-hint", type=float, dest="radius_hint_deg")
    parser.add_argument("--no-parity-flip", action="store_true", help="Disable the parity flip test.")
    parser.add_argument("--full-mode", action="store_true", help="Disable fast_mode heuristics.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    index_root = args.index_root.expanduser().resolve()
    if not index_root.exists():
        parser.error(f"--index-root {index_root} does not exist")
    args.index_root = index_root
    if args.grid:
        args.grid = args.grid.expanduser().resolve()
        if not args.grid.is_file():
            parser.error(f"--grid file {args.grid} not found")
    args.log_level = args.log_level.upper()
    logging.basicConfig(level=getattr(logging, args.log_level, logging.INFO), format="%(levelname)s: %(message)s")

    try:
        sweeps = load_sweeps(args.grid)
    except ValueError as exc:
        parser.error(str(exc))
        return 2
    images = resolve_inputs(args.inputs, args.limit)
    if not images:
        parser.error("no FITS images matched the provided inputs")
        return 2
    logging.info("expanded %d input(s); using %d sweep profiles", len(images), len(sweeps))
    base_config = build_base_config(args)
    results = run_benchmark(images, sweeps, base_config, args)
    write_outputs(results, sweeps, base_config, args)
    logging.info("processed %d images (%d total attempts)", len(images), len(results))
    return 0 if any(entry.success for entry in results) else 1


def _attr_with_fallback(args: argparse.Namespace, primary: str, secondary: str, default: Any) -> Any:
    if hasattr(args, primary):
        value = getattr(args, primary)
        if value is not None:
            return value
    if hasattr(args, secondary):
        value = getattr(args, secondary)
        if value is not None:
            return value
    return default


def build_base_config(args: argparse.Namespace) -> SolveConfig:
    tile_cache = args.tile_cache_size if args.tile_cache_size is not None else DEFAULT_CONFIG.tile_cache_size
    bucket_cap_s = _attr_with_fallback(args, "bucket_cap_S", "bucket_cap_s", DEFAULT_CONFIG.bucket_cap_S)
    bucket_cap_m = _attr_with_fallback(args, "bucket_cap_M", "bucket_cap_m", DEFAULT_CONFIG.bucket_cap_M)
    bucket_cap_l = _attr_with_fallback(args, "bucket_cap_L", "bucket_cap_l", DEFAULT_CONFIG.bucket_cap_L)
    return SolveConfig(
        max_candidates=args.max_candidates,
        max_stars=args.max_stars,
        max_quads=args.max_quads,
        detect_k_sigma=args.detect_k_sigma,
        detect_min_area=args.detect_min_area,
        bucket_cap_S=bucket_cap_s,
        bucket_cap_M=bucket_cap_m,
        bucket_cap_L=bucket_cap_l,
        sip_order=args.sip_order,
        quality_rms=args.quality_rms,
        quality_inliers=args.quality_inliers,
        pixel_tolerance=args.pixel_tolerance,
        log_level=args.log_level,
        verbose=False,
        try_parity_flip=not args.no_parity_flip,
        fast_mode=not args.full_mode,
        downsample=args.downsample,
        tile_cache_size=tile_cache,
        bucket_limit_override=args.bucket_limit_override,
        vote_percentile=args.vote_percentile,
        ra_hint_deg=args.ra_hint_deg,
        dec_hint_deg=args.dec_hint_deg,
        radius_hint_deg=args.radius_hint_deg,
        focal_length_mm=args.focal_length_mm,
        pixel_size_um=args.pixel_size_um,
        pixel_scale_arcsec=args.pixel_scale_arcsec,
        pixel_scale_min_arcsec=args.pixel_scale_min_arcsec,
        pixel_scale_max_arcsec=args.pixel_scale_max_arcsec,
    )


def run_benchmark(
    images: list[Path],
    sweeps: list[SweepProfile],
    base_config: SolveConfig,
    args: argparse.Namespace,
    *,
    log: Callable[[str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    progress: Callable[[int, int, AttemptResult], None] | None = None,
) -> list[AttemptResult]:
    results: list[AttemptResult] = []
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    temp_root: Path | None = None
    emitter = log or print
    if not args.allow_write:
        temp_dir = tempfile.TemporaryDirectory(prefix="zesolver-bench-")
        temp_root = Path(temp_dir.name)
    total_attempts = len(images) * len(sweeps)
    completed = 0
    cancelled = False
    try:
        for image_index, image_path in enumerate(images, 1):
            if cancel_check and cancel_check():
                cancelled = True
                break
            emitter(f"[{image_index}/{len(images)}] {image_path}")
            for sweep_index, profile in enumerate(sweeps, 1):
                if cancel_check and cancel_check():
                    cancelled = True
                    break
                config = merge_config(base_config, profile.overrides)
                target_path = image_path
                if temp_root is not None and looks_like_fits(image_path):
                    target_path = copy_to_temp(image_path, temp_root)
                attempt = execute_attempt(
                    target_path=target_path,
                    image_path=image_path,
                    config=config,
                    sweep_index=sweep_index,
                    profile=profile,
                    index_root=args.index_root,
                )
                results.append(attempt)
                completed += 1
                if progress:
                    try:
                        progress(completed, total_attempts or 1, attempt)
                    except Exception:
                        pass
                status = "OK " if attempt.success else "ERR"
                tile = f" tile={attempt.tile_key}" if attempt.tile_key else ""
                emitter(
                    f"    [{sweep_index}/{len(sweeps)}] {profile.label:<14} "
                    f"{status} {attempt.elapsed_s:5.2f}s{tile} :: {attempt.message}"
                )
                if attempt.success and not args.continue_after_success:
                    break
            if cancelled:
                break
            emitter("")
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
    if cancelled:
        emitter("benchmark cancelled")
    return results


def execute_attempt(
    target_path: Path,
    image_path: Path,
    config: SolveConfig,
    sweep_index: int,
    profile: SweepProfile,
    index_root: Path,
) -> AttemptResult:
    start = time.perf_counter()
    try:
        solution = solve_blind(target_path, index_root, config=config)
        success = solution.success
        message = solution.message
        tile = solution.tile_key
        stats = solution.stats or {}
    except Exception as exc:  # pragma: no cover - defensive logging
        success = False
        message = f"exception: {exc}"
        tile = None
        stats = {}
        logging.exception("attempt failed for %s (%s)", image_path, profile.label)
    elapsed = time.perf_counter() - start
    return AttemptResult(
        image=image_path,
        sweep_index=sweep_index,
        profile_label=profile.label,
        overrides=dict(profile.overrides),
        success=success,
        elapsed_s=elapsed,
        message=message,
        tile_key=tile,
        stats=stats,
    )


def write_outputs(
    results: list[AttemptResult],
    sweeps: list[SweepProfile],
    base_config: SolveConfig,
    args: argparse.Namespace,
) -> None:
    if args.output_json:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "index_root": str(args.index_root),
            "allow_write": bool(args.allow_write),
            "continue_after_success": bool(args.continue_after_success),
            "base_config": asdict(base_config),
            "sweeps": [
                {"label": profile.label, "overrides": profile.overrides, "description": profile.description}
                for profile in sweeps
            ],
            "runs": [
                {
                    "image": str(entry.image),
                    "sweep_index": entry.sweep_index,
                    "profile": entry.profile_label,
                    "success": entry.success,
                    "elapsed_s": entry.elapsed_s,
                    "message": entry.message,
                    "tile_key": entry.tile_key,
                    "overrides": entry.overrides,
                    "stats": make_json_safe(entry.stats),
                }
                for entry in results
            ],
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logging.info("wrote JSON results to %s", args.output_json)
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "image",
            "sweep_index",
            "profile",
            "success",
            "elapsed_s",
            "message",
            "tile_key",
            "overrides",
            "inliers",
            "rms_px",
        ]
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for entry in results:
                stats = entry.stats or {}
                writer.writerow(
                    {
                        "image": str(entry.image),
                        "sweep_index": entry.sweep_index,
                        "profile": entry.profile_label,
                        "success": entry.success,
                        "elapsed_s": f"{entry.elapsed_s:.3f}",
                        "message": entry.message,
                        "tile_key": entry.tile_key or "",
                        "overrides": json.dumps(entry.overrides, sort_keys=True),
                        "inliers": stats.get("inliers"),
                        "rms_px": stats.get("rms_px"),
                    }
                )
        logging.info("wrote CSV results to %s", args.output_csv)


def load_sweeps(grid_path: Path | None) -> list[SweepProfile]:
    if grid_path is None:
        return [SweepProfile(entry.label, dict(entry.overrides), entry.description) for entry in DEFAULT_SWEEP]
    payload = json.loads(grid_path.read_text(encoding="utf-8"))
    raw_entries: Iterable[Any]
    if isinstance(payload, dict) and "sweeps" in payload:
        raw_entries = payload["sweeps"]
    elif isinstance(payload, list):
        raw_entries = payload
    else:
        raise ValueError("grid file must be a list or an object with a 'sweeps' array")
    sweeps: list[SweepProfile] = []
    for idx, entry in enumerate(raw_entries, 1):
        if not isinstance(entry, dict):
            raise ValueError("each grid entry must be a JSON object")
        label = entry.get("label") or entry.get("name") or f"profile-{idx}"
        description = entry.get("description") or entry.get("notes")
        overrides = entry.get("overrides")
        if overrides is None:
            overrides = {
                key: value for key, value in entry.items() if key not in {"label", "name", "description", "notes"}
            }
        if not isinstance(overrides, dict):
            raise ValueError(f"grid entry '{label}' overrides must be an object")
        sweeps.append(SweepProfile(label, normalize_overrides(overrides), description))
    if not sweeps:
        raise ValueError("grid file did not define any sweeps")
    return sweeps


def normalize_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        canonical = _CONFIG_FIELD_LUT.get(key.lower())
        if not canonical:
            raise ValueError(f"unknown SolveConfig field '{key}' in overrides")
        normalized[canonical] = value
    return normalized


def merge_config(base: SolveConfig, overrides: dict[str, Any]) -> SolveConfig:
    if not overrides:
        return base
    return replace(base, **overrides)


def resolve_inputs(raw_inputs: Sequence[str], limit: int | None = None) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    queue: list[str] = list(raw_inputs)
    for item in queue:
        for expanded in expand_input_token(item):
            path = Path(expanded).expanduser()
            if path.is_dir():
                for candidate in iter_dir_fits(path):
                    candidate = candidate.resolve()
                    if candidate in seen:
                        continue
                    seen.add(candidate)
                    resolved.append(candidate)
                    if limit and len(resolved) >= limit:
                        return resolved
            elif path.is_file():
                candidate = path.resolve()
                if not looks_like_fits(candidate):
                    logging.warning("skipping %s (not a FITS file)", candidate)
                    continue
                if candidate in seen:
                    continue
                seen.add(candidate)
                resolved.append(candidate)
                if limit and len(resolved) >= limit:
                    return resolved
            else:
                logging.warning("skipping %s (not a file or directory)", path)
    if limit:
        return resolved[:limit]
    return resolved


def expand_input_token(token: str) -> list[str]:
    token = token.strip()
    if not token:
        return []
    if token.startswith("@"):
        list_path = Path(token[1:]).expanduser()
        if not list_path.is_file():
            logging.warning("list file %s not found", list_path)
            return []
        entries: list[str] = []
        for line in list_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            candidate = Path(stripped)
            if not candidate.is_absolute():
                candidate = (list_path.parent / candidate).resolve()
            else:
                candidate = candidate.expanduser().resolve()
            entries.append(str(candidate))
        return entries
    if any(char in token for char in "*?[]"):
        matches = glob.glob(token)
        if matches:
            return matches
    return [token]


def iter_dir_fits(root: Path) -> Iterable[Path]:
    for candidate in sorted(root.rglob("*")):
        if candidate.is_file() and looks_like_fits(candidate):
            yield candidate


def looks_like_fits(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in _FITS_ENDINGS)


def copy_to_temp(path: Path, temp_root: Path) -> Path:
    suffix = "".join(path.suffixes) or path.suffix or ""
    dest = temp_root / f"{path.stem}_{uuid.uuid4().hex}{suffix}"
    shutil.copy2(path, dest)
    return dest


def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {key: make_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(value) for value in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return str(obj)


if __name__ == "__main__":
    raise SystemExit(main())
