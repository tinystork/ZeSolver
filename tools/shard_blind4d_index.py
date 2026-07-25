#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zeblindsolver.index_manifest_4d import MANIFEST_SCHEMA, MANIFEST_VERSION, sha256_file
from zeblindsolver.quad_index_4d import ASTROMETRY_AB_CODE_4D_SCHEMA


@dataclass(frozen=True)
class ShardPlan:
    shard_id: str
    tile_indices: tuple[int, ...]
    tile_keys: tuple[str, ...]


_CANCELLED = False


def _request_cancel(signum, frame) -> None:  # type: ignore[no-untyped-def]
    del signum, frame
    global _CANCELLED
    _CANCELLED = True


def _metadata_array(metadata: dict[str, Any]) -> np.ndarray:
    text = json.dumps(metadata, sort_keys=True, ensure_ascii=True)
    return np.asarray([text], dtype=f"<U{max(1, len(text))}")


def _load_metadata(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    metadata = json.loads(str(data["metadata"][0]))
    schema = str(metadata.get("schema") or "")
    if schema != ASTROMETRY_AB_CODE_4D_SCHEMA:
        raise SystemExit(f"unsupported 4D schema: {schema!r}")
    return metadata


def _ring_number(tile_key: str) -> int:
    try:
        return int(str(tile_key).split("_", 1)[1][:2])
    except Exception as exc:
        raise ValueError(f"invalid d50 tile key: {tile_key!r}") from exc


def build_plan(tile_keys: Iterable[str], *, topology: str, tiles_per_shard: int, oracle_tile_keys: Iterable[str]) -> list[ShardPlan]:
    keys = tuple(str(v) for v in tile_keys)
    by_key = {key: i for i, key in enumerate(keys)}
    topology = topology.strip().lower()
    plans: list[ShardPlan] = []
    if topology == "ring":
        rings: dict[int, list[int]] = {}
        for i, key in enumerate(keys):
            rings.setdefault(_ring_number(key), []).append(i)
        for ring in sorted(rings):
            idx = tuple(rings[ring])
            plans.append(ShardPlan(f"direct-d50-ring-{ring:02d}", idx, tuple(keys[i] for i in idx)))
        return plans
    if topology == "fixed":
        n = max(1, int(tiles_per_shard))
        for shard_no, start in enumerate(range(0, len(keys), n)):
            idx = tuple(range(start, min(len(keys), start + n)))
            plans.append(ShardPlan(f"direct-d50-fixed{n}-{shard_no:03d}", idx, tuple(keys[i] for i in idx)))
        return plans
    if topology == "oracle":
        wanted = [str(v).strip() for v in oracle_tile_keys if str(v).strip()]
        if not wanted:
            raise ValueError("oracle topology requires --tile-keys")
        for key in wanted:
            if key not in by_key:
                raise ValueError(f"unknown tile key: {key}")
        for shard_no, key in enumerate(wanted):
            idx = (by_key[key],)
            plans.append(ShardPlan(f"direct-d50-oracle-{shard_no:03d}-{key}", idx, (key,)))
        return plans
    raise ValueError(f"unknown topology: {topology!r}")


def _infer_block_sizes(data: Any, tile_count: int) -> tuple[int, int]:
    if tile_count <= 0:
        raise ValueError("source has no tile keys")
    stars_per_tile = int(data["catalog_ra_dec"].shape[0]) // tile_count
    quads_per_tile = int(data["codes_4d"].shape[0]) // tile_count
    if stars_per_tile * tile_count != int(data["catalog_ra_dec"].shape[0]):
        raise ValueError("catalog star array is not evenly tile-blocked")
    if quads_per_tile * tile_count != int(data["codes_4d"].shape[0]):
        raise ValueError("quad code array is not evenly tile-blocked")
    return stars_per_tile, quads_per_tile


def _extract_blocks(data: Any, plan: ShardPlan, *, stars_per_tile: int, quads_per_tile: int) -> dict[str, np.ndarray]:
    codes: list[np.ndarray] = []
    quads: list[np.ndarray] = []
    source_quad_indices: list[np.ndarray] = []
    tile_key_indices: list[np.ndarray] = []
    ratio_hashes: list[np.ndarray] = []
    catalog_ra_dec: list[np.ndarray] = []
    catalog_xy: list[np.ndarray] = []
    for local_tile_index, tile_index in enumerate(plan.tile_indices):
        s0 = int(tile_index) * stars_per_tile
        s1 = s0 + stars_per_tile
        q0 = int(tile_index) * quads_per_tile
        q1 = q0 + quads_per_tile
        catalog_ra_dec.append(np.asarray(data["catalog_ra_dec"][s0:s1], dtype=data["catalog_ra_dec"].dtype))
        catalog_xy.append(np.asarray(data["catalog_xy"][s0:s1], dtype=data["catalog_xy"].dtype))
        codes.append(np.asarray(data["codes_4d"][q0:q1], dtype=data["codes_4d"].dtype))
        qsi = np.asarray(data["quad_star_indices"][q0:q1], dtype=np.int64)
        qsi = qsi - s0 + local_tile_index * stars_per_tile
        quads.append(qsi.astype(np.int32, copy=False))
        source_quad_indices.append(np.asarray(data["source_quad_indices"][q0:q1], dtype=data["source_quad_indices"].dtype))
        tile_key_indices.append(np.full((q1 - q0,), local_tile_index, dtype=np.int32))
        if "ratio_hashes" in data:
            ratio_hashes.append(np.asarray(data["ratio_hashes"][q0:q1], dtype=data["ratio_hashes"].dtype))
    result = {
        "codes_4d": np.concatenate(codes, axis=0),
        "quad_star_indices": np.concatenate(quads, axis=0),
        "source_quad_indices": np.concatenate(source_quad_indices, axis=0),
        "tile_key_indices": np.concatenate(tile_key_indices, axis=0),
        "tile_keys": np.asarray(plan.tile_keys, dtype=f"<U{max(1, max(len(v) for v in plan.tile_keys))}"),
        "catalog_ra_dec": np.concatenate(catalog_ra_dec, axis=0),
        "catalog_xy": np.concatenate(catalog_xy, axis=0),
    }
    result["ratio_hashes"] = (
        np.concatenate(ratio_hashes, axis=0)
        if ratio_hashes
        else np.full(result["codes_4d"].shape[0], -1, dtype=np.int64)
    )
    return result


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with tmp.open("xb") as handle:
            np.savez_compressed(handle, metadata=_metadata_array(metadata), **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    finally:
        if tmp.exists():
            tmp.unlink()


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _copy_zip_member_streaming(zip_path: Path, member: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        with zipfile.ZipFile(zip_path) as archive:
            with archive.open(member) as src, tmp.open("xb") as dst:
                while True:
                    if _CANCELLED:
                        raise KeyboardInterrupt("cancelled")
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
                dst.flush()
                os.fsync(dst.fileno())
        os.replace(tmp, target)
        _fsync_dir(target.parent)
    finally:
        if tmp.exists():
            tmp.unlink()


def _prepare_mmap_source(source: Path, out_dir: Path, source_sha: str) -> dict[str, Any]:
    required = (
        "codes_4d",
        "quad_star_indices",
        "source_quad_indices",
        "tile_key_indices",
        "ratio_hashes",
        "catalog_ra_dec",
        "catalog_xy",
    )
    cache_dir = out_dir / ".source_mmap_cache" / f"{source.stem}-{source_sha[:16]}"
    arrays: dict[str, Any] = {}
    for name in required:
        member = f"{name}.npy"
        target = cache_dir / member
        if not target.exists():
            print(f"extract source member for mmap: {member}", flush=True)
            _copy_zip_member_streaming(source, member, target)
        arrays[name] = np.load(target, mmap_mode="r", allow_pickle=False)
    with np.load(source, allow_pickle=False) as data:
        metadata = _load_metadata(data)
        arrays["metadata"] = data["metadata"]
        arrays["tile_keys"] = np.asarray(data["tile_keys"].astype(str).tolist(), dtype=data["tile_keys"].dtype)
    return {"metadata": metadata, "arrays": arrays, "cache_dir": cache_dir}


def _manifest_entry(path: Path, plan: ShardPlan, arrays: dict[str, np.ndarray], metadata: dict[str, Any], *, priority: int) -> dict[str, Any]:
    return {
        "id": plan.shard_id,
        "enabled": True,
        "path": path.name,
        "filename": path.name,
        "quad_schema": ASTROMETRY_AB_CODE_4D_SCHEMA,
        "index_version": int(metadata.get("version", 1)),
        "level": str(metadata.get("level") or "S"),
        "tile_keys": list(plan.tile_keys),
        "star_count": int(arrays["catalog_ra_dec"].shape[0]),
        "quad_count": int(arrays["codes_4d"].shape[0]),
        "sampler_tag": str(metadata.get("sampler_tag") or ""),
        "code_tol_recommended": float(metadata.get("code_tol_recommended", 0.015) or 0.015),
        "catalog_source": str(metadata.get("source_catalog") or metadata.get("catalog_source") or "astap_raw"),
        "sha256": sha256_file(path),
        "priority": int(priority),
        "file_size_bytes": int(path.stat().st_size),
        "metadata": metadata,
    }


def write_manifest(path: Path, entries: list[dict[str, Any]], *, description: str) -> None:
    payload = {
        "schema": MANIFEST_SCHEMA,
        "manifest_version": MANIFEST_VERSION,
        "description": description,
        "indexes": entries,
    }
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tmp.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    finally:
        if tmp.exists():
            tmp.unlink()


def convert(args: argparse.Namespace) -> int:
    signal.signal(signal.SIGINT, _request_cancel)
    signal.signal(signal.SIGTERM, _request_cancel)
    source = Path(args.source).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = Path(args.manifest_path).expanduser().resolve() if args.manifest_path else out_dir / "blind4d_manifest.json"
    source_sha = sha256_file(source)
    started = time.perf_counter()
    with np.load(source, allow_pickle=False) as data:
        metadata = _load_metadata(data)
        tile_keys = tuple(str(v) for v in data["tile_keys"].astype(str).tolist())
        plans = build_plan(
            tile_keys,
            topology=args.topology,
            tiles_per_shard=args.tiles_per_shard,
            oracle_tile_keys=args.tile_keys.split(",") if args.tile_keys else (),
        )
        if args.limit_shards:
            plans = plans[: int(args.limit_shards)]
        if args.dry_run:
            sizes = [len(plan.tile_keys) for plan in plans]
            print(json.dumps({"shards": len(plans), "tile_counts": sizes, "source_sha256": source_sha}, indent=2))
            return 0
    prepared = _prepare_mmap_source(source, out_dir, source_sha)
    metadata = dict(prepared["metadata"])
    data = prepared["arrays"]
    tile_keys = tuple(str(v) for v in data["tile_keys"].astype(str).tolist())
    stars_per_tile, quads_per_tile = _infer_block_sizes(data, len(tile_keys))
    plans = build_plan(
        tile_keys,
        topology=args.topology,
        tiles_per_shard=args.tiles_per_shard,
        oracle_tile_keys=args.tile_keys.split(",") if args.tile_keys else (),
    )
    if args.limit_shards:
        plans = plans[: int(args.limit_shards)]
    entries: list[dict[str, Any]] = []
    for i, plan in enumerate(plans, start=1):
        if _CANCELLED:
            print("cancelled before shard write", flush=True)
            return 130
        out_path = out_dir / f"{plan.shard_id}.npz"
        if args.resume and out_path.exists():
            print(f"[{i}/{len(plans)}] resume existing {out_path.name}", flush=True)
            with np.load(out_path, allow_pickle=False) as shard_data:
                shard_metadata = json.loads(str(shard_data["metadata"][0]))
                arrays = {
                    "catalog_ra_dec": shard_data["catalog_ra_dec"],
                    "codes_4d": shard_data["codes_4d"],
                }
                entries.append(_manifest_entry(out_path, plan, arrays, shard_metadata, priority=i - 1))
            continue
        arrays = _extract_blocks(data, plan, stars_per_tile=stars_per_tile, quads_per_tile=quads_per_tile)
        shard_metadata = dict(metadata)
        shard_metadata.update(
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source_monolith": str(source),
                "source_monolith_sha256": source_sha,
                "shard_topology": str(args.topology),
                "shard_id": plan.shard_id,
                "shard_tile_keys": list(plan.tile_keys),
                "source_tile_indices": [int(v) for v in plan.tile_indices],
            }
        )
        _write_npz_atomic(out_path, arrays, shard_metadata)
        entries.append(_manifest_entry(out_path, plan, arrays, shard_metadata, priority=i - 1))
        print(
            f"[{i}/{len(plans)}] wrote {out_path.name} tiles={len(plan.tile_keys)} "
            f"quads={arrays['codes_4d'].shape[0]} stars={arrays['catalog_ra_dec'].shape[0]}",
            flush=True,
        )
    write_manifest(
        manifest_path,
        entries,
        description=f"Sharded Blind 4D view from {source.name} topology={args.topology}",
    )
    print(f"done shards={len(entries)} manifest={manifest_path} elapsed_s={time.perf_counter() - started:.3f}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Shard a dense Blind 4D NPZ without rebuilding stars or quads.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-path")
    parser.add_argument("--topology", choices=("ring", "fixed", "oracle"), required=True)
    parser.add_argument("--tiles-per-shard", type=int, default=32)
    parser.add_argument("--tile-keys", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-shards", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return convert(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
