from __future__ import annotations

import logging
import math
from collections import Counter
from pathlib import Path
from typing import Collection, Sequence

import numpy as np

from .quad_index_builder import QuadIndex, lookup_hashes, load_manifest

logger = logging.getLogger(__name__)


def tally_candidates(
    obs_hashes: np.ndarray | tuple[np.ndarray, np.ndarray],
    index_root: Path | str,
    levels: Sequence[str] | None = None,
    allowed_tiles: Collection[int] | None = None,
) -> list[tuple[str, int]]:
    """Return candidate tiles ordered by their hash vote counts."""
    if isinstance(obs_hashes, tuple):
        hashes, counts = obs_hashes
    else:
        hashes = obs_hashes
        counts = None
    if hashes.size == 0:
        return []
    if counts is not None:
        if counts.shape != hashes.shape:
            raise ValueError("hash count vector must match hash array shape")
        weight_array = counts.astype(np.int64, copy=False)
    else:
        weight_array = None
    if levels is None:
        levels = ("L", "M", "S")
    manifest = load_manifest(index_root)
    tile_keys = [tile.get("tile_key", str(idx)) for idx, tile in enumerate(manifest.get("tiles", []))]
    votes: Counter[int] = Counter()
    for level in levels:
        try:
            index = QuadIndex.load(Path(index_root), level)
        except FileNotFoundError:
            logger.debug("quad index level %s missing, skipping", level)
            continue
        slices = lookup_hashes(index_root, level, hashes)
        for hash_idx, slc in enumerate(slices):
            bucket_size = slc.stop - slc.start
            if bucket_size <= 0:
                continue
            step = 1
            multiplier = 1
            if index.bucket_cap > 0 and bucket_size > index.bucket_cap:
                step = max(1, math.ceil(bucket_size / index.bucket_cap))
                sampled = math.ceil(bucket_size / step)
                if sampled > 0:
                    ratio = bucket_size / sampled
                    multiplier = max(1, int(round(ratio)))
                logger.debug(
                    "downsampling bucket %s (size %d > cap %d) with step=%d multiplier=%d",
                    level,
                    bucket_size,
                    index.bucket_cap,
                    step,
                    multiplier,
                )
            weight = 1
            if weight_array is not None:
                weight = int(weight_array[hash_idx])
                if weight <= 0:
                    continue
            effective_weight = weight * multiplier
            for pos in range(slc.start, slc.stop, step):
                tile_index = index.tile_indices[pos]
                idx = int(tile_index)
                if allowed_tiles is not None and idx not in allowed_tiles:
                    continue
                votes[idx] += effective_weight
    results: list[tuple[str, int]] = []
    for tile_index, score in votes.most_common():
        key = tile_keys[tile_index] if tile_index < len(tile_keys) else f"tile_{tile_index}"
        results.append((key, int(score)))
    return results
