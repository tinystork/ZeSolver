from __future__ import annotations

import logging
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
            if slc.start == slc.stop:
                continue
            if index.bucket_cap > 0 and (slc.stop - slc.start) > index.bucket_cap:
                logger.debug("skipping bucket %s (size %d > cap %d)", level, slc.stop - slc.start, index.bucket_cap)
                continue
            weight = 1
            if weight_array is not None:
                weight = int(weight_array[hash_idx])
                if weight <= 0:
                    continue
            for tile_index in index.tile_indices[slc]:
                idx = int(tile_index)
                if allowed_tiles is not None and idx not in allowed_tiles:
                    continue
                votes[idx] += weight
    results: list[tuple[str, int]] = []
    for tile_index, score in votes.most_common():
        key = tile_keys[tile_index] if tile_index < len(tile_keys) else f"tile_{tile_index}"
        results.append((key, int(score)))
    return results
