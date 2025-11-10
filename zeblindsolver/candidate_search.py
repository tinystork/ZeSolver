from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from .quad_index_builder import QuadIndex, lookup_hashes, load_manifest

logger = logging.getLogger(__name__)


def tally_candidates(obs_hashes: np.ndarray, index_root: Path | str, levels: Sequence[str] | None = None) -> list[tuple[str, int]]:
    """Return candidate tiles ordered by their hash vote counts."""
    if obs_hashes.size == 0:
        return []
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
        slices = lookup_hashes(index_root, level, obs_hashes)
        for slc in slices:
            if slc.start == slc.stop:
                continue
            if index.bucket_cap > 0 and (slc.stop - slc.start) > index.bucket_cap:
                logger.debug("skipping bucket %s (size %d > cap %d)", level, slc.stop - slc.start, index.bucket_cap)
                continue
            votes.update(index.tile_indices[slc])
    results: list[tuple[str, int]] = []
    for tile_index, score in votes.most_common():
        key = tile_keys[tile_index] if tile_index < len(tile_keys) else f"tile_{tile_index}"
        results.append((key, int(score)))
    return results
