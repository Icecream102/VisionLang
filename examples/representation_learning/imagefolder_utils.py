"""Dataset splitting helpers for ImageFolder-based representation experiments."""

import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple


def stratified_split_indices(
    targets: Sequence[int], val_fraction: float, seed: int
) -> Tuple[List[int], List[int]]:
    """Create deterministic per-class train/validation indices without copying images."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be strictly between 0 and 1.")
    by_class: Dict[int, List[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        by_class[target].append(index)

    rng = random.Random(seed)
    train_indices, val_indices = [], []
    for indices in by_class.values():
        if len(indices) < 2:
            raise ValueError("Every class needs at least two images for a split.")
        rng.shuffle(indices)
        val_count = min(len(indices) - 1, max(1, round(len(indices) * val_fraction)))
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])
    return sorted(train_indices), sorted(val_indices)
