"""Benchmark world sets. EVALUATOR ONLY (uses hidden ground truth for the acceptance rule)."""
from .groundtruth import ground_truth
from .world import DEV_START, FROZEN_START, World, generate_world


def frozen_worlds(n: int = 20, start_seed: int = FROZEN_START) -> list[World]:
    """First n accepted worlds scanning seeds upward from start_seed.
    Frozen set: start_seed=FROZEN_START. Development set: start_seed=DEV_START.
    The ranges are far apart so the two sets can never share a seed."""
    out, seed = [], start_seed
    while len(out) < n:
        w = generate_world(seed)
        if w is not None and ground_truth(w).minimal_sets:
            out.append(w)
        seed += 1
    return out
