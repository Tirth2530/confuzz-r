"""Perturbation-space helpers. Pure combinatorics, no evaluation results."""
from itertools import combinations


def all_perturbations(ids):
    for k in range(len(ids) + 1):
        for c in combinations(ids, k):
            yield frozenset(c)
