"""Hidden ground truth by exhaustive enumeration. EVALUATOR ONLY.

This module must never be imported by the MCP server, the agents, or any
testing policy. tests/test_isolation.py enforces that.

M(w) = minimal critical failing sets: sets that fail critically while every
proper subset passes. Supersets of a failing set usually also fail, so raw
failing combinations would overstate what a tester "found".
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

from .sim import simulate
from .world import World


@dataclass(frozen=True)
class GroundTruth:
    world_id: str
    n_dims: int
    failing: frozenset[frozenset[str]]
    minimal_sets: frozenset[frozenset[str]]

    def by_order(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for m in self.minimal_sets:
            out[len(m)] = out.get(len(m), 0) + 1
        return dict(sorted(out.items()))


from .space import all_perturbations  # noqa: E402


def outcomes(world: World) -> dict[frozenset[str], bool]:
    return {p: simulate(world, p).critical for p in all_perturbations(world.disruption_ids)}


def minimal_failing(out: dict[frozenset[str], bool]) -> frozenset[frozenset[str]]:
    mins = set()
    for p, fail in out.items():
        if not fail or not p:
            continue
        if all(not out[frozenset(q)] for k in range(1, len(p)) for q in combinations(sorted(p), k)):
            mins.add(p)
    return frozenset(mins)


_CACHE: dict[str, GroundTruth] = {}


def ground_truth(world: World) -> GroundTruth:
    key = world.id + "|" + ",".join(sorted(world.active_standby)) + "|" + str(hash(world.sessions))
    if key not in _CACHE:
        out = outcomes(world)
        _CACHE[key] = GroundTruth(
            world.id, len(world.disruptions),
            frozenset(p for p, f in out.items() if f), minimal_failing(out),
        )
    return _CACHE[key]
