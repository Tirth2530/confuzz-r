"""Repair-attack scenarios for the Critic experiment. EVALUATOR ONLY (uses exhaustive outcomes).

A scenario = (world, target minimal failure m, repair a) where a is one of the 3 cheapest single
actions that make m pass and keep the base passing (what the solver could hand Blue), restricted
to repairs with at least one regression (otherwise there is nothing to find).
Scenario sets (fixed before any Critic condition ran on frozen data): 60 worlds per split.
v0 (cheapest repair only, 20 worlds) gave 12 dev / 5 frozen scenarios: too few; superseded.
Hidden answer: R(a) = perturbations that pass on the original and fail on the patched world.

Critic sees the public spec of the ORIGINAL world, the repair, the diff it causes, and the target
failure. Never R(a).
"""
from __future__ import annotations

from dataclasses import dataclass

from .benchmark_worlds import frozen_worlds
from .groundtruth import ground_truth, outcomes
from .repair import ACTION_TYPES, Action, action_cost, apply_action, concrete_actions
from .sim import simulate
from .world import DEV_START, FROZEN_START, World


@dataclass
class Scenario:
    id: str
    world: World
    target: frozenset[str]
    action: Action
    patched: World
    regressions: frozenset[frozenset[str]]


def fixing_repairs(world: World, target: frozenset[str], k: int = 3) -> list[Action]:
    fixing = []
    for t in ACTION_TYPES:
        if t == "DO_NOTHING":
            continue
        for a in concrete_actions(world, {"type": t}):
            w2 = apply_action(world, a)
            if not simulate(w2, frozenset()).critical and not simulate(w2, target).critical:
                fixing.append(a)
    return sorted(fixing, key=lambda a: (action_cost(world, a), str(a.to_dict())))[:k]


def scenarios(split: str, n_worlds: int = 60) -> list[Scenario]:
    worlds = frozen_worlds(n_worlds, DEV_START if split == "dev" else FROZEN_START)
    out = []
    for w in worlds:
        before = outcomes(w)
        for m in sorted(ground_truth(w).minimal_sets, key=lambda s: sorted(s)):
            for a in fixing_repairs(w, m):
                p = apply_action(w, a)
                after = outcomes(p)
                reg = frozenset(q for q in before if after[q] and not before[q])
                if reg:
                    out.append(Scenario(f"{w.id}:{'+'.join(sorted(m))}:{a.describe(w)}", w, m, a, p, reg))
    return out


def repair_diff(s: Scenario) -> dict:
    """What the repair changed, in organizer terms (public information)."""
    w, p = s.world, s.patched
    moved = []
    for a, b in zip(w.sessions, p.sessions):
        if (a.room, a.slot) != (b.room, b.slot):
            moved.append({"session": a.id, "title": a.title, "speaker": a.speaker,
                          "from": {"room": a.room, "zone": w.room(a.room).zone, "slot": a.slot},
                          "to": {"room": b.room, "zone": p.room(b.room).zone, "slot": b.slot}})
    activated = sorted(p.active_standby - w.active_standby)
    return {"action": s.action.to_dict(), "described": s.action.describe(w),
            "moved_sessions": moved, "activated_standby_rooms": activated}
