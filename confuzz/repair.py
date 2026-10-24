"""Repair actions, deterministic instantiation, and exact bounded-space verification.

Blue (OpenAI) proposes a repair STRATEGY: an action type plus optional fields.
instantiate() enumerates concrete actions matching the strategy, keeps those that
are valid and make every target failure pass, and returns the cheapest. The LLM
never picks the winner and never marks a repair safe.

Repair cost (operator-chosen weights, documented):
  MOVE_SESSION 1, SWAP_ROOMS 2, SHIFT_SESSION 3, ACTIVATE_STANDBY room.standby_cost (rental),
  DO_NOTHING 0.

Verification (bounded world, exact): over all 2^d perturbations compare the
patched world to the original. A repair is VERIFIED_ON_BOUNDED_SPACE iff
  (1) the undisrupted patched schedule passes,
  (2) every target minimal failing set passes on the patched world,
  (3) no perturbation that passed before fails after (no regressions).
This is exhaustive over the modeled perturbation space only. It says nothing
about disruptions outside the model.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .space import all_perturbations
from .domain import simulate
from .world import Session, World

ACTION_TYPES = ("MOVE_SESSION", "SWAP_ROOMS", "SHIFT_SESSION", "ACTIVATE_STANDBY", "DO_NOTHING")
BASE_COST = {"MOVE_SESSION": 1.0, "SWAP_ROOMS": 2.0, "SHIFT_SESSION": 3.0, "DO_NOTHING": 0.0}


@dataclass(frozen=True)
class Action:
    type: str
    session: str | None = None
    session2: str | None = None
    room: str | None = None
    slot: int | None = None
    sku: str | None = None      # inventory domain fields
    src: str | None = None
    dst: str | None = None
    qty: int | None = None
    site: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def describe(self, world: World) -> str:
        if getattr(world, "domain", "") == "inventory":
            from .inventory import describe
            return describe(world, self)
        if self.type == "MOVE_SESSION":
            return f"move {self.session} to room {self.room}"
        if self.type == "SWAP_ROOMS":
            return f"swap rooms of {self.session} and {self.session2}"
        if self.type == "SHIFT_SESSION":
            return f"shift {self.session} to slot {self.slot}, room {self.room}"
        if self.type == "ACTIVATE_STANDBY":
            return f"activate standby room {self.room}"
        return "do nothing"


def action_types(world) -> tuple:
    if getattr(world, "domain", "") == "inventory":
        from .inventory import ACTION_TYPES as T
        return T
    return ACTION_TYPES


def action_cost(world: World, a: Action) -> float:
    if getattr(world, "domain", "") == "inventory":
        from .inventory import action_cost as f
        return f(world, a)
    if a.type == "ACTIVATE_STANDBY":
        return world.room(a.room).standby_cost
    return BASE_COST[a.type]


def apply_action(world: World, a: Action) -> World:
    """Return the patched world. Raises ValueError if the action is invalid."""
    if getattr(world, "domain", "") == "inventory":
        from .inventory import apply_action as f
        return f(world, a)
    occupied = {(s.room, s.slot): s.id for s in world.sessions}
    usable = {r.id for r in world.usable_rooms()}
    if a.type == "DO_NOTHING":
        return world
    if a.type == "ACTIVATE_STANDBY":
        r = world.room(a.room)
        if not r.standby or r.id in world.active_standby:
            raise ValueError(f"{a.room} is not an inactive standby room")
        return replace(world, active_standby=world.active_standby | {r.id})
    s = world.session(a.session)
    if a.type == "MOVE_SESSION":
        if a.room not in usable:
            raise ValueError(f"room {a.room} not usable")
        if (a.room, s.slot) in occupied:
            raise ValueError(f"room {a.room} occupied in slot {s.slot}")
        return world.with_sessions(replace(x, room=a.room) if x.id == s.id else x for x in world.sessions)
    if a.type == "SHIFT_SESSION":
        if a.slot is None or not (0 <= a.slot < world.n_slots) or a.slot == s.slot:
            raise ValueError("invalid target slot")
        if a.room not in usable or (a.room, a.slot) in occupied:
            raise ValueError(f"room {a.room} not free in slot {a.slot}")
        if any(x.speaker == s.speaker and x.slot == a.slot for x in world.sessions if x.id != s.id):
            raise ValueError("speaker already presenting in target slot")
        return world.with_sessions(replace(x, slot=a.slot, room=a.room) if x.id == s.id else x for x in world.sessions)
    if a.type == "SWAP_ROOMS":
        s2 = world.session(a.session2)
        if s2.slot != s.slot or s2.id == s.id:
            raise ValueError("swap requires two different sessions in the same slot")
        return world.with_sessions(
            replace(x, room=s2.room) if x.id == s.id else replace(x, room=s.room) if x.id == s2.id else x
            for x in world.sessions)
    raise ValueError(f"unknown action type {a.type}")


def apply_actions(world: World, actions: list[Action]) -> World:
    for a in actions:
        world = apply_action(world, a)
    return world


def concrete_actions(world: World, strategy: dict) -> list[Action]:
    """All concrete actions consistent with a (possibly partial) strategy."""
    if getattr(world, "domain", "") == "inventory":
        from .inventory import concrete_actions as f
        return f(world, strategy, Action)
    t = strategy.get("type")
    if t not in ACTION_TYPES:
        raise ValueError(f"type must be one of {ACTION_TYPES}")
    rooms = [r.id for r in world.rooms]
    sessions = [s.id for s in world.sessions]
    pick = lambda key, pool: [strategy[key]] if strategy.get(key) is not None else pool
    out: list[Action] = []
    if t == "DO_NOTHING":
        out = [Action("DO_NOTHING")]
    elif t == "ACTIVATE_STANDBY":
        out = [Action(t, room=r) for r in pick("room", [r.id for r in world.rooms if r.standby])]
    elif t == "MOVE_SESSION":
        out = [Action(t, session=s, room=r) for s in pick("session", sessions) for r in pick("room", rooms)]
    elif t == "SWAP_ROOMS":
        out = [Action(t, session=s, session2=s2) for s in pick("session", sessions) for s2 in pick("session2", sessions) if s < s2 or strategy.get("session")]
    elif t == "SHIFT_SESSION":
        out = [Action(t, session=s, slot=k, room=r) for s in pick("session", sessions)
               for k in pick("slot", list(range(world.n_slots))) for r in pick("room", rooms)]
    valid = []
    for a in out:
        try:
            apply_action(world, a)
            valid.append(a)
        except (ValueError, KeyError):
            pass
    return valid


@dataclass
class Instantiation:
    action: Action | None
    cost: float
    checks: int                 # simulator calls spent by the solver (repair budget, not discovery)
    candidates_considered: int
    reason: str


def instantiate(world: World, targets: list[frozenset[str]], strategy: dict,
                prior_actions: list[Action] = ()) -> Instantiation:
    """Cheapest valid concrete action matching strategy that makes all targets pass.
    prior_actions: earlier actions in a multi-step repair, already applied."""
    base = apply_actions(world, list(prior_actions))
    cands = concrete_actions(base, strategy)
    checks, fixing = 0, []
    for a in cands:
        w2 = apply_action(base, a)
        checks += 1
        if simulate(w2, frozenset()).critical:
            continue
        ok = True
        for t in targets:
            checks += 1
            if simulate(w2, t).critical:
                ok = False
                break
        if ok:
            fixing.append(a)
    if not fixing:
        return Instantiation(None, 0.0, checks, len(cands),
                             "no concrete action of this strategy fixes the target failure(s)")
    best = min(fixing, key=lambda a: (action_cost(base, a), str(a.to_dict())))
    return Instantiation(best, action_cost(base, best), checks, len(cands),
                         f"{len(fixing)} of {len(cands)} candidates fix the target; cheapest chosen")


def verify_exact(original: World, patched: World, targets: list[frozenset[str]]) -> dict:
    ids = original.disruption_ids
    before, after = {}, {}
    for p in all_perturbations(ids):
        before[p] = simulate(original, p).critical
        after[p] = simulate(patched, p).critical
    regressions = sorted((sorted(p) for p in before if after[p] and not before[p]), key=lambda x: (len(x), x))
    fixed = [sorted(t) for t in targets if not after[t]]
    still = [sorted(t) for t in targets if after[t]]
    baseline_ok = not after[frozenset()]
    status = "VERIFIED_ON_BOUNDED_SPACE" if (baseline_ok and not still and not regressions) else "REPAIR_REJECTED"
    return {
        "status": status,
        "space_size": len(before),
        "baseline_passes": baseline_ok,
        "targets_fixed": fixed,
        "targets_still_failing": still,
        "regressions": regressions[:20],
        "regression_count": len(regressions),
        "failing_perturbations_before": sum(before.values()),
        "failing_perturbations_after": sum(after.values()),
        "scope": "exhaustive over the modeled perturbation space only",
    }


def instantiate_plan(world: World, targets: list[frozenset[str]], steps: list[dict], max_combos: int = 20000) -> Instantiation:
    """Cheapest concrete plan matching a multi-step strategy. Every concrete choice for earlier steps
    is tried (bounded by max_combos); the last step must make every target pass and keep the base
    passing. Returns the plan's actions via .action (single step) or .plan (all steps)."""
    best: tuple[float, str, list[Action]] | None = None
    checks, considered = 0, 0

    def rec(w: World, i: int, acc: list[Action], cost: float):
        nonlocal best, checks, considered
        if considered >= max_combos:
            return
        cands = concrete_actions(w, steps[i])
        for a in cands:
            considered += 1
            c = cost + action_cost(w, a)
            if best is not None and c > best[0]:
                continue
            w2 = apply_action(w, a)
            if i < len(steps) - 1:
                rec(w2, i + 1, acc + [a], c)
                continue
            checks += 1
            if simulate(w2, frozenset()).critical:
                continue
            ok = True
            for t in targets:
                checks += 1
                if simulate(w2, t).critical:
                    ok = False
                    break
            if ok:
                key = (c, str([x.to_dict() for x in acc + [a]]))
                if best is None or key < best[:2]:
                    best = (c, key[1], acc + [a])

    rec(world, 0, [], 0.0)
    if best is None:
        inst = Instantiation(None, 0.0, checks, considered, "no concrete plan of this strategy fixes the target failure(s)")
        inst.plan = []
        return inst
    inst = Instantiation(best[2][-1], best[0], checks, considered, f"cheapest of {considered} concrete combinations that fixes the target")
    inst.plan = best[2]
    return inst
