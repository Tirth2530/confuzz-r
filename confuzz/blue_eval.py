"""Blue ablation. EVALUATOR ONLY. Deterministic verification decides every outcome; no LLM grading.

Scenario: (world, target minimal failure m). Dev: 20 dev worlds. Frozen: the 20 frozen worlds.

Metrics for a repair plan a (all exact over the 2^8 modeled perturbations, nominal demand):
  C(a)  intervention cost: MOVE_SESSION 1, SWAP_ROOMS 2, SHIFT_SESSION 3, ACTIVATE_STANDBY standby_cost (5)
  E(a)  residual failure exposure = |{p : patched schedule fails under p}| / 256
  W(a)  worst-case residual loss = max over p of simulator severity on the patched schedule
  CE(a) counterexample exists = at least one p passes on the original and fails on the patched schedule
  V(a)  verified = base passes, m passes, and CE(a) is false (same rule as verify_exact)
Per method and scenario:
  proposals_to_verified = number of instantiated plans evaluated until the first verified one (cap K=5;
                          None if none verified). Outcome repair = the first verified plan.
Reference (not a competitor; uses exhaustive search): every verified plan of 1 or 2 actions;
  its Pareto frontier over (C, E). For a method's outcome a: frontier_gap = E(a) - min{E(f) : f on
  frontier, C(f) <= C(a)}; dominated = some reference plan is no worse on both and better on one.
Scalar tradeoff (pre-registered before any OpenAI Blue run): J_lam(a) = C(a) + lam * 256 * E(a),
  i.e. cost plus lam per residual failing perturbation. Primary lam = 0.05 (20 residual failing
  perturbations = 1 session move); sensitivity lam in {0.02, 0.10}. No verified repair within K -> J
  undefined (reported as a miss). Oracle J = min J over the reference verified plans.
  J_regret = J(outcome) - oracle J.
Head-to-head (OpenAI Blue vs cheapest-first), on scenarios where both verified: Blue dominates,
  cheapest dominates, equal, or incomparable on (C, E).
"""
from __future__ import annotations

from dataclasses import dataclass

from .benchmark_worlds import frozen_worlds
from .groundtruth import ground_truth, outcomes
from .repair import ACTION_TYPES, Action, action_cost, apply_action, apply_actions, concrete_actions
from .sim import simulate
from .space import all_perturbations
from .world import DEV_START, FROZEN_START, World

K_PROPOSALS = 5


@dataclass
class BlueScenario:
    id: str
    world: World
    target: frozenset[str]


def blue_scenarios(split: str, n_worlds: int = 20) -> list[BlueScenario]:
    ws = frozen_worlds(n_worlds, DEV_START if split == "dev" else FROZEN_START)
    return [BlueScenario(f"{w.id}:{'+'.join(sorted(m))}", w, m)
            for w in ws for m in sorted(ground_truth(w).minimal_sets, key=lambda s: sorted(s))]


_BEFORE: dict[str, dict] = {}


def evaluate_plan(world: World, plan: list[Action], target: frozenset[str]) -> dict:
    if world.id not in _BEFORE:
        _BEFORE[world.id] = outcomes(world)
    before = _BEFORE[world.id]
    p = apply_actions(world, plan)
    fails, worst, ce = 0, 0.0, False
    for q in all_perturbations(world.disruption_ids):
        r = simulate(p, q)
        if r.critical:
            fails += 1
            worst = max(worst, r.severity)
            if not before[q]:
                ce = True
    base_ok = not simulate(p, frozenset()).critical
    fixed = not simulate(p, target).critical
    cost = 0.0
    w = world
    for a in plan:
        cost += action_cost(w, a)
        w = apply_action(w, a)
    return {"plan": [a.describe(world) for a in plan], "cost": cost, "exposure": round(fails / 256, 4),
            "worst_loss": round(worst, 4), "counterexample_exists": ce, "target_fixed": fixed,
            "verified": base_ok and fixed and not ce}


def fixing_singles(world: World, target: frozenset[str]) -> list[Action]:
    out = []
    for t in ACTION_TYPES:
        if t in ("DO_NOTHING", "CANCEL_SESSION"):  # cancel added after pre-registration; excluded to keep scenario sets fixed
            continue
        for a in concrete_actions(world, {"type": t}):
            w2 = apply_action(world, a)
            if not simulate(w2, frozenset()).critical and not simulate(w2, target).critical:
                out.append(a)
    return sorted(out, key=lambda a: (action_cost(world, a), str(a.to_dict())))


def cheapest_first(s: BlueScenario) -> dict:
    """Deterministic baseline: propose fixing single actions cheapest-first; first verified wins."""
    tried = []
    for a in fixing_singles(s.world, s.target)[:K_PROPOSALS]:
        ev = evaluate_plan(s.world, [a], s.target)
        tried.append(ev)
        if ev["verified"]:
            return {"outcome": ev, "proposals_to_verified": len(tried), "tried": tried}
    return {"outcome": None, "proposals_to_verified": None, "tried": tried}


def reference_frontier(s: BlueScenario) -> dict:
    w, t = s.world, s.target
    plans: list[list[Action]] = [[a] for a in fixing_singles(w, t)]
    for a in [x for k in ("MOVE_SESSION", "SWAP_ROOMS", "SHIFT_SESSION", "ACTIVATE_STANDBY") for x in concrete_actions(w, {"type": k})]:
        wa = apply_action(w, a)
        for b in [x for k in ("MOVE_SESSION", "SWAP_ROOMS", "ACTIVATE_STANDBY") for x in concrete_actions(wa, {"type": k})]:
            wb = apply_action(wa, b)
            if not simulate(wb, frozenset()).critical and not simulate(wb, t).critical:
                plans.append([a, b])
    evs = [e for e in (evaluate_plan(w, p, t) for p in plans) if e["verified"]]
    front = [e for e in evs if not any((o["cost"] <= e["cost"] and o["exposure"] <= e["exposure"])
                                       and (o["cost"] < e["cost"] or o["exposure"] < e["exposure"]) for o in evs)]
    uniq = {}
    for e in sorted(front, key=lambda e: (e["cost"], e["exposure"])):
        uniq.setdefault((e["cost"], e["exposure"]), e)
    return {"n_verified_plans": len(evs), "frontier": list(uniq.values()), "all_verified": evs}


def frontier_position(outcome: dict | None, ref: dict) -> dict:
    if outcome is None:
        return {"frontier_gap": None, "dominated": None}
    feas = [f["exposure"] for f in ref["frontier"] if f["cost"] <= outcome["cost"]]
    gap = round(outcome["exposure"] - min(feas), 4) if feas else 0.0
    dom = any((o["cost"] <= outcome["cost"] and o["exposure"] <= outcome["exposure"])
              and (o["cost"] < outcome["cost"] or o["exposure"] < outcome["exposure"]) for o in ref["all_verified"])
    return {"frontier_gap": gap, "dominated": dom}


def head_to_head(a: dict | None, b: dict | None) -> str:
    if a is None or b is None:
        return "n/a"
    if (a["cost"], a["exposure"]) == (b["cost"], b["exposure"]):
        return "equal"
    if a["cost"] <= b["cost"] and a["exposure"] <= b["exposure"]:
        return "first_dominates"
    if b["cost"] <= a["cost"] and b["exposure"] <= a["exposure"]:
        return "second_dominates"
    return "incomparable"


LAMBDAS = (0.02, 0.05, 0.10)
PRIMARY_LAMBDA = 0.05


def J(e: dict | None, lam: float) -> float | None:
    return None if e is None else round(e["cost"] + lam * 256 * e["exposure"], 4)
