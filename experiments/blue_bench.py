"""Blue ablation: OpenAI Blue strategies vs deterministic cheapest-first repair. Exact verification decides.

  python experiments/blue_bench.py --split dev --conditions cheapest_first
  OPENAI_API_KEY=... python experiments/blue_bench.py --split frozen --conditions cheapest_first,openai_blue --tag blue_frozen  # ONCE

Metric definitions: confuzz/blue_eval.py docstring. OpenAI Blue: up to 3 rounds of up to 3 strategies;
each strategy is instantiated by the same exact planner the live server uses; plans are verified in
order; after a rejection Blue receives its rejected plans with one minimal counterexample each and the
violations it causes; at most K=5 instantiated plans per scenario.
"""
from __future__ import annotations

import argparse, json, subprocess, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from confuzz.blue_eval import (J, K_PROPOSALS, LAMBDAS, blue_scenarios, cheapest_first, evaluate_plan, frontier_position,  # noqa: E402
                               head_to_head, reference_frontier)
from confuzz.groundtruth import outcomes  # noqa: E402
from confuzz.repair import apply_actions, instantiate_plan  # noqa: E402
from confuzz.sim import simulate  # noqa: E402
from confuzz.space import all_perturbations  # noqa: E402


def failure_trace(s) -> dict:
    r = simulate(s.world, s.target).to_dict()
    return {"minimal_set": sorted(s.target), "disruptions": [s.world.disruption(d).label for d in sorted(s.target)],
            "violations": r["violations"], "playbook_moves": r["playbook_moves"], "margins": r["robustness_margins"]}


def one_counterexample(s, plan) -> dict | None:
    before = outcomes(s.world)
    p = apply_actions(s.world, plan)
    for q in all_perturbations(s.world.disruption_ids):  # smallest first
        if not before[q]:
            r = simulate(p, q)
            if r.critical:
                return {"perturbation": sorted(q), "violations": r.violations}
    return None


def openai_blue(s, llm_log: list) -> dict:
    from confuzz.llm import blue_strategies
    tried, history = [], []
    for rnd in range(3):
        rec = blue_strategies(s.world, failure_trace(s), history)
        llm_log.append({k: rec[k] for k in ("model", "mode", "usage", "latency_s", "from_cache")} | {"scenario": s.id, "round": rnd, "output": rec["output"]})
        for st in rec["output"]["strategies"][:3]:
            if len(tried) >= K_PROPOSALS:
                return {"outcome": None, "proposals_to_verified": None, "tried": tried}
            inst = instantiate_plan(s.world, [s.target], st["steps"])
            if not inst.plan:
                history.append({"strategy": st["steps"], "result": "solver: no concrete plan of this strategy fixes the failure"})
                continue
            ev = evaluate_plan(s.world, inst.plan, s.target)
            tried.append(ev | {"strategy": st["steps"], "mechanism_broken": st["mechanism_broken"]})
            if ev["verified"]:
                return {"outcome": ev, "proposals_to_verified": len(tried), "tried": tried}
            history.append({"strategy": st["steps"], "instantiated": ev["plan"], "result": "REJECTED",
                            "counterexample": one_counterexample(s, inst.plan)})
    return {"outcome": None, "proposals_to_verified": None, "tried": tried}


def agg(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return {"mean": round(float(np.mean(v)), 4), "median": round(float(np.median(v)), 4), "n": len(v)} if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "frozen"], required=True)
    ap.add_argument("--conditions", default="cheapest_first")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    conds = a.conditions.split(",")
    t0 = time.time()
    sc = blue_scenarios(a.split)
    llm_log, rows = [], []
    for s in sc:
        ref = reference_frontier(s)
        row = {"scenario": s.id, "order": len(s.target), "reference_frontier": ref["frontier"], "n_verified_plans": ref["n_verified_plans"]}
        res = {}
        for c in conds:
            res[c] = cheapest_first(s) if c == "cheapest_first" else openai_blue(s, llm_log)
            o = res[c]["outcome"]
            row[c] = {"outcome": o, "proposals_to_verified": res[c]["proposals_to_verified"],
                      "J": {str(l): J(o, l) for l in LAMBDAS},
                      "J_regret": {str(l): (round(J(o, l) - min(J(e, l) for e in ref["all_verified"]), 4) if o and ref["all_verified"] else None) for l in LAMBDAS},
                      "first_proposal_counterexample": res[c]["tried"][0]["counterexample_exists"] if res[c]["tried"] else None,
                      "tried": res[c]["tried"]} | frontier_position(o, ref)
        if len(conds) == 2:
            row["head_to_head"] = head_to_head(res[conds[1]]["outcome"], res[conds[0]]["outcome"])
        rows.append(row)
    summary = {}
    for c in conds:
        rs = [r[c] for r in rows]
        flat = [{"cost": r["outcome"]["cost"], "exposure": r["outcome"]["exposure"], "worst_loss": r["outcome"]["worst_loss"],
                 "frontier_gap": r["frontier_gap"], "proposals": r["proposals_to_verified"]} for r in rs if r["outcome"]]
        summary[c] = {"scenarios": len(rs), "verified_within_K": len(flat),
                      "verified_rate": round(len(flat) / len(rs), 4) if rs else None,
                      "first_proposal_counterexample_rate": round(float(np.mean([r["first_proposal_counterexample"] for r in rs if r["first_proposal_counterexample"] is not None])), 4),
                      "cost": agg(flat, "cost"), "exposure": agg(flat, "exposure"), "worst_loss": agg(flat, "worst_loss"),
                      "frontier_gap": agg(flat, "frontier_gap"), "proposals_to_verified": agg(flat, "proposals"),
                      "J_regret_mean": {str(l): agg([{"x": r["J_regret"][str(l)]} for r in rs], "x") for l in LAMBDAS},
                      "J_regret_zero_rate": {str(l): round(float(np.mean([r["J_regret"][str(l)] == 0 for r in rs if r["J_regret"][str(l)] is not None])), 4) if flat else None for l in LAMBDAS},
                      "on_frontier_rate": round(float(np.mean([not r["dominated"] for r in rs if r["dominated"] is not None])), 4) if flat else None}
    h2h = {}
    if len(conds) == 2:
        for r in rows:
            h2h[r["head_to_head"]] = h2h.get(r["head_to_head"], 0) + 1
    no_verified_exists = sum(1 for r in rows if r["n_verified_plans"] == 0)
    try:
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    tag = a.tag or f"blue_{a.split}"
    out = {"config": vars(a) | {"K": K_PROPOSALS}, "git_commit": commit, "summary": summary,
           "head_to_head (first=" + (conds[1] if len(conds) == 2 else "-") + ")": h2h,
           "scenarios_with_no_verified_plan_of_1_or_2_actions": no_verified_exists,
           "llm": llm_log, "rows": rows, "wall_clock_s": round(time.time() - t0, 1)}
    (ROOT / "experiments" / "results" / f"{tag}.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"{len(sc)} scenarios; {no_verified_exists} have no verified plan of <=2 actions (abstention is correct there)")
    for c, s in summary.items():
        print(c, json.dumps(s))
    if h2h:
        print("head_to_head", h2h)


if __name__ == "__main__":
    main()
