"""Critic experiment: queries to first counterexample against an AI-generated repair.

  python experiments/critic_bench.py --split dev --conditions random,size_ordered,diff_guided
  OPENAI_API_KEY=... python experiments/critic_bench.py --split frozen \
      --conditions random,size_ordered,diff_guided,openai_critic --tag critic_frozen   # run ONCE

Each condition attacks the same frozen scenarios with the same budget. Every attack is one
simulator query on the patched schedule (the original outcome is public: the base schedule's
behaviour under that perturbation is what the organizer already had). A counterexample is a
perturbation that passes on the original and fails on the patched schedule.

Conditions (fixed before any run):
  random         uniform over non-empty perturbations, without replacement
  size_ordered   singles, then pairs, then triples..., random within a size
  diff_guided    deterministic structural heuristic, no LLM: a disruption is "relevant" if it touches
                 what the repair changed (a moved session's demand or speaker, its old/new room, the
                 new zone's route, an activated standby room). Order: fewest irrelevant elements,
                 then smallest size, then random. This is the strong no-LLM baseline.
  openai_critic  OpenAI reads spec + repair diff + target failure, returns up to 10 ranked attacks
                 with hypotheses; tested in order; then falls back to random for the rest of the budget.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from confuzz.scenarios import repair_diff, scenarios  # noqa: E402
from confuzz.space import all_perturbations  # noqa: E402

KS = [1, 2, 3, 5, 10, 20]


def relevant(s, did: str) -> bool:
    d = s.world.disruption(did)
    diff = repair_diff(s)
    sess = {m["session"] for m in diff["moved_sessions"]}
    spk = {m["speaker"] for m in diff["moved_sessions"]}
    rooms = {m["from"]["room"] for m in diff["moved_sessions"]} | {m["to"]["room"] for m in diff["moved_sessions"]} | set(diff["activated_standby_rooms"])
    zones = {m["to"]["zone"] for m in diff["moved_sessions"]}
    return ((d.kind == "SURGE" and d.target in sess) or (d.kind == "SPEAKER_DELAY" and d.target in spk)
            or (d.kind in ("ROOM_LOSS", "AV_FAIL") and d.target in rooms) or (d.kind == "ROUTE_CLOSED" and d.target in zones))


def order(cond: str, s, seed: int, llm_list: list[list[str]] | None = None) -> list[frozenset[str]]:
    rng = random.Random(seed)
    allp = [p for p in all_perturbations(s.world.disruption_ids) if p]
    rng.shuffle(allp)
    if cond == "random":
        return allp
    if cond == "size_ordered":
        return sorted(allp, key=len)  # stable: random within size
    if cond == "diff_guided":
        rel = {d: relevant(s, d) for d in s.world.disruption_ids}
        return sorted(allp, key=lambda p: (sum(not rel[d] for d in p), len(p)))
    if cond == "openai_critic":
        head, seen = [], set()
        for c in llm_list or []:
            p = frozenset(c)
            if p and p <= set(s.world.disruption_ids) and p not in seen:
                head.append(p); seen.add(p)
        return head + [p for p in allp if p not in seen]
    raise ValueError(cond)


def queries_to_ce(seq, regressions, budget) -> int | None:
    for i, p in enumerate(seq[:budget], 1):
        if p in regressions:
            return i
    return None


def summarize(vals, budget):
    a = np.array([v if v is not None else budget + 1 for v in vals], float)
    return {"median": float(np.median(a)), "mean_capped": round(float(a.mean()), 3),
            "found_within_budget": int(sum(v is not None for v in vals)), "n": len(vals)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "frozen"], required=True)
    ap.add_argument("--conditions", default="random,size_ordered,diff_guided")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    conds = a.conditions.split(",")
    t0 = time.time()
    sc = scenarios(a.split)
    llm = {}
    llm_records = []
    if "openai_critic" in conds:
        from confuzz.llm import critic_attacks
        for s in sc:
            rec = critic_attacks(s.world, repair_diff(s), sorted(s.target))
            llm[s.id] = [x["perturbation"] for x in rec["output"]["attacks"]]
            llm_records.append({k: rec[k] for k in ("model", "mode", "usage", "latency_s", "from_cache")} | {"scenario": s.id, "attacks": rec["output"]["attacks"]})
    res, rows = {c: [] for c in conds}, []
    for s in sc:
        for c in conds:
            seeds = range(a.seeds)
            for seed in seeds:
                q = queries_to_ce(order(c, s, seed, llm.get(s.id)), s.regressions, a.budget)
                res[c].append(q)
                rows.append({"scenario": s.id, "condition": c, "seed": seed, "queries_to_counterexample": q})
    summary = {c: summarize(v, a.budget) | {f"success@{k}": round(float(np.mean([x is not None and x <= k for x in v])), 4) for k in KS}
               for c, v in res.items()}
    meta = [{"id": s.id, "repair": repair_diff(s)["described"], "target": sorted(s.target),
             "n_regressions": len(s.regressions), "min_regression_order": min(len(r) for r in s.regressions),
             "regression_fraction_of_space": round(len(s.regressions) / 255, 4)} for s in sc]
    try:
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    tag = a.tag or f"critic_{a.split}"
    out = {"config": vars(a), "git_commit": commit, "scenarios": meta, "summary": summary, "llm": llm_records,
           "rows": rows, "wall_clock_s": round(time.time() - t0, 1),
           "note": "openai_critic is deterministic given its cached list; seeds only vary its random fallback"}
    (ROOT / "experiments" / "results" / f"{tag}.json").write_text(json.dumps(out, indent=1))
    print(f"{len(sc)} scenarios; regressions per scenario median {np.median([m['n_regressions'] for m in meta])}, "
          f"min-order counts {dict(zip(*np.unique([m['min_regression_order'] for m in meta], return_counts=True)))}")
    for c in conds:
        s = summary[c]
        print(f"{c:14s} median q-to-CE {s['median']:.1f}  " + " ".join(f"S@{k}={s[f'success@{k}']:.2f}" for k in KS))


if __name__ == "__main__":
    main()
