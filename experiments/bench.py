"""Benchmark runner. Evaluator side: the ONLY place hidden ground truth is used.

  python experiments/bench.py --split dev --conditions random,random_pruned,agt_uniform
  python experiments/bench.py --split frozen --conditions random,random_pruned,agt_uniform,agt_openai --tag v3_frozen

Writes experiments/results/<tag>.json with config, git commit, worlds, per-episode found sets and
query logs, per-condition metrics, and OpenAI usage. agt_openai calls OpenAI directly (mode
recorded as direct_openai_api) once per world; responses are cached.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from confuzz.groundtruth import ground_truth  # noqa: E402
from confuzz.search import (AcquisitionPolicy, AdaptiveGroupTesting, GroupTestingAcquisition, RandomPolicy,  # noqa: E402
                            RandomPrunedPolicy, SemanticPrior, SizeOrderedPolicy, episode_metrics, recall_curve,
                            run_episode)
from confuzz.benchmark_worlds import frozen_worlds
from confuzz.world import DEV_START, FROZEN_START  # noqa: E402

BUDGETS = [5, 10, 15, 20, 25, 30]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def make_policy(cond: str, seed: int, prior: SemanticPrior | None):
    return {
        "random": lambda: RandomPolicy(seed),
        "random_pruned": lambda: RandomPrunedPolicy(seed),
        "size_ordered": lambda: SizeOrderedPolicy(seed),
        "v1": lambda: AcquisitionPolicy(seed, None, name="v1"),
        "v2": lambda: GroupTestingAcquisition(seed, None, name="v2"),
        "agt_uniform": lambda: AdaptiveGroupTesting(seed, None, name="agt_uniform"),
        "agt_openai": lambda: AdaptiveGroupTesting(seed, prior, name="agt_openai"),
    }[cond]()


def summarize(vals):
    a = np.array([v for v in vals if v is not None], float)
    if len(a) == 0:
        return None
    se = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
    return {"mean": round(float(a.mean()), 4), "median": round(float(np.median(a)), 4),
            "sd": round(float(a.std(ddof=1)) if len(a) > 1 else 0.0, 4),
            "ci95": [round(float(a.mean() - 1.96 * se), 4), round(float(a.mean() + 1.96 * se), 4)], "n": len(a)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "frozen"], required=True)
    ap.add_argument("--conditions", default="random,random_pruned,agt_uniform")
    ap.add_argument("--worlds", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    conds = args.conditions.split(",")
    worlds = frozen_worlds(args.worlds, DEV_START if args.split == "dev" else FROZEN_START)
    t0 = time.time()
    priors, llm_records = {}, []
    if "agt_openai" in conds:
        from confuzz.llm import red_interaction_model
        for w in worlds:
            rec = red_interaction_model(w)
            llm_records.append({k: rec[k] for k in ("model", "mode", "world_id", "usage", "latency_s", "from_cache")})
            priors[w.id] = SemanticPrior(rec["output"]["hypotheses"], rec["output"]["mechanism_edges"])

    episodes, per = [], {c: {"curves": [], "m": []} for c in conds}
    for w in worlds:
        gt = ground_truth(w)
        for c in conds:
            for s in range(args.seeds):
                ep = run_episode(w, make_policy(c, s, priors.get(w.id)), args.budget, s)
                curve = recall_curve(ep, gt.minimal_sets, BUDGETS)
                ms = {b: episode_metrics(ep, gt.minimal_sets, b) for b in BUDGETS}
                per[c]["curves"].append(curve)
                per[c]["m"].append(ms)
                episodes.append({
                    "condition": c, "world_id": w.id, "seed": s,
                    "found": [{"at_query": i, "set": sorted(m), "is_minimal_gt": m in gt.minimal_sets} for i, m in ep.found],
                    "queries": [{"i": o.index, "p": sorted(o.perturbation), "fail": o.critical, "purpose": o.purpose} for o in ep.log],
                })

    summary = {}
    for c in conds:
        curves = np.array(per[c]["curves"])
        summary[c] = {
            "recall_at": {str(b): summarize(curves[:, i]) for i, b in enumerate(BUDGETS)},
            "false_robust_rate_at": {str(b): round(float(np.mean([m[b]["false_robust"] for m in per[c]["m"]])), 4) for b in BUDGETS},
            "queries_to_first": summarize([m[BUDGETS[-1]]["queries_to_first"] for m in per[c]["m"]]),
            "runs_with_no_hit_within_budget": int(sum(m[BUDGETS[-1]]["queries_to_first"] is None for m in per[c]["m"])),
            "high_order_recall_at_max": summarize([m[BUDGETS[-1]]["high_order_recall"] for m in per[c]["m"]]),
            "search_queries_at_20": summarize([m[20]["search_queries"] for m in per[c]["m"]]),
            "localization_queries_at_20": summarize([m[20]["localization_queries"] for m in per[c]["m"]]),
        }
    # paired comparison vs random_pruned on recall@10 and @20 (same worlds and seeds)
    paired = {}
    if "random_pruned" in conds:
        from scipy.stats import wilcoxon
        base = np.array(per["random_pruned"]["curves"])
        for c in conds:
            if c == "random_pruned":
                continue
            other = np.array(per[c]["curves"])
            for b in (10, 20):
                i = BUDGETS.index(b)
                d = other[:, i] - base[:, i]
                try:
                    p = float(wilcoxon(d, zero_method="zsplit").pvalue) if np.any(d != 0) else 1.0
                except ValueError:
                    p = None
                paired[f"{c}_minus_random_pruned_recall@{b}"] = {"mean_diff": round(float(d.mean()), 4),
                                                               "wilcoxon_p": p, "n_pairs": len(d)}
    tag = args.tag or f"{args.split}_{'_'.join(conds)}"
    out = {
        "config": vars(args) | {"budgets": BUDGETS}, "git_commit": git_commit(), "created": time.time(),
        "split": args.split, "worlds": [{"id": w.id, "minimal_by_order": ground_truth(w).by_order()} for w in worlds],
        "summary": summary, "paired_tests": paired, "llm": llm_records,
        "notes": "paired Wilcoxon signed-rank (zsplit) on per-(world,seed) recall differences; descriptive, not corrected for multiple comparisons",
        "wall_clock_s": round(time.time() - t0, 1), "episodes": episodes,
    }
    path = ROOT / "experiments" / "results" / f"{tag}.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {path}")
    for c in conds:
        r = summary[c]["recall_at"]
        print(f"{c:14s} " + " ".join(f"R@{b}={r[str(b)]['mean']:.3f}" for b in BUDGETS)
              + f"  FRR@10={summary[c]['false_robust_rate_at']['10']:.3f}  medQ1={summary[c]['queries_to_first']['median'] if summary[c]['queries_to_first'] else None}")
    for k, v in paired.items():
        print(k, v)


if __name__ == "__main__":
    main()
