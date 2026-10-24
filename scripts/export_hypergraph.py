"""Export the failure hypergraph of the demo world from real engine state (no hand-written edges).

Stages:
  hidden    minimal failing sets of the original schedule (exhaustive; evaluator view)
  discovered  minimal sets found by a session (from runs/state.json or a given trace) -- optional
  R1/R2     minimal failing sets after each repair from a session, each tagged
            kept / fixed / subsumed / regression / residual (see code); counterexamples the critic found.
Usage: python scripts/export_hypergraph.py [--state runs/state.json] [--out experiments/results/hypergraph_demo.json]
Without --state it replays the scripted episode's repairs (labeled scripted).
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from confuzz.groundtruth import ground_truth
from confuzz.repair import Action, apply_actions
from confuzz.sim import simulate
from confuzz.world import demo_world

def edges(world, ref=None):
    gt = ground_truth(world)
    out = []
    for m in sorted(gt.minimal_sets, key=lambda s: (len(s), sorted(s))):
        r = simulate(world, m)
        tag = None if ref is None else ("kept" if m in ref else "NEW")
        out.append({"set": sorted(m), "order": len(m), "severity": round(r.severity, 3),
                    "constraints": sorted({v["constraint"] for v in r.violations}),
                    "detail": [v.get("detail", "") for v in r.violations][:3], "status": tag})
    return out, gt

ap = argparse.ArgumentParser()
ap.add_argument("--state", default=None)
ap.add_argument("--out", default=str(ROOT / "experiments" / "results" / "hypergraph_demo.json"))
a = ap.parse_args()
w = demo_world()
hidden, gt0 = edges(w)
stages = [{"id": "hidden", "title": "Hidden failure structure (exhaustive, evaluator only)", "edges": hidden}]
source = "scripted"
if a.state:
    st = json.loads(Path(a.state).read_text()); source = "session " + a.state
    disc = [f["minimal_set"] for f in st["failures"]]
    repairs = [(r["id"], [Action(**x) for x in r["actions"]], r["described"], r["status"],
                [x["perturbation"] for x in r["attacks"] if "COUNTEREXAMPLE" in str(x.get("verdict", ""))]) for r in st["repairs"]]
else:
    disc = [["D3", "D5", "D6"], ["D1", "D2", "D4"]]
    repairs = [("R1", [Action("SWAP_ROOMS", session="S10", session2="S12")], ["swap rooms of S10 and S12"], "REPAIR_REJECTED", [["D4"]]),
               ("R2", [Action("ACTIVATE_STANDBY", room="F"), Action("MOVE_SESSION", session="S07", room="F")],
                ["activate standby room F", "move S07 to room F"], "VERIFIED_ON_TEST_SET", [])]
stages.append({"id": "discovered", "title": "Discovered by search (paid queries only)",
               "edges": [e | {"status": "discovered" if e["set"] in disc else "undiscovered"} for e in hidden]})
for rid, acts, desc, status, ces in repairs:
    p = apply_actions(w, acts)
    es, gtp = edges(p, ref=gt0.minimal_sets)
    # original minimal sets that are no longer minimal: FIXED (now passes) or SUBSUMED (still fails,
    # because a smaller set now fails on the patched schedule)
    removed = [e | {"status": "subsumed" if simulate(p, e["set"]).critical else "fixed"}
               for e in hidden if frozenset(e["set"]) not in gtp.minimal_sets]
    # patched minimal sets: KEPT (was minimal before), REGRESSION (passed on the original),
    # RESIDUAL (failed before too, newly minimal because a sub-failure was fixed)
    for e in es:
        if e["status"] == "NEW":
            e["status"] = "residual" if simulate(w, e["set"]).critical else "regression"
    stages.append({"id": rid, "title": f"After {rid}: {'; '.join(desc)}", "repair_status": status,
                   "counterexamples_found_by_critic": ces, "edges": es + removed})
nodes = [{"id": d.id, "kind": d.kind, "label": d.label} for d in w.disruptions]
Path(a.out).write_text(json.dumps({"source": source, "world": w.id, "nodes": nodes, "stages": stages}, indent=1))
for s in stages:
    print(s["id"], [(("+".join(e["set"])), e["status"]) for e in s["edges"]])
