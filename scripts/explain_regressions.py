"""Causal account of each regression a repair introduces, from deterministic simulator state only.

For every singleton (or minimal) regression r of repair a on the demo world:
  1. original outcome under r (pass, with margins)
  2. what a changed (sessions moved: room/zone/slot)
  3. the mechanism: playbook moves and assignments that differ between original and patched under r
  4. the violated invariant and its margin on the patched schedule
  5. introduced vs exposed: INTRODUCED if the violated entity's state under r differs only because of a
     (the original schedule under the same r satisfies that invariant); EXPOSED if the original
     schedule already had a negative or zero-slack margin on that invariant that r merely tips.
Usage: python scripts/explain_regressions.py   (R1 = swap S10/S12 on the demo world)
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from confuzz.repair import Action, apply_actions
from confuzz.sim import entity_margin, simulate
from confuzz.space import all_perturbations
from confuzz.world import demo_world

w = demo_world()
plan = [Action("SWAP_ROOMS", session="S10", session2="S12")]
p = apply_actions(w, plan)
changed = [(a.id, a.room, w.room(a.room).zone, b.room, p.room(b.room).zone) for a, b in zip(w.sessions, p.sessions) if a.room != b.room]
regs = [q for q in all_perturbations(w.disruption_ids) if q and simulate(p, q).critical and not simulate(w, q).critical]
minimal = sorted([r for r in regs if not any(o < r for o in regs)], key=lambda r: (len(r), sorted(r)))
out = {"repair": "swap rooms of S10 and S12", "changed": [{"session": s, "from": f"{r1} ({z1})", "to": f"{r2} ({z2})"} for s, r1, z1, r2, z2 in changed],
       "minimal_regressions": []}
for r in minimal:
    o, n = simulate(w, r), simulate(p, r)
    viol = n.violations
    rec = {"perturbation": sorted(r), "disruptions": [w.disruption(d).label for d in sorted(r)],
           "original": {"critical": o.critical, "margins": {c: round(v, 4) for c, v in o.margins.items() if c in {x["constraint"] for x in viol}}},
           "patched_violations": viol,
           "assignment_diff": {s: {"original": o.assignment[s], "patched": n.assignment[s]} for s in o.assignment if o.assignment[s] != n.assignment[s]}}
    # introduced vs exposed, per violated invariant, by comparing the same invariant on the original under r
    verdicts = []
    for v in viol:
        c = v["constraint"]
        key = "speaker" if "speaker" in v and c in ("transition", "speaker_overlap") else "session"
        orig_m = entity_margin(o, c, key, v.get(key))
        orig_m = 1.0 if orig_m is None else orig_m  # invariant not checked for that entity before = satisfied
        verdicts.append({"constraint": c, "entity": v.get(key), "patched_margin": v["normalized_margin"], "original_margin_same_r": round(orig_m, 4),
                         "verdict": "INTRODUCED by the repair" if orig_m > 0 else "EXPOSED (original already at or below zero slack)"})
    rec["verdicts"] = verdicts
    out["minimal_regressions"].append(rec)
Path(ROOT / "experiments/results/r1_regression_explanations.json").write_text(json.dumps(out, indent=1))
print("R1 changed:", out["changed"])
for rec in out["minimal_regressions"]:
    print("\n", rec["perturbation"], rec["disruptions"])
    print("   original critical:", rec["original"]["critical"], "| margins on the same invariant:", rec["original"]["margins"])
    for v in rec["patched_violations"]:
        print("   patched violation:", v["constraint"], v.get("session") or v.get("speaker"), "|", v.get("detail"), "| margin", v["normalized_margin"])
    print("   assignment diff:", rec["assignment_diff"])
    for v in rec["verdicts"]:
        print("   ->", v["constraint"], v["entity"], v["verdict"], f"(same entity: original margin {v['original_margin_same_r']} -> patched {v['patched_margin']})")
