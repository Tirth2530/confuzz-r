"""TrueForge checkpoint helper. Reads/writes the same state file the MCP server uses.

  python scripts/checkpoint.py status              # phase, budgets, repairs, applied, state fingerprint
  python scripts/checkpoint.py prepare-verified    # SCRIPTED state with R1 rejected and R2 verified (approval test)
  python scripts/checkpoint.py reset               # fresh demo session

The fingerprint hashes everything except the event log, so `status` before and after a Deny must
print the same fingerprint, and after Allow the applied list must contain the repair exactly once.
State prepared by this script is marked prepared_by_script=true and logged as such in the trace;
it is for testing the approval path only and must not be presented as an agent run.
"""
import hashlib, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from confuzz.session import STATE_PATH, Session  # honours CONFUZZ_STATE


def fingerprint(st: dict) -> str:
    core = {k: v for k, v in st.items() if k != "events"}
    return hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()[:16]


def status():
    if not STATE_PATH.exists():
        print(f"no state at {STATE_PATH}"); return
    st = json.loads(STATE_PATH.read_text())
    print(f"state file   {STATE_PATH}")
    print(f"fingerprint  {fingerprint(st)}   (excludes event log)")
    print(f"scripted     {st.get('prepared_by_script', False)}")
    print(f"phase        {st['phase']}   discovery {st['used']}/{st['budget']}   critic {st['attack_used']}/{st['attack_budget']}")
    for f in st["failures"]:
        print(f"failure      {f['id']} {f['minimal_set']} order {f['order']}")
    for r in st["repairs"]:
        print(f"repair       {r['id']} {r['status']:24s} cost {r['cost']}  {'; '.join(r['described'])}")
    print(f"applied      {st['applied']}   (count of each: {[st['applied'].count(x) for x in st['applied']]})")
    last = [e for e in st["events"] if e["kind"] in ("approved_and_applied", "apply_refused")][-3:]
    for e in last:
        print(f"event        {e['actor']} {e['kind']} {e.get('repair_id')} {e.get('status', '')}")


def prepare_verified():
    s = Session.new("demo", budget=20, attack_budget=10)
    s.s["prepared_by_script"] = True
    s.event("SCRIPTED_PREP", "notice", text="events until 'prep_done' were produced by scripts/checkpoint.py, not by an agent")
    s.query(["D1", "D2", "D4"], "test", "SCRIPTED_PREP", "prep")
    s.minimize(["D1", "D2", "D4"], actor="SCRIPTED_PREP")
    s.propose_repair(["F1"], [{"type": "SWAP_ROOMS"}], "prep: cheapest swap", None)
    s.attack("R1", ["D4"], "prep", "transition")
    s.propose_repair(["F1"], [{"type": "ACTIVATE_STANDBY", "room": "F"}, {"type": "MOVE_SESSION", "session": "S07"}], "prep",
                     {"previous_repair_id": "R1", "counterexample": ["D4"], "diagnosis": "prep", "what_changed_in_my_reasoning": "prep"})
    s.attack("R2", ["D2", "D3", "D6"], "prep", "accessibility")
    v = s.verify("R2")
    s.event("SCRIPTED_PREP", "prep_done")
    s.save()
    print("prepared:", v["status"], "for R2; R1 is", s.repair("R1")["status"])
    status()


def reset():
    s = Session.new("demo", budget=20, attack_budget=10); s.save(); status()


if __name__ == "__main__":
    {"status": status, "prepare-verified": prepare_verified, "reset": reset}[sys.argv[1] if len(sys.argv) > 1 else "status"]()
