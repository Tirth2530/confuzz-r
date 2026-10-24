"""Persisted experiment session: budgets, phase state machine, evidence log.

Every rule that matters is enforced HERE, not in prompts:
  * discovery query budget (run_experiment, minimize_failure, explain_interaction)
  * separate critic attack budget (attack_repair)
  * phase order: a repair needs a reproduced minimal failure; verification needs at least
    one critic attack; apply needs VERIFIED_ON_TEST_SET
  * after a rejection, the next repair must state previous repair, counterexample,
    diagnosis and what changed in its reasoning
This module never imports confuzz.groundtruth (hidden answers).
"""
from __future__ import annotations

import json
from dataclasses import replace
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from .certificate import raw_evidence, risk_certificate
from .repair import Action, action_types, apply_actions, instantiate_plan
from .search import AdaptiveGroupTesting, SemanticPrior, candidates as _candidates, Oracle
from .domain import simulate
from .world import World, demo_world, generate_world

STATE_PATH = Path(os.environ.get("CONFUZZ_STATE", Path(__file__).resolve().parent.parent / "runs" / "state.json"))


class RuleViolation(Exception):
    """Raised when a call breaks a server-enforced rule. Message is returned to the agent."""


def load_world(world_id: str) -> World:
    if world_id == "demo":
        return demo_world()
    if world_id == "inventory":
        from .inventory import demo_inventory_world
        return demo_inventory_world()
    if world_id.startswith("gen-"):
        w = generate_world(int(world_id[4:]))
        if w is None:
            raise RuleViolation(f"{world_id} is not an accepted world")
        return w
    raise RuleViolation(f"unknown world {world_id}")


def _fs(x) -> frozenset[str]:
    return frozenset(str(i) for i in x)


class Session:
    def __init__(self, state: dict):
        self.s = state

    # ---- persistence ------------------------------------------------------
    @classmethod
    def new(cls, world_id: str = "demo", budget: int = 20, attack_budget: int = 10, seed: int = 0) -> "Session":
        load_world(world_id)
        return cls({
            "world_id": world_id, "seed": seed, "budget": budget, "used": 0,
            "attack_budget": attack_budget, "attack_used": 0, "solver_checks": 0,
            "phase": "DISCOVER", "queries": [], "failures": [], "prior": None,
            "repairs": [], "applied": [], "events": [], "created": time.time(),
            "incidents": [], "notifications": [],
        })

    @classmethod
    def load(cls) -> "Session":
        if not STATE_PATH.exists():
            s = cls.new()
            s.save()
            return s
        return cls(json.loads(STATE_PATH.read_text()))

    def save(self):
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=STATE_PATH.parent)
        with os.fdopen(fd, "w") as f:
            json.dump(self.s, f, indent=1)
        os.replace(tmp, STATE_PATH)

    def event(self, actor: str, kind: str, **data):
        self.s["events"].append({"t": round(time.time(), 2), "actor": actor, "kind": kind, **data})

    def set_phase(self, phase: str):
        if self.s["phase"] != phase:
            self.event("ENGINE", "phase", from_=self.s["phase"], to=phase)
            self.s["phase"] = phase

    # ---- worlds -------------------------------------------------------------
    def base_world(self, include_open_incidents: bool = True) -> World:
        """Original world plus approved (applied) repairs plus reported incidents.
        With include_open_incidents=False: the schedule as it was before the still-open incidents
        (the reference a fix for an incident is checked against for regressions)."""
        w = load_world(self.s["world_id"])
        for rid in self.s["applied"]:
            w = apply_actions(w, [Action(**a) for a in self.repair(rid)["actions"]])
        inc = [i for i in self.s.get("incidents", []) if include_open_incidents or i["status"] != "open"]
        if inc and hasattr(w, "incidents"):
            w = replace(w, incidents=tuple((i["kind"], i["room"], i["label"]) for i in inc))
        return w

    def _targets_incident(self, r: dict) -> bool:
        return any(self.failure(fid).get("incident_id") for fid in r["failure_ids"])

    def incident_rooms_for(self, r: dict) -> set[str]:
        rooms = set()
        for fid in r["failure_ids"]:
            iid = self.failure(fid).get("incident_id")
            rooms |= {i["room"] for i in self.s.get("incidents", []) if i["id"] == iid}
        return rooms

    def reference_world(self, r: dict) -> World:
        """What 'before' means for a repair: for an incident fix, the schedule before the incident
        (so a regression = something that used to survive and no longer does); otherwise the current state."""
        return self.base_world(include_open_incidents=False) if self._targets_incident(r) else self.base_world()

    def report_incident(self, kind: str, room: str, note: str = "") -> dict:
        from .world import INCIDENT_KINDS
        w = self.base_world()
        if not hasattr(w, "incidents"):
            raise RuleViolation("incident reporting is implemented for the conference domain")
        if kind not in INCIDENT_KINDS:
            raise RuleViolation(f"issue must be one of {INCIDENT_KINDS}")
        try:
            rr = w.room(room)
        except KeyError:
            raise RuleViolation(f"unknown room {room}; rooms: {[r.id for r in w.rooms]}")
        label = {"ROOM_CLOSED": f"{rr.name} closed", "PROJECTOR_BROKEN": f"{rr.name} projector broken",
                 "ACCESS_LOST": f"{rr.name} lost step-free access"}[kind] + (f" ({note})" if note else "")
        iid = f"I{len(self.s['incidents']) + 1}"
        inc = {"id": iid, "kind": kind, "room": room, "label": label, "status": "open", "failure_id": None,
               "reported_at": round(time.time(), 1)}
        self.s["incidents"].append(inc)
        r = simulate(self.base_world(), frozenset())
        moves = r.to_dict()["playbook_moves"]
        if not r.critical:
            inc["status"] = "absorbed"
            self.event("SUPERVISOR", "incident_reported", incident_id=iid, status="absorbed")
            return {"incident_id": iid, "label": label, "status": "ABSORBED",
                    "message": "the operations playbook handles this incident with every constraint satisfied",
                    "playbook_moves": moves,
                    "next": "optionally check what-if robustness with run_experiment; tell affected people about the moves"}
        fid = f"F{len(self.s['failures']) + 1}"
        self.s["failures"].append({"id": fid, "minimal_set": [], "order": 0, "incident_id": iid, "label": label,
                                   "found_at_query": self.s["used"], "minimization_trace": [],
                                   "violations": r.violations, "margins": r.to_dict()["robustness_margins"],
                                   "playbook_moves": moves, "seed": self.s["seed"]})
        inc["failure_id"] = fid
        self.event("SUPERVISOR", "incident_reported", incident_id=iid, failure_id=fid, status="breaks schedule")
        self.set_phase("REPAIR")
        return {"incident_id": iid, "label": label, "status": "BREAKS_SCHEDULE", "failure_id": fid,
                "violations": r.violations, "playbook_moves": moves,
                "next": f"BLUE proposes a repair for {fid}; CRITIC attacks it with what-if disruptions; "
                        "a counterexample is a what-if the schedule survived before the incident but not after the fix. "
                        "If nothing holds, consider CANCEL_SESSION."}

    def notifications(self, before: World, after: World) -> list[dict]:
        """Who must be told: sessions whose room changed (by repair or playbook) or that were canceled."""
        a0 = simulate(before, frozenset()).assignment
        a1 = simulate(after, frozenset()).assignment
        out = []
        for s in before.sessions:
            if s.id not in a1:
                out.append({"session": s.id, "title": s.title, "speaker": s.speaker, "change": "CANCELED",
                            "from": a0.get(s.id), "to": None, "slot": s.slot})
            elif a0.get(s.id) != a1[s.id] or (s.slot != after.session(s.id).slot):
                out.append({"session": s.id, "title": s.title, "speaker": s.speaker, "change": "MOVED",
                            "from": a0.get(s.id), "to": a1[s.id], "slot": after.session(s.id).slot})
        return out

    def repair(self, rid: str) -> dict:
        for r in self.s["repairs"]:
            if r["id"] == rid:
                return r
        raise RuleViolation(f"unknown repair {rid}")

    def failure(self, fid: str) -> dict:
        for f in self.s["failures"]:
            if f["id"] == fid:
                return f
        raise RuleViolation(f"unknown failure {fid}")

    def patched_world(self, rid: str) -> World:
        return apply_actions(self.base_world(), [Action(**a) for a in self.repair(rid)["actions"]])

    # ---- discovery ----------------------------------------------------------
    def _oracle(self) -> Oracle:
        """Rebuild an oracle whose log mirrors this session's paid queries."""
        o = Oracle(self.base_world(), self.s["budget"])
        for q in self.s["queries"]:
            o.query(q["perturbation"], q["purpose"])
        return o

    def _validate_ids(self, ids) -> frozenset[str]:
        valid = set(self.base_world().disruption_ids)
        p = _fs(ids)
        bad = p - valid
        if bad:
            raise RuleViolation(f"unknown disruption ids {sorted(bad)}; valid: {sorted(valid)}")
        return p

    def query(self, ids, purpose: str, actor: str, rationale: str = "") -> dict:
        p = self._validate_ids(ids)
        for q in self.s["queries"]:
            if _fs(q["perturbation"]) == p:
                return q | {"cached": True}
        if self.s["used"] >= self.s["budget"]:
            raise RuleViolation(f"discovery budget exhausted ({self.s['used']}/{self.s['budget']} queries)")
        self.s["used"] += 1
        r = simulate(self.base_world(), p)
        q = {"index": self.s["used"], "perturbation": sorted(p), "purpose": purpose, "actor": actor,
             "rationale": rationale, **r.to_dict()}
        self.s["queries"].append(q)
        self.event(actor, "simulate", index=q["index"], perturbation=q["perturbation"], purpose=purpose,
                   critical=r.critical, violations=[v["constraint"] for v in r.violations])
        return q

    def submit_prior(self, hypotheses: list[dict], edges: list[dict], notes: str) -> dict:
        ids = set(self.base_world().disruption_ids)
        clean = []
        for h in hypotheses:
            cs = [str(x) for x in h.get("candidate_set", [])]
            if not cs or not set(cs) <= ids:
                raise RuleViolation(f"hypothesis candidate_set must be non-empty disruption ids from {sorted(ids)}")
            rp = float(h.get("risk_prior", -1))
            if not 0.0 <= rp <= 1.0:
                raise RuleViolation("risk_prior must be in [0, 1] (an LLM prior, not a probability)")
            clean.append({"candidate_set": sorted(set(cs)), "risk_prior": rp,
                          "interaction_hypothesis": str(h.get("interaction_hypothesis", "")),
                          "predicted_constraints": [str(c) for c in h.get("predicted_constraints", [])]})
        for e in edges:
            if not {"src", "dst"} <= set(e):
                raise RuleViolation("each mechanism edge needs src and dst")
        self.s["prior"] = {"hypotheses": clean, "edges": [{"src": str(e["src"]), "dst": str(e["dst"]),
                                                          "relation": str(e.get("relation", ""))} for e in edges],
                           "notes": notes, "submitted_after_query": self.s["used"]}
        self.event("RED", "interaction_model", n_hypotheses=len(clean), n_edges=len(edges),
                   top=[h["candidate_set"] for h in sorted(clean, key=lambda h: -h["risk_prior"])[:3]])
        return {"accepted": True, "hypotheses": len(clean), "edges": len(edges),
                "label": "LLM semantic prior (uncalibrated score, not a probability)"}

    def rank(self, top_k: int = 5) -> dict:
        prior = SemanticPrior(self.s["prior"]["hypotheses"], self.s["prior"]["edges"]) if self.s["prior"] else None
        pol = AdaptiveGroupTesting(self.s["seed"], prior)
        o = self._oracle()
        found = [_fs(f["minimal_set"]) for f in self.s["failures"]]
        cands = _candidates(o.world, o, found, prune_passed=True)
        if not cands:
            return {"ranked": [], "note": "no untested candidates remain"}
        ranked = pol.rank(cands, o, top=top_k, found=found)
        self.event("SCORER", "rank", top=ranked[0]["perturbation"], score=ranked[0]["score"],
                   covers=ranked[0]["unresolved_interactions_covered"], uses_llm_prior=prior is not None)
        return {"uses_llm_prior": prior is not None, "candidates": len(cands), "ranked": ranked,
                "formula": "A(S) = V(S) / (1 + P(fail|S) * (|S|-1)); V = weighted unresolved pairs/triples inside S",
                "assumption": "monotone disruption effects: a passing set clears all its subsets"}

    def minimize(self, ids, actor: str = "RED") -> dict:
        p = self._validate_ids(ids)
        first = self.query(p, "confirm", actor)
        if not first["critical_failure"]:
            raise RuleViolation(f"{sorted(p)} does not fail; nothing to minimize")
        self.set_phase("REPRODUCE")
        current, trace = set(p), []
        for d in sorted(p):
            if len(current) == 1:
                break
            trial = sorted(current - {d})
            q = self.query(trial, "minimize", actor)
            trace.append({"probe": trial, "critical": q["critical_failure"],
                          "violations": [v["constraint"] for v in q["violations"]]})
            if q["critical_failure"]:
                current.remove(d)
        m = sorted(current)
        for f in self.s["failures"]:
            if f["minimal_set"] == m:
                return {"failure_id": f["id"], "minimal_set": m, "already_known": True, "trace": trace}
        rep = simulate(self.base_world(), m)
        f = {"id": f"F{len(self.s['failures']) + 1}", "minimal_set": m, "order": len(m),
             "found_at_query": self.s["used"], "minimization_trace": trace,
             "violations": rep.violations, "margins": rep.to_dict()["robustness_margins"],
             "playbook_moves": rep.to_dict()["playbook_moves"], "seed": self.s["seed"]}
        self.s["failures"].append(f)
        self.event("ENGINE", "minimal_failure", failure_id=f["id"], minimal_set=m, order=len(m),
                   violations=[v["constraint"] for v in rep.violations])
        self.set_phase("REPAIR")
        return {"failure_id": f["id"], "minimal_set": m, "interaction_order": len(m), "trace": trace,
                "violations": rep.violations, "playbook_moves": f["playbook_moves"],
                "note": "1-minimal: removing any single disruption makes it pass (verified by the probes above)"}

    def explain(self, fid: str) -> dict:
        """Proper-subset table for a minimal failure. Unqueried subsets are paid from the budget."""
        from itertools import combinations
        f = self.failure(fid)
        rows = []
        for k in range(1, len(f["minimal_set"])):
            for c in combinations(f["minimal_set"], k):
                q = self.query(list(c), "explain", "ENGINE")
                rows.append({"subset": list(c), "result": "FAIL" if q["critical_failure"] else "PASS"})
        rows.append({"subset": f["minimal_set"], "result": "CRITICAL FAILURE"})
        return {"failure_id": fid, "table": rows, "budget": f"{self.s['used']}/{self.s['budget']}"}

    # ---- repair ------------------------------------------------------------
    def propose_repair(self, failure_ids: list[str], steps: list[dict], rationale: str,
                       revision: dict | None) -> dict:
        if not self.s["failures"]:
            raise RuleViolation("no reproduced minimal failure yet; discover and minimize first")
        targets = [_fs(self.failure(fid)["minimal_set"]) for fid in failure_ids]
        rejected = [r for r in self.s["repairs"] if r["status"] == "REPAIR_REJECTED"]
        if rejected:
            need = ("previous_repair_id", "counterexample", "diagnosis", "what_changed_in_my_reasoning")
            if not revision or any(not revision.get(k) for k in need):
                raise RuleViolation(f"a previous repair was rejected; include revision fields {list(need)}")
            self.repair(revision["previous_repair_id"])
        if not steps:
            raise RuleViolation("give at least one strategy step")
        base = self.base_world()
        for st in steps:
            if st.get("type") not in action_types(base):
                raise RuleViolation(f"step type must be one of {action_types(base)}")
        inst = instantiate_plan(base, targets, steps)
        checks = inst.checks
        if not inst.plan:
            self.s["solver_checks"] += checks
            self.event("SOLVER", "no_instantiation", steps=steps, reason=inst.reason)
            raise RuleViolation(f"strategy {steps}: {inst.reason}")
        actions, total, notes = inst.plan, inst.cost, [inst.reason]
        self.s["solver_checks"] += checks
        rid = f"R{len(self.s['repairs']) + 1}"
        patched = apply_actions(base, actions)
        sc = []
        for fid, t in zip(failure_ids, targets):
            b, a = simulate(base, t), simulate(patched, t)
            sc.append({"failure_id": fid, "perturbation": sorted(t),
                       "before": {"critical": b.critical, "margins": b.to_dict()["robustness_margins"]},
                       "after": {"critical": a.critical, "margins": a.to_dict()["robustness_margins"]}})
        r = {"id": rid, "failure_ids": failure_ids, "strategy": steps, "actions": [a.to_dict() for a in actions],
             "described": [a.describe(base) for a in actions], "cost": total, "rationale": rationale,
             "revision": revision, "status": "PROPOSED", "attacks": [], "certificate": None,
             "same_condition": sc, "solver_checks": checks}
        self.s["repairs"].append(r)
        self.event("BLUE", "repair_proposed", repair_id=rid, actions=r["described"], cost=total,
                   revision=bool(revision))
        self.set_phase("ATTACK")
        return {"repair_id": rid, "actions": r["described"], "cost": total, "solver": notes,
                "same_condition_on_target": sc,
                "next": "CRITIC must attack this repair (attack_repair) before it can be verified"}

    def attack(self, rid: str, ids, hypothesis: str, vulnerable_constraint: str) -> dict:
        r = self.repair(rid)
        if r["status"] != "PROPOSED":
            raise RuleViolation(f"repair {rid} is {r['status']}; only PROPOSED repairs can be attacked")
        p = self._validate_ids(ids)
        if self.s["attack_used"] >= self.s["attack_budget"]:
            raise RuleViolation(f"critic attack budget exhausted ({self.s['attack_used']}/{self.s['attack_budget']})")
        self.s["attack_used"] += 1
        before = simulate(self.reference_world(r), p)
        after = simulate(self.patched_world(rid), p)
        from .repair import attributable_failure
        is_ce = attributable_failure(after, self.incident_rooms_for(r)) and not before.critical
        res = {"attack": len(r["attacks"]) + 1, "perturbation": sorted(p), "hypothesis": hypothesis,
               "vulnerable_constraint": vulnerable_constraint,
               "original_critical": before.critical, "patched_critical": after.critical,
               "patched_violations": after.violations,
               "verdict": "COUNTEREXAMPLE: repair introduces a new failure" if is_ce else
                          ("pre-existing failure, not caused by the repair" if after.critical else "repair survives this attack")}
        r["attacks"].append(res)
        self.event("CRITIC", "attack", repair_id=rid, perturbation=sorted(p), counterexample=is_ce,
                   violations=[v["constraint"] for v in after.violations])
        if is_ce:
            r["status"] = "REPAIR_REJECTED"
            self.event("ENGINE", "repair_rejected", repair_id=rid, counterexample=sorted(p))
            self.set_phase("REPAIR")
        else:
            self.set_phase("VERIFY")
        return res | {"repair_status": r["status"], "critic_budget": f"{self.s['attack_used']}/{self.s['attack_budget']}"}

    def verify(self, rid: str) -> dict:
        r = self.repair(rid)
        if r["status"] != "PROPOSED":
            raise RuleViolation(f"repair {rid} is {r['status']}")
        if not r["attacks"]:
            raise RuleViolation("the critic must attack the repair at least once before verification")
        targets = [_fs(self.failure(fid)["minimal_set"]) for fid in r["failure_ids"]]
        cert = risk_certificate(self.reference_world(r), self.patched_world(rid), targets, seed=self.s["seed"] + 7,
                                ignore_rooms=self.incident_rooms_for(r))
        if self._targets_incident(r):
            cert["reference"] = "regressions are measured against the schedule before the incident"
        r["certificate"] = cert
        r["status"] = cert["status"]
        self.event("RISK", "certificate", repair_id=rid, status=cert["status"],
                   regressions=cert["exact_bounded_space"]["regression_count"],
                   failure_rate_after=cert["sampled_demand_uncertainty"]["failure_rate_after"])
        if cert["status"] == "REPAIR_REJECTED":
            ex = cert["exact_bounded_space"]
            ce = ex["regressions"][0] if ex["regressions"] else None
            if ce:
                r["attacks"].append({"attack": "exhaustive", "perturbation": ce, "verdict": "COUNTEREXAMPLE found by exact verification"})
            self.set_phase("REPAIR")
        else:
            self.set_phase("APPROVE")
        return {"repair_id": rid, "status": r["status"], "certificate": cert}

    def apply(self, rid: str, evidence_summary: str) -> dict:
        r = self.repair(rid)
        if r["status"] != "VERIFIED_ON_TEST_SET":
            self.event("ENGINE", "apply_refused", repair_id=rid, status=r["status"])
            raise RuleViolation(f"refused: repair {rid} is {r['status']}, not VERIFIED_ON_TEST_SET")
        if rid in self.s["applied"]:
            raise RuleViolation(f"{rid} already applied")
        before = load_world(self.s["world_id"])
        self.s["applied"].append(rid)
        r["status"] = "APPLIED"
        for fid in r["failure_ids"]:
            iid = self.failure(fid).get("incident_id")
            for inc in self.s.get("incidents", []):
                if inc["id"] == iid:
                    inc["status"] = "resolved"
        notes = self.notifications(before, self.base_world()) if hasattr(before, "sessions") else []
        self.s["notifications"] = notes
        r["evidence_summary_shown_to_human"] = evidence_summary
        self.event("HUMAN", "approved_and_applied", repair_id=rid, actions=r["described"])
        self.set_phase("APPLIED")
        return {"applied": rid, "actions": r["described"],
                "active_schedule_changed": True,
                "notify": notes,
                "note": "executed only because the human approved this tool call in TrueForge"}

    def evidence(self, rid: str) -> dict:
        """Raw rows behind a repair's certificate (read-only; for independent recomputation)."""
        r = self.repair(rid)
        if not r.get("certificate"):
            raise RuleViolation(f"repair {rid} has no certificate yet; call verify_repair first")
        ev = raw_evidence(self.reference_world(r) if r["status"] != "APPLIED" else self._pre_apply_world(rid),
                          self.patched_world(rid) if r["status"] != "APPLIED" else self.base_world(),
                          seed=self.s["seed"] + 7)
        self.event("ENGINE", "evidence_exported", repair_id=rid, rows=len(ev["exact_rows"]), draws=len(ev["sampled_draws"]))
        cert = r["certificate"]
        return {"repair_id": rid, "certificate_claims": {
                    "status": cert["status"],
                    "regression_count": cert["exact_bounded_space"]["regression_count"],
                    "failing_before": cert["exact_bounded_space"]["failing_perturbations_before"],
                    "failing_after": cert["exact_bounded_space"]["failing_perturbations_after"],
                    **{k: v for k, v in cert["sampled_demand_uncertainty"].items() if k not in ("process", "same_condition")}},
                **ev}

    def _pre_apply_world(self, rid: str) -> World:
        w = load_world(self.s["world_id"])
        for x in self.s["applied"]:
            if x == rid:
                break
            w = apply_actions(w, [Action(**a) for a in self.repair(x)["actions"]])
        return w

    def trace(self) -> list[dict]:
        return self.s["events"]
