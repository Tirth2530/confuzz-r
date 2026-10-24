"""CONFUZZ-R MCP server (stdio). TrueForge agents reach the deterministic engine only through these tools.

Run:  python -m confuzz.mcp_server --http --port 8800   # TrueForge: remote MCP URL http://127.0.0.1:8800/mcp
      python -m confuzz.mcp_server                       # stdio (local clients, tests)
Env:  CONFUZZ_STATE=/path/state.json  (persisted session; survives restarts)

TrueForge connects to remote MCP servers by URL only (Streamable HTTP or SSE) and its outbound
URL guard blocks loopback addresses unless allowed: start TrueForge with
  OUTBOUND_URL_ALLOWED_HOSTS='["127.0.0.1"]'
(verified against trueforge main @ 1090880, packages/trueforge-core/src/core/util/ssrfGuard.ts).

Hidden ground truth is never reachable from here (tests/test_isolation.py checks the imports).
Works with the MCP Python SDK 1.x (FastMCP) and 2.x (MCPServer).
"""
from __future__ import annotations

import json
from typing import Any

try:  # mcp 2.x
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server
from mcp.types import ToolAnnotations

from .session import RuleViolation, Session

mcp = _Server("confuzz-r")

READ = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
SIM = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False)
DESTRUCTIVE = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False)


def _run(fn, *a, **k) -> str:
    """Load session, run, persist. Rule violations go back to the agent as an error payload."""
    s = Session.load()
    try:
        out = fn(s, *a, **k)
        s.save()
        return json.dumps(out, indent=1, default=str)
    except RuleViolation as e:
        s.save()
        return json.dumps({"error": "RULE_VIOLATION", "message": str(e)})


@mcp.tool(annotations=READ)
def get_conference() -> str:
    """Conference spec (rooms, sessions, speakers, zones, travel, AV, step-free needs), the 8 atomic
    disruptions, the operations playbook, constraint definitions, current phase and budgets.
    Never contains test results you have not paid for."""
    def f(s: Session):
        return {"phase": s.s["phase"], "discovery_budget": f"{s.s['used']}/{s.s['budget']}",
                "critic_budget": f"{s.s['attack_used']}/{s.s['attack_budget']}",
                "applied_repairs": s.s["applied"], "spec": s.base_world().public_spec()}
    return _run(f)


@mcp.tool(annotations=SIM)
def submit_interaction_model(hypotheses: list[dict[str, Any]], mechanism_edges: list[dict[str, Any]], notes: str = "") -> str:
    """RED: submit your semantic interaction model before spending budget.
    hypotheses: [{candidate_set: [disruption ids], interaction_hypothesis: str,
                  predicted_constraints: [constraint names], risk_prior: 0..1}]
    mechanism_edges: causal graph edges [{src, dst, relation}] linking disruption ids to
                     consequence nodes (e.g. "D2" -> "displaced:S07" -> "capacity").
    Pairs of disruptions whose paths converge on a shared node raise their acquisition score.
    risk_prior is your judgment, treated as an uncalibrated prior, never as a probability."""
    return _run(lambda s: s.submit_prior(hypotheses, mechanism_edges, notes))


@mcp.tool(annotations=READ)
def rank_experiments(top_k: int = 5) -> str:
    """SCORER: rank untested perturbations by the acquisition function. Returns each candidate's
    score with raw and weighted components (llm prior, empirical risk, coverage of open
    interaction hypotheses, novelty, minimization cost). Free: does not run the simulator."""
    return _run(lambda s: s.rank(top_k))


@mcp.tool(annotations=SIM)
def run_experiment(disruptions: list[str], rationale: str) -> str:
    """SIMULATOR: apply a set of disruption ids to the active schedule and return the
    deterministic robustness vector (normalized margin per constraint, raw margins, violations,
    playbook moves). Costs 1 discovery query unless this exact set was already run."""
    return _run(lambda s: s.query(disruptions, "test", "RED", rationale))


@mcp.tool(annotations=SIM)
def minimize_failure(disruptions: list[str]) -> str:
    """Shrink a failing set to a 1-minimal failing set (every single removal passes).
    Each probe costs 1 discovery query. Records the failure (F1, F2, ...) with its interaction order."""
    return _run(lambda s: s.minimize(disruptions))


@mcp.tool(annotations=SIM)
def explain_interaction(failure_id: str) -> str:
    """Run every proper subset of a minimal failure and return the PASS/FAIL table that shows it is
    a genuine interaction. Unqueried subsets cost discovery queries."""
    return _run(lambda s: s.explain(failure_id))


@mcp.tool(annotations=SIM)
def propose_repair(failure_ids: list[str], strategy_steps: list[dict[str, Any]], rationale: str,
                   previous_repair_id: str = "", counterexample: list[str] | None = None,
                   diagnosis: str = "", what_changed_in_my_reasoning: str = "") -> str:
    """BLUE: propose a repair strategy for one or more failures. strategy_steps is a list of
    {type, session?, session2?, room?, slot?} with type in MOVE_SESSION, SWAP_ROOMS, SHIFT_SESSION,
    ACTIVATE_STANDBY, DO_NOTHING. Leave fields out to let the deterministic solver choose the
    cheapest concrete action of that type that fixes the target(s). Returns the instantiated
    actions, cost, and a same-condition before/after on each target.
    If any earlier repair was rejected you MUST fill previous_repair_id, counterexample,
    diagnosis and what_changed_in_my_reasoning."""
    rev = None
    if previous_repair_id or counterexample or diagnosis or what_changed_in_my_reasoning:
        rev = {"previous_repair_id": previous_repair_id, "counterexample": counterexample or [],
               "diagnosis": diagnosis, "what_changed_in_my_reasoning": what_changed_in_my_reasoning}
    return _run(lambda s: s.propose_repair(failure_ids, strategy_steps, rationale, rev))


@mcp.tool(annotations=SIM)
def attack_repair(repair_id: str, disruptions: list[str], hypothesis: str, vulnerable_constraint: str) -> str:
    """CRITIC: test a perturbation against the PATCHED schedule. Reason about which constraint the
    repair made newly vulnerable. A counterexample is a set that passes on the original schedule but
    fails on the patched one; it rejects the repair. Costs 1 critic query."""
    return _run(lambda s: s.attack(repair_id, disruptions, hypothesis, vulnerable_constraint))


@mcp.tool(annotations=SIM)
def verify_repair(repair_id: str) -> str:
    """RISK: exact verification over all 2^d modeled perturbations (no regressions, targets fixed,
    base passes) plus a same-condition sampled certificate under demand noise with a Clopper-Pearson
    upper bound. Requires at least one critic attack first. Sets VERIFIED_ON_TEST_SET or REPAIR_REJECTED."""
    return _run(lambda s: s.verify(repair_id))


@mcp.tool(annotations=DESTRUCTIVE)
def apply_repair(repair_id: str, evidence_summary: str) -> str:
    """Apply a verified repair to the ACTIVE conference schedule. Requires human approval in
    TrueForge. The server refuses any repair that is not VERIFIED_ON_TEST_SET.
    evidence_summary is what the human reads on the approval card: write the discovered failure,
    its minimal set, the proposed repair and why, rejected earlier repairs with their
    counterexamples, critic attacks tried, exact bounded-space result, sampled failure rate and
    upper bound, and known limitations."""
    return _run(lambda s: s.apply(repair_id, evidence_summary))


@mcp.tool(annotations=READ)
def get_trace() -> str:
    """Full event trace of this session: who (RED, SCORER, ENGINE, BLUE, CRITIC, RISK, HUMAN) did what."""
    return _run(lambda s: s.trace())


@mcp.tool(annotations=DESTRUCTIVE)
def reset_session(world_id: str = "demo", budget: int = 20, attack_budget: int = 10, seed: int = 0) -> str:
    """Start a fresh experiment session (discards current session state)."""
    def f(s: Session):
        n = Session.new(world_id, budget, attack_budget, seed)
        s.s = n.s
        return {"reset": True, "world_id": world_id, "budget": budget, "attack_budget": attack_budget}
    return _run(f)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", action="store_true", help="serve Streamable HTTP instead of stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8800)
    a = ap.parse_args()
    if not a.http:
        mcp.run()
        return
    try:  # mcp 2.x
        mcp.run("streamable-http", host=a.host, port=a.port)
    except TypeError:  # mcp 1.x: host/port live on settings
        mcp.settings.host, mcp.settings.port = a.host, a.port
        mcp.run("streamable-http")


if __name__ == "__main__":
    main()
