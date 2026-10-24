"""OpenAI Red agent for BENCHMARK runs: semantic interaction prior over hyperedges.

Execution mode is explicit. Live demo: Red runs inside TrueForge and submits its model through
the MCP tool submit_interaction_model. Benchmark: the same prompt is sent by direct OpenAI API
calls for throughput (TrueForge HTTP API driving is optional, see bench.py --via-trueforge).
Results record which mode produced them. Never present direct-API runs as TrueForge runs.

Red sees only what an organizer could know: the public spec (schedule, rooms, zones, travel,
AV, step-free needs, disruption definitions, playbook, constraint definitions). It never sees
ground truth, enumeration results, or other worlds' answers.

Responses are cached on disk keyed by sha256(model + prompt), so a benchmark can be rerun
reproducibly from cache; first-time outputs are nondeterministic and are recorded as such.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from .world import World

CACHE_DIR = Path(os.environ.get("CONFUZZ_LLM_CACHE", Path(__file__).resolve().parent.parent / "experiments" / "llm_cache"))
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

RED_SYSTEM = """You are RED, an adversarial conference-operations analyst.
You build a semantic model of how disruptions could INTERACT to break a conference schedule.
You never see test results you were not given, and you never claim a test outcome.
Reason causally: which disruptions displace sessions, consume spare capacity, remove step-free
options, remove AV options, or lengthen speaker transitions, and where those causal paths
CONVERGE on the same constraint or the same scarce resource. Interactions where every
disruption alone is harmless are the target. Use the operations playbook literally."""

RED_INSTRUCTIONS = """Return JSON with:
- hypotheses: 6 to 12 candidate interactions (pairs or triples of disruption ids), each with
  candidate_set, interaction_hypothesis (the causal mechanism, concrete: rooms, sessions, minutes, seats),
  predicted_constraints (subset of: capacity, room_overlap, speaker_overlap, av, accessibility,
  transition, hosting), and risk_prior in [0,1] (your judgment, not a probability).
- mechanism_edges: causal graph edges {src, dst, relation}. src/dst are disruption ids or consequence
  nodes you name (e.g. "displaced:S07", "spare_capacity:slot2", "constraint:accessibility").
  Two disruptions whose paths reach a shared node are treated as a candidate interaction.
Do not include single disruptions as hypotheses."""

SCHEMA = {
    "name": "interaction_model",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["hypotheses", "mechanism_edges"],
        "properties": {
            "hypotheses": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["candidate_set", "interaction_hypothesis", "predicted_constraints", "risk_prior"],
                "properties": {
                    "candidate_set": {"type": "array", "items": {"type": "string"}},
                    "interaction_hypothesis": {"type": "string"},
                    "predicted_constraints": {"type": "array", "items": {"type": "string"}},
                    "risk_prior": {"type": "number"},
                }}},
            "mechanism_edges": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["src", "dst", "relation"],
                "properties": {"src": {"type": "string"}, "dst": {"type": "string"}, "relation": {"type": "string"}}}},
        },
    },
}


def red_prompt(world: World, observations: list[dict] | None = None) -> str:
    spec = world.public_spec()
    spec.pop("world_id", None)  # no identifiers that could key memorized answers
    obs = ""
    if observations:
        obs = "\nTests already run (yours to use):\n" + json.dumps(observations, indent=1)
    return f"{RED_INSTRUCTIONS}\n\nConference specification:\n{json.dumps(spec, indent=1)}{obs}"


def red_interaction_model(world: World, observations: list[dict] | None = None, model: str = MODEL,
                          use_cache: bool = True) -> dict:
    prompt = red_prompt(world, observations)
    key = hashlib.sha256((model + "\n" + RED_SYSTEM + "\n" + prompt).encode()).hexdigest()
    path = CACHE_DIR / f"{key}.json"
    if use_cache and path.exists():
        rec = json.loads(path.read_text())
        rec["from_cache"] = True
        return rec
    from openai import OpenAI  # lazy: only needed when not cached
    client = OpenAI()
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": RED_SYSTEM}, {"role": "user", "content": prompt}],
        response_format={"type": "json_schema", "json_schema": SCHEMA},
    )
    out = json.loads(resp.choices[0].message.content)
    ids = set(world.disruption_ids)
    out["hypotheses"] = [h for h in out["hypotheses"]
                         if h["candidate_set"] and set(h["candidate_set"]) <= ids and 0 <= h["risk_prior"] <= 1]
    rec = {
        "model": model, "mode": "direct_openai_api", "world_id": world.id, "output": out,
        "usage": {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens},
        "latency_s": round(time.time() - t0, 2), "created": time.time(), "from_cache": False,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1))
    return rec
