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


# ---------------------------------------------------------------------------
# Critic (benchmark mode). Live runs use attack_repair inside TrueForge with the same reasoning brief.
# ---------------------------------------------------------------------------
CRITIC_SYSTEM = """You are CRITIC, an adversarial reviewer of a schedule repair proposed by another AI.
Your job: find a perturbation (set of disruption ids) that PASSES on the original schedule but FAILS
on the patched schedule. Ask what the repair made newly vulnerable: which sessions moved, which
zones, rooms, speaker transitions, step-free routes or AV now matter that did not before, and what
spare capacity the repair consumed that the operations playbook used to rely on. Use the playbook
literally. Single disruptions count. You never see test results and never claim an outcome."""

CRITIC_SCHEMA = {
    "name": "critic_attacks", "strict": True,
    "schema": {"type": "object", "additionalProperties": False, "required": ["attacks"],
               "properties": {"attacks": {"type": "array", "items": {
                   "type": "object", "additionalProperties": False,
                   "required": ["perturbation", "hypothesis", "vulnerable_constraint"],
                   "properties": {"perturbation": {"type": "array", "items": {"type": "string"}},
                                  "hypothesis": {"type": "string"},
                                  "vulnerable_constraint": {"type": "string"}}}}}},
}


def critic_prompt(world: World, diff: dict, target: list[str]) -> str:
    spec = world.public_spec()
    spec.pop("world_id", None)
    return ("Return JSON {attacks: [...]} with up to 10 attacks, most likely counterexample first. Each attack: "
            "perturbation (disruption ids), hypothesis (concrete mechanism), vulnerable_constraint.\n\n"
            f"Original conference specification:\n{json.dumps(spec, indent=1)}\n\n"
            f"The repair was proposed to fix the failure caused by disruptions {target}.\n"
            f"Repair and what it changed:\n{json.dumps(diff, indent=1)}")


def critic_attacks(world: World, diff: dict, target: list[str], model: str = MODEL, use_cache: bool = True) -> dict:
    prompt = critic_prompt(world, diff, target)
    key = hashlib.sha256((model + "\n" + CRITIC_SYSTEM + "\n" + prompt).encode()).hexdigest()
    path = CACHE_DIR / f"critic_{key}.json"
    if use_cache and path.exists():
        rec = json.loads(path.read_text()); rec["from_cache"] = True
        return rec
    from openai import OpenAI
    t0 = time.time()
    resp = OpenAI().chat.completions.create(
        model=model, messages=[{"role": "system", "content": CRITIC_SYSTEM}, {"role": "user", "content": prompt}],
        response_format={"type": "json_schema", "json_schema": CRITIC_SCHEMA})
    out = json.loads(resp.choices[0].message.content)
    ids = set(world.disruption_ids)
    out["attacks"] = [x for x in out["attacks"] if x["perturbation"] and set(x["perturbation"]) <= ids][:10]
    rec = {"model": model, "mode": "direct_openai_api", "world_id": world.id, "output": out,
           "usage": {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens},
           "latency_s": round(time.time() - t0, 2), "created": time.time(), "from_cache": False}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1))
    return rec


# ---------------------------------------------------------------------------
# Blue (benchmark mode). Live runs use propose_repair inside TrueForge with the same brief.
# ---------------------------------------------------------------------------
BLUE_SYSTEM = """You are BLUE, a conference-operations repair strategist. You receive a confirmed failure
(the minimal set of disruptions that breaks the schedule, the simulator's violations and the
operations playbook's moves). Propose repair STRATEGIES that break the causal mechanism, not just the
symptom, and that avoid creating new failures elsewhere. A deterministic solver instantiates each
strategy (cheapest concrete action of that type that fixes the target) and an exact verifier decides
whether it is accepted. You never see verification results you were not given."""

_STEP = {"type": "object", "additionalProperties": False,
         "required": ["type", "session", "session2", "room", "slot"],
         "properties": {"type": {"type": "string", "enum": ["MOVE_SESSION", "SWAP_ROOMS", "SHIFT_SESSION", "ACTIVATE_STANDBY"]},
                        "session": {"type": ["string", "null"]}, "session2": {"type": ["string", "null"]},
                        "room": {"type": ["string", "null"]}, "slot": {"type": ["integer", "null"]}}}
BLUE_SCHEMA = {"name": "blue_strategies", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["strategies"],
    "properties": {"strategies": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["steps", "mechanism_broken", "risk_of_new_failure", "what_changed_in_my_reasoning"],
        "properties": {"steps": {"type": "array", "items": _STEP},
                       "mechanism_broken": {"type": "string"},
                       "risk_of_new_failure": {"type": "string"},
                       "what_changed_in_my_reasoning": {"type": "string"}}}}}}}


def blue_prompt(world: World, trace: dict, history: list[dict]) -> str:
    spec = world.public_spec()
    spec.pop("world_id", None)
    h = ""
    if history:
        h = ("\nYour earlier proposals and what the verifier found (a counterexample is a disruption set that "
             "passed before your repair and fails after it):\n" + json.dumps(history, indent=1) +
             "\nFill what_changed_in_my_reasoning and propose materially different strategies.")
    return ("Return JSON {strategies: [...]}: up to 3 strategies, best first. Each has 1 or 2 steps. "
            "Leave a field null to let the solver choose it. Use ACTIVATE_STANDBY only for standby rooms.\n\n"
            f"Conference specification:\n{json.dumps(spec, indent=1)}\n\nConfirmed failure:\n{json.dumps(trace, indent=1)}{h}")


def blue_strategies(world: World, trace: dict, history: list[dict], model: str = MODEL, use_cache: bool = True) -> dict:
    prompt = blue_prompt(world, trace, history)
    key = hashlib.sha256((model + "\n" + BLUE_SYSTEM + "\n" + prompt).encode()).hexdigest()
    path = CACHE_DIR / f"blue_{key}.json"
    if use_cache and path.exists():
        rec = json.loads(path.read_text()); rec["from_cache"] = True
        return rec
    from openai import OpenAI
    t0 = time.time()
    resp = OpenAI().chat.completions.create(
        model=model, messages=[{"role": "system", "content": BLUE_SYSTEM}, {"role": "user", "content": prompt}],
        response_format={"type": "json_schema", "json_schema": BLUE_SCHEMA})
    out = json.loads(resp.choices[0].message.content)
    for st in out["strategies"]:
        st["steps"] = [{k: v for k, v in step.items() if v is not None} for step in st["steps"]][:2]
    rec = {"model": model, "mode": "direct_openai_api", "world_id": world.id, "output": out,
           "usage": {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens},
           "latency_s": round(time.time() - t0, 2), "created": time.time(), "from_cache": False}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1))
    return rec
