"""Domain dispatch. The engine (search, repair, verification, certificate, session, MCP server) is
domain-independent; each domain supplies a simulator, its noise keys and its action set.
Adding a domain = one module with simulate(), noise_keys(), ACTION_TYPES, concrete_actions(),
apply_action(), action_cost(), describe(), and a demo world."""
from __future__ import annotations


def _inv(world) -> bool:
    return getattr(world, "domain", "conference") == "inventory"


def simulate(world, perturbation, demand_noise=None):
    if _inv(world):
        from .inventory import simulate as f
    else:
        from .sim import simulate as f
    return f(world, perturbation, demand_noise)


def noise_keys(world) -> list[str]:
    if _inv(world):
        from .inventory import noise_keys as f
        return f(world)
    return [s.id for s in world.sessions]


def domain_name(world) -> str:
    return "inventory" if _inv(world) else "conference"
