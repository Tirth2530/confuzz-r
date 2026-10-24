"""Risk certificate for a repair. Two clearly separated kinds of evidence:

1. EXACT over the modeled perturbation space (2^d sets, nominal demand): from verify_exact.
2. SAMPLED over demand uncertainty: n draws, each a uniformly random perturbation and an
   independent per-session demand multiplier ~ Uniform(1 - noise, 1 + noise). Each draw is
   run on the ORIGINAL and the PATCHED world with identical perturbation, seed and demand
   (same-condition comparison). Draws are i.i.d. Bernoulli trials for "critical failure"
   under this sampling process, so an exact one-sided Clopper-Pearson upper bound applies
   to the failure probability UNDER THIS SAMPLING PROCESS ONLY.

Nothing here is a guarantee about real conferences or disruptions outside the model.
"""
from __future__ import annotations

import math
import random

import numpy as np
from scipy.stats import beta

from .repair import verify_exact
from .sim import simulate
from .world import World


def clopper_pearson_upper(x: int, n: int, conf: float = 0.95) -> float:
    if n == 0:
        return 1.0
    if x >= n:
        return 1.0
    return float(beta.ppf(conf, x + 1, n - x))


def cvar(losses: list[float], alpha: float = 0.95) -> tuple[float, int]:
    """CVaR_alpha = mean of the worst ceil((1 - alpha) * n) losses. Returns (value, tail size)."""
    if not losses:
        return 0.0, 0
    k = max(1, math.ceil(round((1 - alpha) * len(losses), 9)))  # round: (1-0.95)*100 = 5.000000000000004
    tail = sorted(losses, reverse=True)[:k]
    return float(np.mean(tail)), k


def sampled_draws(world: World, n: int, seed: int, noise: float):
    rng = random.Random(seed)
    ids = world.disruption_ids
    for _ in range(n):
        p = frozenset(d for d in ids if rng.random() < 0.5)
        dn = {s.id: rng.uniform(1 - noise, 1 + noise) for s in world.sessions}
        yield p, dn


def risk_certificate(original: World, patched: World, targets: list[frozenset[str]],
                     n: int = 400, seed: int = 7, noise: float = 0.10, conf: float = 0.95) -> dict:
    exact = verify_exact(original, patched, targets)
    before_f, after_f, before_l, after_l, acc_after = 0, 0, [], [], 0
    for p, dn in sampled_draws(original, n, seed, noise):
        rb, ra = simulate(original, p, dn), simulate(patched, p, dn)
        before_f += rb.critical
        after_f += ra.critical
        before_l.append(rb.severity)
        after_l.append(ra.severity)
        acc_after += any(v["constraint"] == "accessibility" for v in ra.violations)
    cv_b, k = cvar(before_l)
    cv_a, _ = cvar(after_l)
    if exact["status"] != "VERIFIED_ON_BOUNDED_SPACE":
        status = "REPAIR_REJECTED"
    elif n < 100:
        status = "INSUFFICIENT_EVIDENCE"
    else:
        status = "VERIFIED_ON_TEST_SET"
    return {
        "status": status,
        "exact_bounded_space": exact,
        "sampled_demand_uncertainty": {
            "process": f"{n} i.i.d. draws: uniform random perturbation, per-session demand x U({1 - noise:.2f}, {1 + noise:.2f}), seed {seed}",
            "same_condition": "each draw run on original and patched world with identical perturbation, seed and demand",
            "n": n,
            "failures_before": before_f,
            "failures_after": after_f,
            "failure_rate_before": round(before_f / n, 4),
            "failure_rate_after": round(after_f / n, 4),
            f"clopper_pearson_{int(conf * 100)}_upper_after": round(clopper_pearson_upper(after_f, n, conf), 4),
            "accessibility_violation_rate_after": round(acc_after / n, 4),
            "mean_severity_before": round(float(np.mean(before_l)), 4),
            "mean_severity_after": round(float(np.mean(after_l)), 4),
            "cvar95_severity_before": round(cv_b, 4),
            "cvar95_severity_after": round(cv_a, 4),
            "cvar_note": f"NOISY: tail contains only {k} observations" if k < 30 else f"tail contains {k} observations",
        },
        "assumptions": [
            "disruptions are limited to the 8 modeled atomic disruptions",
            "the operations playbook is followed exactly",
            "sampled bound holds only for the stated sampling process (uniform perturbations, +/-10% demand)",
        ],
        "limitations": [
            "uniform sampling over perturbations does not reflect real disruption frequencies",
            "severity weights are operator choices, not measured costs",
            "no claim about disruptions, constraints or attendee behavior outside the model",
        ],
    }


def raw_evidence(original: World, patched: World, n: int = 400, seed: int = 7, noise: float = 0.10) -> dict:
    """The rows behind risk_certificate, so an independent agent can recompute every number.
    Uses the same perturbations, seeds and demand draws as risk_certificate with the same arguments."""
    from .space import all_perturbations
    exact_rows = []
    for p in all_perturbations(original.disruption_ids):
        exact_rows.append({"perturbation": sorted(p), "fails_before": simulate(original, p).critical,
                           "fails_after": simulate(patched, p).critical})
    draws = []
    for p, dn in sampled_draws(original, n, seed, noise):
        rb, ra = simulate(original, p, dn), simulate(patched, p, dn)
        draws.append({"perturbation": sorted(p), "fails_before": rb.critical, "fails_after": ra.critical,
                      "severity_before": round(rb.severity, 6), "severity_after": round(ra.severity, 6),
                      "accessibility_after": any(v["constraint"] == "accessibility" for v in ra.violations)})
    return {"exact_rows": exact_rows, "sampled_draws": draws,
            "how_to_check": ("regression = fails_after and not fails_before over exact_rows; failure rate = mean(fails_after) "
                             "over sampled_draws; 95% one-sided Clopper-Pearson upper bound for x failures in n trials = "
                             "the p solving P(Binomial(n, p) <= x) = 0.05; CVaR95 = mean of the worst ceil(0.05 n) severities")}
