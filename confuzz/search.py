"""Budgeted active testing: oracle with query accounting, minimization, policies, acquisition.

Every simulator query a method makes counts against its budget B, including the
queries spent minimizing a failure. Repeating an already-asked query is answered
from the method's own log at no cost (same for every method).

Shared harness rules (identical for every policy):
  * the empty perturbation is never proposed (the base schedule is known to pass)
  * a perturbation that contains an already-found minimal failing set is never proposed
    (it would fail for a known reason)
  * when a query fails, the harness minimizes it with a 1-minimal reduction
    (remove one element at a time, keep the removal if the set still fails critically;
    this is the final phase of Zeller's ddmin, adequate for |delta| <= 8)
  * a found set counts toward recall only if minimization finished within budget
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .domain import simulate
from .sim import SimResult
from .space import all_perturbations
from .world import World


class BudgetExhausted(Exception):
    pass


@dataclass
class Observation:
    index: int                 # 1-based query count at which this was paid for
    perturbation: frozenset[str]
    critical: bool
    margins: dict[str, float]
    severity: float
    purpose: str


class Oracle:
    """The only path from a testing policy to the simulator."""

    def __init__(self, world: World, budget: int):
        self.world, self.budget, self.used = world, budget, 0
        self.log: list[Observation] = []
        self._cache: dict[frozenset[str], Observation] = {}

    @property
    def remaining(self) -> int:
        return self.budget - self.used

    def query(self, p, purpose: str = "test") -> Observation:
        p = frozenset(p)
        if p in self._cache:
            return self._cache[p]
        if self.used >= self.budget:
            raise BudgetExhausted()
        self.used += 1
        r: SimResult = simulate(self.world, p)
        ob = Observation(self.used, p, r.critical, dict(r.margins), r.severity, purpose)
        self.log.append(ob)
        self._cache[p] = ob
        return ob

    def known(self, p) -> Observation | None:
        return self._cache.get(frozenset(p))


def minimize(oracle: Oracle, failing: frozenset[str]) -> tuple[frozenset[str], list[dict]]:
    """1-minimal reduction. Returns (set, trace of probes). Raises BudgetExhausted."""
    current = set(failing)
    trace = []
    for d in sorted(failing):
        if len(current) == 1:
            break
        trial = frozenset(current - {d})
        ob = oracle.query(trial, purpose="minimize")
        trace.append({"probe": sorted(trial), "critical": ob.critical})
        if ob.critical:
            current.remove(d)
    return frozenset(current), trace


# ---------------------------------------------------------------------------
# Semantic prior (from Red / OpenAI). Not a probability: an LLM-assigned score.
# ---------------------------------------------------------------------------
@dataclass
class SemanticPrior:
    hypotheses: list[dict] = field(default_factory=list)   # {candidate_set, risk_prior, predicted_constraints, ...}
    edges: list[dict] = field(default_factory=list)        # {src, dst, relation}

    def disruption_reach(self, dids: list[str]) -> dict[str, set[str]]:
        """Nodes reachable from each disruption in the causal graph."""
        adj: dict[str, set[str]] = {}
        for e in self.edges:
            adj.setdefault(str(e["src"]), set()).add(str(e["dst"]))
        reach = {}
        for d in dids:
            seen, stack = set(), [d]
            while stack:
                n = stack.pop()
                for m in adj.get(n, ()):
                    if m not in seen:
                        seen.add(m)
                        stack.append(m)
            reach[d] = seen
        return reach

    def score(self, p: frozenset[str], reach: dict[str, set[str]]) -> float:
        """R_LLM(p) in [0,1]:
        0.5 * best hypothesis match: max_h risk_prior_h * |p & h| / |p | h|  (Jaccard-weighted)
        0.5 * graph convergence: fraction of disruption pairs in p whose causal paths share a node"""
        hyp = 0.0
        for h in self.hypotheses:
            hs = frozenset(h.get("candidate_set", []))
            if hs:
                hyp = max(hyp, float(h.get("risk_prior", 0.0)) * len(p & hs) / len(p | hs))
        pairs = list(combinations(sorted(p), 2))
        conv = sum(1 for a, b in pairs if reach.get(a, set()) & reach.get(b, set())) / len(pairs) if pairs else 0.0
        return 0.5 * hyp + 0.5 * conv


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
class Policy:
    name = "base"

    def propose(self, cands: list[frozenset[str]], oracle: Oracle, found: list[frozenset[str]]) -> frozenset[str]:
        raise NotImplementedError


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def propose(self, cands, oracle, found):
        return self.rng.choice(cands)


class RandomPrunedPolicy(RandomPolicy):
    """Random with the same monotone pruning CONFUZZ-R uses. Isolates the acquisition's contribution."""
    name = "random_pruned"
    prune_passed = True


class SizeOrderedPolicy(Policy):
    """Classic combinatorial testing order: all singles, then pairs, then triples...
    Random order within a size. A strong, simple baseline."""
    name = "size_ordered"

    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def propose(self, cands, oracle, found):
        k = min(len(c) for c in cands)
        return self.rng.choice([c for c in cands if len(c) == k])


DEFAULT_WEIGHTS = {"llm": 0.40, "emp": 0.20, "unc": 0.15, "nov": 0.10, "plaus": 0.15}


class AcquisitionPolicy(Policy):  # v1, kept for the record (lost to random on frozen worlds)
    """CONFUZZ-R acquisition over the finite candidate set:

      A_t(p) = w_llm * n(R_LLM) + w_emp * n(R_emp) + w_unc * n(U) + w_nov * n(N) + w_plaus * n(P)

    n(.) = min-max normalization over current candidates (constant -> 0).
    R_LLM  semantic interaction prior from Red (0 if no prior: the no-LLM ablation)
    R_emp  empirical risk score from an additive ridge surrogate of normalized margins
           fit on this run's observations: risk = clip(1 - min_c predicted_margin_c, 0, 2) / 2
           (additive, so it cannot see pure interactions; that is U's and R_LLM's job)
    U      fraction of disruption pairs in p never observed together in a query
    N      1 - max Jaccard similarity to any queried perturbation
    P      plausibility: 1 / |p| (fewer simultaneous disruptions are more plausible)
    Ties broken by a seeded RNG. Weights fixed a priori; tuning only on development worlds.
    """
    name = "confuzz_v1"

    def __init__(self, seed: int, prior: SemanticPrior | None = None, weights: dict | None = None, name: str | None = None):
        self.rng = random.Random(seed)
        self.prior = prior
        self.w = dict(DEFAULT_WEIGHTS if weights is None else weights)
        if prior is None:
            self.w["llm"] = 0.0
        if name:
            self.name = name
        self.last_decomposition: dict | None = None

    def components(self, cands, oracle) -> dict[str, np.ndarray]:
        ids = oracle.world.disruption_ids
        idx = {d: i for i, d in enumerate(ids)}
        X = np.array([[1.0 if d in c else 0.0 for d in ids] for c in cands])
        obs = oracle.log
        # empirical additive surrogate
        if obs:
            Xo = np.array([[1.0 if d in o.perturbation else 0.0 for d in ids] for o in obs])
            cons = sorted(obs[0].margins)
            Y = np.array([[o.margins[c] for c in cons] for o in obs])
            Xo1 = np.hstack([np.ones((len(obs), 1)), Xo])
            lam = np.eye(Xo1.shape[1]); lam[0, 0] = 0.0
            # intercept anchored at the known undisrupted schedule margins via a pseudo-observation
            base = simulate(oracle.world, frozenset()).margins
            Xo1 = np.vstack([Xo1, np.r_[1.0, np.zeros(len(ids))]])
            Y = np.vstack([Y, [base[c] for c in cons]])
            W = np.linalg.solve(Xo1.T @ Xo1 + lam, Xo1.T @ Y)
            pred = np.hstack([np.ones((len(cands), 1)), X]) @ W
            r_emp = np.clip(1.0 - pred.min(axis=1), 0, 2) / 2
        else:
            r_emp = np.zeros(len(cands))
        seen_pairs = set()
        for o in obs:
            for a, b in combinations(sorted(o.perturbation), 2):
                seen_pairs.add((a, b))
        unc, nov = [], []
        for c in cands:
            pairs = list(combinations(sorted(c), 2))
            unc.append(1.0 if not pairs else sum(1 for p in pairs if p not in seen_pairs) / len(pairs))
            nov.append(1.0 - max((len(c & o.perturbation) / len(c | o.perturbation) for o in obs), default=0.0))
        if self.prior is not None:
            reach = self.prior.disruption_reach(ids)
            r_llm = np.array([self.prior.score(c, reach) for c in cands])
        else:
            r_llm = np.zeros(len(cands))
        plaus = np.array([1.0 / len(c) for c in cands])
        return {"llm": r_llm, "emp": r_emp, "unc": np.array(unc), "nov": np.array(nov), "plaus": plaus}

    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray:
        lo, hi = float(v.min()), float(v.max())
        return np.zeros_like(v) if hi - lo < 1e-12 else (v - lo) / (hi - lo)

    def rank(self, cands, oracle, top: int = 5) -> list[dict]:
        comp = self.components(cands, oracle)
        normed = {k: self._norm(v) for k, v in comp.items()}
        score = sum(self.w[k] * normed[k] for k in comp)
        jitter = np.array([self.rng.random() * 1e-9 for _ in cands])
        order = np.argsort(-(score + jitter))
        out = []
        for i in order[:top]:
            out.append({
                "perturbation": sorted(cands[i]),
                "score": round(float(score[i]), 4),
                "components_raw": {k: round(float(comp[k][i]), 4) for k in comp},
                "components_weighted": {k: round(float(self.w[k] * normed[k][i]), 4) for k in comp},
            })
        return out

    def propose(self, cands, oracle, found):
        ranked = self.rank(cands, oracle, top=1)
        self.last_decomposition = ranked[0]
        return frozenset(ranked[0]["perturbation"])


GT_WEIGHTS = {"llm": 0.35, "emp": 0.15, "cov": 0.35, "nov": 0.05, "cost": 0.10}


class GroupTestingAcquisition(AcquisitionPolicy):
    """CONFUZZ-R v2: acquisition as adaptive group testing.

    Open hypotheses H_t = disruption pairs and triples not contained in any passing query
    (under monotonicity a passing set clears every subset of it).

      A_t(p) = w_llm * n(R_LLM) + w_emp * n(R_emp) + w_cov * n(Cov) + w_nov * n(N) - w_cost * n(Cost)

    Cov   number of open hypotheses contained in p (pairs weighted 1, triples 0.5):
          a failing p localizes a minimal set by minimization, a passing p clears all of them
    Cost  (|p| - 1) / (d - 1): queries minimization would spend if p fails
    R_LLM coverage form: 0.5 * max_h prior_h * |p & h| / |h|
                         + 0.5 * open converging pairs in p / all open converging pairs
    R_emp, N as in v1. Candidates contained in a passing query are pruned (prune_passed).
    """
    name = "confuzz"
    prune_passed = True

    def __init__(self, seed: int, prior: SemanticPrior | None = None, weights: dict | None = None, name: str | None = None):
        super().__init__(seed, prior, dict(GT_WEIGHTS if weights is None else weights), name)

    def components(self, cands, oracle):
        base = super().components(cands, oracle)
        ids = oracle.world.disruption_ids
        passed = [o.perturbation for o in oracle.log if not o.critical]
        open_pairs = {frozenset(c) for c in combinations(ids, 2) if not any(set(c) <= q for q in passed)}
        open_tri = {frozenset(c) for c in combinations(ids, 3) if not any(set(c) <= q for q in passed)}
        cov, cost = [], []
        for c in cands:
            cov.append(sum(1 for x in combinations(sorted(c), 2) if frozenset(x) in open_pairs)
                       + 0.5 * sum(1 for x in combinations(sorted(c), 3) if frozenset(x) in open_tri))
            cost.append((len(c) - 1) / max(1, len(ids) - 1))
        out = {"emp": base["emp"], "cov": np.array(cov, float), "nov": base["nov"], "cost": np.array(cost)}
        if self.prior is not None:
            reach = self.prior.disruption_reach(ids)
            conv = [frozenset((a, b)) for a, b in combinations(ids, 2) if reach.get(a, set()) & reach.get(b, set())]
            conv_open = [x for x in conv if x in open_pairs]
            r = []
            for c in cands:
                hyp = max((float(h.get("risk_prior", 0)) * len(c & frozenset(h.get("candidate_set", []))) /
                           max(1, len(h.get("candidate_set", []))) for h in self.prior.hypotheses), default=0.0)
                g = (sum(1 for x in conv_open if x <= c) / len(conv_open)) if conv_open else 0.0
                r.append(0.5 * hyp + 0.5 * g)
            out["llm"] = np.array(r)
        else:
            out["llm"] = np.zeros(len(cands))
        return out

    def rank(self, cands, oracle, top: int = 5):
        comp = self.components(cands, oracle)
        normed = {k: self._norm(v) for k, v in comp.items()}
        sign = {"cost": -1.0}
        score = sum(sign.get(k, 1.0) * self.w[k] * normed[k] for k in comp)
        jitter = np.array([self.rng.random() * 1e-9 for _ in cands])
        order = np.argsort(-(score + jitter))
        return [{
            "perturbation": sorted(cands[i]),
            "score": round(float(score[i]), 4),
            "components_raw": {k: round(float(comp[k][i]), 4) for k in comp},
            "components_weighted": {k: round(float(sign.get(k, 1.0) * self.w[k] * normed[k][i]), 4) for k in comp},
        } for i in order[:top]]


class AdaptiveGroupTesting(Policy):
    """CONFUZZ-R v3: adaptive group testing over an explicit hyperedge hypothesis space.

    Model (monotone case): FAIL(S) = 1 iff some hidden minimal set m is contained in S.
    Hypothesis state H_t: every pair and triple of disruptions that is
      * not contained in any passing query (a passing S clears all its subsets), and
      * not explained by a found minimal set (h contains a found m).
    Each h has weight w_h: 1 under the uniform prior; 1 + lam * prior_h under an OpenAI
    semantic prior (prior_h in [0, 1], from Red's hypotheses and causal-graph convergence).

    Failure model: each unresolved h independently is a failing hyperedge with probability
    q * w_h, so P(fail | S) = 1 - prod_{h in H_t, h <= S} (1 - min(0.95, q * w_h)).
    q is estimated from this run's own evidence: q = (failing tests + 1) / (sum of weighted
    unresolved hyperedges covered at test time + 2) (Laplace-smoothed).

    Value and cost of testing S:
      V(S) = sum_{h in H_t, h <= S} w_h       weighted hypotheses resolved either way:
                                               a pass clears them, a fail localizes one by minimization
      C(S) = 1 + P(fail | S) * (|S| - 1)      the test plus expected linear-minimization probes
      A(S) = V(S) / C(S)                       expected weighted hypotheses resolved per query
    Ties broken by smaller |S|, then a seeded RNG. No tunable weights besides lam (fixed a priori, 3.0).
    """
    name = "agt"
    prune_passed = True

    def __init__(self, seed: int, prior: SemanticPrior | None = None, lam: float = 3.0, name: str | None = None):
        self.rng = random.Random(seed)
        self.prior, self.lam = prior, lam
        if name:
            self.name = name
        self._hw: dict[frozenset[str], float] | None = None
        self.last_decomposition: dict | None = None

    def hyperedge_prior(self, ids: list[str]) -> dict[frozenset[str], float]:
        """prior_h in [0,1]. Hypothesis of size 2-3: its risk_prior on that hyperedge; larger
        hypotheses spread 0.5 * risk_prior over their pairs and triples; causal-graph convergence
        (two disruptions reaching a shared node) gives a pair at least 0.3."""
        pr: dict[frozenset[str], float] = {}
        if self.prior is None:
            return pr
        for h in self.prior.hypotheses:
            cs = frozenset(h.get("candidate_set", []))
            rp = float(h.get("risk_prior", 0.0))
            if 2 <= len(cs) <= 3:
                pr[cs] = max(pr.get(cs, 0.0), rp)
            elif len(cs) > 3:
                for k in (2, 3):
                    for c in combinations(sorted(cs), k):
                        pr[frozenset(c)] = max(pr.get(frozenset(c), 0.0), 0.5 * rp)
        reach = self.prior.disruption_reach(ids)
        for a, b in combinations(ids, 2):
            if reach.get(a, set()) & reach.get(b, set()):
                pr[frozenset((a, b))] = max(pr.get(frozenset((a, b)), 0.0), 0.3)
        return pr

    def _weights(self, ids):
        if self._hw is None:
            pr = self.hyperedge_prior(ids)
            self._hw = {frozenset(c): 1.0 + self.lam * pr.get(frozenset(c), 0.0)
                        for k in (2, 3) for c in combinations(ids, k)}
        return self._hw

    def state(self, oracle: Oracle, found: list[frozenset[str]]):
        ids = oracle.world.disruption_ids
        hw = self._weights(ids)
        passed = [o.perturbation for o in oracle.log if not o.critical]
        H = {h: w for h, w in hw.items()
             if not any(h <= q for q in passed) and not any(m <= h for m in found)}
        # estimate q from history: weighted unresolved coverage at each search test
        fails, cover = 0, 0.0
        seen_passed: list[frozenset[str]] = []
        for o in oracle.log:
            if o.purpose == "test":
                cov = sum(w for h, w in hw.items() if h <= o.perturbation and not any(h <= q for q in seen_passed))
                cover += cov
                fails += o.critical
            if not o.critical:
                seen_passed.append(o.perturbation)
        q = (fails + 1) / (cover + 2)
        return H, q

    def score_all(self, cands, oracle, found):
        H, q = self.state(oracle, found)
        rows = []
        for c in cands:
            inside = [(h, w) for h, w in H.items() if h <= c]
            V = sum(w for _, w in inside)
            logpass = sum(math.log(1 - min(0.95, q * w)) for _, w in inside)
            pf = 1 - math.exp(logpass)
            C = 1 + pf * (len(c) - 1)
            rows.append((V / C, -len(c), self.rng.random(), c, len(inside), V, pf, C))
        rows.sort(reverse=True)
        return rows, len(H), q

    def rank(self, cands, oracle, top: int = 5, found=()):
        rows, nH, q = self.score_all(cands, oracle, list(found))
        return [{"perturbation": sorted(r[3]), "score": round(r[0], 4),
                 "unresolved_interactions_covered": r[4], "weighted_coverage": round(r[5], 3),
                 "p_fail_estimate": round(r[6], 3), "expected_queries_if_tested": round(r[7], 3),
                 "unresolved_interactions_total": nH, "q_hat": round(q, 4)} for r in rows[:top]]

    def propose(self, cands, oracle, found):
        top = self.rank(cands, oracle, 1, found)[0]
        self.last_decomposition = top
        return frozenset(top["perturbation"])


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------
def candidates(world: World, oracle: Oracle, found: list[frozenset[str]], prune_passed: bool = False) -> list[frozenset[str]]:
    """Shared rules for all policies, plus optional monotone pruning (a method feature):
    prune_passed drops any set contained in an already-passing query, assuming that removing
    disruptions cannot create a failure. The benchmark measures what that assumption costs."""
    passed = [o.perturbation for o in oracle.log if not o.critical] if prune_passed else []
    return [p for p in all_perturbations(world.disruption_ids)
            if p and oracle.known(p) is None and not any(m <= p for m in found)
            and not any(p <= q for q in passed)]


@dataclass
class EpisodeResult:
    policy: str
    world_id: str
    seed: int
    budget: int
    found: list[tuple[int, frozenset[str]]]   # (query index when minimization completed, set)
    queries: int
    log: list[Observation]


def run_episode(world: World, policy: Policy, budget: int, seed: int) -> EpisodeResult:
    oracle = Oracle(world, budget)
    found: list[frozenset[str]] = []
    found_at: list[tuple[int, frozenset[str]]] = []
    try:
        while oracle.remaining > 0:
            cands = candidates(world, oracle, found, getattr(policy, "prune_passed", False))
            if not cands:
                break
            p = policy.propose(cands, oracle, found)
            ob = oracle.query(p)
            if ob.critical:
                m, _ = minimize(oracle, p)
                if not any(m == f for f in found):
                    found.append(m)
                    found_at.append((oracle.used, m))
    except BudgetExhausted:
        pass
    return EpisodeResult(policy.name, world.id, seed, budget, found_at, oracle.used, oracle.log)


# ---------------------------------------------------------------------------
# Metrics (need hidden ground truth: evaluator side only)
# ---------------------------------------------------------------------------
def recall_curve(ep: EpisodeResult, minimal_sets, budgets: list[int]) -> list[float]:
    M = set(minimal_sets)
    return [len({m for (i, m) in ep.found if i <= b and m in M}) / len(M) for b in budgets]


def episode_metrics(ep: EpisodeResult, minimal_sets, budget: int) -> dict:
    M = set(minimal_sets)
    hits = [(i, m) for (i, m) in ep.found if i <= budget and m in M]
    high = {m for m in M if len(m) >= 3}
    return {
        "recall": len(hits) / len(M),
        "queries_to_first": min((i for i, _ in hits), default=None),
        "high_order_recall": (len({m for _, m in hits if m in high}) / len(high)) if high else None,
        # False-robust run: the world has hidden critical minimal failures and the run
        # ends its budget having found none of them, so on its evidence it would report
        # "no critical failure found".
        "false_robust": len(hits) == 0,
        "found_not_minimal": sum(1 for i, m in ep.found if i <= budget and m not in M),
        "queries_used": min(ep.queries, budget),
        "search_queries": sum(1 for o in ep.log if o.index <= budget and o.purpose == "test"),
        "localization_queries": sum(1 for o in ep.log if o.index <= budget and o.purpose == "minimize"),
    }
