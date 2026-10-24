# CONFUZZ-R

An adversarial verification engine for conference operations. OpenAI agents running on TrueForge
search for **interaction failures** (disruptions that are harmless alone but break the schedule
together), propose repairs, attack their own repairs, and stop at a human approval boundary.
A deterministic simulator decides every outcome.

**OpenAI reasons. TrueForge orchestrates. Tools measure. Humans authorize.**

```
TrueForge agent loop (OpenAI)
  RED     submit_interaction_model -> rank_experiments -> run_experiment -> minimize_failure -> explain_interaction
  BLUE    propose_repair  (strategy type; deterministic solver instantiates the cheapest action that fixes the target)
  CRITIC  attack_repair   (counterexample = passes on original, fails on patched -> REJECTED)
  RISK    verify_repair   (exact over all 2^8 perturbations + same-condition sampled certificate)
  HUMAN   apply_repair    (TrueForge require_approval_for_tools; server refuses anything not verified)
                    |
          confuzz-r MCP server (stdio) -- budgets, phase rules, persisted state, no ground truth
```

## Run

```bash
python -m unittest discover -s tests -v          # integrity tests
python scripts/scripted_episode.py               # engine loop over real stdio MCP (scripted, no LLM)
python experiments/bench.py --split dev --conditions random,random_pruned,agt_uniform --tag my_dev
OPENAI_API_KEY=... python experiments/bench.py --split frozen \
    --conditions random,random_pruned,agt_uniform,agt_openai --tag v3_frozen   # run ONCE
python experiments/figures.py experiments/results/v3_frozen.json
OPENAI_API_KEY=... python experiments/critic_bench.py --split frozen \
    --conditions random,size_ordered,diff_guided,openai_critic --tag critic_frozen          # run ONCE
```
TrueForge wiring (verified against TrueForge source; HTTP MCP, loopback allowlist, approval defaults): `trueforge/SETUP.md`. Agent instructions: `trueforge/AGENT_INSTRUCTIONS.md`.

## Problem and definitions

- **World**: schedule (12 sessions, 5 rooms + 1 standby, 4 slots, 2 zones), 8 atomic disruptions
  (demand surge, room loss, step-free route closure, AV failure, speaker delay).
- **Perturbation**: any subset of the 8 disruptions (256 total). The space is bounded **on purpose**:
  an offline evaluator can enumerate it and know the true failure structure. At 20 binary disruptions
  it would be 1,048,576 sets, and real systems add categorical and continuous dimensions, so the
  methods are evaluated by queries spent, not by whether enumeration is possible here.
- **Simulator** `s' = F(s, delta)`: applies disruptions, re-hosts displaced sessions with a fixed
  operations playbook, returns a normalized margin per constraint (capacity, room/speaker overlap,
  AV, accessibility, transition, hosting). Critical failure = any margin < 0. No failure is coded;
  `tests/` checks the simulator never references a disruption id.
- **Minimal failing set**: fails critically while every proper subset passes. Ground truth
  `M(w)` = all minimal failing sets (evaluator only; `confuzz/groundtruth.py`, never imported by
  agent-facing code, enforced by a test).
- **Minimal Failure Recall@B** = |found ∩ M(w)| / |M(w)|, where B counts **every** simulator query:
  search tests, confirmations and minimization probes. Repeating a query is free for every method.
- **False-robust run** (per world-seed run, worlds with ≥1 hidden minimal failure): the run spends
  budget B and finds none of them, so on its own evidence it would report "no critical failure found".
  Reported as a rate over runs. Not a real-world safety probability.
- **Monotonicity**: all 41 benchmark worlds (demo + 20 dev + 20 frozen) were measured monotone
  (adding a disruption never turns a failure into a pass). Pruning assumes this; **the engine's
  pruning is only valid under monotone disruption effects**. We have no non-monotone world class yet.

## Research history (kept, not overwritten)

1. **Generator v1** produced fragile worlds: 19/20 had only first-order failures, so checking single
   disruptions first won. Population redefined *a priori* to the research question: base schedule
   passes, every single disruption passes, ≥1 failure exists. Generated worlds pre-mitigate known
   single points of failure (step-free attendees kept out of the closable zone, no AV-dependent
   session in the AV-fragile room). The demo world is hand-designed and disclosed as such.
2. **CONFUZZ-R v1 lost to random** on the frozen worlds (`experiments/results/v1_frozen_historical.json`):
   Recall@10 random 0.57, v1 0.30, size-ordered 0.07; median queries to first failure 7 / 10 / 17.
   Diagnosis: with minimization in the loop, large random sets act as **group tests**: one test covers
   many pairs, a failure is then localized in about |S| - 1 probes. v1's plausibility term 1/|S|
   pushed it toward small sets, the opposite of what works.
3. **Reformulation as adaptive group testing** (`AdaptiveGroupTesting`; not our invention, see
   combinatorial group testing and hidden hyperedge discovery). Explicit hypothesis state: unresolved
   pairs and triples; a passing test clears every hyperedge it contains; a found minimal set explains
   its supersets. `A(S) = V(S) / (1 + P(fail|S)(|S|-1))`: weighted unresolved hyperedges resolved per
   expected query. One parameter, prior strength λ = 3, fixed before any run. Developed on dev worlds only.
4. **v3 does not beat random + pruning on development worlds** (`experiments/results/v3_dev.json`):

| Method (dev, 20 worlds x 10 seeds) | R@10 | R@15 | R@20 | R@30 | False-robust@10 | Median Q to 1st | Search / localization queries (first 20) |
|---|---|---|---|---|---|---|---|
| Random | 0.383 | 0.584 | 0.730 | 0.914 | 0.000 | 6 | 10.8 / 9.2 |
| Random + pruning | 0.386 | 0.590 | 0.779 | 0.949 | 0.035 | 7 | 10.0 / 10.0 |
| v3 group testing, uniform prior | 0.355 | 0.355 | 0.710 | 0.941 | 0.000 | 9 | 3.5 / 15.8 |

   Paired Wilcoxon, v3 minus random + pruning: recall@20 −0.069 (p = 3.8e-8). Localization consumes
   about 80% of v3's budget; in 8-disruption worlds with 2-3 hidden pairs, random 4-element bundles
   are already near the information floor. **Frozen evaluation of v3 has not been run.** It will run
   once, with the OpenAI-prior condition, and be reported whatever it shows.

## The open sponsor question

> Does an OpenAI semantic prior over interaction hyperedges reduce the simulator queries adaptive
> group testing needs? Condition `agt_openai` vs `agt_uniform`, same worlds, seeds and budget.

Red receives only the public spec (`World.public_spec()`): schedule, rooms, zones, travel, AV and
step-free needs, disruption definitions, playbook, constraint definitions. Never ground truth.
Benchmark priors are produced by **direct OpenAI API calls** (mode recorded as `direct_openai_api`)
with on-disk caching; live demo runs happen inside TrueForge. The two are never conflated.

## Sponsor gate (hard rule from here on)

Every improvement must pass two gates. Research: does it make verification more rigorous or
measurable? Sponsor: does it make OpenAI reasoning or a native TrueForge capability more
structurally necessary? Improvements that move intelligence or orchestration into plain Python
are not prioritized. Each OpenAI role has a measurable ablation against a no-LLM alternative:

| OpenAI role | No-LLM alternative | Metric | Status |
|---|---|---|---|
| Red: semantic prior over interaction hyperedges | uniform prior (same group-testing engine) | Minimal Failure Recall@B | needs key; frozen run pending |
| Critic: targeted attack on a proposed repair | random, size-ordered, **diff-guided heuristic** | queries to first counterexample, Success@k | needs key; frozen run pending |
| Blue: repair strategy choice | cheapest fixing action from the solver | verified-repair rate, cost | not built yet |

## Critic experiment

Scenario: a world, one of its minimal failures, and one of the 3 cheapest single repairs that fix it
but introduce at least one regression (hidden answer: the regression set, by exhaustive enumeration).
Each attack is one simulator query on the patched schedule. Scenario sets: 200 worlds per split
(214 dev, 213 frozen scenarios), fixed before any Critic condition ran on frozen data. Earlier
definitions (20 and 60 worlds) gave too few frozen indirect scenarios (5 total, then 7 indirect;
counts only, no method run) and were superseded.

**Pre-registered DIRECT / INDIRECT split** (`confuzz/scenarios.py`, simulator-derived, no LLM):
a minimal regression is DIRECT if any violation the simulator reports for it lands on a session,
room or speaker the repair changed; otherwise INDIRECT (the failure surfaces on untouched entities,
via consumed or freed slack and playbook choices). A scenario is DIRECT if any minimal regression is.
Frozen: 192 direct, 21 indirect.

Development results, no-LLM conditions (`experiments/results/critic_dev_noLLM.json`, 10 seeds):

| Class | Critic | Success@1 | @3 | @10 | Median queries |
|---|---|---|---|---|---|
| Direct (n=185) | random | 0.12 | 0.43 | 0.84 | 4 |
| Direct (n=185) | diff_guided | 0.37 | 0.70 | 0.86 | 2 |
| Indirect (n=29) | random | 0.12 | 0.36 | 0.66 | 6 |
| Indirect (n=29) | size_ordered | 0.06 | 0.07 | 0.32 | 16 |
| Indirect (n=29) | diff_guided | 0.00 | 0.01 | 0.28 | 21 |

**Pre-registered hypotheses for the OpenAI Critic (frozen, run once):**
- H-C1 (primary): on INDIRECT frozen scenarios, OpenAI Critic Success@3 exceeds both random and diff-guided.
- H-C2: on DIRECT frozen scenarios, OpenAI Critic Success@3 is at least diff-guided's.
- Analysis: Success@k per class with exact counts; paired comparison per scenario (OpenAI's list is
  deterministic per scenario, so its unit is the scenario: n = 21 indirect). With n = 21, H-C1 is
  powered only for large effects; a null result will be reported as such.

## Blue experiment

Scenario: (world, target minimal failure); 20 dev / 20 frozen worlds (63 dev scenarios).
Metrics, all exact over the 256 modeled perturbations (definitions in `confuzz/blue_eval.py`):
intervention cost C, residual exposure E (fraction of perturbations that still fail), worst-case
residual loss W (max severity), counterexample exists, verified, proposals to first verified
(cap K = 5). Reference frontier: every verified plan of 1-2 actions, Pareto over (C, E).
Pre-registered scalar: J_λ = C + λ · (residual failing perturbations), primary λ = 0.05,
sensitivity λ ∈ {0.02, 0.10}; J-regret = J(outcome) - best J among reference plans.

Deterministic cheapest-first baseline on dev (`experiments/results/blue_dev_deterministic.json`):
verified 54/63 (7 scenarios have no verified plan of ≤2 actions, so
abstaining is correct there); first proposal has a counterexample 21% of the time;
median cost 5; mean exposure 0.207; mean worst loss 1.31;
mean proposals to verified 1.30; J-regret = 0 in 67% of
scenarios at λ = 0.05 (mean regret 0.43).

**Pre-registered hypothesis for OpenAI Blue:** H-B1: at λ = 0.05, mean J-regret of OpenAI Blue's
verified outcome is lower than cheapest-first's, with verified rate no lower. Report also the
head-to-head counts (dominates / dominated / equal / incomparable on (C, E)) and all λ.

## Failure hypergraph

`scripts/export_hypergraph.py` exports minimal failing sets per stage (hidden truth, discovered,
after each repair) from engine computation; `scripts/build_hypergraph_page.py` renders
`viz/failure_hypergraph.html`. With `--state runs/state.json` it draws a real TrueForge session.

## Demo world (hand-designed, every number below is computed)

Minimal failures (exhaustive): {D2,D5}, {D2,D6} pairwise; {D1,D2,D3}, {D1,D2,D4}, {D3,D5,D6} third-order.
Scripted engine check (`scripts/scripted_episode.py`, no LLM): on {keynote surge, Hall A loss, speaker
delay} the cheapest fixing repair (swap S10/S12, cost 2) is rejected because *speaker delay alone*
becomes a new failure; activating standby Ballroom F and moving the keynote there (cost 6) passes
exact verification (0 regressions over 256 perturbations; failing perturbations 124 -> 16) and the
same-condition sampled certificate (400 draws, ±10% demand: failure rate 65.5% -> 20.3%, 95%
Clopper-Pearson upper bound 23.8%). **Residual risk remains**; VERIFIED means "no regressions and
target fixed on the modeled space", not "safe".

## Certificate semantics

- Exact part: exhaustive over the 256 modeled perturbations at nominal demand.
- Sampled part: i.i.d. draws (uniform perturbation, per-session demand × U(0.9, 1.1)), each run on
  original and patched worlds with identical inputs. Clopper-Pearson bound applies to that sampling
  process only. CVaR95 is flagged NOISY when the tail has < 30 observations.
- Statuses: VERIFIED_ON_TEST_SET, INSUFFICIENT_EVIDENCE, REPAIR_REJECTED. No universal certification.

## Limitations

Bounded 8-disruption model; monotone worlds only; uniform perturbation sampling is not a real
disruption frequency; severity and repair-cost weights are operator choices; greedy 1-minimal
reduction can return one minimal set while masking others in the same failing group (later search
continues; rediscoveries are not counted); generated worlds share one structural template.
