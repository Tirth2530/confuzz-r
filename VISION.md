# CONFUZZ-R: prove-before-apply for AI operations agents

## The problem

Companies are starting to let AI agents change live operations: reschedule sessions, move stock between
warehouses, reroute shipments, roll out config changes. The dangerous failures are not the obvious ones.
They are second-order: a fix that solves today's incident and silently removes the slack another part
of the system was relying on.

We saw the same verification problem across both systems we built. In the conference, a cheap repair
can solve the immediate scheduling problem while removing slack elsewhere, so CONFUZZ-R attacks the
repair before it can reach a human. In the inventory network, RED found a genuine three-way infant
formula stockout; BLUE repaired it with a 100-unit transfer from North DC to City Hub, and exhaustive
verification found zero new regressions across the modeled 256-event space. The sandbox then surfaced
the dominant residual risk: carrier-strike scenarios remained far more fragile than the rest of the space.

## Why now

AI agents are already taking actions nobody intended. On September 24, 2026, Australia's prime minister
said an OpenAI model, asked to search for data on government medicine spending during internal testing,
got past restrictions on an old health statistics website and reached non-public files; OpenAI said its
models "took actions we did not intend", and the government said there was no evidence personal
information was accessed ([CBS News](https://www.cbsnews.com/news/australia-openai-rogue-model-hack-healthcare-system-sam-altman/),
[ABC News](https://abcnews.com/Technology/extreme-concern-openai-agent-hacked-australian-public-health/story?id=136707027),
[Nature](https://www.nature.com/articles/d41586-026-03024-z)). The same reporting describes unintended-access
incidents in testing involving models from other labs as well. This is an industry-wide problem, not one
vendor's.

CONFUZZ-R would not have stopped that specific incident: it was a model browsing the web during an
evaluation, and we gate operational changes made through tools. The lesson we take from it is narrower
and general: an agent pursuing a reasonable goal can take actions outside its intent, so the boundary
between an agent's intention and a real-world action must be enforced by infrastructure, not by a prompt.

## The product

A harness that sits between an agent's proposal and the live system and refuses to let the change
through until it has been attacked and verified:

1. **Discover**: reproduce the incident and shrink it to the minimal set of events that actually cause it.
2. **Repair**: an OpenAI agent proposes a fix that targets the causal mechanism; a solver makes it concrete.
3. **Attack**: a second OpenAI agent asks what the fix made newly vulnerable and tries to break it.
4. **Verify**: exact checking over the modeled event space, plus a sampled risk bound; the agent
   recomputes the evidence with its own code in a sandbox.
5. **Approve**: only a verified fix can even reach the human, and TrueForge stops it there.

Every rule that matters (budgets, phase order, "no apply without verification") is enforced in the
tool server, not in a prompt. The model reasons; the harness decides what it is allowed to do.

**Category:** a verification gateway between autonomous operations agents and real systems.
CI/CD verifies code before it ships. CONFUZZ-R verifies an agent's operational change before it is applied.

```
        AI AGENTS (propose changes)
                  |
        +-------------------+
        |     CONFUZZ-R     |   attack  ->  verify  ->  certify  ->  approval eligibility
        +-------------------+
                  |
        TrueForge approval gate  ->  human  ->  REAL SYSTEMS
```

## The proof package: an Operational Safety Certificate

Every change that reaches a human ships with one (today: `verify_repair` + `export_evidence` + the approval card):
the change and its target; the incident's minimal cause; repairs rejected and the counterexample that
killed each; the space tested (all 256 modeled combinations); regressions found (must be 0); residual
failures that remain; a sampled failure bound under demand uncertainty; the agent's own sandbox analysis
(top residual-risk drivers, demand envelope); model assumptions; and who approved it.

## Three modes

- **Incident commander (built):** something broke; find a fix, attack it, verify it, gate it.
- **Preflight (same engine):** an agent proposes a change with nothing broken yet; attack it before it ships.
- **Watchdog (next):** the certificate records the conditions it was verified under (modeled disruptions,
  demand band). When live conditions leave that envelope, the certificate is invalidated and the change
  is sent back for re-verification. This turns a one-time check into continuous assurance.

## Proven today: one engine, two domains

| | Conference operations | Inventory network |
|---|---|---|
| System | 12 sessions, 6 rooms, 4 slots, 2 zones | 3 sites, 4 SKUs (cold-chain insulin, formula), 7-day horizon |
| Disruptions | room loss, demand surge, route closure, AV failure, speaker delay | supplier delay, DC power outage, recall, dock closure, carrier strike, heatwave |
| Minimal failures (exhaustive) | 2 pairwise, 3 third-order | 3 pairwise, 1 third-order |
| Repair / Critic result | cheap room swap can create new scheduling regressions | 100-unit formula transfer survives Critic attacks and introduces 0 modeled regressions |
| Verified result | standby-room repair verified within the modeled space | transfer 100 FOR North DC -> City Hub; exact failures 128 -> 96, 0 regressions |

Same MCP server, same agents, same TrueForge approval gate, same live console. Switching domains is
`reset_session(world_id="inventory")`.

## Adding a domain

A domain is one module (`confuzz/inventory.py` is about 300 lines): a deterministic simulator that returns
per-constraint margins, the list of atomic disruptions, the operator's playbook, and the action set
with costs. The discovery, repair, attack, verification, certificate, approval and console layers are
reused unchanged (`confuzz/domain.py` is the whole dispatch layer).

## Roadmap

1. **Live digital twins instead of hand-built worlds.** Read the real schedule or stock position through
   MCP connectors (event platforms, ERP and WMS systems, Shopify) and build the simulator state from it.
   The approval gate then writes the verified change back through the same connector.
2. **Larger event spaces.** Exhaustive verification works at 8 disruptions (256 combinations). Beyond
   that, verification becomes budgeted adversarial search with statistical bounds, which is where the
   Critic's measured advantage matters most. Our benchmarks already measure methods by queries spent,
   not by whether enumeration is possible.
3. **Next markets, in order:** cloud infrastructure and SRE (an agent shifts traffic to region B; what if
   B loses a zone?), airline and rail recovery (an aircraft swap fixes one flight and times out a crew
   three hours later), hospital operations (a nurse reassignment fixes one unit and breaks ICU coverage
   under two admissions). Same four inputs each time: system state, disruptions, invariants, actions.
4. **A hybrid Critic.** Our pre-registered data shows deterministic locality heuristics are strong on
   direct regressions and blind on indirect ones; the product Critic routes between them.

## Where this ends up

Organizations will run many agents that change live systems: stock, schedules, infrastructure, staffing,
routing. Something has to sit between an agent's intention and a real-world action, try to break the
change, and refuse to pass it on without evidence. That is CONFUZZ-R.

## Why TrueForge and OpenAI are structural

- **OpenAI** does the work no solver does: modeling how events interact, choosing a repair that breaks
  the mechanism, and reasoning adversarially about what a fix made fragile. We measure each role against
  a deterministic baseline instead of assuming it helps.
- **TrueForge** provides what makes this safe to run: MCP tool execution against the live system, subagents
  that challenge each other with clean contexts, a sandbox for the agent's own verification code, persistent
  sessions for multi-phase investigations, and the approval gate that no prompt can talk its way past.

## Honest limits

Both demo worlds are hand-designed and disclosed as such; benchmarks run on generated conference worlds.
VERIFIED means "no regressions across the modeled events", not "safe". Pruning assumes monotone event
effects (measured true on all benchmark worlds). The inventory network keeps a high residual risk
(carrier-strike combinations), and the approval card says so.
