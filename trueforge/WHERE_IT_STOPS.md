# Where CONFUZZ-R stops

The job: on conference day, find a schedule fix for a live incident that does not break anything
else, and change the live schedule only with an organizer's approval.

| Action | Who may do it | Gate | Why |
|---|---|---|---|
| Read the schedule, disruptions, trace, evidence | any agent | none (read-only tools) | no effect on anything |
| Run simulated disruptions, minimize failures | RED | server budget (20 queries) | simulation only; budget stops runaway search |
| Propose a fix | BLUE | server: needs a reproduced minimal failure; after a rejection, must state the counterexample and what changed in its reasoning | stops blind retries |
| Attack a fix | CRITIC | server budget (10 attacks) | simulation of the patched schedule only |
| Certify a fix | engine (verify_repair) | server: needs at least one critic attack; exhaustive check of all 256 modeled combinations | the LLM never marks its own fix safe |
| Recompute the certificate | agent, in the TrueForge sandbox | Daytona sandbox, disposable | generated code runs isolated, can't touch the host or the state |
| **Change the live schedule** | **organizer only** | **two independent gates: the server refuses anything not VERIFIED, and TrueForge pauses `apply_repair` for Allow/Deny with the evidence on the card** | irreversible for attendees |
| Reset the session | organizer only | TrueForge approval | discards evidence |

**What if the agent is wrong?**
- Wrong hypothesis: costs budget, nothing else; the simulator decides outcomes.
- Wrong fix: the critic attacks it and exact verification checks every modeled combination. In the demo,
  the first fix (cost 2) fixed its target but made four single disruptions fail alone. It was rejected
  before anyone saw an approval card.
- Wrong certificate (an engine bug): the agent recomputes it from raw rows in the sandbox and reports mismatches.
- Wrong approval click: Deny leaves state byte-identical (fingerprint check); Allow applies exactly once.
- Outside the model: VERIFIED means "no regressions over the 8 modeled disruptions", not "safe". The
  card lists what is still at risk, including the residual 4-way failure in the demo.
