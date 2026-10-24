# 5-minute demo

Setup before going on stage: server running, TrueForge open, `python scripts/checkpoint.py reset`,
the live console (http://127.0.0.1:8800/console) beside the TrueForge chat, the hypergraph page in a second tab, fallback recording ready.

**0:00 The job (30 s).** "It's 10am at DevConf. Hall A just lost its HVAC. The keynote is surging, a lift
is out, the speaker is running late. Each problem alone is fine. Together they break the schedule.
Organizers want to hand this to an agent, but the agent must never change the live schedule on its own."

**0:30 Discover (60 s).** Ask a judge to pick the incident from the 8 disruptions (or use D1+D2+D4).
Prompt: *"Incident report: <their picks> are happening now. Run the CONFUZZ-R investigation."*
Show RED's interaction model, the scorer's "covers N unresolved interactions", run_experiment returning
real margins, minimize_failure, and the subset table: every pair PASS, the triple CRITICAL.

**1:30 The first fix is wrong (75 s).** BLUE proposes the cheapest fix (swap two sessions, cost 2).
It fixes the target. CRITIC asks what the swap made vulnerable, tests "speaker delay alone", and the
simulator returns -2 minutes transition: rejected. Cut to the hypergraph "After repair R1" tab:
four red rings, four disruptions that used to be harmless now break the schedule alone.

**2:45 The second fix (45 s).** BLUE states its diagnosis and what changed in its reasoning, proposes
standby Ballroom F plus moving the keynote (cost 6). CRITIC attacks it and fails to break it.
verify_repair: 0 regressions over all 256 combinations, failing combinations 124 -> 16, sampled
failure rate 65.5% -> 20.3% (95% bound 23.8%). Sandbox: the agent's own code recomputes the numbers.

**3:30 Where it stops (60 s).** apply_repair hits the TrueForge approval card with the evidence.
Click Deny: state unchanged. Ask again, Allow: applied once. Say what is still at risk (the residual
4-way failure; 8 modeled disruptions).

**4:15 Same harness, second domain (15 s).** Switch the console to the inventory network
(`reset_session(world_id="inventory")`, run recorded earlier from a real agent session): the 200-unit formula
transfer fixes City Hub, and the Critic shows a South DC power outage alone now overflows City Hub storage.
"Same agents, same gate, a different business."

**4:30 How we know (30 s).** One slide: the Critic result on indirect regressions (strongest frozen
number), with the deterministic baseline next to it. "We measured where the LLM helps and where code is enough."

If live fails at any point: switch to the recording, keep narrating the same beats.
