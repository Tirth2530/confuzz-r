# 5-minute demo

Setup: MCP server running (`--http`), TrueForge chat on the left, the console
(http://127.0.0.1:8800/console) on the right, `python scripts/checkpoint.py reset`, fallback recording ready.

**0:00 The job (25 s).** "It's 10am at DevConf. Something always breaks: a leak, a dead projector, a broken lift.
The event supervisor has to fix the schedule in minutes, and the obvious fix can quietly break something else.
CONFUZZ-R is the supervisor's agent: it finds a fix, attacks its own fix, and touches nothing until they approve."

**0:25 Report the incident (30 s).** On the console: Room = Grand Hall A, Issue = Room closed, note "water leak", Report.
The incidents panel says *absorbed*: the playbook moved 4 talks to Rooms E and B. "Most tools stop here. Looks fine."

**0:55 What it cost (60 s).** In TrueForge: *"Handle the reported incident."* RED checks what could still go wrong.
Console experiments pane: before the leak, every single problem was survivable; now a broken projector in Room E
or losing Room B would each break the day, and a keynote surge plus a late speaker breaks it too (subset table:
each alone PASS, together FAIL).

**1:55 The first fix is wrong (60 s).** BLUE proposes the cheap fix: swap two talks (cost 2). CRITIC asks what it made
fragile, tests "speaker 5 minutes late" alone: transition -2 minutes. Rejected, before any human saw it. Red ring on the console.

**2:55 The fix that holds (45 s).** BLUE states what it learned and proposes: open the standby ballroom, move the keynote
there (cost 6). CRITIC fails to break it. verify_repair: 0 regressions over all 256 combinations, failing combinations
216 -> 32. The agent's own sandbox code double-checks the numbers and names what is still at risk.

**3:40 Where it stops (45 s).** apply_repair: TrueForge pauses with the evidence. Deny: nothing changes. Allow: applied
once. The console's "who to notify" panel lists the 4 speakers and their new rooms.

**4:25 Bigger than conferences (35 s).** "Same engine, same agents, same gate, a different business": switch to the
inventory network (recorded real run). RED found a genuine three-way formula stockout. BLUE proposed a 100-unit
North-DC-to-City-Hub transfer; CRITIC attacked it, exhaustive verification found 0 new regressions across 256 modeled
combinations, and the sandbox surfaced carrier strike as the dominant residual-risk driver. "Before an AI changes a
real system, it has to prove the change won't create a worse problem somewhere else."

If live fails: switch to the recording and keep the same beats.
