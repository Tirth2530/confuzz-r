# TrueForge setup (checkpoint 1)

Requires Node 22.14+ and Python 3.11+ with `mcp`, `numpy`, `scipy` (and `openai` for benchmark priors).

1. `npx @truefoundry/trueforge@latest` → UI at http://localhost:8790. Add an OpenAI model with your credits.
2. Add an MCP connector, type **stdio**:
   - command: `python` (the interpreter that has `mcp`, `numpy`, `scipy`)
   - args: `-m confuzz.mcp_server`
   - working directory: this repo root (or set `PYTHONPATH` to it)
   - env: `CONFUZZ_STATE=<repo>/runs/state.json`
3. Create an agent: OpenAI model, the `confuzz-r` connector with all tools, instructions =
   `trueforge/AGENT_INSTRUCTIONS.md`. Enable subagents if your version offers it.
4. Approval gate on `apply_repair` (and `reset_session`):
   - if the tool picker shows a per-tool shield or approval toggle, turn it on for those two tools;
   - otherwise: `python trueforge/set_approval.py --agent <id> --dry-run`, check, then run without `--dry-run`.
   `apply_repair` is also annotated `destructiveHint=true`, which TrueForge's default gates. The server
   refuses unverified repairs regardless.
5. Checkpoint test in chat: "Call get_conference, then run_experiment with D1 and report the margins."
   Pass = the agent returns the real margins (capacity 0.045, transition 0.8 for D1 on the demo world).
6. Approval test: "Call apply_repair with repair_id R1." Pass = TrueForge pauses for Allow/Deny,
   and on Allow the server answers RULE_VIOLATION (nothing verified yet).
7. Subagents: ask the agent to delegate the RED brief to a subagent; confirm in Sessions that a
   subagent ran with its own context. If not supported, keep role labels in one agent.
8. Sandbox (optional, needs Daytona key): ask the agent to load `experiments/results/*.json` into
   the sandbox and run `experiments/figures.py`. If Daytona is unavailable, say so; do not fake it.

Reset between demo runs: ask the agent to call `reset_session` (approval-gated), or delete `runs/state.json`.
