# TrueForge setup (checkpoint 1)

Verified against the TrueForge source, `truefoundry/trueforge` main @ `1090880` (Oct 23 2026):
agent spec schema, tool-approval selectors, subagent docs, MCP transport and outbound URL guard.
Not yet verified by running it (this build machine cannot reach npm).

Requirements: Node 22.14+, Python 3.11+ with `mcp numpy scipy openai`.

## 1. Start the CONFUZZ-R MCP server (HTTP)

TrueForge only connects to **remote** MCP servers by URL (Streamable HTTP or SSE). There is no stdio connector.

```bash
cd confuzz-r
CONFUZZ_STATE=$PWD/runs/state.json python -m confuzz.mcp_server --http --port 8800
# serves http://127.0.0.1:8800/mcp
```

## 2. Start TrueForge with loopback allowed for that one host

TrueForge's outbound URL guard (`NETWORK_POLICY_ENABLED`, default on) blocks loopback and private
addresses. Allow only our host instead of turning the policy off:

```bash
OUTBOUND_URL_ALLOWED_HOSTS='["127.0.0.1"]' npx @truefoundry/trueforge@latest   # UI: http://localhost:8790
```

Use `127.0.0.1`, not `localhost` (a dotless hostname is denied unless listed).

## 3. Connect model and MCP server

- Settings → Models: add OpenAI with your credits.
- Settings → Connectors → **Add MCP Server**: URL `http://127.0.0.1:8800/mcp`, no auth.

## 4. Create the agent

- Model: your OpenAI model. Instructions: paste `trueforge/AGENT_INSTRUCTIONS.md`.
- **Dynamic sub-agents: on** (default). The root agent spawns RED, BLUE and CRITIC with the built-in
  `create_sub_agent` tool. Subagents share all MCP tools, cannot nest, and their approval-gated calls
  still pause for the human. Role separation is by instructions; the server's phase rules (critic
  attack before verify, verified before apply, revision fields after a rejection) hold regardless of which agent calls.
- **Select MCP Tools → confuzz-r**, enable all tools, then set approvals. This step matters:

  TrueForge's default is `require_approval_for_tools: ["@write", "@destructive"]`. `@write` = any tool
  with `readOnlyHint=false`. Our experiment tools (`run_experiment`, `minimize_failure`, `attack_repair`,
  `propose_repair`, `verify_repair`, `submit_interaction_model`, `explain_interaction`) honestly declare
  `readOnlyHint=false` because they change session state, **so by default every experiment would pause
  for approval**. Turn the shield **off** for those (the "Other Actions" header switch does it in one
  click) and keep it **on** for `apply_repair` and `reset_session` (the "Destructive Actions" group).
  Target state: `require_approval_for_tools = ["apply_repair", "reset_session"]`.
  Check it on the agent's **Overview** tab (MCP Servers & Tools shows the shield per tool).
  API fallback: `python trueforge/set_approval.py --agent <id> --dry-run`, then without `--dry-run`.

## 5. Checkpoint tests (in chat)

1. "Call get_conference, then run_experiment with D1 and report the margins."
   Pass: real margins come back without an approval prompt (demo world: capacity 0.045, transition 0.8).
2. "Call apply_repair with repair_id R1 and evidence_summary 'test'."
   Pass: TrueForge pauses with Allow/Deny showing the arguments; on Allow the server returns
   RULE_VIOLATION (nothing is verified). This proves both gates: the harness and the server.
3. "Delegate this to a subagent: call rank_experiments and report the top candidate."
   Pass: Sessions shows a subagent run with its own context.
4. Reset: delete `runs/state.json` or call `reset_session` (approval-gated).

## 6. Sandbox (no Daytona needed in standalone mode)

The docs say Daytona is the only provider, but the source (`packages/trueforge/src/sandbox/localRuntime.ts`)
shows that **standalone mode (`npx`) falls back to a local sandbox automatically** when no Daytona provider
is configured and the platform probe passes. It isolates commands with `@anthropic-ai/sandbox-runtime`
(Seatbelt on macOS, bubblewrap + seccomp on Linux) in a per-session directory with a Python venv.

- Check: `curl -s http://localhost:8790/api/v1/capabilities` should show `"sandbox":{"enabled":true}`.
- Enable **Sandbox** on the agent (Create/Edit agent → Sandbox on).
- In the run, the VERIFY phase calls `export_evidence` (87 KB, offloaded to a sandbox file) and the agent
  writes and runs its own Python to recompute the certificate. That is the sandbox moment for the demo.
- If capabilities shows sandbox disabled, the probe failed on this machine: say so in the demo, do not fake it.
  Daytona remains the alternative (API key scopes `write:sandboxes`, `write:snapshots`, `manage:secrets`).

## 7. Live console

With the server in HTTP mode, open **http://127.0.0.1:8800/console**. It polls the same state file the
tools write and shows only what agents have paid for: experiments, discovered minimal failures, repairs,
critic counterexamples, certificates, approval status, trace. Put it next to the TrueForge chat on stage.
