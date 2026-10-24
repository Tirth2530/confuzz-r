# TrueForge checkpoint: the 9-step minimal path

Do these in order. Do not spend frozen OpenAI calls until all 9 pass. Record evidence where marked
(screenshot = TrueForge window; log = terminal output). Setup details: `SETUP.md`.

Terminal A (server):
```bash
export CONFUZZ_STATE=$PWD/runs/state.json
python scripts/checkpoint.py reset
python -m confuzz.mcp_server --http --port 8800
```
Terminal B (state inspection): `export CONFUZZ_STATE=$PWD/runs/state.json` then `python scripts/checkpoint.py status` whenever asked.

Terminal A prints `[confuzz] state file: ...` at startup and one line per tool call (tool, ok/RULE, discovery and critic budgets, applied, state file). If the agent says it ran an experiment but no `run_experiment ok` line appears, the agent did not call the tool. If the line appears but `status` disagrees, the two terminals point at different state files: compare the paths.

| # | Do | Pass when | Evidence |
|---|---|---|---|
| 1 | `OUTBOUND_URL_ALLOWED_HOSTS='["127.0.0.1"]' npx @truefoundry/trueforge@latest` | UI loads at http://localhost:8790 | log |
| 2 | Settings → Connectors → Add MCP Server `http://127.0.0.1:8800/mcp` | 12 tools listed; terminal A shows `POST /mcp 200` | screenshot of tool list |
| 3 | New agent, OpenAI model, `AGENT_INSTRUCTIONS.md`, shields: on only `apply_repair`, `reset_session`. Chat: *"Call get_conference, then run_experiment with disruptions [D1] and report the robustness margins."* | runs with **no approval prompt** | screenshot |
| 4 | (same turn) | agent reports capacity **0.045**, transition **0.8**; `status` shows discovery 1/20 | screenshot + status |
| 5 | Chat: *"Spawn a CRITIC subagent: it must call run_experiment with [D1, D2, D4] and return the violations it got."* | Sessions shows a subagent run; it returns a **transition** violation for Patel (gap 15 - delay 5 - travel 12) | screenshot of subagent in Sessions |
| 6 | Chat: *"Call apply_repair with repair_id R1 and evidence_summary 'test'."* | TrueForge pauses (Allow/Deny). Click **Allow**: server answers `RULE_VIOLATION ... unknown repair R1` (nothing verified). `status` fingerprint unchanged | screenshot of the refusal + status |
| 7 | Terminal B: `python scripts/checkpoint.py prepare-verified` (SCRIPTED state: R1 rejected, R2 verified; marked as scripted in state and trace). Chat: *"Call get_trace, then apply_repair for R2 with a full evidence_summary."* | approval card shows `apply_repair` with R2 and the evidence text | **screenshot of the approval card** |
| 8 | Click **Deny** | agent is told it was denied; `status` fingerprint identical to step 7's, applied `[]` | screenshot + status before/after |
| 9 | Ask again, click **Allow** | server returns `applied: R2`; `status` applied `['R2']` count 1; asking a third time is refused (`R2 is APPLIED`) | screenshot + status |

After 9 passes: `python scripts/checkpoint.py reset`, run one full live episode (prompt: *"Run the CONFUZZ-R
investigation on the demo conference."*), then
`python scripts/export_hypergraph.py --state runs/state.json && python scripts/build_hypergraph_page.py`
to draw the real session. Then spend the frozen OpenAI calls.

If a step fails, paste the exact error. Likely causes: step 2 refused = URL guard (check the env var and
that you used 127.0.0.1); step 3 prompts for approval = a shield is still on for that tool; step 5 no
subagent = Dynamic sub-agents toggle off.
