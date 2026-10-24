"""SCRIPTED episode over stdio MCP: no LLM, hard-coded tool arguments. Verifies the engine loop
end to end (discovery -> minimization -> repair -> critic -> rejection -> revision -> verification).
Its output is an engine check, NOT an agent run, and must never be presented as one.
Stops before apply_repair: approval belongs to a human in TrueForge."""
import asyncio, json, os, sys, tempfile
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    env = dict(os.environ, CONFUZZ_STATE=os.path.join(tempfile.mkdtemp(), "state.json"))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    params = StdioServerParameters(command=sys.executable, args=["-m", "confuzz.mcp_server"], env=env, cwd=root)
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            async def call(name, **a):
                out = json.loads((await s.call_tool(name, a)).content[0].text)
                if isinstance(out, dict) and out.get("error"): print(f"  [{name}] RULE: {out['message']}")
                return out
            await call("reset_session", world_id="demo", budget=20, attack_budget=10)
            top = (await call("rank_experiments", top_k=1))["ranked"][0]
            print("SCORER top:", top["perturbation"], "covers", top["unresolved_interactions_covered"], "unresolved interactions")
            q = await call("run_experiment", disruptions=top["perturbation"], rationale="scripted")
            print("SIM:", "CRITICAL" if q["critical_failure"] else "pass", [v["constraint"] for v in q["violations"]])
            m = await call("minimize_failure", disruptions=top["perturbation"])
            print("MINIMAL:", m["failure_id"], m["minimal_set"], "order", m["interaction_order"])
            # go after the stage example explicitly too
            q = await call("run_experiment", disruptions=["D1","D2","D4"], rationale="scripted")
            m2 = await call("minimize_failure", disruptions=["D1","D2","D4"])
            print("MINIMAL:", m2["failure_id"], m2["minimal_set"], [v["detail"] for v in m2["violations"]])
            ex = await call("explain_interaction", failure_id=m2["failure_id"])
            print("TABLE:", " | ".join(f"{'+'.join(t['subset'])}:{t['result']}" for t in ex["table"]))
            r1 = await call("propose_repair", failure_ids=[m2["failure_id"]], strategy_steps=[{"type": "SWAP_ROOMS"}], rationale="cheapest swap")
            print("BLUE R1:", r1["actions"], "cost", r1["cost"])
            a = await call("attack_repair", repair_id="R1", disruptions=["D4"], hypothesis="panel moved South; delay alone now crosses zones", vulnerable_constraint="transition")
            print("CRITIC:", a["verdict"], "->", a["repair_status"])
            r2 = await call("propose_repair", failure_ids=[m2["failure_id"]], strategy_steps=[{"type": "ACTIVATE_STANDBY", "room": "F"}, {"type": "MOVE_SESSION", "session": "S07"}],
                            rationale="add North capacity so the keynote never re-hosts South", previous_repair_id="R1", counterexample=["D4"],
                            diagnosis="R1 moved the follow-on session South, exposing every delay", what_changed_in_my_reasoning="fix the source (keynote placement) not the destination")
            print("BLUE R2:", r2.get("actions"), "cost", r2.get("cost"))
            a = await call("attack_repair", repair_id="R2", disruptions=["D2","D3","D6"], hypothesis="two room losses plus route closure exhaust step-free capacity", vulnerable_constraint="accessibility")
            print("CRITIC:", a["verdict"], "->", a["repair_status"])
            v = await call("verify_repair", repair_id="R2")
            c = v["certificate"]; ex = c["exact_bounded_space"]; sm = c["sampled_demand_uncertainty"]
            print("RISK:", v["status"], f"| exact: {ex['space_size']} perturbations, regressions {ex['regression_count']}, failing {ex['failing_perturbations_before']}->{ex['failing_perturbations_after']}",
                  f"| sampled n={sm['n']}: failure rate {sm['failure_rate_before']}->{sm['failure_rate_after']}, 95% CP upper {sm['clopper_pearson_95_upper_after']}, {sm['cvar_note']}")
            print("STOP: apply_repair requires human approval in TrueForge (not called by this script)")
            json.dump(await call("get_trace"), open(os.path.join(root, "experiments", "results", "scripted_episode_trace.json"), "w"), indent=1)

asyncio.run(main())
