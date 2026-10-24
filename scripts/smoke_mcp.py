"""Drive the MCP server over stdio exactly as TrueForge would (no LLM). Checks the tool surface and rules."""
import asyncio, json, os, sys, tempfile
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    env = dict(os.environ, CONFUZZ_STATE=os.path.join(tempfile.mkdtemp(), "state.json"))
    params = StdioServerParameters(command=sys.executable, args=["-m", "confuzz.mcp_server"], env=env,
                                   cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = (await s.list_tools()).tools
            print("TOOLS:", [(t.name, "destructive" if t.annotations and (getattr(t.annotations,"destructive_hint",None) or getattr(t.annotations,"destructiveHint",None)) else "") for t in tools])
            async def call(name, **args):
                res = await s.call_tool(name, args)
                return json.loads(res.content[0].text)
            await call("reset_session", world_id="demo", budget=20)
            q = await call("run_experiment", disruptions=["D1"], rationale="smoke")
            print("D1 margins:", q["robustness_margins"], "critical:", q["critical_failure"])
            print("apply before verify ->", await call("apply_repair", repair_id="R1", evidence_summary="x"))
            q = await call("run_experiment", disruptions=["D1","D2","D3"], rationale="smoke")
            print("D1+D2+D3 critical:", q["critical_failure"], [v["detail"] for v in q["violations"]])
            m = await call("minimize_failure", disruptions=["D1","D2","D3"])
            print("minimal:", m["minimal_set"], "order", m["interaction_order"])
            ex = await call("explain_interaction", failure_id=m["failure_id"])
            print("table:", [(t["subset"], t["result"]) for t in ex["table"]])
            r1 = await call("propose_repair", failure_ids=["F1"], strategy_steps=[{"type":"MOVE_SESSION","session":"S07"}], rationale="smoke")
            print("R1:", r1)
            print("verify w/o attack ->", (await call("verify_repair", repair_id="R1")).get("message"))
            print("budget after:", (await call("get_conference"))["discovery_budget"])

asyncio.run(main())
