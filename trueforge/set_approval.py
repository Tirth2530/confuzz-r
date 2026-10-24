"""Set require_approval_for_tools on a TrueForge agent via its HTTP API (needed if your installed
TrueForge has no per-tool approval toggle in the UI). GETs the agent, edits only the confuzz-r
MCP server entry, PUTs it back. Field names follow the TrueForge docs and issue #507; CHECK the
GET output first and adjust if your version differs.

  python trueforge/set_approval.py --agent <agent_id> [--base http://localhost:8790] [--dry-run]
"""
import argparse, json, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--agent", required=True)
ap.add_argument("--base", default="http://localhost:8790")
ap.add_argument("--server-name", default="confuzz-r")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

url = f"{a.base}/api/v1/agents/{a.agent}"
agent = json.load(urllib.request.urlopen(url))
print("GET agent keys:", list(agent.keys()))
servers = agent.get("mcp_servers") or agent.get("spec", {}).get("mcp_servers")
if servers is None:
    raise SystemExit("no mcp_servers field found; inspect the GET output above and edit this script")
hit = [s for s in servers if s.get("name") == a.server_name]
if not hit:
    raise SystemExit(f"no MCP server named {a.server_name}; found {[s.get('name') for s in servers]}")
hit[0]["require_approval_for_tools"] = ["apply_repair", "reset_session"]
hit[0].setdefault("enable_tools", ["@all"])
print(json.dumps(hit[0], indent=1))
if a.dry_run:
    raise SystemExit("dry run: nothing sent")
req = urllib.request.Request(url, data=json.dumps(agent).encode(), method="PUT", headers={"Content-Type": "application/json"})
print(urllib.request.urlopen(req).status, "updated")
