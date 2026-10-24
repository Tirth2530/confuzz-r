"""API fallback for setting tool approvals (the UI shield in Select MCP Tools does the same thing).

Verified against TrueForge docs/openapi.json @ 1090880:
  GET /api/v1/agents/{agent_id}  -> {"data": {"id", "name", "description", "manifest": {..., "mcp_servers": [...]}}}
  PUT /api/v1/agents/{agent_id}  <- {"description": ..., "manifest": {...}}   (manifest required)
mcp_servers[].require_approval_for_tools accepts "@all", "@write", "@destructive" or literal tool names;
the default is ["@write", "@destructive"], which would gate every CONFUZZ-R experiment tool.

  python trueforge/set_approval.py --agent <agent_id> [--base http://localhost:8790] [--token <bearer>] [--dry-run]
"""
import argparse
import json
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--agent", required=True)
ap.add_argument("--base", default="http://localhost:8790")
ap.add_argument("--server-name", default="confuzz-r", help="the MCP server name as configured in Connectors")
ap.add_argument("--token", default=None, help="bearer token if your TrueForge has auth enabled")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

headers = {"Content-Type": "application/json"}
if a.token:
    headers["Authorization"] = f"Bearer {a.token}"
url = f"{a.base}/api/v1/agents/{a.agent}"
data = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=headers)))["data"]
manifest = data["manifest"]
servers = manifest.get("mcp_servers", [])
hit = [s for s in servers if s.get("name") == a.server_name]
if not hit:
    raise SystemExit(f"no MCP server named {a.server_name!r} on this agent; found {[s.get('name') for s in servers]}")
hit[0]["require_approval_for_tools"] = ["apply_repair", "send_announcements", "reset_session"]
print(json.dumps(hit[0], indent=1))
if a.dry_run:
    raise SystemExit("dry run: nothing sent")
body = {"description": data.get("description") or "", "manifest": manifest}
req = urllib.request.Request(url, data=json.dumps(body).encode(), method="PUT", headers=headers)
print(urllib.request.urlopen(req).status, "updated")
