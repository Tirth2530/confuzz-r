"""Inject exported hypergraph JSON into the page template. Usage: python scripts/build_hypergraph_page.py [json] [out.html]"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "experiments/results/hypergraph_demo.json"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "viz/failure_hypergraph.html"
data = json.loads(src.read_text())
html = (ROOT / "viz/hypergraph_template.html").read_text().replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
out.write_text(html); print("wrote", out)
