"""Figures and tables from persisted results only.

  python experiments/figures.py experiments/results/v3_dev.json [more.json ...]

Per input: <tag>_recall.png (minimal-failure recall vs simulator-query budget, mean with 95% CI band,
legend) and <tag>_table.md (same numbers as a table, so color never carries meaning alone).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
# fixed entity -> color (validated categorical slots, light mode); never reassigned by rank
COLOR = {"random": "#2a78d6", "random_pruned": "#eb6834", "agt_uniform": "#1baf7a", "agt_openai": "#eda100",
         "v1": "#e87ba4", "v2": "#6250d6", "size_ordered": "#008300"}
LABEL = {"random": "Random", "random_pruned": "Random + pruning", "agt_uniform": "Group testing (uniform prior)",
         "agt_openai": "Group testing + OpenAI prior", "v1": "CONFUZZ-R v1", "v2": "CONFUZZ-R v2", "size_ordered": "Size-ordered"}


def plot(path: Path):
    res = json.loads(path.read_text())
    tag = path.stem
    budgets = res["config"]["budgets"]
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    rows = []
    for cond, s in res["summary"].items():
        m = [s["recall_at"][str(b)]["mean"] for b in budgets]
        lo = [s["recall_at"][str(b)]["ci95"][0] for b in budgets]
        hi = [s["recall_at"][str(b)]["ci95"][1] for b in budgets]
        c = COLOR.get(cond, "#52514e")
        ax.fill_between(budgets, lo, hi, color=c, alpha=0.12, linewidth=0)
        ax.plot(budgets, m, color=c, linewidth=2, marker="o", markersize=4, label=LABEL.get(cond, cond))
        rows.append((cond, m, s["false_robust_rate_at"], s["queries_to_first"]))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_xlabel("Simulator queries (all queries counted, including minimization)", color="#52514e")
    ax.set_ylabel("Minimal failure recall", color="#52514e")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(budgets[0] - 1, budgets[-1] + 1)
    n_w = len(res["worlds"])
    ax.set_title(f"Hidden minimal interaction failures found vs budget ({res['split']} worlds, n={n_w}, "
                 f"{res['config']['seeds']} seeds)", fontsize=10, color="#0b0b0b", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{tag}_recall.png")
    lines = [f"# {tag} ({res['split']}, {n_w} worlds x {res['config']['seeds']} seeds, commit {res['git_commit'][:8]})", "",
             "| Method | " + " | ".join(f"Recall@{b}" for b in budgets) + " | False-robust@10 | Median queries to 1st |",
             "|---|" + "---|" * (len(budgets) + 2)]
    for cond, m, fr, q in rows:
        lines.append(f"| {LABEL.get(cond, cond)} | " + " | ".join(f"{x:.3f}" for x in m)
                     + f" | {fr['10']:.3f} | {q['median'] if q else '-'} |")
    if res.get("paired_tests"):
        lines += ["", "Paired Wilcoxon vs Random + pruning (per world-seed recall difference; descriptive, uncorrected):", ""]
        for k, v in res["paired_tests"].items():
            lines.append(f"- {k}: mean diff {v['mean_diff']:+.3f}, p={v['wilcoxon_p']:.2g}" if v["wilcoxon_p"] is not None else f"- {k}: n/a")
    (OUT / f"{tag}_table.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / f"{tag}_recall.png", OUT / f"{tag}_table.md")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        plot(Path(p))
