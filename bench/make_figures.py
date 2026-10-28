"""Figures for the paper: NAFEMS heat benchmark (field + mesh convergence) and the extraction results."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402

sys.path.insert(0, "/home/vladimir/development/dealii-agent")
from femagent.verification.vtu import read_vtu  # noqa: E402

HERE = Path(__file__).parent
FIG = Path(sys.argv[1])
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "font.family": "serif"})

rows = [r for r in json.loads((HERE / "nafems_results.json").read_text()) if r["degree"] == 1] if (HERE / "nafems_results.json").exists() else []
if not rows:  # the probe was interrupted after r = 9; the log holds the values
    import ast
    rows = [ast.literal_eval(l) for l in (HERE / "nafems.log").read_text().splitlines() if l.startswith("{") and "'degree': 1" in l]
    json.dump(rows, open(HERE / "nafems_q1.json", "w"))

# --- Figure: temperature field and probe-value convergence
vtu = sorted((HERE / "nafems" / "r6_q1").glob("*/build/*.vtu"))[-1]
d = read_vtu(vtu)
x = [p[0] for p in d.points]
y = [p[1] for p in d.points]
t = next(iter(d.arrays.values()))
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [0.8, 1.2]})
tri = mtri.Triangulation(x, y)
cs = a.tricontourf(tri, t, levels=20, cmap="inferno")
a.tricontour(tri, t, levels=[18.25], colors="white", linewidths=0.8)
a.plot([0.6], [0.2], "o", color="cyan", ms=4)
a.annotate("P", (0.6, 0.2), xytext=(0.47, 0.17), color="white", fontsize=9)
a.set_aspect("equal"); a.set_xlabel("x [m]"); a.set_ylabel("y [m]"); a.grid(False)
a.set_title("(a) $T$ on $64\\times64$ cells", fontsize=9)
fig.colorbar(cs, ax=a, shrink=0.8, label="T [°C]")
ref = 18.25
cells = [4 ** r["refinement"] for r in rows]
err = [abs(r["value"] - ref) / ref for r in rows]
b.loglog([2 ** r["refinement"] for r in rows], err, "o-", color="#2563eb", label="generated solver ($Q_1$)")
b.axhline(1e-3, color="#b42318", ls="--", lw=1, label="acceptance tolerance $10^{-3}$")
b.set_xlabel("cells per direction"); b.set_ylabel("$|T_h(P)-T_{ref}|/T_{ref}$")
b.set_title("(b) probe value at $P=(0.6,0.2)$ m", fontsize=9)
b.legend(fontsize=7.5, loc="lower left")
fig.tight_layout()
fig.savefig(FIG / "nafems.pdf")

# --- Figure: extraction results per model
S = json.loads((HERE / "summary.json").read_text())
llms = ["llama3_latest", "qwen3-coder_30b", "devstral-small-2_latest", "qwen3.6_27b-coding", "laguna-xs-2.1_q4_K_M"]
labels = ["Llama 3\n8B", "Qwen3-Coder\n30B", "Devstral\n24B", "Qwen3.6\n27B (think)", "Laguna XS\n33B (think)"]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
for i, mod in enumerate(("html", "pdf")):
    w = 0.27
    for j, (key, lab, col) in enumerate((("valid", "accepted by validators", "#fbbf24"), ("correct", "equivalent to gold", "#2563eb"), ("exact", "solver output identical to gold", "#16803c"))):
        vals = [S["v1"][f"{l}|{mod}"][key] for l in llms]
        ax[i].bar([k + (j - 1) * w for k in range(len(llms))], vals, w, label=lab, color=col)
    ax[i].set_xticks(range(len(llms))); ax[i].set_xticklabels(labels, fontsize=7)
    ax[i].set_title("HTML (MathML)" if mod == "html" else "PDF text", fontsize=9)
    ax[i].set_ylim(0, 6.5); ax[i].set_yticks(range(7))
ax[0].set_ylabel("documents (of 6)")
ax[1].legend(fontsize=7, loc="upper left", ncol=1)
fig.tight_layout()
fig.savefig(FIG / "extraction.pdf")
print("figures written to", FIG)
