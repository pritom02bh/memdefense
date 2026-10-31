# Regenerates the results figure: benign accuracy change vs token overhead.
#   python3 fig.py                              # the v2 paper figure (results/)
#   MEMDEF_RESULTS=results_v3 python3 fig.py    # from results_v3
# Default root: the verified v2 numbers below (scripts.stats over the v2
# cells), written to paper/figures/. Any other root: computed from
# <root>/stats.json (python3 -m scripts.stats --json), <root>/tau.json
# (Stage 4 report) and the cells' rerank totals, written to <root>/figures/.
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import manifest as mf
from core.paths import CELLS, REPO_ROOT, RESULTS

DEFAULT_ROOT = RESULTS == REPO_ROOT / "results"
out = REPO_ROOT / "paper" / "figures" if DEFAULT_ROOT else RESULTS / "figures"
out.mkdir(parents=True, exist_ok=True)
NAMES = {"sanitize": "Sanitize", "provenance": "Provenance", "anomaly": "Anomaly",
         "rerank": "Rerank", "stacked": "Stacked"}
STAGE = {"sanitize": "write", "provenance": "write", "anomaly": "write",
         "rerank": "read", "stacked": "both"}


def rows_from_root():
    """(name, tau %, delta pp, ci lo, ci hi, stage, fqr label) per defense."""
    stats = json.loads((RESULTS / "stats.json").read_text())
    core = stats["anchor_gemini/gemini-backbone"]["core4"]
    # the manifest's tau (Stage 4b) when there is one, else Stage 4's
    tm = RESULTS / "tau_manifest.json"
    tau = ({c: {"new": v["tau"]} for c, v in json.loads(tm.read_text())["all"].items()
            if isinstance(v, dict) and "tau" in v}
           if tm.exists() else json.loads((RESULTS / "tau.json").read_text()))
    drops, judged, blocked, turns = {}, {}, {}, {}
    for d in mf.cell_dirs(("main",)):
        s = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
        t = s.get("rerank_totals")
        if t:
            drops[s["condition"]] = drops.get(s["condition"], 0) + t["dropped"]
            judged[s["condition"]] = judged.get(s["condition"], 0) + t["retrieved"]
        ws = s.get("write_screen") or {}
        if ws:
            blocked[s.get("condition")] = blocked.get(s.get("condition"), 0) + len(ws.get("blocked") or [])
            turns[s.get("condition")] = turns.get(s.get("condition"), 0) + (ws.get("turns_screened") or 0)
    rows = []
    for cond, name in NAMES.items():
        c = core[cond]
        if judged.get(cond):      # read-time drops (stacked: its reranker stage)
            fqr = f"FQR {100 * drops[cond] / judged[cond]:.1f}%"
        elif blocked.get(cond):   # write-time blocks, when there are any
            fqr = f"FQR {100 * blocked[cond] / turns[cond]:.1f}%"
        else:
            fqr = None
        rows.append((name, 100 * tau[cond]["new"], 100 * c["point"], 100 * c["ci"][0],
                     100 * c["ci"][1], STAGE[cond], fqr))
    return rows


# name, tau %, delta pp, ci lo, ci hi, stage, fqr label
rows = [
    ("Sanitize",   0.0, 1.79, -1.72, 5.56, "write", None),
    ("Provenance", 0.0, 1.90, -2.20, 6.04, "write", None),
    ("Anomaly",    1.2, 3.06, -0.89, 7.10, "write", None),
    ("Rerank",     2.7, -4.43, -9.03, -0.05, "read", "FQR 9.9%"),
    ("Stacked",    4.1, -3.08, -7.37, 0.99, "both", "FQR 8.9%"),
] if DEFAULT_ROOT else rows_from_root()
style = {  # validated categorical slots 1-3, plus a shape so print/CVD still works
    "write": ("#2a78d6", "o", "Write-time (FQR 0%)" if DEFAULT_ROOT else "Write-time"),
    "read":  ("#eb6834", "s", "Read-time"),
    "both":  ("#1baf7a", "D", "Write + read"),
}
nudge = {"Sanitize": -0.13, "Provenance": 0.13}  # both sit at 0%; offset so both show
label_at = {  # (dx, dy, ha) in data units; hand-placed to dodge the bars
    "Sanitize":   (-0.14, 0.0, "right"),
    "Provenance": (0.0, 4.85, "center"),  # above its bar, clear of Anomaly's
    "Anomaly":    (0.12, 0.0, "left"),
    "Rerank":     (-0.14, 0.0, "right"),
    "Stacked":    (0.14, 0.0, "left"),
}
if not DEFAULT_ROOT:  # results_v3: Provenance under its bar, Anomaly left of its point
    label_at["Provenance"] = (0.0, -3.9, "center")
    label_at["Anomaly"] = (-0.12, 1.0, "right")

plt.rcParams.update({"font.size": 8, "font.family": "serif",
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(3.4, 2.5))
ax.axhline(0, color="#8a8984", lw=0.8, ls=(0, (3, 3)), zorder=1)

seen = set()
for name, tau, d, lo, hi, stage, fqr in rows:
    c, m, lab = style[stage]
    x = tau + nudge.get(name, 0.0)
    ax.errorbar(x, d, yerr=[[d - lo], [hi - d]], fmt=m, ms=5.5, color=c,
                mec="white", mew=0.8, elinewidth=1.2, capsize=0, zorder=3,
                label=None if stage in seen else lab)
    seen.add(stage)
    dx, dy, ha = label_at[name]
    text = name if fqr is None else f"{name}\n{fqr}"
    ax.text(x + dx, d + dy, text, ha=ha, va="center", fontsize=7, color="#2b2b29")

if DEFAULT_ROOT:   # hand-tuned for the v2 numbers
    ax.set_xlim(-1.7, 5.5)
    ax.set_ylim(-10.5, 8.5)
    ax.set_yticks([-10, -5, 0, 5])
    ax.set_xticks([0, 1, 2, 3, 4])
else:              # fit the data; label positions may need hand-tuning
    ax.set_xlim(-1.7, max(r[1] for r in rows) + 1.5)
    ax.set_ylim(min(r[3] for r in rows) - 1.5, max(r[4] for r in rows) + 1.5)
ax.set_xlabel(r"Token overhead $\tau_d$ (%)")
ax.set_ylabel(r"$\Delta_d$, core accuracy (points)")
ax.grid(axis="y", color="#e6e5e0", lw=0.6, zorder=0)
ax.tick_params(colors="#52514e", length=2.5)
ax.legend(loc="upper right", frameon=False, fontsize=6.5, handletextpad=0.3,
          borderaxespad=0.1)

for ext in ("pdf", "png"):
    fig.savefig(out / f"fig_cost_utility.{ext}", dpi=300, bbox_inches="tight")
print(f"wrote {out}/fig_cost_utility.pdf and .png")