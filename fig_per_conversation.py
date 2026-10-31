# Per-conversation benign accuracy change, one marker per defense.
#   python3 fig_per_conversation.py                              # v2 paper (results/)
#   MEMDEF_RESULTS=results_v3 python3 fig_per_conversation.py    # from results_v3
# Default root: the v2 numbers below (core metric, majority vote over
# replicates), written to paper/figures/. Any other root: the
# per-conversation deltas in <root>/stats.json (python3 -m scripts.stats
# --json; same majority vote), written to <root>/figures/.
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.paths import REPO_ROOT, RESULTS

DEFAULT_ROOT = RESULTS == REPO_ROOT / "results"
out = REPO_ROOT / "paper" / "figures" if DEFAULT_ROOT else RESULTS / "figures"
out.mkdir(parents=True, exist_ok=True)

if DEFAULT_ROOT:
    convs = ["conv-26", "conv-30", "conv-41", "conv-42", "conv-43"]
    delta = {  # points; the row means reproduce Table 4 exactly
        "Sanitize":   [-1.5,   0.0, 4.0, 9.1,  -2.6],
        "Provenance": [ 3.0,   2.6, 0.0, 9.1,  -5.3],
        "Anomaly":    [ 0.0,   7.9, 4.0, 6.1,  -2.6],
        "Rerank":     [-4.5, -13.2, 0.0, 6.1, -10.5],
        "Stacked":    [ 3.0,  -7.9, 0.0, 0.0, -10.5],
    }
else:
    core = json.loads((RESULTS / "stats.json").read_text())["anchor_gemini/gemini-backbone"]["core4"]
    convs = sorted(core["sanitize"]["per_conv"])
    delta = {name: [round(100 * core[cond]["per_conv"][c], 1) for c in convs]
             for cond, name in (("sanitize", "Sanitize"), ("provenance", "Provenance"),
                                ("anomaly", "Anomaly"), ("rerank", "Rerank"),
                                ("stacked", "Stacked"))}
# same colors as Figure 2 (stage), shapes tell the write-time defenses apart
style = {
    "Sanitize":   ("#2a78d6", "o", 4.5),
    "Provenance": ("#2a78d6", "^", 4.8),
    "Anomaly":    ("#2a78d6", "v", 4.8),
    "Rerank":     ("#eb6834", "s", 5.5),
    "Stacked":    ("#1baf7a", "D", 5.0),
}
offs = [-0.26, -0.13, 0.0, 0.13, 0.26]

plt.rcParams.update({"font.size": 8, "font.family": "serif",
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(3.4, 2.45))
if "conv-42" in convs:   # shaded: read in the caption
    i42 = convs.index("conv-42")
    ax.axvspan(i42 - 0.45, i42 + 0.45, color="#f1f0ec", zorder=0, lw=0)
ax.axhline(0, color="#8a8984", lw=0.8, ls=(0, (3, 3)), zorder=1)

for (name, vals), dx in zip(delta.items(), offs):
    c, m, ms = style[name]
    ax.plot([i + dx for i in range(len(convs))], vals, ls="none", marker=m, ms=ms,
            color=c, mec="white", mew=0.7, zorder=3, label=name)

ax.set_xticks(range(len(convs)))
ax.set_xticklabels(convs)
ax.set_xlim(-0.55, len(convs) - 0.45)
if DEFAULT_ROOT:   # hand-tuned for the v2 numbers
    ax.set_ylim(-15, 12)
else:
    lo = min(min(v) for v in delta.values())
    hi = max(max(v) for v in delta.values())
    ax.set_ylim(min(-15, lo - 2), max(12, hi + 2))
ax.set_yticks([-10, -5, 0, 5, 10])
ax.set_ylabel(r"$\Delta_{d,c}$, core accuracy (points)")
ax.grid(axis="y", color="#e6e5e0", lw=0.6, zorder=0)
ax.tick_params(colors="#52514e", length=2.5)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False,
          fontsize=6.3, handletextpad=0.15, columnspacing=0.7, borderaxespad=0.2)

for ext in ("pdf", "png"):
    fig.savefig(out / f"fig_per_conversation.{ext}", dpi=300, bbox_inches="tight")
print(f"wrote {out}/fig_per_conversation.pdf and .png")