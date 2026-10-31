#!/usr/bin/env python3
# Rebuilds the per-cell appendix table from $MEMDEF_RESULTS (default
# results/). No API calls.
#   python3 -m scripts.make_percell_table
# Writes tab_percell.tex (longtable) -- to paper/figures/ for the default
# root, to <root>/figures/ for any other (so a new run never overwrites
# the paper) -- and prints pooled checks to compare with Table 4.
import collections
import json
import re

from core import manifest as mf
from core.paths import ADVERSARIAL_RESCORED as RESCORED, CELLS, QUARANTINE as QUAR, REPO_ROOT, RESULTS

OUT_DIR = REPO_ROOT / "paper" / "figures" if RESULTS == REPO_ROOT / "results" else RESULTS / "figures"
OUT = OUT_DIR / "tab_percell.tex"
TOP_K = 10  # fixme: must match the reranker's search(top_k=...)
ORDER = ["none", "sanitize", "provenance", "anomaly", "rerank", "stacked", "mini"]
NAME = {"none": "None", "sanitize": "Sanitize", "provenance": "Provenance",
        "anomaly": "Anomaly", "rerank": "Rerank", "stacked": "Stacked",
        "mini": "Baseline-mini"}
WRITE = {"sanitize", "provenance", "anomaly"}


def jl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


if not RESCORED.exists():
    raise SystemExit("Run scripts.rescore_abstention first; the Adv. column needs it.")
adv = {k: v["adv_abst_corrected"] for k, v in json.loads(RESCORED.read_text())["per_cell"].items()}

rows = []
# with a manifest: its main cells and the baseline-mini cells (this root only)
for d in mf.cell_dirs(local_only=True):
    s = d / "summary.json"
    if not s.exists():
        continue
    j = json.loads(s.read_text())
    cond = j["condition"]
    # replicate from the "__s<k>" part, wherever it sits (v3 ids end in __v3 / __mini)
    m = re.search(r"__s(\d+)(?:__|$)", d.name)
    rep = int(m.group(1)) if m else 1
    answers = jl(d / "answers.jsonl")
    core = [bool(r["correct"]) for r in answers
            if r.get("category") != "adversarial" and r.get("correct") is not None]

    log = jl(QUAR / f"{d.name}.jsonl")
    reads = [r for r in log if r.get("phase") == "read"]
    writes = [r for r in log if r.get("phase") != "read"]
    cands = [r.get("memories_candidates") for r in answers]
    if cond in ("rerank", "stacked") and answers and all(c is not None for c in cands):
        # v3 rows record the memories judged per question directly
        seen, quar = sum(cands), len(reads)
    elif cond in ("rerank", "stacked"):
        # decisions = memories the reranker judged, kept + dropped, per question
        per_q = collections.Counter(r["question"] for r in reads)
        seen, answered = 0, set()
        for r in answers:
            answered.add(r["question"])
            kept = r.get("memories_retrieved")
            seen += max(TOP_K, per_q[r["question"]]) if kept is None else kept + per_q[r["question"]]
        seen += sum(max(TOP_K, n) for q, n in per_q.items() if q not in answered)
        quar = len(reads)
    elif cond in WRITE:
        seen, quar = j.get("turns_ingested") or 150, len(writes)
    else:
        seen, quar = None, 0
    rows.append({"cond": cond, "conv": j["conv_id"], "rep": rep,
                 "mem": j.get("memories_stored"), "nq": len(answers),
                 "core": sum(core) / len(core) if core else None,
                 "adv": adv.get(d.name), "quar": quar, "seen": seen})

rows.sort(key=lambda r: (ORDER.index(r["cond"]), r["conv"], r["rep"]))

f3 = lambda x: "--" if x is None else f"{x:.3f}"
lines = [
    r"{\footnotesize",
    r"\setlength{\tabcolsep}{5pt}",
    r"\begin{longtable}{@{}l l r r r r r r r@{}}",
    rf"\caption{{Per-cell results, all {len(rows)} runs. Core is core accuracy within"
    r" the run; Adv.\ is adversarial abstention with hand labels applied."
    r" Quar.\ counts quarantines, and FQR divides them by the decisions"
    r" made in that cell (memories judged, for the reranker stage).}"
    r"\label{tab:percell}\\",
    r"\toprule",
    r"\textbf{Condition} & \textbf{Conv.} & \textbf{Rep.} & \textbf{Mem.}"
    r" & $\boldsymbol{n_q}$ & \textbf{Core} & \textbf{Adv.} & \textbf{Quar.}"
    r" & \textbf{FQR} \\",
    r"\midrule",
    r"\endfirsthead",
    r"\toprule",
    r"\textbf{Condition} & \textbf{Conv.} & \textbf{Rep.} & \textbf{Mem.}"
    r" & $\boldsymbol{n_q}$ & \textbf{Core} & \textbf{Adv.} & \textbf{Quar.}"
    r" & \textbf{FQR} \\",
    r"\midrule",
    r"\endhead",
    r"\bottomrule",
    r"\endlastfoot",
]
prev = None
for r in rows:
    if prev and r["cond"] != prev:
        lines.append(r"\midrule")
    prev = r["cond"]
    fqr = "--" if not r["seen"] else f"{100 * r['quar'] / r['seen']:.1f}\\%"
    lines.append(f"{NAME[r['cond']]} & {r['conv']} & {r['rep']} & {r['mem']} & {r['nq']}"
                 f" & {f3(r['core'])} & {f3(r['adv'])} & {r['quar']} & {fqr} \\\\")
lines += [r"\end{longtable}", r"}"]

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines) + "\n")
print(f"wrote {OUT} ({len(rows)} cells)")

# pooled checks: these must match Table 4 / Section 7.3
print("\npooled check (must match the paper):")
for cond in ORDER:
    rs = [r for r in rows if r["cond"] == cond]
    q, s = sum(r["quar"] for r in rs), sum(r["seen"] or 0 for r in rs)
    print(f"  {cond:11} cells {len(rs):>2}  quarantines {q:>4} / {s:>5} decisions"
          + (f"  FQR {100 * q / s:.1f}%" if s else ""))
if RESULTS == REPO_ROOT / "results":
    print("expected: rerank 756 / 7650 (9.9%), stacked 681 / 7650 (8.9%), write-time 0 / 2250")