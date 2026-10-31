# scripts/diagnostics/stage4_report.py
#
# STAGE 4, READ-ONLY. No API calls. Run after the v3 run has finished:
#
#   MEMDEF_RESULTS=results_v3 python3 -m scripts.diagnostics.stage4_report
#
# 1. The integrity check on every v3 cell (must pass).
# 2. Token overhead tau_d, new against old. tau_d = defense tokens of
#    condition d / baseline base tokens. Baseline base tokens = the
#    extraction calls of the 15 old baseline cells' final attempts (old
#    log, backbone only) + the answer calls of the 15 none__v3 cells.
#    Old tau uses the old defense tokens over the old baseline (old
#    extractions + old answers). Only final attempts count: a
#    re-ingested cell's superseded Phase A screening does not.
#    Also writes <root>/tau.json (read by fig.py).
# 3. Base tokens per condition, computed the same way: each cell's store
#    extraction (old final attempt, or the v3 re-ingest) + its answers.
# 4. Store check: every old namespace's fingerprint equals the one taken
#    before the run, every cell's store size was the same before and
#    after answering, and every re-ingested store holds what its summary
#    says.
# 5. Nothing under results/ changed since the Stage 0 backup.
# Exit 0 only if every check passes.

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict

from core.paths import CELLS, QDRANT, REPO_ROOT, RESULTS
from scripts.check_integrity import print_report, run_checks
from scripts.store_fingerprint import fingerprint

OLD = REPO_ROOT / "results"
GEMINI = "gemini-3.1-flash-lite"
CONDS = ["none", "sanitize", "provenance", "anomaly", "rerank", "stacked", "mini"]
DEFENDED = ["sanitize", "provenance", "anomaly", "rerank", "stacked"]


def tokens(rows, pred) -> int:
    return sum(r.get("tokens_total") or 0 for r in rows if pred(r))


def is_answer(r):
    return r["_step"] == "answer" and r["status"] == "ok"


def is_defense(r):
    return r.get("purpose") == "defense"


def main() -> int:
    if RESULTS.name != "results_v3":
        sys.exit("run with MEMDEF_RESULTS=results_v3")
    ok = True

    # ---------------------------------------------------------------- 1
    print("=" * 78 + "\n1. INTEGRITY CHECK, v3 cells\n" + "=" * 78)
    new = run_checks("*", quiet=True)
    print_report(new, {})
    bad = [r for r in new if r.violations]
    print(f"\n-> {'PASS' if not bad else 'FAIL'}: {len(new)} cells, {len(bad)} with violations")
    ok &= not bad and len(new) == 105

    old = run_checks("anchor_gemini__*", quiet=True, log_path=OLD / "proxy" / "calls.jsonl",
                     cells_dir=OLD / "cells", quarantine_dir=OLD / "quarantine")
    old_by = {r.cell_id: r for r in old}
    new_by = {r.cell_id: r for r in new}
    summ = {r.cell_id: json.loads((CELLS / r.cell_id / "summary.json").read_text()) for r in new}

    def old_extract(cid):
        return tokens(old_by[cid].final_extract, lambda r: r["model_resolved"] == GEMINI)

    def store_extract(cid):
        """Extraction tokens of the store a v3 cell answered from."""
        s = summ[cid]
        if s["reingested"]:
            return tokens(new_by[cid].final_extract, lambda r: r["model_resolved"] == GEMINI)
        if s["source_namespace"].endswith("__v3"):          # mini on a re-ingested none store
            return store_extract(s["source_namespace"])
        return old_extract(s["source_namespace"])

    # ---------------------------------------------------------------- 2
    print("\n" + "=" * 78 + "\n2. TOKEN OVERHEAD tau_d, new vs old\n" + "=" * 78)
    none_new = [c for c in summ if summ[c]["condition"] == "none"]
    none_old = [summ[c]["old_cell_id"] for c in none_new]
    base_new = (sum(old_extract(c) for c in none_old)
                + sum(tokens(new_by[c].final_rows, is_answer) for c in none_new))
    base_old = sum(old_extract(c) + tokens(old_by[c].final_rows, is_answer) for c in none_old)
    print(f"baseline base tokens  new = {base_new:,}  (old extraction of the 15 none cells "
          f"+ answers of the 15 none__v3 cells)")
    print(f"                      old = {base_old:,}  (old extraction + old answers)")
    tau = {"baseline_base_tokens": {"new": base_new, "old": base_old}}
    print(f"\n  {'condition':11} {'defense tok new':>16} {'tau new':>9} {'defense tok old':>16} "
          f"{'tau old':>9}")
    for cond in DEFENDED:
        cells = [c for c in summ if summ[c]["condition"] == cond]
        d_new = sum(tokens(new_by[c].final_rows, is_defense) for c in cells)
        d_old = sum(tokens(old_by[summ[c]["old_cell_id"]].final_rows, is_defense) for c in cells)
        tau[cond] = {"new": d_new / base_new, "old": d_old / base_old,
                     "defense_tokens_new": d_new, "defense_tokens_old": d_old, "cells": len(cells)}
        print(f"  {cond:11} {d_new:>16,} {100 * d_new / base_new:>8.2f}% {d_old:>16,} "
              f"{100 * d_old / base_old:>8.2f}%")
    (RESULTS / "tau.json").write_text(json.dumps(tau, indent=2))
    print(f"\nwrote {RESULTS / 'tau.json'}")

    # ---------------------------------------------------------------- 3
    print("\n" + "=" * 78 + "\n3. BASE TOKENS PER CONDITION (store extraction + answers)\n" + "=" * 78)
    print(f"  {'condition':11} {'cells':>5} {'extraction':>13} {'answers new':>12} {'base new':>13} "
          f"{'answers old':>12} {'base old':>13}")
    for cond in CONDS:
        cells = [c for c in summ if summ[c]["condition"] == cond]
        ex = sum(store_extract(c) for c in cells)
        an = sum(tokens(new_by[c].final_rows, is_answer) for c in cells)
        if cond == "mini":
            print(f"  {cond:11} {len(cells):>5} {ex:>13,} {an:>12,} {ex + an:>13,} {'-':>12} {'-':>13}")
            continue
        olds = [summ[c]["old_cell_id"] for c in cells]
        ex_old = sum(old_extract(c) for c in olds)
        an_old = sum(tokens(old_by[c].final_rows, is_answer) for c in olds)
        print(f"  {cond:11} {len(cells):>5} {ex:>13,} {an:>12,} {ex + an:>13,} {an_old:>12,} "
              f"{ex_old + an_old:>13,}")
    print("  (extraction = the store each cell answered from: old final attempt, or the v3 "
          "re-ingest; old base uses the old extraction)")

    # ---------------------------------------------------------------- 4
    print("\n" + "=" * 78 + "\n4. STORES UNCHANGED BY ANSWERING\n" + "=" * 78)
    before = json.loads((RESULTS / "store_fingerprint_before.json").read_text())["namespaces"]
    now = fingerprint(QDRANT)
    changed = [ns for ns in before if now.get(ns) != before[ns]]
    print(f"namespaces fingerprinted before the run: {len(before)}; changed since: {len(changed)}"
          + (f"  {changed}" if changed else ""))
    ok &= not changed
    print(f"\n  {'cell':58} {'source':6} {'before':>6} {'after':>6} {'store now':>9}  check")
    for cid, s in summ.items():
        src = s["source_namespace"]
        n_now = (now.get(src) or {}).get("points", 0)
        same = s["memories_before_answering"] == s["memories_after_answering"] == n_now
        if not s["reingested"] and not src.endswith("__v3"):
            same &= before.get(src) == now.get(src)
        ok &= same
        print(f"  {cid:58} {'v3' if src.endswith('__v3') else 'old':6} "
              f"{s['memories_before_answering']:>6} {s['memories_after_answering']:>6} "
              f"{n_now:>9}  {'ok' if same else 'CHANGED'}")

    # ---------------------------------------------------------------- 5
    print("\n" + "=" * 78 + "\n5. results/ UNTOUCHED\n" + "=" * 78)
    backup = REPO_ROOT / "results_backup_20261029.tar.gz"
    newer = subprocess.run(["find", str(OLD), "-newer", str(backup)], capture_output=True,
                           text=True).stdout.split()
    print(f"files under results/ newer than {backup.name}: {len(newer)}" + (f" {newer[:5]}" if newer else ""))
    ok &= not newer

    print(f"\nSTAGE 4: {'ALL CHECKS PASS' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
