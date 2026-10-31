# runner/analyse.py
#
# Turns results/cells/*/summary.json into paper-ready tables: main
# results, cost-by-purpose, and deltas (BUD, token overhead) relative to
# the "none" baseline. Safe to run on a PARTIAL matrix.
#
#   python3 -m runner.analyse
#   python3 -m runner.analyse --csv out.csv

import argparse
import csv
import json
import pathlib
from core import manifest as mf
from core.paths import RESULTS
from scripts.check_integrity import require_integrity
import sys
from collections import defaultdict

RESULTS = RESULTS / "cells"


def load_summaries():
    out = []
    # with a manifest: its main cells and the baseline-mini cells (this root only)
    for d in mf.cell_dirs(local_only=True) if RESULTS.exists() else []:
        f = d / "summary.json"
        if f.exists():
            out.append(json.loads(f.read_text()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    require_integrity()

    summaries = load_summaries()
    if not summaries:
        print("No completed cells yet.")
        return 1

    by_group_backbone = defaultdict(dict)
    for s in summaries:
        key = (s["group"], s["backbone"])
        by_group_backbone[key].setdefault(s["condition"], []).append(s)

    print(f"\n{len(summaries)} cell(s) complete.\n")
    rows_for_csv = []

    for (group, backbone), conditions in by_group_backbone.items():
        print("=" * 78)
        print(f"{group} / {backbone}")
        print("=" * 78)

        baseline = conditions.get("none", [])
        base_acc = (sum(s["accuracy_core4"] for s in baseline) / len(baseline)
                    if baseline else None)
        base_tokens = (sum(s["cost_by_purpose"].get("base", {}).get("tokens", 0)
                            for s in baseline) / len(baseline) if baseline else None)

        header = (f"{'condition':12} {'n':>3} {'core4':>7} {'±':>6} "
                  f"{'adv_abst':>9} {'BUD':>7} {'tok_ovh':>8} "
                  f"{'defense$':>9} {'base$':>8} {'p95ms':>7} {'FQR':>6}")
        print(header)
        print("-" * len(header))

        for cond_name in ["none", "sanitize", "provenance", "anomaly", "rerank", "stacked"]:
            if cond_name not in conditions:
                continue
            cells = conditions[cond_name]
            n = len(cells)
            accs = [c["accuracy_core4"] for c in cells if c["accuracy_core4"] is not None]
            acc = sum(accs) / len(accs) if accs else None
            spread = (max(accs) - min(accs)) / 2 if len(accs) > 1 else 0.0

            advs = [c["adversarial_abstention"] for c in cells
                    if c.get("adversarial_abstention") is not None]
            adv = sum(advs) / len(advs) if advs else None

            bud = (acc - base_acc) if (acc is not None and base_acc is not None) else None

            def_usd = sum(c["cost_by_purpose"].get("defense", {}).get("usd", 0) for c in cells) / n
            base_usd = sum(c["cost_by_purpose"].get("base", {}).get("usd", 0) for c in cells) / n
            def_tok = sum(c["cost_by_purpose"].get("defense", {}).get("tokens", 0) for c in cells) / n
            tok_ovh = (def_tok / base_tokens) if base_tokens else None

            p95s = [c["latency_p95_ms"] for c in cells if c.get("latency_p95_ms")]
            p95 = sum(p95s) / len(p95s) if p95s else None
            fqrs = [c["false_quarantine_rate"] for c in cells]
            fqr = sum(fqrs) / len(fqrs) if fqrs else None

            def fmt(x, spec="{:.3f}"):
                return spec.format(x) if x is not None else "  --  "

            print(f"{cond_name:12} {n:>3} {fmt(acc):>7} {fmt(spread,'{:.3f}'):>6} "
                  f"{fmt(adv):>9} {fmt(bud):>7} {fmt(tok_ovh):>8} "
                  f"${def_usd:>7.4f} ${base_usd:>6.4f} {fmt(p95,'{:.0f}'):>7} {fmt(fqr):>6}")

            rows_for_csv.append({
                "group": group, "backbone": backbone, "condition": cond_name, "n_cells": n,
                "accuracy_core4": acc, "accuracy_spread": spread,
                "adversarial_abstention": adv, "BUD": bud, "token_overhead": tok_ovh,
                "defense_usd_per_conv": def_usd, "base_usd_per_conv": base_usd,
                "latency_p95_ms": p95, "false_quarantine_rate": fqr,
            })
        print()

    all_conditions = ["none", "sanitize", "provenance", "anomaly", "rerank", "stacked"]
    print("Coverage (conversations completed per condition):")
    for (group, backbone), conditions in by_group_backbone.items():
        line = f"  {group}/{backbone}: "
        line += ", ".join(f"{c}={len(conditions.get(c, []))}" for c in all_conditions)
        print(line)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_for_csv[0].keys()))
            w.writeheader()
            w.writerows(rows_for_csv)
        print(f"\nWrote {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
