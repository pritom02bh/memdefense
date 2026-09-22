#!/usr/bin/env python3
"""
Audit every cell in results/cells/ and report GOOD / BAD / MISSING.

Run from the project root:
    python3 audit_cells.py            # report only, changes nothing
    python3 audit_cells.py --delete   # delete the BAD cells (asks first)

A cell is BAD if any of these hold:
  - summary.json missing or unreadable   (crashed mid-write)
  - accuracy_core4 is null               (every question failed to grade)
  - memories_stored < 150                (ingest ran during an API outage,
                                          so the agent answered from a
                                          fraction of the memory it should
                                          have had -- this is the dangerous
                                          case, because the cell still
                                          reports a plausible-looking score)
  - answers.jsonl missing                (nothing to pair on)

Healthy cells in this project store roughly 230-330 memories from 150 turns.
"""
import json
import pathlib
import shutil
import sys

RESULTS = pathlib.Path("results/cells")
GROUP = "anchor_gemini"
BACKBONE = "gemini-backbone"
CONDITIONS = ["none", "sanitize", "provenance", "anomaly", "rerank", "stacked"]
CONVS = [0, 1, 2, 3, 4]
SEEDS = [1, 2, 3]
MIN_MEMORIES = 150


def cell_dir(cond, conv, seed):
    base = f"{GROUP}__{BACKBONE}__{cond}__conv{conv}"
    return RESULTS / (base if seed == 1 else f"{base}__s{seed}")


def classify(d):
    """Return (status, detail)."""
    if not d.exists():
        return "MISSING", ""
    s = d / "summary.json"
    if not s.exists():
        return "BAD", "no summary.json (crashed mid-cell)"
    try:
        j = json.loads(s.read_text())
    except Exception as e:
        return "BAD", f"unreadable summary.json ({type(e).__name__})"
    if not (d / "answers.jsonl").exists():
        return "BAD", "no answers.jsonl"
    core = j.get("accuracy_core4")
    mem = j.get("memories_stored", 0)
    if core is None:
        return "BAD", f"core4=None, mem={mem} (all questions failed)"
    if mem < MIN_MEMORIES:
        return "BAD", f"core4={core:.4f} but mem={mem} (POISONED INGEST)"
    return "GOOD", f"core4={core:.4f} mem={mem} q={j.get('questions_graded')}"


def main():
    if not RESULTS.exists():
        print("ERROR: results/cells not found. Run from the project root.")
        return 1

    buckets = {"GOOD": [], "BAD": [], "MISSING": []}
    for cond in CONDITIONS:
        for seed in SEEDS:
            for conv in CONVS:
                d = cell_dir(cond, conv, seed)
                status, detail = classify(d)
                buckets[status].append((cond, seed, conv, d, detail))

    # unexpected extras (anything on disk not in the matrix)
    expected = {cell_dir(c, v, s) for c in CONDITIONS for v in CONVS for s in SEEDS}
    extras = [p for p in sorted(RESULTS.iterdir()) if p.is_dir() and p not in expected]

    print("=" * 78)
    print("BAD CELLS -- must be deleted and rerun")
    print("=" * 78)
    if not buckets["BAD"]:
        print("  none")
    for cond, seed, conv, d, detail in buckets["BAD"]:
        print(f"  {d.name}")
        print(f"      {detail}")

    print()
    print("=" * 78)
    print("PER-CONDITION SUMMARY  (out of 5 conversations per seed)")
    print("=" * 78)
    print(f"{'condition':<12} {'seed':>4} {'good':>5} {'bad':>4} {'missing':>8}")
    print("-" * 78)
    for cond in CONDITIONS:
        for seed in SEEDS:
            g = sum(1 for c, s, *_ in buckets["GOOD"] if c == cond and s == seed)
            b = sum(1 for c, s, *_ in buckets["BAD"] if c == cond and s == seed)
            m = sum(1 for c, s, *_ in buckets["MISSING"] if c == cond and s == seed)
            flag = "" if g == 5 else ("   <-- incomplete" if g else "   <-- not run")
            print(f"{cond:<12} {seed:>4} {g:>5} {b:>4} {m:>8}{flag}")

    ng, nb, nm = len(buckets["GOOD"]), len(buckets["BAD"]), len(buckets["MISSING"])
    print("-" * 78)
    print(f"TOTAL: {ng} good, {nb} bad, {nm} missing   (full matrix = 90)")

    if extras:
        print("\nUnrecognised directories (not in the expected matrix):")
        for p in extras:
            print(f"  {p.name}")

    if buckets["BAD"] and "--delete" in sys.argv:
        print(f"\nAbout to DELETE {nb} bad cell directories listed above.")
        if input("Type 'yes' to confirm: ").strip() == "yes":
            for *_, d, _ in buckets["BAD"]:
                shutil.rmtree(d)
                print(f"  deleted {d.name}")
            print("\nDone. Rerun those conditions; good cells will be skipped.")
        else:
            print("Aborted. Nothing deleted.")
    elif buckets["BAD"]:
        print("\nRerun with --delete to remove the bad cells.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())