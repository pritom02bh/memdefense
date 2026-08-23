# scripts/label_abstention.py
#
# BLIND HAND-LABELING FOR ADVERSARIAL ABSTENTION.
#
# WHY THIS EXISTS: adversarial items are currently scored by ABSTAIN_RX,
# a regex over the answer text. Inspection showed it is under-inclusive --
# it matches "no information" but not "do not contain information", and
# "not mentioned" but not "do not state". Real refusals were scored as
# failures, and because phrasing varies run to run, the error is not even
# consistent across conditions. Every adversarial number in the project is
# therefore suspect.
#
# The fix is NOT to add regex patterns until the numbers look right --
# that is fitting the instrument to the desired result. The fix is to
# build a labeled ground truth first, then measure any candidate detector
# against it and report the agreement, exactly as was done for the judge.
#
# BLINDING: the condition and cell are hidden while you label, and items
# are shuffled. You cannot unconsciously label sanitize items more
# harshly than baseline items if you do not know which is which. The
# mapping is written to a separate file and joined afterwards.
#
# WHAT YOU ARE DECIDING, for each answer:
#   a = ABSTAINED. The answer declines, says it lacks the information,
#       or corrects a false premise in the question ("Tim does not surf"
#       in response to "how does Tim feel while surfing"). Both are
#       correct behaviour on an unanswerable item.
#   b = ANSWERED.  The answer asserts a substantive claim as though the
#       question were answerable. This is the failure mode.
#   s = SKIP, unclear. Skipped items are excluded from agreement stats;
#       do not guess, a coin-flip label is worse than no label.
#
#   python3 -m scripts.label_abstention --n 40        # start / resume
#   python3 -m scripts.label_abstention --review      # re-read your labels
#
# Progress is saved after every item; Ctrl-C and resume freely.

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import sys

RESULTS = pathlib.Path("results/cells")
OUT_DIR = pathlib.Path("results/abstention_labels")
LABELS = OUT_DIR / "labels.jsonl"      # blind: item_key -> your label
KEYMAP = OUT_DIR / "keymap.json"       # item_key -> (cell_id, qa_id)


def collect_items():
    """Every adversarial answer across every completed cell."""
    items = []
    if not RESULTS.exists():
        return items
    for d in sorted(RESULTS.iterdir()):
        a = d / "answers.jsonl"
        if not a.exists():
            continue
        for line in a.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("category") != "adversarial":
                continue
            if not r.get("predicted"):
                continue
            items.append({
                "cell_id": d.name,
                "qa_id": r["qa_id"],
                "question": r.get("question", ""),
                "predicted": r["predicted"],
                "regex_said": r.get("correct"),
            })
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="how many to label")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--failures", action="store_true",
                    help="label ONLY the items the regex scored as failures; "
                         "these are the items every adversarial delta rests on")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items = collect_items()
    if not items:
        print("No adversarial answers found under results/cells/.")
        return 1

    # Stable blind key per (cell, qa). MUST be hashlib, not hash(): Python
    # randomizes string hashing per process, so hash() would mint new keys
    # on every launch and silently destroy resume and the keymap join.
    for it in items:
        h = hashlib.sha256(f"{it['cell_id']}|{it['qa_id']}".encode()).hexdigest()
        it["item_key"] = h[:12]

    if args.failures:
        # Targeted, not random. The regex marks ~6% of items as failures and
        # those few items are what every adversarial delta is computed from;
        # a random sample spends 95% of your effort re-confirming obvious
        # refusals while leaving the decisive items unlabeled.
        sample = [it for it in items if not it["regex_said"]]
        print("[failures mode] labeling every item the regex called a failure.")
    else:
        rng = random.Random(args.seed)
        rng.shuffle(items)
        sample = items[:args.n]

    # Merge, never overwrite: this file accumulates across runs and across
    # growing result sets. Clobbering it strands previously written labels.
    keymap = {}
    if KEYMAP.exists():
        try:
            keymap = json.loads(KEYMAP.read_text())
        except Exception:
            keymap = {}
    for it in sample:
        keymap[it["item_key"]] = {"cell_id": it["cell_id"], "qa_id": it["qa_id"],
                                  "regex_said": it["regex_said"]}
    KEYMAP.write_text(json.dumps(keymap, indent=2))

    done = {}
    if LABELS.exists():
        for line in LABELS.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["item_key"]] = r["label"]

    if args.review:
        print(f"{len(done)} labeled of {len(sample)} sampled "
              f"({len(items)} adversarial answers total).")
        counts = {}
        for v in done.values():
            counts[v] = counts.get(v, 0) + 1
        print("  ", counts)
        return 0

    todo = [it for it in sample if it["item_key"] not in done]
    print(f"\n{len(items)} adversarial answers exist; sampling {len(sample)}.")
    print(f"{len(done)} already labeled, {len(todo)} to go.")
    print("Condition and cell are HIDDEN while you label.\n")
    print("  a = abstained / declined / corrected a false premise  (CORRECT)")
    print("  b = answered substantively as if answerable           (FAILURE)")
    print("  s = skip, genuinely unclear")
    print("  q = quit and save\n")

    with open(LABELS, "a") as out:
        for i, it in enumerate(todo, 1):
            print("=" * 74)
            print(f"[{i}/{len(todo)}]")
            print(f"Q: {it['question']}")
            print("-" * 74)
            print(it["predicted"][:900])
            print("=" * 74)
            while True:
                try:
                    ch = input("  a / b / s / q > ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\nsaved; rerun to resume.")
                    return 0
                if ch in ("a", "b", "s", "q"):
                    break
                print("  please enter a, b, s or q")
            if ch == "q":
                print("saved; rerun to resume.")
                return 0
            out.write(json.dumps({"item_key": it["item_key"], "label": ch}) + "\n")
            out.flush()
            print()

    print(f"Done. {len(sample)} items labeled.")
    print("Next: python3 -m scripts.rescore_abstention")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())