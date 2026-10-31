# scripts/check_abstention.py
#
# STEP ONE: measure the CURRENT detector against the human labels before
# building a replacement. Fitting a new regex before seeing the confusion
# matrix leads to tuning patterns until the numbers look right.
#
# Reports:
#   - confusion matrix (regex verdict x human label) on the labeled sample
#   - raw agreement and Cohen's kappa
#   - how many of ALL adversarial answers the regex calls a failure, since
#     those are the items driving every adversarial delta in the results
#
#   python3 -m scripts.check_abstention

from __future__ import annotations

import hashlib
import json
import pathlib
from core import labels as lb
from core import manifest as mf
from core.paths import ABSTENTION_LABELS, CELLS
from collections import Counter

RESULTS = CELLS
LAB_DIR = ABSTENTION_LABELS


def kappa(a_vals, b_vals):
    """Cohen's kappa for two binary raters."""
    n = len(a_vals)
    if n == 0:
        return float("nan")
    agree = sum(1 for x, y in zip(a_vals, b_vals) if x == y) / n
    pa = Counter(a_vals)
    pb = Counter(b_vals)
    exp = sum((pa[k] / n) * (pb[k] / n) for k in set(pa) | set(pb))
    if abs(1 - exp) < 1e-12:
        return float("nan")
    return (agree - exp) / (1 - exp)


def full_index():
    """Rebuild key -> meta for EVERY adversarial answer.

    Do not trust keymap.json: it only ever held the sampled subset, and an
    earlier version of the labeling script overwrote it on each run. The
    item key is a deterministic sha256 of cell_id|qa_id, so the whole
    mapping can be regenerated from the cells themselves and joined
    against labels written at any point in the past.
    """
    idx = {}
    for d in mf.cell_dirs():
        a, s = d / "answers.jsonl", d / "summary.json"
        if not (a.exists() and s.exists()):
            continue
        cond = json.loads(s.read_text()).get("condition", "?")
        for line in a.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("category") != "adversarial" or not r.get("predicted"):
                continue
            k = hashlib.sha256(f"{d.name}|{r['qa_id']}".encode()).hexdigest()[:12]
            idx[k] = {"cell_id": d.name, "qa_id": r["qa_id"],
                      "condition": cond, "regex_said": r.get("correct"),
                      "predicted": r["predicted"], "question": r.get("question", "")}
    return idx


def main():
    keymap = full_index()
    # final labels (core/labels.py), not the last row written
    labels = lb.final(lb.ABSTENTION).labels

    human, machine, rows = [], [], []
    for k, lab in labels.items():
        if lab == "s" or k not in keymap:
            continue
        meta = keymap[k]
        # human: 'a' = abstained (correct);  regex: correct=True = abstained
        h = "abstain" if lab == "a" else "answer"
        m = "abstain" if meta.get("regex_said") else "answer"
        human.append(h)
        machine.append(m)
        rows.append((k, meta, h, m))

    n = len(human)
    print(f"labeled and usable: {n}")
    if n == 0:
        return 1

    cm = Counter(zip(machine, human))
    print("\nconfusion  (rows = regex, cols = human)")
    print(f"{'':>12} {'human:abstain':>14} {'human:answer':>13}")
    for m in ("abstain", "answer"):
        print(f"{'regex:'+m:>12} {cm[(m,'abstain')]:>14} {cm[(m,'answer')]:>13}")

    agree = sum(1 for h, m in zip(human, machine) if h == m) / n
    print(f"\nraw agreement: {agree:.3f}")
    print(f"Cohen's kappa: {kappa(machine, human):.3f}")

    # the items that matter: regex said "answer" but the human label says "abstain"
    misses = [(k, meta) for k, meta, h, m in rows
              if m == "answer" and h == "abstain"]
    print(f"\nregex called a FAILURE where the human label says refusal: {len(misses)}")
    for k, meta in misses:
        print(f"   {meta['cell_id']}  {meta['qa_id']}")

    # scope: how many adversarial items does the regex call failures overall?
    tot = fail = 0
    per_cond = Counter()
    for d in mf.cell_dirs():
        a = d / "answers.jsonl"
        s = d / "summary.json"
        if not (a.exists() and s.exists()):
            continue
        cond = json.loads(s.read_text()).get("condition", "?")
        for line in a.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("category") != "adversarial" or not r.get("predicted"):
                continue
            tot += 1
            if not r.get("correct"):
                fail += 1
                per_cond[cond] += 1

    print(f"\nacross ALL cells: {tot} adversarial answers, "
          f"{fail} scored as failures by the regex")
    for c, v in sorted(per_cond.items()):
        print(f"   {c:<12} {v}")
    print("\nThose", fail, "items are what every adversarial delta rests on.")
    print("Labeling all of them is cheap and removes the sampling question.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())