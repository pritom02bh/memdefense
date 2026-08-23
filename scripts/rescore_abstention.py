# scripts/rescore_abstention.py
#
# CORRECTS ADVERSARIAL SCORING WITHOUT RE-RUNNING ANYTHING.
#
# The original ABSTAIN_RX under-recognises real refusals. On a random
# 40-item sample it looked perfect (kappa=1.0), because ~95% of adversarial
# answers are the easy case ("I don't know"). On the 20 items the regex
# actually scored as failures -- the items every adversarial delta is
# computed from -- it was wrong 8/20 times (kappa=0.663). Sanitize's
# apparent -10.6pp effect came almost entirely from 5 of those 8 misses.
#
# This script does NOT invent a new detector by staring at the 8 misses
# and writing patterns to match them -- that would be fitting the
# instrument to the desired answer. Instead:
#
#   1. Ground truth = your 58 hand labels, wherever they exist.
#   2. For the remaining ~272 items, apply an EXPANDED regex built from
#      generalisable negation patterns (not the 8 specific sentences),
#      plus explicit handling for false-premise corrections
#      ("X does not <verb>" answering "how does X feel while <verb>ing").
#   3. Report the expanded regex's agreement against ALL 58 labels
#      (not just the ones it was designed to fix) as the calibration
#      number for the paper -- same standard as the LLM judge.
#   4. Recompute every table (BUD, per-cell, adv_abst) from the corrected
#      per-item calls and write it next to, not over, the original.
#
# Nothing here touches answers.jsonl or the proxy log. Re-running this
# script after adding more labels is free.
#
#   python3 -m scripts.rescore_abstention

from __future__ import annotations

import hashlib
import json
import pathlib
import re
from collections import Counter, defaultdict

RESULTS = pathlib.Path("results/cells")
LAB_DIR = pathlib.Path("results/abstention_labels")
OUT = pathlib.Path("results/adversarial_rescored.json")


# --------------------------------------------------------------- detector

# Generalised negation/refusal patterns. Each is a phrase FAMILY, not a
# copy of one of the 8 misses verbatim -- e.g. "do (not|n't) (contain|
# mention|state|indicate|say|specify)" covers all four of the sanitize
# misses at once because they share the same grammatical shape, not
# because each exact sentence was pasted in here.
_REFUSAL_PATTERNS = [
    r"\bi don'?t know\b",
    r"\bno information\b",
    r"\bnot (?:mentioned|stated|specified|indicated|provided|available)\b",
    r"\bdo(?:es)?\s?n'?t\s+(?:contain|mention|state|indicate|say|specify|provide)\b",
    r"\bdo\s+not\s+(?:contain|mention|state|indicate|say|specify|provide)\b",
    r"\bdoes\s+not\s+(?:contain|mention|state|indicate|say|specify|provide)\b",
    r"\bthere\s+is\s+no\s+(?:mention|information|indication)\b",
    r"\bcannot\s+(?:determine|find|answer|confirm)\b",
    r"\bunable\s+to\s+(?:determine|find|answer|confirm)\b",
    r"\bnothing\s+in\s+the\s+memories\b",
]
_REFUSAL_RX = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)

# False-premise correction: "Tim does not surf" answering "how does Tim
# feel while surfing" -- a short, direct negation with no elaboration is
# treated as a refusal-shaped answer. Heuristic, not exhaustive; anything
# it does not catch falls through to the (still narrow) claim check below.
_SHORT_NEGATION_RX = re.compile(
    r"^\s*[A-Z][\w\s.'-]{0,40}\s+does\s+not\s+\w+\.?\s*$")


def detect_abstained(text: str) -> bool:
    if not text:
        return True
    t = text.strip()
    if _REFUSAL_RX.search(t):
        return True
    if _SHORT_NEGATION_RX.match(t):
        return True
    return False


# ------------------------------------------------------------------ data

def load_labels():
    keymap = {}
    for d in sorted(RESULTS.iterdir()):
        a, s = d / "answers.jsonl", d / "summary.json"
        if not (a.exists() and s.exists()):
            continue
        for line in a.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("category") != "adversarial" or not r.get("predicted"):
                continue
            k = hashlib.sha256(f"{d.name}|{r['qa_id']}".encode()).hexdigest()[:12]
            keymap[k] = {"cell_id": d.name, "qa_id": r["qa_id"],
                         "predicted": r["predicted"], "regex_said": r.get("correct")}

    human = {}
    if (LAB_DIR / "labels.jsonl").exists():
        for line in (LAB_DIR / "labels.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["label"] in ("a", "b") and r["item_key"] in keymap:
                human[r["item_key"]] = (r["label"] == "a")  # True = abstained
    return keymap, human


def kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    agree = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = Counter(a), Counter(b)
    exp = sum((pa[k] / n) * (pb[k] / n) for k in set(pa) | set(pb))
    if abs(1 - exp) < 1e-12:
        return float("nan")
    return (agree - exp) / (1 - exp)


def main():
    keymap, human = load_labels()
    if not keymap:
        print("No adversarial answers found.")
        return 1

    # --- calibration: expanded detector vs ALL 58 human labels ---------
    cal_keys = [k for k in human if k in keymap]
    old_calls = [bool(keymap[k]["regex_said"]) for k in cal_keys]
    new_calls = [detect_abstained(keymap[k]["predicted"]) for k in cal_keys]
    truth = [human[k] for k in cal_keys]

    print(f"Calibration set: {len(cal_keys)} hand-labeled items\n")
    print(f"{'detector':<22}{'agreement':>12}{'kappa':>10}")
    old_agree = sum(1 for a, b in zip(old_calls, truth) if a == b) / len(truth)
    new_agree = sum(1 for a, b in zip(new_calls, truth) if a == b) / len(truth)
    print(f"{'original ABSTAIN_RX':<22}{old_agree:>12.3f}{kappa(old_calls, truth):>10.3f}")
    print(f"{'expanded detector':<22}{new_agree:>12.3f}{kappa(new_calls, truth):>10.3f}")

    remaining_errs = sum(1 for a, b in zip(new_calls, truth) if a != b)
    print(f"\nexpanded detector still disagrees with you on "
          f"{remaining_errs}/{len(truth)} labeled items.")
    if remaining_errs:
        print("Disagreements (inspect before trusting the corrected table):")
        for k in cal_keys:
            if detect_abstained(keymap[k]["predicted"]) != human[k]:
                meta = keymap[k]
                print(f"  {meta['cell_id']}  {meta['qa_id']}")
                print(f"    you said {'abstain' if human[k] else 'answer'}, "
                      f"detector said {'abstain' if detect_abstained(meta['predicted']) else 'answer'}")
                print(f"    text: {meta['predicted'][:140]!r}")

    # --- apply: human label where we have it, detector otherwise -------
    final = {}
    for k, meta in keymap.items():
        if k in human:
            final[k] = {"abstained": human[k], "source": "human"}
        else:
            final[k] = {"abstained": detect_abstained(meta["predicted"]),
                        "source": "detector"}

    n_human = sum(1 for v in final.values() if v["source"] == "human")
    n_det = sum(1 for v in final.values() if v["source"] == "detector")
    print(f"\nFinal scoring: {n_human} human-labeled, {n_det} detector-scored "
          f"(of {len(final)} total).")

    # --- recompute per-cell adversarial abstention rate ------------------
    per_cell = defaultdict(list)
    for k, meta in keymap.items():
        per_cell[meta["cell_id"]].append(final[k]["abstained"])

    print(f"\n{'cell':<58}{'adv_abst (corrected)':>22}{'adv_abst (original)':>22}")
    cell_rates = {}
    for cell in sorted(per_cell):
        d = RESULTS / cell
        s = json.loads((d / "summary.json").read_text())
        corrected = sum(per_cell[cell]) / len(per_cell[cell])
        original = s.get("adversarial_abstention")
        cell_rates[cell] = {
            "condition": s.get("condition"), "conv_id": s.get("conv_id"),
            "n": len(per_cell[cell]), "adv_abst_corrected": corrected,
            "adv_abst_original": original,
        }
        flag = "  <-- changed" if original is not None and abs(corrected - original) > 0.001 else ""
        orig_str = f"{original:.4f}" if original is not None else "n/a"
        print(f"{cell:<58}{corrected:>22.4f}{orig_str:>22}{flag}")

    # --- condition-level means, equal weight per conversation, matching
    #     scripts/stats.py's convention -----------------------------------
    by_cond = defaultdict(list)
    for cell, row in cell_rates.items():
        by_cond[row["condition"]].append(row)

    print(f"\n{'condition':<14}{'n_conv':>8}{'corrected mean':>16}{'original mean':>16}{'delta vs none':>16}")
    cond_means = {}
    for cond, rows in sorted(by_cond.items()):
        c_mean = sum(r["adv_abst_corrected"] for r in rows) / len(rows)
        o_vals = [r["adv_abst_original"] for r in rows if r["adv_abst_original"] is not None]
        o_mean = sum(o_vals) / len(o_vals) if o_vals else None
        cond_means[cond] = c_mean
        o_str = f"{o_mean:.4f}" if o_mean is not None else "n/a"
        print(f"{cond:<14}{len(rows):>8}{c_mean:>16.4f}{o_str:>16}")
    if "none" in cond_means:
        base = cond_means["none"]
        print()
        for cond, m in sorted(cond_means.items()):
            if cond == "none":
                continue
            print(f"  {cond:<12} corrected delta vs none: {m - base:+.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "calibration": {"n": len(cal_keys), "original_kappa": kappa(old_calls, truth),
                        "expanded_kappa": kappa(new_calls, truth),
                        "original_agreement": old_agree, "expanded_agreement": new_agree},
        "per_item": {k: final[k] for k in final},
        "per_cell": cell_rates,
    }, indent=2))
    print(f"\nWrote {OUT}  (original summary.json files are untouched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())