# core/evidence_filter.py
#
# EVIDENCE-ALIGNED QUESTION FILTERING.
#
# THE PROBLEM: we truncate conversations to control cost, but LoCoMo
# questions were written against the FULL conversation. A question whose
# answer lives in turn 340 is unanswerable when only turns 1-150 were
# ingested -- the agent correctly says "I don't know" and is scored
# wrong. That is why the first matrix run scored 25.4% against a
# validated full-conversation baseline of 58.6%.
#
# THE FIX: keep only questions whose evidence turns all survive
# truncation. This does not inflate accuracy -- it stops penalising the
# system for a limitation we imposed on it.
#
# VALIDITY REQUIREMENT: apply to EVERY condition including the
# undefended baseline. Filtered defense cells compared against an
# unfiltered baseline would make BUD meaningless.
#
#   python3 -m core.evidence_filter
#   python3 -m core.evidence_filter --turns 100 150 200

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

from core.dataset import load_locomo


def evidence_refs(qa) -> list[str]:
    """Evidence turn identifiers, normalised to a list of strings.
    LoCoMo entries look like 'D2:7' (session 2, turn 7)."""
    ev = getattr(qa, "evidence", None)
    if not ev:
        return []
    if isinstance(ev, str):
        ev = [ev]
    return [str(e).strip() for e in ev if str(e).strip()]


def is_answerable(qa, kept_ids: set[str]) -> bool:
    """A question survives iff EVERY evidence turn survives.

    Requiring ALL (not ANY) is the conservative choice: a multi-hop
    question missing one of three supporting turns is not answerable in
    the sense the benchmark intends, even though one hop remains.

    Questions with NO evidence field are kept. In LoCoMo these are
    typically open_domain items drawing on world knowledge plus general
    context rather than one specific turn; dropping them would silently
    remove a whole category.
    """
    refs = evidence_refs(qa)
    if not refs:
        return True
    return all(r in kept_ids for r in refs)


def filter_questions(conv, max_turns: int | None) -> list:
    """The evaluation set for a conversation truncated to max_turns."""
    if not max_turns or max_turns >= len(conv.turns):
        return list(conv.qa_pairs)
    kept_ids = {str(t.turn_id) for t in conv.turns[:max_turns]}
    return [qa for qa in conv.qa_pairs if is_answerable(qa, kept_ids)]


def retention_report(convs, conv_indices, turn_points) -> None:
    """How many questions survive at each truncation point, per category.
    Report this table in the paper: disclosed, truncation is a
    cost-control design choice; undisclosed, it looks like an
    unexplained deviation from the benchmark."""
    cats = ["single_hop", "multi_hop", "temporal", "open_domain", "adversarial"]

    full = defaultdict(int)
    for ci in conv_indices:
        for qa in convs[ci].qa_pairs:
            full[qa.category] += 1
    full_total = sum(full.values())

    print(f"\nConversations: {', '.join(str(i) for i in conv_indices)}")
    print(f"Full question set: {full_total} "
          f"({', '.join(f'{c}={full[c]}' for c in cats if full[c])})\n")

    head = f"{'turns':>6} | " + " | ".join(f"{c[:11]:>11}" for c in cats) + " |    total"
    print(head)
    print("-" * len(head))
    for n in turn_points:
        counts = defaultdict(int)
        total = 0
        for ci in conv_indices:
            for qa in filter_questions(convs[ci], n):
                counts[qa.category] += 1
                total += 1
        print(f"{n:>6} | " + " | ".join(f"{counts[c]:>11}" for c in cats)
              + f" | {total:>8}")

    print("\nRetention (% of the full question set):\n")
    print(head)
    print("-" * len(head))
    for n in turn_points:
        counts = defaultdict(int)
        total = 0
        for ci in conv_indices:
            for qa in filter_questions(convs[ci], n):
                counts[qa.category] += 1
                total += 1
        cells = " | ".join(
            f"{(counts[c]/full[c]*100 if full.get(c) else 0):>10.0f}%" for c in cats)
        print(f"{n:>6} | {cells} | {total/full_total*100:>7.0f}%")

    print("\nNOTE: multi_hop degrades fastest -- its evidence is spread")
    print("across sessions by construction. The filtered set is therefore")
    print("slightly easier in composition than the full benchmark, which")
    print("is why this table belongs in the paper alongside the results.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, nargs="+",
                    default=[100, 150, 200, 250, 300, 419])
    ap.add_argument("--convs", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--data", default="data/locomo/locomo10.json")
    args = ap.parse_args()
    convs = load_locomo(args.data)
    retention_report(convs, args.convs, args.turns)
    return 0


if __name__ == "__main__":
    sys.exit(main())
