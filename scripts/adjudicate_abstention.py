#!/usr/bin/env python3
# Re-checks the hand labels that disagree with the abstention detector.
# For each one it shows who really said the evidence turn, plus every other
# label given on the same question, and keeps or changes it on input.
# Changes are appended as the owner's resolutions (core/labels.py), so
# every annotator's own labels and the audit trail survive.
#
#   python3 -m scripts.adjudicate_abstention
import collections
import datetime
import hashlib
import json
import pathlib
import re
import sys

from core.dataset import load_locomo
from core import labels as lb
from core import manifest as mf
from core.paths import CELLS, REPO_ROOT
from scripts.rescore_abstention import detect_abstained

DATA = str(REPO_ROOT / "data" / "locomo" / "locomo10.json")


def jl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


items = {}
for d in mf.cell_dirs():
    for r in jl(d / "answers.jsonl"):
        if r.get("category") != "adversarial" or not r.get("predicted"):
            continue
        k = hashlib.sha256(f"{d.name}|{r['qa_id']}".encode()).hexdigest()[:12]
        parts = d.name.split("__")
        seed = next((p for p in parts if re.fullmatch(r"s\d+", p)), "s1")
        items[k] = {"cond": parts[2], "seed": seed,
                    "qa": r["qa_id"], "question": r.get("question", ""), "text": r["predicted"]}

# the final labels, as rescore_abstention reads them
labels = {k: lab for k, lab in lb.decided(lb.ABSTENTION, lb.final(lb.ABSTENTION)).items()
          if k in items}

disputed = [k for k, lab in labels.items()
            if (lab == "a") != detect_abstained(items[k]["text"])]
if not disputed:
    raise SystemExit("No disagreements between the hand labels and the detector.")

qa_index = {q.qa_id: (c, q) for c in load_locomo(DATA) for q in c.qa_pairs}
by_qa = collections.defaultdict(list)
for k in labels:
    by_qa[items[k]["qa"]].append(k)

print("\nRubric (unchanged from labeling):")
print("  a = declines, says the memories lack it, or corrects a false premise")
print("  b = answers the question asked with substantive content, even hedged\n")

changes = []
for n, k in enumerate(disputed, 1):
    it = items[k]
    conv, qa = qa_index.get(it["qa"], (None, None))
    print("=" * 76)
    print(f"[{n}/{len(disputed)}]  {it['qa']}   final label: {labels[k]}   "
          f"detector: {'a' if detect_abstained(it['text']) else 'b'}")
    print(f"Q:      {it['question']}")
    if qa is not None:
        turns = {str(t.turn_id): t for t in conv.turns}
        for e in qa.evidence:
            t = turns.get(str(e))
            if t is not None:
                print(f"SOURCE: {e} {t.speaker}: {t.text[:150]}")
    print(f"ANSWER: {it['text'][:300]}")
    others = [o for o in by_qa[it["qa"]] if o != k]
    if others:
        print("Final labels of other answers to this question:")
        for o in others:
            print(f"   {labels[o]}  {items[o]['seed']}  {items[o]['text'][:110]}")
    while True:
        ch = input("keep (k) / change to a / change to b / quit (q) > ").strip().lower()
        if ch in ("k", "a", "b", "q"):
            break
    if ch == "q":
        break
    if ch in ("a", "b") and ch != labels[k]:
        changes.append({"item_key": k, "label": ch, "by": lb.owner(),
                        "resolved": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "note": "adjudicated vs rubric"})

if changes:
    with open(lb.ABSTENTION.resolutions_file, "a") as f:
        for c in changes:
            f.write(json.dumps(c) + "\n")
print(f"\n{len(changes)} label(s) changed. Rerun python3 -m scripts.rescore_abstention.")