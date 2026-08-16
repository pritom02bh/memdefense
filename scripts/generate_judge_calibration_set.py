# scripts/generate_judge_calibration_set.py
#
# Component 4, step 1: produce ~30 real Mem0 answers spanning every
# question category, for YOU to hand-grade.

import dataclasses
import json
import pathlib
import sys
from collections import defaultdict

from adapters.mem0_adapter import Mem0Adapter
from core.dataset import load_locomo

TURNS_PER_CONV = 60
TARGET_PER_CATEGORY = 6
CONV_INDICES = [0, 1]

OUT_DIR = pathlib.Path("results/judge_calibration")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "answers.jsonl"


def main():
    convs = load_locomo("data/locomo/locomo10.json")
    by_category = defaultdict(list)

    with open(OUT_FILE, "w") as out:
        for idx in CONV_INDICES:
            conv_full = convs[idx]
            conv = dataclasses.replace(conv_full, turns=conv_full.turns[:TURNS_PER_CONV])

            print(f"Ingesting {conv.conv_id}: first {len(conv.turns)} of "
                  f"{len(conv_full.turns)} turns...", flush=True)
            adapter = Mem0Adapter(run_id="dev", cell_id="judge_calibration")
            adapter.reset(conv.conv_id)
            stats = adapter.ingest(conv)
            print(f"  {stats}", flush=True)

            for qa in conv_full.qa_pairs:
                if len(by_category[qa.category]) >= TARGET_PER_CATEGORY:
                    continue
                answer_text, answer_stats = adapter.answer(qa.question)
                record = {
                    "conv_id": conv.conv_id,
                    "qa_id": qa.qa_id,
                    "category": qa.category,
                    "question": qa.question,
                    "gold": qa.answer,
                    "predicted": answer_text,
                    "human_grade": None,
                }
                out.write(json.dumps(record) + "\n")
                out.flush()
                by_category[qa.category].append(record)
                print(f"  [{qa.category}] {qa.question[:60]}", flush=True)

            if sum(len(v) for v in by_category.values()) >= TARGET_PER_CATEGORY * 5:
                break

    print("\nCategory coverage:")
    for cat, items in sorted(by_category.items()):
        print(f"  {cat:15} {len(items)}")
    print(f"\nWrote {sum(len(v) for v in by_category.values())} items to {OUT_FILE}")
    print("\nNext: fill in \"human_grade\": true/false for each line.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
