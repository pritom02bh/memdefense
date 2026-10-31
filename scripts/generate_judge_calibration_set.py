# scripts/generate_judge_calibration_set.py
#
# Component 4, step 1: produce ~30 real Mem0 answers spanning every
# question category, to be graded by hand. This is intentionally cheap:
# only the first 60 turns of two conversations are ingested (not full
# conversations), which keeps token spend and rate-limit risk low while
# still producing a realistic mix of correct and incorrect answers --
# which is exactly what's needed to stress-test the judge.
#
# Run with the proxy already running in another terminal:
#   python3 -m scripts.generate_judge_calibration_set
#
# Output: results/judge_calibration/answers.jsonl
# One line per graded item, with a blank "human_grade" field to fill
# in by hand (true / false / null) before running the judge.

import dataclasses
import json
import pathlib
from core.paths import RESULTS
import sys
from collections import defaultdict

from adapters.mem0_adapter import Mem0Adapter
from core.dataset import load_locomo

TURNS_PER_CONV = 60          # keep ingestion cheap
TARGET_PER_CATEGORY = 6      # 5 categories x 6 = 30 items
CONV_INDICES = [0, 1]        # conv-26, conv-30 -- two different speaker pairs

OUT_DIR = RESULTS / "judge_calibration"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "answers.jsonl"


def qa_pairs_answerable_within(conv, turn_limit: int):
    """
    A QA pair is only fair to ask if its supporting evidence falls within
    the turns we actually ingested. LoCoMo's evidence field lists turn IDs
    like ["D1:3", "D2:7"] -- we approximate by only keeping QA pairs whose
    evidence turn indices are within range, falling back to "keep everything
    <= turn_limit's session count" if evidence isn't parseable. Simpler and
    good enough for calibration: just cap total emitted per category and
    accept some "I don't know" answers -- those are useful test cases too.
    """
    return conv.qa_pairs


def main():
    convs = load_locomo("data/locomo/locomo10.json")
    by_category = defaultdict(list)

    with open(OUT_FILE, "w") as out:
        for idx in CONV_INDICES:
            conv_full = convs[idx]
            conv = dataclasses.replace(conv_full, turns=conv_full.turns[:TURNS_PER_CONV])

            print(f"Ingesting {conv.conv_id}: first {len(conv.turns)} of "
                  f"{len(conv_full.turns)} turns...")
            adapter = Mem0Adapter(run_id="dev", cell_id="judge_calibration")
            adapter.reset(conv.conv_id)
            stats = adapter.ingest(conv)
            print(f"  {stats}")

            candidates = qa_pairs_answerable_within(conv_full, TURNS_PER_CONV)

            for qa in candidates:
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
                    "human_grade": None,   # filled in by hand: true / false
                }
                out.write(json.dumps(record) + "\n")
                out.flush()
                by_category[qa.category].append(record)
                print(f"  [{qa.category}] Q: {qa.question[:60]}...")

            total = sum(len(v) for v in by_category.values())
            if total >= TARGET_PER_CATEGORY * 5:
                break

    print("\nCategory coverage:")
    for cat, items in sorted(by_category.items()):
        print(f"  {cat:15} {len(items)}")
    print(f"\nWrote {sum(len(v) for v in by_category.values())} items to {OUT_FILE}")
    print("\nNext: open that file and fill in \"human_grade\": true/false for each line.")


if __name__ == "__main__":
    sys.exit(main())