# scripts/grade_calibration.py
#
# Component 4, step 2: hand-grade the calibration set.
#
# Shows one item at a time, takes y/n, and writes grades back safely.
# Resumable -- already-graded items are skipped, so grading can stop and
# restart. Nothing is written until an answer is given, and the file is rewritten
# atomically after each grade so a crash can't corrupt it.
#
#   python3 -m scripts.grade_calibration
#
# GRADING RULE (must match prompts/judge.txt, or kappa measures the grader's
# inconsistency rather than the judge's quality):
#   y = the predicted answer conveys the same fact as gold, any wording
#   n = different fact, too vague to inform, or an unwarranted "I don't know"
#
# For adversarial items, gold is the corrected premise. If the agent
# confidently invented an event that never happened, that is n.

import json
import pathlib
from core.paths import RESULTS
import sys

IN_FILE = RESULTS / "judge_calibration" / "answers.jsonl"


def load_rows():
    return [json.loads(line) for line in IN_FILE.read_text().splitlines() if line.strip()]


def save_rows(rows):
    tmp = IN_FILE.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(IN_FILE)


def main():
    if not IN_FILE.exists():
        print(f"No file at {IN_FILE}. Run generate_judge_calibration_set first.")
        return 1

    rows = load_rows()
    ungraded = [i for i, r in enumerate(rows) if r.get("human_grade") is None]

    if not ungraded:
        graded = sum(1 for r in rows if r.get("human_grade") is not None)
        print(f"All {graded} items already graded. Run scripts/compute_kappa.py next.")
        return 0

    print(f"{len(ungraded)} items to grade ({len(rows) - len(ungraded)} already done).")
    print("y = correct, n = incorrect, s = skip for now, q = save and quit\n")

    for count, i in enumerate(ungraded, 1):
        row = rows[i]
        print("=" * 70)
        print(f"Item {count} of {len(ungraded)}   [{row['category']}]   {row['conv_id']}")
        print("=" * 70)
        print(f"\nQ:         {row['question']}")
        print(f"\nGOLD:      {row['gold']}")
        print(f"\nPREDICTED: {row['predicted']}\n")

        while True:
            choice = input("correct? (y/n/s/q): ").strip().lower()
            if choice in ("y", "n", "s", "q"):
                break
            print("  please enter y, n, s, or q")

        if choice == "q":
            save_rows(rows)
            remaining = sum(1 for r in rows if r.get("human_grade") is None)
            print(f"\nSaved. {remaining} items still ungraded -- rerun to continue.")
            return 0
        if choice == "s":
            continue

        rows[i]["human_grade"] = (choice == "y")
        save_rows(rows)
        print()

    remaining = sum(1 for r in rows if r.get("human_grade") is None)
    if remaining:
        print(f"\nDone for now. {remaining} skipped items still ungraded.")
    else:
        print("\nAll items graded. Next: python3 -m scripts.compute_kappa")
    return 0


if __name__ == "__main__":
    sys.exit(main())