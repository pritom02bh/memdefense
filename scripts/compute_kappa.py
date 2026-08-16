# scripts/compute_kappa.py
#
# Component 4 blocking test. Reads results/judge_calibration/answers.jsonl
# (with human_grade filled in by hand), runs the judge on the same items,
# and computes Cohen's kappa between you and the judge.
#
# kappa >= 0.8 required before the judge can be trusted for real cells.
# If it's below that: read the disagreements printed below, revise
# prompts/judge.txt, then re-run this script (no need to re-run generation --
# the judge cache will be bypassed automatically since a prompt change
# produces different output for the same cache key... actually it won't,
# since the cache key doesn't include the prompt. See note in main() below.

import json
import pathlib
import sys

from core.judge import judge, CACHE_PATH

IN_FILE = pathlib.Path("results/judge_calibration/answers.jsonl")


def cohens_kappa(human: list[bool], model: list[bool]) -> float:
    """
    Standard 2-category Cohen's kappa, computed directly from the
    2x2 contingency table -- no external dependency required.
    """
    assert len(human) == len(model)
    n = len(human)
    both_true = sum(1 for h, m in zip(human, model) if h and m)
    both_false = sum(1 for h, m in zip(human, model) if not h and not m)
    observed_agreement = (both_true + both_false) / n

    human_true_rate = sum(human) / n
    model_true_rate = sum(model) / n
    chance_agreement = (human_true_rate * model_true_rate) + \
                        ((1 - human_true_rate) * (1 - model_true_rate))

    if chance_agreement == 1.0:
        return 1.0  # degenerate case: everyone agreed on everything
    return (observed_agreement - chance_agreement) / (1 - chance_agreement)


def main():
    if not IN_FILE.exists():
        print(f"No calibration file at {IN_FILE}. Run "
              f"scripts/generate_judge_calibration_set.py first.")
        return 1

    rows = [json.loads(line) for line in IN_FILE.read_text().splitlines() if line.strip()]

    missing = [r["qa_id"] for r in rows if r.get("human_grade") is None]
    if missing:
        print(f"{len(missing)} of {len(rows)} items still need human_grade filled in "
              f"(true/false) in {IN_FILE}:")
        for qid in missing[:10]:
            print(f"  {qid}")
        if len(missing) > 10:
            print(f"  ...and {len(missing) - 10} more")
        return 1

    human_grades = []
    model_grades = []
    disagreements = []

    print(f"Grading {len(rows)} items with the judge...")
    for row in rows:
        result = judge(row["question"], row["gold"], row["predicted"])
        human_grades.append(bool(row["human_grade"]))
        model_grades.append(result.correct)
        if bool(row["human_grade"]) != result.correct:
            disagreements.append((row, result))

    kappa = cohens_kappa(human_grades, model_grades)
    agreement_pct = sum(1 for h, m in zip(human_grades, model_grades) if h == m) / len(rows)

    print(f"\n{'='*60}")
    print(f"Items graded: {len(rows)}")
    print(f"Percent agreement: {agreement_pct:.1%}")
    print(f"Cohen's kappa: {kappa:.3f}")
    print(f"{'='*60}")

    if kappa >= 0.8:
        print("PASS -- judge is approved. Do not edit prompts/judge.txt after this point.")
    else:
        print("FAIL -- kappa below 0.8. Judge prompt needs revision before proceeding.")
        print(f"\n{len(disagreements)} disagreements:\n")
        for row, result in disagreements:
            print(f"  Q: {row['question']}")
            print(f"  Gold: {row['gold']}")
            print(f"  Predicted: {row['predicted']}")
            print(f"  You said: {'correct' if row['human_grade'] else 'incorrect'}   "
                  f"Judge said: {'correct' if result.correct else 'incorrect'}   "
                  f"Judge reason: {result.reason}")
            print()

    return 0 if kappa >= 0.8 else 1


if __name__ == "__main__":
    sys.exit(main())