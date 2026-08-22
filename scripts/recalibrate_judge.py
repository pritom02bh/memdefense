# scripts/recalibrate_judge.py
#
# Tests an ALTERNATIVE judge model against the calibration items you
# already hand-graded. Reports Cohen's kappa so a cheaper (or
# cross-provider) judge can be adopted on evidence rather than hope.
#
# WHY THIS IS SAFE TO RUN: it does not touch prompts/judge.txt, and it
# writes to a separate cache keyed by model, so the approved gpt-4o judge
# and its kappa=0.841 remain intact regardless of the outcome.
#
# WHY IT MATTERS: kappa is a property of (judge model + prompt + task).
# Swapping the judge invalidates the existing calibration completely. A
# cheaper judge that agrees with you at kappa=0.62 would save money and
# quietly destroy the trustworthiness of every accuracy number.
#
# NOTE ON SCOPE: adversarial items are excluded. They are scored by
# deterministic abstention detection, not by the judge, so they no longer
# belong in judge calibration. This means kappa here is computed over the
# ~24 core-4 items -- report that n honestly in the paper.
#
#   python3 -m scripts.recalibrate_judge --model judge-claude
#   python3 -m scripts.recalibrate_judge --model judge          # re-check gpt-4o

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

from openai import OpenAI

PROXY = "http://localhost:4000/v1"
CAL_FILE = pathlib.Path("results/judge_calibration/answers.jsonl")
PROMPT_PATH = pathlib.Path("prompts/judge.txt")


def cohens_kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    obs = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    chance = pa * pb + (1 - pa) * (1 - pb)
    if chance >= 1.0:
        return 1.0
    return (obs - chance) / (1 - chance)


def grade(client, model: str, template: str, q: str, gold: str, pred: str) -> bool:
    prompt = template.format(question=q, gold=gold, predicted=pred)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
        extra_headers={"x-purpose": "judge", "x-run-id": "recal",
                       "x-cell-id": f"recal_{model}"},
    )
    raw = (resp.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = "\n".join(l for l in raw.splitlines() if not l.strip().startswith("```")).strip()
    parsed = json.loads(raw)
    return parsed["grade"].strip().lower() == "correct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="judge-claude",
                    help="alias of the judge model to test")
    args = ap.parse_args()

    if not CAL_FILE.exists():
        print(f"No calibration file at {CAL_FILE}")
        return 1

    rows = [json.loads(l) for l in CAL_FILE.read_text().splitlines() if l.strip()]
    items = [r for r in rows
             if r.get("human_grade") is not None and r["category"] != "adversarial"]
    print(f"Testing judge '{args.model}' on {len(items)} core-4 hand-graded items "
          f"({len(rows) - len(items)} excluded: adversarial or ungraded).\n")

    template = PROMPT_PATH.read_text()
    client = OpenAI(base_url=PROXY, api_key="proxy-dummy-key")

    human, model_grades, disagreements = [], [], []
    for i, r in enumerate(items, 1):
        try:
            g = grade(client, args.model, template,
                      r["question"], r["gold"], r["predicted"])
        except Exception as e:
            print(f"  item {i} failed: {type(e).__name__}: {str(e)[:120]}")
            return 1
        human.append(bool(r["human_grade"]))
        model_grades.append(g)
        if bool(r["human_grade"]) != g:
            disagreements.append((r, g))

    k = cohens_kappa(human, model_grades)
    agree = sum(1 for x, y in zip(human, model_grades) if x == y) / len(items)

    print("=" * 58)
    print(f"  judge model      : {args.model}")
    print(f"  items            : {len(items)}")
    print(f"  percent agreement: {agree:.1%}")
    print(f"  Cohen's kappa    : {k:.3f}")
    print("=" * 58)

    if k >= 0.8:
        print(f"\nPASS -- '{args.model}' is adequate as a judge.")
        print("To adopt it, point core/judge.py at this alias and record the")
        print("new kappa, item count, and date in versions.md. The prompt is")
        print("unchanged, so the cache key (which includes the prompt hash)")
        print("stays valid -- but clear results/judge_cache.jsonl anyway, since")
        print("the cache key does NOT include the judge model.")
    else:
        print(f"\nFAIL -- kappa {k:.3f} is below 0.8. Do not adopt this judge.")
        print("Keep the calibrated one. A cheaper judge that disagrees with you")
        print("saves money and invalidates every accuracy number in the paper.")

    if disagreements:
        print(f"\n{len(disagreements)} disagreement(s):\n")
        for r, g in disagreements:
            print(f"  [{r['category']}] {r['question']}")
            print(f"    gold: {r['gold']}")
            print(f"    pred: {(r['predicted'] or '')[:110]}")
            print(f"    you: {'correct' if r['human_grade'] else 'incorrect'}   "
                  f"{args.model}: {'correct' if g else 'incorrect'}\n")
    return 0 if k >= 0.8 else 1


if __name__ == "__main__":
    sys.exit(main())