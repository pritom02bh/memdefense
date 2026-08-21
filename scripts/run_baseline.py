# scripts/run_baseline.py
#
# THE VALIDATION RUN. First time Components 1-4 execute together on a
# full conversation, defenses off.
#
# WHY THIS MATTERS MORE THAN ANY OTHER SINGLE STEP: every number in the
# paper is a difference from this baseline. If the baseline is wrong,
# every BUD and every dASR is wrong, and no amount of care downstream
# recovers it. Comparing this against Mem0's published LoCoMo accuracy is
# the only way to catch a broken harness BEFORE spending real money on
# ~20 cells.
#
#   python3 -m scripts.run_baseline --conv 0
#   python3 -m scripts.run_baseline --conv 0 --max-turns 120   # cheaper probe
#   python3 -m scripts.run_baseline --conv 0 --resume          # continue after a crash
#
# Outputs:
#   results/baseline/{cell_id}/answers.jsonl    one line per question, checkpointed
#   results/baseline/{cell_id}/summary.json     accuracy overall and per category
#
# Rate limiting: ingest hits OpenAI's 200k TPM cap on long conversations
# (each turn resends the accumulated memory context, ~8.5k tokens/call by
# turn 400). We retry with exponential backoff rather than dropping turns
# -- a silently skipped turn is a missing memory, which corrupts accuracy
# in a way that looks like a real result.

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import time
from collections import defaultdict

from adapters.mem0_adapter import Mem0Adapter
from core.dataset import load_locomo
from core.judge import judge

MAX_RETRIES = 6
BASE_BACKOFF = 2.0   # seconds; doubles each attempt


def with_backoff(fn, label: str):
    """Retry on rate limits. Anything else raises immediately -- we want
    real errors loud, not buried under retries."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            is_rate_limit = "rate limit" in msg or "ratelimit" in msg or "429" in msg
            if not is_rate_limit or attempt == MAX_RETRIES - 1:
                raise
            wait = BASE_BACKOFF * (2 ** attempt)
            print(f"    rate limited on {label}; retry {attempt+1}/{MAX_RETRIES-1} "
                  f"in {wait:.0f}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def ingest_with_progress(adapter: Mem0Adapter, conv, every: int = 25):
    """Mem0's own ingest is silent for minutes. Drive it turn by turn so
    progress is visible and a rate limit on one turn doesn't lose the run."""
    written = 0
    errors = 0
    total = len(conv.turns)
    start = time.monotonic()

    for i, turn in enumerate(conv.turns, 1):
        single = dataclasses.replace(conv, turns=(turn,))
        try:
            stats = with_backoff(lambda: adapter.ingest(single), f"turn {i}")
            written += stats.memories_written
            errors += stats.errors
        except Exception as e:
            errors += 1
            print(f"    turn {i} failed permanently: {type(e).__name__}: {e}", flush=True)

        if i % every == 0 or i == total:
            elapsed = time.monotonic() - start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0
            print(f"  ingest {i}/{total} turns  ({elapsed/60:.1f}m elapsed, "
                  f"~{eta/60:.1f}m left, {errors} errors)", flush=True)

    return written, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", type=int, default=0, help="conversation index (0-9)")
    ap.add_argument("--max-turns", type=int, default=None,
                    help="cap turns for a cheaper probe run")
    ap.add_argument("--max-questions", type=int, default=None,
                    help="cap questions for a cheaper probe run")
    ap.add_argument("--resume", action="store_true",
                    help="skip questions already answered in answers.jsonl")
    ap.add_argument("--data", default="data/locomo/locomo10.json")
    args = ap.parse_args()

    convs = load_locomo(args.data)
    conv_full = convs[args.conv]
    conv = conv_full
    if args.max_turns:
        conv = dataclasses.replace(conv_full, turns=conv_full.turns[:args.max_turns])

    cell_id = f"baseline_mem0_{conv.conv_id}"
    if args.max_turns:
        cell_id += f"_t{args.max_turns}"

    out_dir = pathlib.Path("results/baseline") / cell_id
    out_dir.mkdir(parents=True, exist_ok=True)
    answers_file = out_dir / "answers.jsonl"

    done_ids = set()
    if args.resume and answers_file.exists():
        for line in answers_file.read_text().splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["qa_id"])
        print(f"Resuming: {len(done_ids)} questions already answered.")

    questions = list(conv_full.qa_pairs)
    if args.max_questions:
        questions = questions[:args.max_questions]

    print(f"\n{'='*64}")
    print(f"BASELINE RUN  cell_id={cell_id}")
    print(f"  conversation : {conv.conv_id}")
    print(f"  turns        : {len(conv.turns)} of {len(conv_full.turns)}")
    print(f"  questions    : {len(questions)} of {len(conv_full.qa_pairs)}")
    print(f"  defense      : NONE (baseline)")
    print(f"{'='*64}\n")

    adapter = Mem0Adapter(run_id="baseline", cell_id=cell_id)

    # Ingest only if we're not resuming into an already-populated namespace.
    if not (args.resume and done_ids):
        print("Clearing namespace and ingesting...", flush=True)
        adapter.reset(conv.conv_id)
        t0 = time.monotonic()
        written, errors = ingest_with_progress(adapter, conv)
        print(f"\nIngest complete: {written} memories written, {errors} errors, "
              f"{(time.monotonic()-t0)/60:.1f} minutes.\n", flush=True)
        stored = adapter.dump_memories()
        print(f"Memories in store: {len(stored)}\n", flush=True)
    else:
        adapter._namespace = conv.conv_id  # resume into the existing namespace
        print("Resuming -- skipping ingest.\n", flush=True)

    # Answer + judge, checkpointing every question.
    print("Answering and grading...", flush=True)
    t0 = time.monotonic()
    with open(answers_file, "a") as out:
        for i, qa in enumerate(questions, 1):
            if qa.qa_id in done_ids:
                continue
            try:
                answer_text, astats = with_backoff(
                    lambda: adapter.answer(qa.question), f"answer q{i}")
                verdict = with_backoff(
                    lambda: judge(qa.question, qa.answer, answer_text), f"judge q{i}")
                row = {
                    "qa_id": qa.qa_id,
                    "category": qa.category,
                    "question": qa.question,
                    "gold": qa.answer,
                    "predicted": answer_text,
                    "correct": verdict.correct,
                    "judge_reason": verdict.reason,
                    "latency_ms": astats.latency_ms,
                    "memories_retrieved": astats.memories_retrieved,
                }
            except Exception as e:
                row = {
                    "qa_id": qa.qa_id,
                    "category": qa.category,
                    "question": qa.question,
                    "gold": qa.answer,
                    "predicted": None,
                    "correct": None,
                    "error": f"{type(e).__name__}: {e}",
                }
                print(f"  q{i} failed: {row['error']}", flush=True)

            out.write(json.dumps(row) + "\n")
            out.flush()

            if i % 20 == 0:
                elapsed = time.monotonic() - t0
                print(f"  {i}/{len(questions)} questions "
                      f"({elapsed/60:.1f}m elapsed)", flush=True)

    # Summarise.
    rows = [json.loads(l) for l in answers_file.read_text().splitlines() if l.strip()]
    graded = [r for r in rows if r.get("correct") is not None]
    failed = [r for r in rows if r.get("correct") is None]

    by_cat = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in graded:
        by_cat[r["category"]]["n"] += 1
        by_cat[r["category"]]["correct"] += 1 if r["correct"] else 0

    overall = sum(1 for r in graded if r["correct"]) / len(graded) if graded else 0.0

    summary = {
        "cell_id": cell_id,
        "conv_id": conv.conv_id,
        "turns_ingested": len(conv.turns),
        "questions_graded": len(graded),
        "questions_failed": len(failed),
        "accuracy_overall": round(overall, 4),
        "accuracy_by_category": {
            cat: round(v["correct"] / v["n"], 4) for cat, v in sorted(by_cat.items())
        },
        "counts_by_category": {cat: v["n"] for cat, v in sorted(by_cat.items())},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*64}")
    print(f"BASELINE RESULT  ({len(graded)} graded, {len(failed)} failed)")
    print(f"{'='*64}")
    print(f"  overall accuracy : {overall:.1%}")
    print(f"\n  by category:")
    for cat, v in sorted(by_cat.items()):
        acc = v["correct"] / v["n"]
        print(f"    {cat:15} {acc:>6.1%}   ({v['correct']}/{v['n']})")
    print(f"\nWrote {out_dir}/summary.json")
    print("\nNEXT: compare overall accuracy against Mem0's published LoCoMo")
    print("figure. Close => harness is sound. Far off => diagnose before")
    print("building anything on top of this.")
    print("\nToken cost for this run:")
    print("  python3 -c \"import json;rows=[json.loads(l) for l in "
          "open('results/proxy/calls.jsonl')];r=[x for x in rows if "
          f"x.get('cell_id')=='{cell_id}'];"
          "print('calls',len(r));print('tokens',sum(x['tokens_total'] for x in r));"
          "print('usd',round(sum(x['cost_usd'] for x in r),4))\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())