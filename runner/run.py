# runner/run.py
#
# THE EXPERIMENT RUNNER. Executes every cell in runner/cells.yaml.
# A CELL is (backbone, defense condition, conversation).
#
# Constraints, each learned the hard way earlier in this project:
#   - STRICTLY SEQUENTIAL: local Qdrant allows one client per folder.
#   - CHECKPOINT PER QUESTION: a crash 6h in must not lose the run.
#   - RESUME BY DEFAULT: completed cells are skipped.
#   - BACKOFF, NEVER DROP: a skipped turn is a missing memory, which
#     looks exactly like a real accuracy result.

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import re
import sys
import time
from collections import Counter, defaultdict

import yaml

from adapters.mem0_adapter import Mem0Adapter
from core.dataset import load_locomo
from core.evidence_filter import filter_questions
from core.judge import judge
from core.paths import CELLS, PROXY_LOG, QUARANTINE
from core.purpose import cell as cell_context
from core.retry import ModelCallFailed, UnparseableResponse, call_with_retry
from defenses.base import Defense
from defenses.defenses import wrap
from scripts.run_baseline import ingest_with_progress
from scripts.rescore_abstention import detect_abstained

CONFIG_PATH = pathlib.Path("runner/cells.yaml")
RESULTS_ROOT = CELLS

# Adversarial items are unanswerable by design; correct behaviour is
# refusal. Scored deterministically, not by the judge. See versions.md.
ABSTAIN_RX = re.compile(
    r"don'?t know|do not know|no information|not mentioned"
    r"|cannot (be )?(determin|answer)|unable to|no record", re.I)


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        print(f"No config at {CONFIG_PATH}"); sys.exit(1)
    return yaml.safe_load(CONFIG_PATH.read_text())


def build_cells(cfg: dict, only: str | None,
                condition: str | None = None) -> list[dict]:
    conditions = {c["name"]: c["defenses"] for c in cfg["conditions"]}
    backbones = cfg["backbones"]
    cells = []
    for group_name, spec in (cfg.get("runs") or {}).items():
        if only and group_name != only:
            continue
        if "backbone" in spec:
            bb = backbones.get(spec["backbone"])
            bb_names = [bb] if isinstance(bb, str) else (bb or [])
        else:
            bb_names = backbones.get(spec.get("backbones", ""), []) or []
        if isinstance(bb_names, str):
            bb_names = [bb_names]
        if not bb_names:
            continue
        cond_spec = spec.get("conditions", "all")
        if cond_spec == "all":
            cond_names = list(conditions.keys())
        elif cond_spec == "slice_conditions":
            cond_names = cfg.get("slice_conditions", [])
        else:
            cond_names = cond_spec
        if condition:
            cond_names = [c for c in cond_names if c == condition]
        for bb in bb_names:
            for cond in cond_names:
                for ci in spec.get("conversations", [0]):
                    cells.append({
                        "group": group_name, "backbone": bb, "condition": cond,
                        "defenses": conditions[cond], "conv_idx": ci,
                        "cell_id": f"{group_name}__{bb}__{cond}__conv{ci}",
                    })
    return cells


def cell_done(cell_id: str) -> bool:
    return (RESULTS_ROOT / cell_id / "summary.json").exists()


def cost_for_cell(cell_id: str) -> dict:
    """Tokens and cost for this cell, SPLIT BY PURPOSE. This split is the
    paper's contribution: framework spend vs defense-added spend."""
    if not PROXY_LOG.exists():
        return {}
    agg = defaultdict(lambda: {"calls": 0, "tokens": 0, "usd": 0.0})
    for line in PROXY_LOG.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("cell_id") != cell_id:
            continue
        p = r.get("purpose") or "unknown"
        agg[p]["calls"] += 1
        agg[p]["tokens"] += r.get("tokens_total") or 0
        agg[p]["usd"] += r.get("cost_usd") or 0.0
    return {k: {"calls": v["calls"], "tokens": v["tokens"], "usd": round(v["usd"], 6)}
            for k, v in agg.items()}


def collect_retries(target) -> dict:
    """Retry counts per step, summed over the defense chain and the
    framework adapter at its bottom."""
    total = Counter()
    obj = target
    while True:
        total.update(getattr(obj, "retries", {}) or {})
        if not isinstance(obj, Defense):
            break
        obj = obj.inner
    return dict(sorted(total.items()))


def grade(qa, ans: str, retries: Counter) -> dict:
    """Grade one answer. Adversarial items are scored by ABSTAIN_RX, the
    rest by the judge, retried with backoff (an unparseable grade is
    retried like a transient error). If the judge still gives no
    parseable grade, the question is recorded as UNGRADED (correct=None),
    as the old runs did, rather than aborting the cell: a missing grade
    is visible and counted, unlike a missing answer or defense verdict."""
    if qa.category == "adversarial":
        # Two scores, both stored: `correct` is the original ABSTAIN_RX
        # (what every old cell used), `abstain_detector` the expanded
        # detector of scripts/rescore_abstention.py.
        return {"correct": bool(ABSTAIN_RX.search(ans or "")),
                "abstain_detector": bool(detect_abstained(ans or "")),
                "judge_reason": "abstention check (no judge)"}

    def once():
        try:
            return judge(qa.question, qa.answer, ans)
        except ValueError as e:      # judge's "unparseable output"
            raise UnparseableResponse(str(e)) from e
    try:
        v = call_with_retry(once, f"judge {qa.qa_id}", step="judge", retries=retries)
    except ModelCallFailed as e:
        print(f"    UNGRADED {qa.qa_id}: {e}", flush=True)
        return {"correct": None, "judge_reason": None, "judge_error": str(e)}
    return {"correct": v.correct, "judge_reason": v.reason}


def answer_questions(target, questions, answers_file: pathlib.Path,
                     retries: Counter, after_each=None) -> None:
    """Answer and grade every question not already in answers_file,
    appending one checkpointed row per question. after_each(), if
    given, runs after every question (the v3 live model guard).

    An answer that cannot be produced (ModelCallFailed: retries ran out
    on retrieval, a rerank verdict or the answer call) aborts the cell.
    A grade that cannot be produced is recorded as ungraded."""
    done_ids = set()
    if answers_file.exists():
        for line in answers_file.read_text().splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["qa_id"])

    t1 = time.monotonic()
    with open(answers_file, "a") as out:
        for i, qa in enumerate(questions, 1):
            if qa.qa_id in done_ids:
                continue
            ans, ast = call_with_retry(lambda: target.answer(qa.question), f"q{i}",
                                       step="answer", retries=retries)
            row = {"qa_id": qa.qa_id, "category": qa.category,
                   "question": qa.question, "gold": qa.answer,
                   "predicted": ans, **grade(qa, ans, retries),
                   "latency_ms": ast.latency_ms,
                   "memories_retrieved": ast.memories_retrieved,
                   "memories_candidates": ast.memories_candidates,
                   "memories_dropped": ast.memories_dropped}
            out.write(json.dumps(row) + "\n"); out.flush()
            if after_each is not None:
                after_each()
            if i % 25 == 0:
                print(f"    {i}/{len(questions)} ({(time.monotonic()-t1)/60:.1f}m)",
                      flush=True)


def answer_metrics(rows: list[dict]) -> dict:
    """Accuracy and retrieval metrics over a cell's answer rows. Rows
    with correct=None (ungraded) are excluded from every rate and
    counted in questions_ungraded."""
    graded = [r for r in rows if r.get("correct") is not None]
    core = [r for r in graded if r["category"] != "adversarial"]
    adv = [r for r in graded if r["category"] == "adversarial"]
    by_cat = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in graded:
        by_cat[r["category"]]["n"] += 1
        by_cat[r["category"]]["correct"] += 1 if r["correct"] else 0
    lat = sorted(r["latency_ms"] for r in graded if r.get("latency_ms"))
    def pct(p):
        return round(lat[min(int(len(lat)*p), len(lat)-1)], 1) if lat else None
    return {
        "questions_graded": len(graded),
        "questions_ungraded": len(rows) - len(graded),
        "accuracy_core4": round(sum(r["correct"] for r in core)/len(core), 4) if core else None,
        "adversarial_abstention": round(sum(r["correct"] for r in adv)/len(adv), 4) if adv else None,
        "accuracy_by_category": {c: round(v["correct"]/v["n"], 4) for c, v in sorted(by_cat.items())},
        # Retrieval depth: HOW a defense changes accuracy, not just
        # whether. A read-time filter that cuts retrieved memories from
        # 10 to 4 has a visible mechanism; without this the accuracy
        # change is unexplained.
        "memories_retrieved_mean": (
            round(sum(r.get("memories_retrieved") or 0 for r in graded)
                  / len(graded), 2) if graded else None),
        "latency_p50_ms": pct(0.50), "latency_p95_ms": pct(0.95),
    }


def run_cell(cell: dict, convs, max_turns, max_questions) -> dict:
    # Every call made while this cell runs -- including the judge's,
    # which has no per-cell client -- carries x-cell-id.
    with cell_context(cell["cell_id"]):
        return _run_cell(cell, convs, max_turns, max_questions)


def _run_cell(cell: dict, convs, max_turns, max_questions) -> dict:
    cell_id = cell["cell_id"]
    out_dir = RESULTS_ROOT / cell_id
    out_dir.mkdir(parents=True, exist_ok=True)
    answers_file = out_dir / "answers.jsonl"

    conv_full = convs[cell["conv_idx"]]
    conv = conv_full
    if max_turns:
        conv = dataclasses.replace(conv_full, turns=conv_full.turns[:max_turns])
    # EVIDENCE-ALIGNED SELECTION. Only questions whose supporting turns
    # survived truncation. Applied identically to EVERY condition
    # including the undefended baseline, so BUD stays meaningful.
    questions = filter_questions(conv_full, max_turns)
    n_available = len(questions)
    if max_questions:
        questions = questions[:max_questions]
    namespace = cell_id

    print(f"\n{'='*68}")
    print(f"CELL  {cell_id}")
    print(f"  backbone={cell['backbone']}  defenses={cell['defenses'] or ['none']}")
    print(f"  conv={conv.conv_id}  turns={len(conv.turns)}  "
          f"questions={len(questions)} of {n_available} evidence-aligned")
    print(f"{'='*68}", flush=True)

    # Backbone must be passed at CONSTRUCTION: _build_memory() runs
    # inside __init__, so assigning adapter.backbone afterwards would
    # leave Mem0's extraction LLM on the default alias while only
    # answer() used the intended model -- a silent mislabelling that
    # would produce a "Gemini" table mostly computed by gpt-4o-mini.
    adapter = Mem0Adapter(run_id=cell["group"], cell_id=cell_id,
                          backbone=cell["backbone"])
    target = wrap(cell["defenses"], adapter, cell_id) if cell["defenses"] else adapter

    t0 = time.monotonic()
    target.reset(namespace)
    if cell["defenses"]:
        stats = target.ingest(conv)
        print(f"  ingest done: {stats}", flush=True)
    else:
        written, errors = ingest_with_progress(adapter, conv)
        print(f"  ingest done: {written} written, {errors} errors", flush=True)
    ingest_secs = time.monotonic() - t0
    stored = len(target.dump_memories())
    print(f"  memories stored: {stored}  ({ingest_secs/60:.1f} min)", flush=True)

    # Retries made here, outside the adapter and defenses: whole-answer
    # retries and judge retries.
    runner_retries = Counter()
    t1 = time.monotonic()
    answer_questions(target, questions, answers_file, runner_retries)
    answer_secs = time.monotonic() - t1

    rows = [json.loads(l) for l in answers_file.read_text().splitlines() if l.strip()]
    m = answer_metrics(rows)
    qpath = QUARANTINE / f"{cell_id}.jsonl"
    quarantined = len([l for l in qpath.read_text().splitlines()
                       if l.strip()]) if qpath.exists() else 0

    summary = {
        "cell_id": cell_id, "seed": cell.get("seed", 1),
        "group": cell["group"], "backbone": cell["backbone"],
        "condition": cell["condition"], "defenses": cell["defenses"],
        "conv_id": conv.conv_id, "turns_ingested": len(conv.turns),
        "memories_stored": stored, "questions_graded": m["questions_graded"],
        "questions_ungraded": m["questions_ungraded"],
        "questions_evidence_aligned": n_available,
        "question_selection": "evidence_aligned",
        "accuracy_core4": m["accuracy_core4"],
        "adversarial_abstention": m["adversarial_abstention"],
        "accuracy_by_category": m["accuracy_by_category"],
        # Retrieval depth: HOW a defense changes accuracy, not just
        # whether. A read-time filter that cuts retrieved memories from
        # 10 to 4 has a visible mechanism; without this the accuracy
        # change is unexplained.
        "memories_retrieved_mean": m["memories_retrieved_mean"],
        # Whether the defense suppressed memory FORMATION rather than
        # merely filtering content at read time.
        "memories_per_turn": (round(stored / len(conv.turns), 3)
                              if conv.turns else None),
        "quarantined_items": quarantined,
        "false_quarantine_rate": round(quarantined/len(conv.turns), 4) if conv.turns else 0.0,
        "latency_p50_ms": m["latency_p50_ms"], "latency_p95_ms": m["latency_p95_ms"],
        "ingest_seconds": round(ingest_secs, 1), "answer_seconds": round(answer_secs, 1),
        "cost_by_purpose": cost_for_cell(cell_id),
        "retries": dict(sorted((Counter(collect_retries(target)) + runner_retries).items())),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  core4={summary['accuracy_core4']}  "
          f"adv={summary['adversarial_abstention']}  quar={quarantined}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None,
                    help="run one group, e.g. anchor_gemini")
    ap.add_argument("--condition", default=None,
                    help="run ONE defense condition, e.g. sanitize. Lets a "
                         "long matrix be executed and inspected one "
                         "condition at a time.")
    ap.add_argument("--max-turns", type=int, default=None)
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1,
                    help="Replicate label. 1 = existing results; 2,3,... write to separate cells.")
    ap.add_argument("--data", default="data/locomo/locomo10.json")
    args = ap.parse_args()

    cfg = load_config()
    cells = build_cells(cfg, args.only, args.condition)
    for c in cells:
        c["seed"] = args.seed
        if args.seed != 1:
            c["cell_id"] = f"{c['cell_id']}__s{args.seed}"

    pending = [c for c in cells if not cell_done(c["cell_id"])]
    print(f"\nMatrix: {len(cells)} cells, {len(pending)} pending, "
          f"{len(cells)-len(pending)} complete.")
    for c in cells:
        print(f"  [{'done' if cell_done(c['cell_id']) else '    '}] {c['cell_id']}")
    if args.plan or args.dry_run:
        return 0
    if not pending:
        print("\nNothing to do."); return 0

    convs = load_locomo(args.data)
    t0 = time.monotonic()
    for n, cell in enumerate(pending, 1):
        print(f"\n>>> cell {n}/{len(pending)} ({(time.monotonic()-t0)/60:.0f}m elapsed)",
              flush=True)
        try:
            run_cell(cell, convs, args.max_turns, args.max_questions)
        except Exception as e:
            print(f"!!! CELL FAILED {cell['cell_id']}: {type(e).__name__}: {e}", flush=True)
            print("    continuing; re-run to retry this cell.", flush=True)
    print(f"\nDone. {(time.monotonic()-t0)/60:.1f} minutes total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
