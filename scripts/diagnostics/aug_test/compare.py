# scripts/diagnostics/aug_test/compare.py
#
# READ-ONLY, NO API CALLS. Today's rebuild of none conv3 s1 with the
# prompt date pinned to 2026-08-22 (repro__none__conv3__aug22, this
# folder's store) vs the August store anchor_gemini__gemini-backbone__none__conv3
# (its byte-identical copy in results_v3/mem0_qdrant; its calls in the
# read-only results/proxy/calls.jsonl): memory count, texts per turn_id,
# and each turn's extraction tokens_out. Verdict: REPRODUCED if the
# stored texts are identical on at least 145 of 150 turns, else NOT.
#
#   .venv/bin/python scripts/diagnostics/aug_test/compare.py

from __future__ import annotations

import collections
import json
import pathlib
import sys

sys.path.insert(0, ".")
from core.dataset import load_locomo
from scripts.diagnostics.repro_oct31.step1_compare import compare, memories
from scripts.diagnostics.gemini_change.step4_logs import per_turn

R = pathlib.Path(".").resolve()
HERE = R / "results_diag" / "aug_test"
TEST = "repro__none__conv3__aug22"
AUG = "anchor_gemini__gemini-backbone__none__conv3"
NEAR = 145


def main():
    turns = load_locomo("data/locomo/locomo10.json")[3].turns[:150]
    new = memories(HERE / "root" / "mem0_qdrant", [TEST])[TEST]
    old = memories(R / "results_v3" / "mem0_qdrant", [AUG])[AUG]
    log_new, _ = per_turn(R / "results_v3" / "proxy" / "calls.jsonl", TEST, turns)
    log_old, prior = per_turn(R / "results" / "proxy" / "calls.jsonl", AUG, turns)
    print(f"AUGUST TEST. none conv3 s1 (conv-42), today with MEMDEF_PROMPT_DATE=2026-08-22 vs the "
          f"August store\n  (August build: {prior} earlier extraction calls before its final ingestion)")
    compare("today", new, log_new, "august", old, log_old, turns)
    same = sum(new.get(t.turn_id, []) == old.get(t.turn_id, []) for t in turns)
    same_tok = sum((log_new.get(i) or {}).get("tokens_out") == (log_old.get(i) or {}).get("tokens_out")
                   for i in range(len(turns)))
    ts_new = [v["ts"] for v in log_new.values() if v]
    ts_old = [v["ts"] for v in log_old.values() if v]
    print(f"  today's ingestion:  {min(ts_new)[:19]} .. {max(ts_new)[:19]}")
    print(f"  August ingestion:   {min(ts_old)[:19]} .. {max(ts_old)[:19]}")
    rs = [json.loads(l) for l in (R / "results_v3" / "proxy" / "calls.jsonl").read_text().splitlines()
          if f'"{TEST}"' in l]
    cost = collections.Counter()
    for r in rs:
        cost[r["model_resolved"]] += r.get("cost_usd") or 0
    print(f"  test cost (logged): ${sum(cost.values()):.4f}  "
          + ", ".join(f"{m} ${c:.4f}" for m, c in sorted(cost.items()))
          + f"; {len(rs)} calls, {sum(r['status'] != 'ok' for r in rs)} not ok")
    verdict = "REPRODUCED" if same >= NEAR else "NOT REPRODUCED"
    print(f"\nVERDICT: {verdict} -- stored texts identical on {same}/150 turns, tokens_out equal on "
          f"{same_tok}/150 (threshold for near-identical: {NEAR}/150 turns)")


if __name__ == "__main__":
    main()
