# scripts/diagnostics/repro_oct31/step3_compare.py
#
# READ-ONLY, NO API CALLS. Today's rebuild (repro__none__conv0__oct31, in
# this folder's store) vs none__conv0__s2__v3b (Oct 30, results_v3 store):
# memory count, texts as multisets and per turn_id, and each turn's
# extraction tokens_out from results_v3/proxy/calls.jsonl. Then the
# verdict and the test's logged cost.
#
#   python3 scripts/diagnostics/repro_oct31/step3_compare.py

from __future__ import annotations

import collections
import json
import pathlib
import sys

sys.path.insert(0, ".")
from core.dataset import load_locomo
from scripts.diagnostics.repro_oct31.step1_compare import LOG, S2, STORE, compare, memories
from scripts.diagnostics.gemini_change.step4_logs import per_turn

HERE = pathlib.Path("results_diag/repro_oct31").resolve()
TEST = "repro__none__conv0__oct31"


def main():
    turns = load_locomo("data/locomo/locomo10.json")[0].turns[:150]
    new = memories(HERE / "root" / "mem0_qdrant", [TEST])[TEST]
    old = memories(STORE, [S2])[S2]
    log_new, log_old = per_turn(LOG, TEST, turns)[0], per_turn(LOG, S2, turns)[0]
    print("STEP 3. today's rebuild (Oct 31) vs none__conv0__s2__v3b (Oct 30)")
    first_mem, first_tok = compare("today", new, log_new, "s2_v3b", old, log_old, turns)
    n_new = sum(len(v) for v in new.values())
    n_old = sum(len(v) for v in old.values())
    mn = collections.Counter(x for v in new.values() for x in v)
    mo = collections.Counter(x for v in old.values() for x in v)
    ts = [v["ts"] for v in log_new.values() if v]
    print(f"  today's ingestion: {min(ts)[:19]} .. {max(ts)[:19]}")

    print("\nVERDICT")
    if n_new == n_old and mn == mo and first_mem is None and first_tok is None:
        print("  IDENTICAL: same count, same texts (as sets and per turn_id), same tokens_out "
              "on every turn.")
    else:
        print("  NOT IDENTICAL")
        if first_tok:
            print(f"  first turn whose tokens_out differs: turn {first_tok[0]} ({first_tok[1]})")
        if first_mem:
            print(f"  first turn whose memories differ: turn {first_mem[0]} ({first_mem[1]})")
        print(f"  memories: today {n_new} vs {n_old}; differing memories: "
              f"{sum((mn - mo).values())} only today, {sum((mo - mn).values())} only Oct 30")

    rows = [json.loads(l) for l in LOG.read_text().splitlines()]
    mine = [r for r in rows if r.get("cell_id") == TEST]
    by = collections.Counter()
    for r in mine:
        by[(r.get("step"), r["model_resolved"], r["status"])] += 1
    cost = collections.Counter()
    for r in mine:
        cost["google" if "gemini" in r["model_resolved"] else "openai"] += r.get("cost_usd") or 0
    print(f"\nCOST (logged price, this test's rows only): {len(mine)} calls "
          + ", ".join(f"{s}/{m}/{st}: {n}" for (s, m, st), n in sorted(by.items())))
    print(f"  Google ${cost['google']:.4f}, OpenAI ${cost['openai']:.4f}, total "
          f"${sum(cost.values()):.4f}")


if __name__ == "__main__":
    main()
