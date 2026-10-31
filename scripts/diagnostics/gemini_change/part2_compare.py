# scripts/diagnostics/gemini_change/part2_compare.py
#
# READ-ONLY. Three-way per-turn comparison for conv-30, first 150 turns:
#   old   = none conv1 s1 final ingestion (results/proxy/calls.jsonl, Aug 22)
#   v3    = none__conv1__v3 (results_v3/proxy/calls.jsonl, Oct 29)
#   today = the c72464e code today (results_diag/proxy/calls.jsonl)
# Memories per turn: old and v3 from results_diag/mem0_qdrant_copy, today
# from the worktree's own store. Run after the old-code process has exited.
#
#   python scripts/diagnostics/gemini_change/part2_compare.py

from __future__ import annotations

import collections
import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from core.dataset import load_locomo
from scripts.diagnostics.gemini_change.step4_logs import per_turn

R = pathlib.Path(".").resolve()
OLD = "anchor_gemini__gemini-backbone__none__conv1"
NEW = OLD + "__v3"
TODAY = "diag__none__conv1__oldcode"
WT_STORE = R.parent / "memdefense_oldcode" / "results" / "mem0_qdrant"


def by_turn(store, ns):
    c = QdrantClient(path=str(store))
    pts, off = [], None
    try:
        while True:
            got, off = c.scroll("memdefense", limit=1000, offset=off, with_payload=True,
                                with_vectors=False, scroll_filter=Filter(must=[
                                    FieldCondition(key="user_id", match=MatchValue(value=ns))]))
            pts += got
            if off is None:
                break
    finally:
        c.close()
    out = collections.defaultdict(list)
    for p in pts:
        out[p.payload.get("turn_id")].append(p.payload.get("data"))
    return out, len(pts)


def main():
    turns = load_locomo("data/locomo/locomo10.json")[1].turns[:150]
    logs = {"old": per_turn(R / "results" / "proxy" / "calls.jsonl", OLD, turns)[0],
            "v3": per_turn(R / "results_v3" / "proxy" / "calls.jsonl", NEW, turns)[0],
            "today": per_turn(R / "results_diag" / "proxy" / "calls.jsonl", TODAY, turns)[0]}
    mems = {}
    n = {}
    mems["old"], n["old"] = by_turn(R / "results_diag" / "mem0_qdrant_copy", OLD)
    mems["v3"], n["v3"] = by_turn(R / "results_diag" / "mem0_qdrant_copy", NEW)
    mems["today"], n["today"] = by_turn(WT_STORE, TODAY)
    K = ("old", "v3", "today")

    print("PART 2 -- conv-30, first 150 turns, no defense\n")
    print(f"  memories stored   old (Aug 22, c72464e code)   {n['old']}")
    print(f"                    v3  (Oct 29, current code)   {n['v3']}")
    print(f"                    old code today (Oct 30)      {n['today']}\n")
    tot = {k: sum(logs[k][t]["tokens_out"] for t in range(150)) for k in K}
    empty = {k: sum(logs[k][t]["tokens_out"] <= 10 for t in range(150)) for k in K}
    fact_tok = {k: sum(logs[k][t]["fact_embed_tokens"] for t in range(150)) for k in K}
    tin = {k: sum(logs[k][t]["tokens_in"] for t in range(150)) for k in K}
    turns_with = {k: sum(bool(mems[k].get(t.turn_id)) for t in turns) for k in K}
    for label, d in (("extraction tokens_out", tot), ("empty extractions (<=10 tokens)", empty),
                     ("turns that stored >=1 memory", turns_with),
                     ("fact tokens embedded", fact_tok), ("extraction tokens_in", tin)):
        print(f"  {label:34} " + "  ".join(f"{k} {d[k]:>9,}" for k in K))
    same_v3 = sum(logs["today"][t]["tokens_out"] == logs["v3"][t]["tokens_out"] for t in range(150))
    same_old = sum(logs["today"][t]["tokens_out"] == logs["old"][t]["tokens_out"] for t in range(150))
    mem_v3 = sum(len(mems["today"].get(t.turn_id, [])) == len(mems["v3"].get(t.turn_id, [])) for t in turns)
    mem_old = sum(len(mems["today"].get(t.turn_id, [])) == len(mems["old"].get(t.turn_id, [])) for t in turns)
    print(f"\n  turns where today's tokens_out equals v3's: {same_v3}/150; equals old's: {same_old}/150")
    print(f"  turns where today's memory count equals v3's: {mem_v3}/150; equals old's: {mem_old}/150")
    ident = sum(sorted(mems["today"].get(t.turn_id, [])) == sorted(mems["v3"].get(t.turn_id, []))
                for t in turns)
    print(f"  turns whose memory TEXTS are identical today and in v3: {ident}/150")

    print("\n  PER TURN: extraction tokens_out and memories stored (old / v3 / today)\n")
    print(f"  {'turn':>4} {'id':>6}   {'out old':>7} {'out v3':>6} {'out today':>9}   "
          f"{'mem old':>7} {'mem v3':>6} {'mem today':>9}")
    for i, t in enumerate(turns):
        o = [logs[k][i]["tokens_out"] for k in K]
        m = [len(mems[k].get(t.turn_id, [])) for k in K]
        print(f"  {i + 1:>4} {t.turn_id:>6}   {o[0]:>7} {o[1]:>6} {o[2]:>9}   "
              f"{m[0]:>7} {m[1]:>6} {m[2]:>9}")
    calls = [json.loads(l) for l in (R / "results_diag" / "proxy" / "calls.jsonl").read_text()
             .splitlines() if l.strip()]
    mine = [r for r in calls if r.get("cell_id") == TODAY]
    bad = [r for r in mine if r["status"] != "ok"]
    models = collections.Counter((r.get("alias_requested"), r["model_resolved"]) for r in mine)
    cost = collections.Counter()
    for r in mine:
        cost["google" if "gemini" in r["model_resolved"] else "openai"] += r.get("cost_usd") or 0
    print(f"\n  CALLS (today's run): {len(mine)} logged, {len(bad)} not ok")
    for (a, m), c in sorted(models.items()):
        print(f"    alias {a!s:16} -> {m:28} {c}")
    for r in bad:
        print(f"    FAILED {r['ts']} {r['model_resolved']} {r.get('error')!r}"[:300])
    print(f"    cost: Google ${cost['google']:.3f}, OpenAI ${cost['openai']:.4f}")
    print(f"    other cells in this log: {sorted({r.get('cell_id') for r in calls} - {TODAY})}")


if __name__ == "__main__":
    main()
