# scripts/diagnostics/prompt_date/part1.py
#
# READ-ONLY, NO API CALLS. Mem0 2.0.15 puts "## Observation Date" and
# "## Current Date" = datetime.now(timezone.utc).date() into the second
# (user-role) message of every extraction call. For every store builder (both roots)
# and the test namespace: the UTC date of each final-ingestion extraction
# call (= the prompt date), and turn-by-turn tokens_out, compared within
# each conversation and write-time pipeline (same turns ingested).
#
#   MEMDEF_RESULTS=results_v3 .venv/bin/python scripts/diagnostics/prompt_date/part1.py

from __future__ import annotations

import collections
import datetime as dt
import json
import pathlib
import sys

sys.path.insert(0, ".")
from core import manifest as mf
from core.dataset import load_locomo
from scripts.check_integrity import (DATA, _quarantine, alias_models, check_cell, load_log)
from scripts.diagnostics.gemini_change.step4_logs import per_turn

R = pathlib.Path(".").resolve()
TEST = "repro__none__conv0__oct31"
P = "anchor_gemini__gemini-backbone__"
convs = {c.conv_id: c for c in load_locomo(str(DATA))}
models = alias_models()
m = mf.load()
MAIN = [e for e in m["cells"] if e["role"] == "main"]
logs = {}


def log(root):
    if root not in logs:
        logs[root] = load_log(R / root / "proxy" / "calls.jsonl")
    return logs[root]


def start_date(r):
    t = dt.datetime.fromisoformat(r["ts"]) - dt.timedelta(milliseconds=r.get("latency_ms") or 0)
    return t.date().isoformat()


def builder(root, cid):
    """final-ingestion extraction rows mapped to turn ids, prompt dates."""
    d = R / root / "cells" / cid
    s = json.loads((d / "summary.json").read_text())
    if s.get("source_namespace", cid) != cid:
        return None
    by_cell, judge = log(root)
    rep = check_cell(d, by_cell.get(cid, []), judge, convs, models, R / root / "quarantine")
    x = rep.final_extract
    if not x:
        return None
    conv = convs[s["conv_id"]]
    turns = conv.turns[: s["turns_ingested"]]
    blocked = _quarantine(cid, R / root / "quarantine")[0]
    exp = [t.turn_id for t in turns if t.turn_id not in blocked]
    unp = (s.get("retries") or {}).get("extract:unparseable", 0)
    tok = {}
    if len(x) == len(exp):
        tok = {t: r["tokens_out"] for t, r in zip(exp, x)}
    return dict(root=root, cid=cid, conv=s["conv_id"], cond=s["condition"], seed=s.get("seed"),
                blocked=frozenset(blocked), n=s.get("memories_stored"), unp=unp,
                dates=collections.Counter(r["ts"][:10] for r in x),
                start_dates=collections.Counter(start_date(r) for r in x),
                span=(x[0]["ts"][:19], x[-1]["ts"][:19]), tok=tok, mapped=len(x) == len(exp))


def test_builder():
    conv = convs["conv-26"]
    turns = conv.turns[:150]
    pt, prior = per_turn(R / "results_v3" / "proxy" / "calls.jsonl", TEST, turns)
    rows = [v for v in pt.values() if v]
    done = json.loads((R / "results_diag/repro_oct31/rebuild_done.json").read_text())
    return dict(root="results_diag/repro_oct31/root", cid=TEST, conv="conv-26", cond="none", seed=None,
                blocked=frozenset(), n=done.get("memories") or done.get("memories_stored"),
                unp=0, dates=collections.Counter(v["ts"][:10] for v in rows),
                start_dates=collections.Counter(v["ts"][:10] for v in rows),
                span=(min(v["ts"] for v in rows)[:19], max(v["ts"] for v in rows)[:19]),
                tok={turns[i].turn_id: v["tokens_out"] for i, v in pt.items() if v},
                mapped=all(pt.values()))


def short(c):
    return c.replace(P, "")


def same(a, b):
    ks = [k for k in a["tok"] if k in b["tok"]]
    n = sum(a["tok"][k] == b["tok"][k] for k in ks)
    first = next((i + 1 for i, k in enumerate(ks) if a["tok"][k] != b["tok"][k]), None)
    return n, len(ks), first


def pd(b):
    return "+".join(f"{d}" + (f"({n})" if len(b["dates"]) > 1 else "") for d, n in sorted(b["dates"].items()))


def main():
    # every builder: main-manifest stores, all results_v3 cells that built a store, the test
    bs = {}
    for e in MAIN:
        b = e["store_built_by"]
        bs[(b["root"], b["cell_id"])] = None
    for d in sorted((R / "results_v3" / "cells").iterdir()):
        if (d / "summary.json").exists():
            bs[("results_v3", d.name)] = None
    for k in list(bs):
        bs[k] = builder(*k)
    bs = {k: v for k, v in bs.items() if v}
    bs[("test", TEST)] = test_builder()
    in_main = {(e["store_built_by"]["root"], e["store_built_by"]["cell_id"]) for e in MAIN}

    print("PART 1.2  PROMPT DATE IN EFFECT DURING EACH STORE'S FINAL INGESTION")
    print("  (UTC date of every ok extraction call; the date at request start agrees with the logged"
          " date for every call unless flagged)")
    print(f"  {'store (builder)':44} {'root':10} {'main':4} {'prompt date(s)':26} {'final ingestion (UTC)':42} mem")
    for k, b in sorted(bs.items(), key=lambda kv: (kv[1]["conv"], kv[1]["span"][0])):
        flag = "" if b["dates"] == b["start_dates"] else f"  START-DATES {dict(b['start_dates'])}"
        print(f"  {short(b['cid']):44} {k[0][:10]:10} {'yes' if k in in_main else '':4} {pd(b):26} "
              f"{b['span'][0]} .. {b['span'][1][11:]:9}   {b['n']}{flag}")

    print("\nPART 1.3  TURN-BY-TURN EXTRACTION OUTPUT (tokens_out), WITHIN EACH CONVERSATION AND PIPELINE")
    print("  pipeline = the set of turns its write-time chain blocked (same set -> same extraction inputs)")
    groups = collections.defaultdict(list)
    for k, b in bs.items():
        groups[(b["conv"], b["blocked"])].append(b)
    for (conv, blocked), g in sorted(groups.items(), key=lambda kv: (kv[0][0], len(kv[0][1]))):
        if len(g) < 2:
            continue
        g.sort(key=lambda b: b["span"][0])
        print(f"\n  {conv}  blocked {sorted(blocked) or 'none'}  ({len(g)} stores)")
        for b in g:
            if not b["mapped"]:
                print(f"    {short(b['cid'])}: final extraction rows not one per turn; skipped")
        g = [b for b in g if b["mapped"]]
        within, across = [], []
        for i, a in enumerate(g):
            for b in g[i + 1:]:
                n, tot, first = same(a, b)
                (within if a["dates"].keys() == b["dates"].keys() and len(a["dates"]) == 1
                 else across).append((n, tot, first, a, b))
        def show(tag, lst):
            if not lst:
                return
            ns = [n for n, *_ in lst]
            print(f"    {tag}: {len(lst)} pairs, identical turns min {min(ns)}  median "
                  f"{sorted(ns)[len(ns)//2]}  max {max(ns)} (of {lst[0][1]})")
        show("same prompt date     ", within)
        show("different prompt date", across)
        for n, tot, first, a, b in sorted(within + across, key=lambda x: x[0]):
            print(f"      {n:3}/{tot}  first diff turn {str(first):4}  {short(a['cid']):38} [{pd(a)}] "
                  f"{a['n']}  vs  {short(b['cid']):38} [{pd(b)}] {b['n']}")

    print("\nPART 1.4  AUGUST: PROMPT DATES PER CONVERSATION AND REPLICATE (main stores)")
    aug = [e for e in MAIN if e["store_period"] == "Aug"]
    by = collections.defaultdict(dict)
    for e in aug:
        b = bs[(e["store_built_by"]["root"], e["store_built_by"]["cell_id"])]
        by[(e["conv_id"], e["seed"])][e["condition"]] = b
    for (conv, seed), d in sorted(by.items()):
        allds = set().union(*(b["dates"].keys() for b in d.values()))
        print(f"  {conv} s{seed}: {'ONE prompt date ' + allds.pop() if len(allds) == 1 else 'DATES DIFFER'}")
        for cond, b in sorted(d.items()):
            print(f"      {cond:10} {pd(b):26} {short(b['cid'])}  {b['span'][0]}..{b['span'][1][11:]}")

    out = {f"{k[0]}/{k[1]}": {"dates": dict(b["dates"]), "span": b["span"], "n": b["n"]}
           for k, b in bs.items()}
    (R / "results_diag/prompt_date/prompt_dates.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
