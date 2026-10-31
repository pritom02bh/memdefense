# scripts/diagnostics/repro_oct31/step1_compare.py
#
# READ-ONLY, NO API CALLS. Opens results_v3/mem0_qdrant (one process) and
# reads results_v3/proxy/calls.jsonl; writes nothing outside this folder.
#   a. none__conv0__s2__v3b vs none__conv0__s3__v3b (both built Oct 30)
#   b. none__conv0__v3 (night of Oct 29) vs none__conv0__s2__v3b (Oct 30)
# Memory texts as multisets and per turn_id, and each turn's extraction
# tokens_out from the proxy log (the final ingestion of each cell).
#
#   python3 scripts/diagnostics/repro_oct31/step1_compare.py [STORE NS LOG] ...

from __future__ import annotations

import collections
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
P = "anchor_gemini__gemini-backbone__none__conv0"
V3, S2, S3 = P + "__v3", P + "__s2__v3b", P + "__s3__v3b"
STORE = R / "results_v3" / "mem0_qdrant"
LOG = R / "results_v3" / "proxy" / "calls.jsonl"


def memories(store, namespaces):
    """namespace -> {turn_id: sorted [texts]}; one client, closed after."""
    c = QdrantClient(path=str(store))
    out = {}
    try:
        for ns in namespaces:
            pts, off = [], None
            while True:
                got, off = c.scroll("memdefense", limit=1000, offset=off, with_payload=True,
                                    with_vectors=False, scroll_filter=Filter(must=[
                                        FieldCondition(key="user_id", match=MatchValue(value=ns))]))
                pts += got
                if off is None:
                    break
            by = collections.defaultdict(list)
            for p in pts:
                by[p.payload.get("turn_id")].append(p.payload.get("data"))
            out[ns] = {t: sorted(v) for t, v in by.items()}
    finally:
        c.close()
    return out


def compare(name_a, a, log_a, name_b, b, log_b, turns):
    """Print the comparison; return the first differing turn (memories, tokens_out)."""
    ma = collections.Counter(x for v in a.values() for x in v)
    mb = collections.Counter(x for v in b.values() for x in v)
    only_a, only_b = sum((ma - mb).values()), sum((mb - ma).values())
    print(f"  memories: {name_a} {sum(ma.values())}, {name_b} {sum(mb.values())}")
    print(f"  as multisets: identical={ma == mb}; only in {name_a}: {only_a}, only in "
          f"{name_b}: {only_b}, shared: {sum((ma & mb).values())}")
    same_turns, first_mem, diff_turns = 0, None, []
    for i, t in enumerate(turns, 1):
        x, y = a.get(t.turn_id, []), b.get(t.turn_id, [])
        if x == y:
            same_turns += 1
        else:
            diff_turns.append(i)
            if first_mem is None:
                first_mem = (i, t.turn_id, x, y)
    stray = (set(a) | set(b)) - {t.turn_id for t in turns}
    print(f"  per turn_id: {same_turns}/{len(turns)} turns store identical memory texts; "
          f"{len(diff_turns)} differ" + (f"; turn_ids outside the 150 turns: {sorted(stray)}"
                                         if stray else ""))
    first_tok = None
    same_tok = 0
    for i in range(len(turns)):
        ta = (log_a.get(i) or {}).get("tokens_out")
        tb = (log_b.get(i) or {}).get("tokens_out")
        if ta == tb and ta is not None:
            same_tok += 1
        elif first_tok is None:
            first_tok = (i + 1, turns[i].turn_id, ta, tb)
    print(f"  extraction tokens_out equal on {same_tok}/{len(turns)} turns; totals "
          f"{sum((log_a.get(i) or {}).get('tokens_out') or 0 for i in range(len(turns))):,} vs "
          f"{sum((log_b.get(i) or {}).get('tokens_out') or 0 for i in range(len(turns))):,}")
    if first_tok:
        i, tid, ta, tb = first_tok
        print(f"  first turn whose tokens_out differs: turn {i} ({tid}): {ta} vs {tb}")
    if first_mem:
        i, tid, x, y = first_mem
        print(f"  first turn whose stored memories differ: turn {i} ({tid})")
        print(f"    {name_a}: {x}")
        print(f"    {name_b}: {y}")
        print(f"  differing turns: {diff_turns[:40]}{' ...' if len(diff_turns) > 40 else ''}")
    return first_mem, first_tok


def main():
    turns = load_locomo("data/locomo/locomo10.json")[0].turns[:150]
    mems = memories(STORE, [V3, S2, S3])
    logs = {ns: per_turn(LOG, ns, turns)[0] for ns in (V3, S2, S3)}
    print("STEP 1a. none conv0 s2 v3b vs s3 v3b (both Oct 30)")
    compare("s2_v3b", mems[S2], logs[S2], "s3_v3b", mems[S3], logs[S3], turns)
    print("\nSTEP 1b. none conv0 v3 (night of Oct 29) vs s2 v3b (Oct 30)")
    compare("v3", mems[V3], logs[V3], "s2_v3b", mems[S2], logs[S2], turns)
    for ns in (V3, S2, S3):
        ts = [v["ts"] for v in logs[ns].values() if v]
        print(f"  {ns.replace(P, 'none_conv0')}: final ingestion {min(ts)[:19]} .. {max(ts)[:19]}")


if __name__ == "__main__":
    main()
