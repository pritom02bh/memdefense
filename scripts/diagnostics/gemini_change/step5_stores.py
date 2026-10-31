# scripts/diagnostics/gemini_change/step5_stores.py -- READ-ONLY on a COPY of results_v3/mem0_qdrant
# (results_diag/mem0_qdrant_copy). Memories per turn_id, old namespace vs v3 namespace.
import json, sys, warnings, collections
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from core.dataset import load_locomo
OLD = "anchor_gemini__gemini-backbone__none__conv1"; NEW = OLD + "__v3"
c = QdrantClient(path="results_diag/mem0_qdrant_copy")
def load(ns):
    pts, off = [], None
    while True:
        got, off = c.scroll("memdefense", scroll_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=ns))]),
                            limit=1000, offset=off, with_payload=True, with_vectors=False)
        pts += got
        if off is None: break
    by = collections.defaultdict(list)
    for p in pts: by[p.payload.get("turn_id")].append((p.payload.get("created_at"), p.payload.get("data")))
    return pts, by
po, bo = load(OLD); pn, bn = load(NEW); c.close()
conv = load_locomo("data/locomo/locomo10.json")[1]; turns = conv.turns[:150]; tid = {t.turn_id: (i, t) for i, t in enumerate(turns)}
print(f"STEP 5. STORES (read from a copy of results_v3/mem0_qdrant)\n")
print(f"  {OLD}: {len(po)} memories in {len(bo)} turns")
print(f"  {NEW}: {len(pn)} memories in {len(bn)} turns")
stray = [k for k in set(bo) | set(bn) if k not in tid]
print(f"  memories whose turn_id is not among the 150 ingested turns: {stray or 'none'}\n")
diff = sorted(((len(bo.get(t.turn_id, [])) - len(bn.get(t.turn_id, [])), i, t) for i, t in enumerate(turns)), key=lambda x: (-x[0], x[1]))
dist = collections.Counter(d for d, _, _ in diff)
print("  per-turn change (old - v3) -> number of turns: " + ", ".join(f"{k:+d}:{v}" for k, v in sorted(dist.items())))
early = sum(len(bo.get(t.turn_id, [])) for t in turns[:10]), sum(len(bn.get(t.turn_id, [])) for t in turns[:10])
print(f"  turns 1-10 (where the old run's leftover message history could reach): old {early[0]}, v3 {early[1]}")
print(f"  turns 11-150: old {len(po) - early[0]}, v3 {len(pn) - early[1]}\n")
print("  per-turn counts, all 150 turns (turn: old/v3):")
line = []
for i, t in enumerate(turns):
    line.append(f"{i+1}:{len(bo.get(t.turn_id, []))}/{len(bn.get(t.turn_id, []))}")
for k in range(0, 150, 15): print("    " + "  ".join(f"{x:>9}" for x in line[k:k+15]))
print("\n  THE 5 TURNS WITH THE LARGEST DROP\n")
for d, i, t in diff[:5]:
    print(f"  turn {i+1} ({t.turn_id}), {t.speaker}: {t.text!r}")
    print(f"    old: {len(bo.get(t.turn_id, []))} memories, v3: {len(bn.get(t.turn_id, []))} memories  (drop {d})")
    for _, m in sorted(bo.get(t.turn_id, [])): print(f"      old  - {m}")
    for _, m in sorted(bn.get(t.turn_id, [])): print(f"      v3   - {m}")
    print()
