# scripts/diagnostics/gemini_change/diff_add_args.py -- compare recorded add() calls, old vs new
import json, sys
def load(p): return [json.loads(l) for l in open(p)]
for kind in ("none", "chain"):
    old, new = load(f"old_{kind}.jsonl"), load(f"new_{kind}.jsonl")
    print(f"== {kind} path: old {len(old)} records, new {len(new)} records")
    raw_same = norm_same = 0
    diffs = []
    for i, (o, n) in enumerate(zip(old, new)):
        ob, nb = json.dumps(o, sort_keys=True), json.dumps(n, sort_keys=True)
        raw_same += ob == nb
        on = ob.replace("__conv1\"", "__conv1__v3\"")
        norm_same += on == nb
        if ob != nb:
            diffs.append((i, o, n))
    print(f"   byte-identical: {raw_same}/{len(old)}; identical once the namespace "
          f"'...__conv1' is renamed '...__conv1__v3': {norm_same}/{len(old)}")
    keys = set()
    for i, o, n in diffs:
        for side in ("args", "kwargs"):
            if side == "args" and o.get("args") != n.get("args"): keys.add("args (messages)")
            if side == "kwargs":
                for k in set(o["kwargs"]) | set(n["kwargs"]):
                    if o["kwargs"].get(k) != n["kwargs"].get(k): keys.add(f"kwargs.{k}")
        if "delete_all" in o or "delete_all" in n: keys.add("delete_all record")
    print(f"   fields that differ: {sorted(keys)}")
    for i, o, n in diffs[:3]:
        print(f"   [{i}] old: {json.dumps(o, sort_keys=True)[:400]}")
        print(f"   [{i}] new: {json.dumps(n, sort_keys=True)[:400]}")
    if len(diffs) > 3: print(f"   ... {len(diffs)-3} more, all in the same fields")
    msgs_same = all(o.get("args") == n.get("args") and o["kwargs"].get("metadata") == n["kwargs"].get("metadata")
                    and set(o["kwargs"]) == set(n["kwargs"]) for o, n in zip(old, new))
    print(f"   messages, metadata and kwarg names identical on every call: {msgs_same}")
