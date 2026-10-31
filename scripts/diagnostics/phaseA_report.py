# scripts/diagnostics/phaseA_report.py
#
# READ-ONLY report on the v3 Phase A write-time screening: every blocked
# turn with its speaker, full text and the anomaly model's reply, whether
# each block repeats across replicates and in stacked, and whether every
# block is in the quarantine log.
#
#   MEMDEF_RESULTS=results_v3 python3 -m scripts.diagnostics.phaseA_report

from __future__ import annotations

import json
import sys
from collections import defaultdict

from core.dataset import load_locomo
from core.paths import CELLS, QUARANTINE, REPO_ROOT

DATA = REPO_ROOT / "data" / "locomo" / "locomo10.json"


def main() -> int:
    convs = {c.conv_id: c for c in load_locomo(str(DATA))}
    screened = {}
    for d in sorted(CELLS.iterdir()):
        f = d / "write_verdicts.json"
        if f.exists():
            screened[d.name] = json.loads(f.read_text())
    if not screened:
        print(f"no write_verdicts.json under {CELLS}")
        return 1

    def meta(cid):
        # ..._<condition>__conv<i>[__s<k>]__v3
        parts = cid.split("__")
        cond, conv = parts[2], parts[3]
        seed = int(parts[4][1:]) if parts[4].startswith("s") else 1
        return cond, conv, seed

    print(f"PHASE A REPORT -- {len(screened)} cells screened (anomaly and stacked, "
          f"not already planned for re-ingest)\n")

    # 1. every blocked turn
    blocked_by_turn = defaultdict(list)
    q_ok = q_missing = 0
    print("1. Blocked turns\n")
    for cid, res in screened.items():
        cond, conv_key, seed = meta(cid)
        old = json.loads((REPO_ROOT / "results" / "cells" / cid.removesuffix("__v3")
                          / "summary.json").read_text())
        conv = convs[old["conv_id"]]
        turns = {t.turn_id: t for t in conv.turns}
        qpath = QUARANTINE / f"{cid}.jsonl"
        qrows = [json.loads(l) for l in qpath.read_text().splitlines() if l.strip()] \
            if qpath.exists() else []
        for b in res["blocked"]:
            t = turns[b["turn_id"]]
            idx = conv.turns.index(t) + 1
            qmatch = [q for q in qrows if q.get("turn_id") == b["turn_id"]]
            q_ok += bool(qmatch)
            q_missing += not qmatch
            blocked_by_turn[(old["conv_id"], conv_key, b["turn_id"])].append((cid, cond, seed))
            print(f"  cell     {cid}")
            print(f"  turn     {b['turn_id']} (turn {idx} of {len(conv.turns[:150])} ingested), "
                  f"{t.session_id}, {t.timestamp}")
            print(f"  speaker  {t.speaker}")
            print(f"  text     {t.text}")
            print(f"  blocked  by {b['defense']}: {b['reason']}")
            print(f"  reply    Gemini's raw reply was NOT recorded: Phase A ran code that kept "
                  f"only the parsed keyword (SUSPICIOUS).")
            print(f"           The fixed code records the raw reply; the re-ingest of this cell "
                  f"will capture it.")
            print(f"  quarantine  {'logged' if qmatch else 'MISSING'} in {qpath.relative_to(REPO_ROOT)}"
                  + (f": {qmatch[0]['reason']!r}" if qmatch else ""))
            print()

    # 2. repetition across replicates and in stacked
    print("2. Does each block repeat across replicates and in stacked?\n")
    for (conv_id, conv_key, turn_id), hits in sorted(blocked_by_turn.items()):
        print(f"  {conv_id} ({conv_key}) turn {turn_id}:")
        for cond in ("anomaly", "stacked"):
            row = []
            for seed in (1, 2, 3):
                cid = (f"anchor_gemini__gemini-backbone__{cond}__{conv_key}"
                       + ("" if seed == 1 else f"__s{seed}") + "__v3")
                if cid not in screened:
                    row.append(f"s{seed}: not screened (already planned for re-ingest)")
                else:
                    hit = any(h[0] == cid for h in hits)
                    row.append(f"s{seed}: {'BLOCKED' if hit else 'passed'}")
            print(f"    {cond:8} " + " | ".join(row))
        screened_here = [c for c in screened if meta(c)[1] == conv_key]
        print(f"    -> blocked in {len(hits)} of {len(screened_here)} screened cells of this "
              f"conversation")
        print()
    other = [c for c in screened if not screened[c]["blocked"]]
    print(f"  Cells screened with no block: {len(other)} "
          f"(conversations {sorted({meta(c)[1] for c in other})})\n")

    # 3. quarantine confirmation
    print("3. Quarantine log\n")
    print(f"  {q_ok} of {q_ok + q_missing} blocks found in {QUARANTINE.relative_to(REPO_ROOT)}/"
          + ("" if not q_missing else f"  -- {q_missing} MISSING"))
    stray = []
    for cid, res in screened.items():
        qpath = QUARANTINE / f"{cid}.jsonl"
        n = sum(1 for l in qpath.read_text().splitlines() if l.strip()) if qpath.exists() else 0
        if n != len(res["blocked"]):
            stray.append((cid, n, len(res["blocked"])))
    print(f"  quarantine lines == blocked turns in every screened cell: "
          f"{'yes' if not stray else 'NO ' + str(stray)}")
    print(f"\n  Total: {sum(len(r['blocked']) for r in screened.values())} blocks in "
          f"{sum(bool(r['blocked']) for r in screened.values())} cells, "
          f"{len(blocked_by_turn)} distinct turns.")
    return 0 if not q_missing and not stray else 1


if __name__ == "__main__":
    sys.exit(main())
