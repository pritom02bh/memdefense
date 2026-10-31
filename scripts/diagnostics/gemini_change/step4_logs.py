# scripts/diagnostics/gemini_change/step4_logs.py
#
# DIAGNOSTIC, READ-ONLY, NO API CALLS. Turn-by-turn comparison of the
# final ingestion of none conv1 s1 in the old log (results/) and of its
# re-ingest none__conv1__v3 (results_v3/). Reads both proxy logs; writes
# only under results_diag/.
#
#   python scripts/diagnostics/gemini_change/step4_logs.py

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, ".")
from core.dataset import load_locomo
from scripts.check_integrity import add_calls, align_final_ingestion, load_log, turn_signatures

R = pathlib.Path(".")
OLD_ID = "anchor_gemini__gemini-backbone__none__conv1"
NEW_ID = OLD_ID + "__v3"
OUT = R / "results_diag" / "step4_turns.json"


def per_turn(log_path, cell_id, turns):
    by_cell, _ = load_log(log_path)
    rs = by_cell[cell_id]
    sigs = turn_signatures(turns)
    calls = add_calls(rs)
    assign, first = align_final_ingestion(calls, list(enumerate(sigs)))
    final = calls[first:]
    # each add(): its fact embeddings are the embedding rows after its
    # extraction and before the next add()'s search embedding
    nxt_search = {id(c): (final[k + 1].search or final[k + 1].rows[0])["ts"]
                  for k, c in enumerate(final[:-1])}
    out = {}
    for t in range(len(turns)):
        cs = assign.get(t, [])
        ok = [r for c in cs for r in c.ok]
        if not ok:
            out[t] = None
            continue
        c = cs[0]
        x = ok[-1]
        end = nxt_search.get(id(c), "9999")
        emb = [r for r in rs if r["_step"] == "embedding" and x["ts"] < r["ts"] < end]
        out[t] = {"tokens_in": x["tokens_in"], "tokens_out": x["tokens_out"],
                  "extract_calls": len(c.rows), "extract_ok": len(ok),
                  "fact_embed_calls": len(emb),
                  "fact_embed_ok": sum(r["status"] == "ok" for r in emb),
                  "fact_embed_tokens": sum(r["tokens_in"] or 0 for r in emb if r["status"] == "ok"),
                  "ts": x["ts"]}
    return out, len(calls[:first])


def main():
    conv = load_locomo("data/locomo/locomo10.json")[1]
    turns = conv.turns[:150]
    old, old_prior = per_turn(R / "results" / "proxy" / "calls.jsonl", OLD_ID, turns)
    new, new_prior = per_turn(R / "results_v3" / "proxy" / "calls.jsonl", NEW_ID, turns)
    print(f"STEP 4. LOGS -- none conv1 s1 ({conv.conv_id}), final ingestion, turn by turn\n")
    print(f"  add() calls before the final ingestion in the log: old {old_prior}, v3 {new_prior}")
    missing = [t + 1 for t in range(150) if old[t] is None or new[t] is None]
    print(f"  turns with no ok extraction in the final ingestion: {missing or 'none'}\n")
    print("  fact_embed = embedding calls between this extraction and the next add()'s search "
          "embedding.\n  mem0's OpenAI embedder sends all of a turn's facts in ONE batched call, so "
          "this is 1 when the\n  turn sent facts for storage and 0 when it sent none; its tokens "
          "are the facts' total tokens.\n")
    hdr = (f"  {'turn':>4} {'id':>6}  {'out old':>7} {'out v3':>6}  {'emb old':>7} {'emb v3':>6}  "
           f"{'facttok old':>11} {'facttok v3':>10}  {'in old':>6} {'in v3':>6}")
    print(hdr)
    tot = {k: [0, 0] for k in ("tokens_out", "fact_embed_calls", "fact_embed_tokens", "tokens_in")}
    old_only = []
    for t in range(150):
        o, n = old[t], new[t]
        for k in tot:
            tot[k][0] += o[k]
            tot[k][1] += n[k]
        if o["fact_embed_calls"] and not n["fact_embed_calls"]:
            old_only.append(t)
        print(f"  {t + 1:>4} {turns[t].turn_id:>6}  {o['tokens_out']:>7} {n['tokens_out']:>6}  "
              f"{o['fact_embed_calls']:>7} {n['fact_embed_calls']:>6}  "
              f"{o['fact_embed_tokens']:>11} {n['fact_embed_tokens']:>10}  "
              f"{o['tokens_in']:>6} {n['tokens_in']:>6}")
    print(f"\n  TOTALS          tokens_out  old {tot['tokens_out'][0]:,}  v3 {tot['tokens_out'][1]:,}"
          f"  ({100 * (tot['tokens_out'][1] / tot['tokens_out'][0] - 1):+.1f}%)")
    print(f"                  fact-embedding calls (turns that sent facts)  old "
          f"{tot['fact_embed_calls'][0]}  v3 {tot['fact_embed_calls'][1]}")
    print(f"                  fact-embedding tokens  old {tot['fact_embed_tokens'][0]:,}  v3 "
          f"{tot['fact_embed_tokens'][1]:,}")
    print(f"                  extraction tokens_in  old {tot['tokens_in'][0]:,}  v3 "
          f"{tot['tokens_in'][1]:,}")
    zero_out = lambda d: sum(d[t]["fact_embed_calls"] == 0 for t in range(150))
    print(f"                  turns that sent no facts  old {zero_out(old)}  v3 {zero_out(new)}")
    new_only = [t for t in range(150) if new[t]["fact_embed_calls"] and not old[t]["fact_embed_calls"]]
    print(f"\n  Turns where OLD stored facts and v3 stored none: {len(old_only)}")
    for t in old_only:
        print(f"    turn {t + 1:>3} {turns[t].turn_id:>6}  out old {old[t]['tokens_out']:>4} v3 "
              f"{new[t]['tokens_out']:>4}  {turns[t].speaker}: {turns[t].text[:90]!r}")
    print(f"\n  Turns where v3 stored facts and OLD stored none: {len(new_only)}  "
          f"{[t + 1 for t in new_only]}")
    print(f"\n  Turn 1 extraction tokens_in: old {old[0]['tokens_in']}, v3 {new[0]['tokens_in']} "
          f"(equal => same prompt size at turn 1, i.e. no leftover message history in the old run)")
    print(f"  Old final ingestion ran {old[0]['ts'][:19]} .. {old[149]['ts'][:19]}; "
          f"v3 {new[0]['ts'][:19]} .. {new[149]['ts'][:19]}")
    OUT.write_text(json.dumps({"old": old, "v3": new}, indent=1))


if __name__ == "__main__":
    main()
