# scripts/diagnostics/gemini_change/population.py -- READ-ONLY. Final-ingestion extraction stats of
# every old Gemini cell vs every v3 re-ingest: store size, empty extractions, tokens_out,
# prior attempts (leftover message history). Writes nothing outside stdout.
import json, pathlib, statistics as st, sys, io, contextlib
sys.path.insert(0, ".")
from scripts.check_integrity import run_checks
R = pathlib.Path(".")
with contextlib.redirect_stdout(io.StringIO()):
    old = run_checks("anchor_gemini__*", quiet=True, log_path=R/"results/proxy/calls.jsonl",
                     cells_dir=R/"results/cells", quarantine_dir=R/"results/quarantine")
    new = run_checks("*__v3", quiet=True, log_path=R/"results_v3/proxy/calls.jsonl",
                     cells_dir=R/"results_v3/cells", quarantine_dir=R/"results_v3/quarantine")
def stats(rep, root):
    s = json.loads((R/root/"cells"/rep.cell_id/"summary.json").read_text())
    x = rep.final_extract
    if not x: return None
    n = s.get("memories_stored", s.get("memories_before_answering"))
    return dict(cid=rep.cell_id, conv=s["conv_id"], cond=s["condition"], n=n, calls=len(x),
                empty=sum(r["tokens_out"] <= 10 for r in x), out=sum(r["tokens_out"] for r in x),
                prior=rep.info.get("prior_attempts"), stale=len(rep.info.get("stale_turns") or []),
                start=x[0]["ts"][:10])
O = [s for r in old if (s := stats(r, "results"))]
N = [s for r in new if s_ok(r) ] if False else [s for r in new if json.loads((R/"results_v3/cells"/r.cell_id/"summary.json").read_text()).get("reingested") and (s := stats(r, "results_v3"))]
print(f"{'cell':52} {'store':>5} {'empty':>5} {'tok_out':>7} {'prior':>5} {'stale':>5} start")
for conv in sorted({s['conv'] for s in O}):
    for grp, rows in (("old", [s for s in O if s['conv']==conv]), ("v3", [s for s in N if s['conv']==conv])):
        for s in sorted(rows, key=lambda s: s['cid']):
            print(f"{grp:3} {s['cid'].replace('anchor_gemini__gemini-backbone__',''):48} {s['n']:>5} {s['empty']:>5} {s['out']:>7} {s['prior']!s:>5} {s['stale']:>5} {s['start']}")
    print()
def summ(rows, label):
    if not rows: return
    print(f"  {label:44} cells {len(rows):>2}  store mean {st.mean(s['n'] for s in rows):6.1f}  "
          f"empty/150 mean {st.mean(s['empty'] for s in rows):5.1f}  tokens_out mean {st.mean(s['out'] for s in rows):8.0f}")
print("SUMMARY by conversation (old cells: all; then only those with no earlier attempt = no leftover history)")
for conv in sorted({s['conv'] for s in N}):
    summ([s for s in O if s['conv']==conv], f"{conv} old, all")
    summ([s for s in O if s['conv']==conv and not s['prior']], f"{conv} old, no prior attempt")
    summ([s for s in N if s['conv']==conv], f"{conv} v3 re-ingests")
