# scripts/stage4b_report.py
#
# STAGE 4b / 4c, READ-ONLY. No API calls. Run after a batch (__v3b, or
# with --batch v3c / v3d the third / fourth) has finished;
# scripts/run_v3.sh runs it on exit 0:
#
#   MEMDEF_RESULTS=results_v3 python3 -m scripts.stage4b_report > results_v3/stage4b_report.txt
#   MEMDEF_RESULTS=results_v3 python3 -m scripts.stage4b_report --batch v3c > results_v3/stage4c_report.txt
#   MEMDEF_RESULTS=results_v3 python3 -m scripts.stage4b_report --batch v3d > results_v3/stage4c_report.txt
#   (--snapshot [--batch v3c|v3d]: the read-only snapshot taken before the batch)
#
# --batch v3d is Stage 4c rerun after the fourth batch: the same report,
# with "unchanged" measured against the snapshot taken before __v3d.
# Stage 4c adds: each main cell's store's final ingestion (measured) against
# the October cutoff, and every write-time block in a main cell's store
# outside the known three (conv0 D1:12, conv1 D2:4, conv2 D4:11), and the
# extraction-prompt date of each main cell's store (October must be
# 2026-10-30).
#
# Everything is taken from the manifest (results_v3/final_cells.json):
# 1. The integrity check on the manifest cells (main + split), including
#    the store, build-period and split rules (must pass).
# 2. The build period per conversation, measured from each store
#    builder's final ingestion.
# 3. Store size per manifest cell.
# 4. Token overhead tau_d from the manifest: defense tokens of the 15
#    manifest cells of condition d / baseline base tokens, where baseline
#    base tokens = the store extraction tokens (the builder's final
#    ingestion, backbone only) + the answer tokens of the 15 manifest
#    none cells. Also per build period. Writes <root>/tau_manifest.json.
# 5. Stores unchanged by answering: every namespace fingerprinted before
#    the second batch is unchanged; every manifest cell's store had the
#    same size before and after answering and has it now; every cell
#    directory that existed before the batch (superseded ones included)
#    is byte-identical.
# 6. Nothing under results/ changed since the Stage 0 backup.
# Exit 0 only if every check passes.

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import defaultdict

from core import manifest as mf
from core.paths import CELLS, QDRANT, REPO_ROOT, RESULTS
from scripts.check_integrity import print_known, print_manifest_report, print_report, run_manifest_checks
from scripts.store_fingerprint import fingerprint

OLD = REPO_ROOT / "results"
GEMINI = "gemini-3.1-flash-lite"
MAIN = ["none", "sanitize", "provenance", "anomaly", "rerank", "stacked"]
DEFENDED = MAIN[1:]
BATCH = "v3b"
FP_BEFORE = RESULTS / "store_fingerprint_before_v3b.json"
CELLS_BEFORE = RESULTS / "cells_before_v3b.sha256"


def set_batch(batch: str) -> None:
    """The batch this report (or snapshot) is for: v3b, v3c or v3d."""
    global BATCH, FP_BEFORE, CELLS_BEFORE
    assert batch in ("v3b", "v3c", "v3d"), batch
    BATCH = batch
    FP_BEFORE = RESULTS / f"store_fingerprint_before_{batch}.json"
    CELLS_BEFORE = RESULTS / f"cells_before_{batch}.sha256"
short = lambda c: c.replace("anchor_gemini__gemini-backbone__", "")


def tokens(rows, pred) -> int:
    return sum(r.get("tokens_total") or 0 for r in rows if pred(r))


def is_answer(r):
    return r["_step"] == "answer" and r["status"] == "ok"


def is_defense(r):
    return r.get("purpose") == "defense"


def cell_files_digest(d) -> dict:
    return {str(p.relative_to(CELLS)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file()}


def main() -> int:
    if RESULTS.name != "results_v3":
        sys.exit("run with MEMDEF_RESULTS=results_v3")
    m = mf.load()
    if m is None:
        sys.exit(f"no manifest at {mf.MANIFEST}")
    ok = True
    stage = {"v3b": "4b", "v3c": "4c", "v3d": "4c"}[BATCH]
    print(f"STAGE {stage} REPORT (batch __{BATCH}"
          + (", rerun after the fourth batch" if BATCH == "v3d" else "") + "; read-only)\n")

    # ---------------------------------------------------------------- 1
    print("=" * 78 + f"\n1. INTEGRITY CHECK, manifest cells ({mf.MANIFEST.name}, "
          f"written {m['created'][:19]} at {m['git_commit'][:12]})\n" + "=" * 78)
    mr = run_manifest_checks(m)
    print_report(mr.reports, mr.by_cell)
    print_manifest_report(mr)
    bad = [r for r in mr.reports if r.violations]
    n_main = sum(r.info["entry"]["role"] == "main" for r in mr.reports)
    n_split = len(mr.reports) - n_main
    passed = not bad and not mr.violations and n_main == 90 and n_split == 45
    print(f"\n-> {'PASS' if passed else 'FAIL'}: {len(mr.reports)} cells ({n_main} main, "
          f"{n_split} split), {len(bad)} with violations, {len(mr.violations)} manifest violations")
    ok &= passed

    # ---------------------------------------------------------------- 2
    print("\n" + "=" * 78 + "\n2. BUILD PERIOD PER CONVERSATION (measured from each store "
          "builder's final ingestion)\n" + "=" * 78)
    print(f"  {'conversation':13} {'declared':8} {'main cells':>10}  measured")
    for conv, declared in m["periods"].items():
        rs = [r for r in mr.reports if r.info["entry"]["role"] == "main"
              and r.info["entry"]["conv_id"] == conv]
        got = sorted({mr.store_period.get(r.cell_id, "?") for r in rs})
        flag = "ok" if got == [declared] else "MISMATCH"
        ok &= got == [declared]
        print(f"  {conv:13} {declared:8} {len(rs):>10}  {', '.join(got)}  {flag}")
    split = [r for r in mr.reports if r.info["entry"]["role"] == "split"]
    sp = sorted({mr.store_period.get(r.cell_id, "?") for r in split})
    print(f"  split ({len(split)} cells): {', '.join(sp)}  {'ok' if sp == ['Aug'] else 'NOT AUGUST-ONLY'}")
    ok &= sp == ["Aug"]
    if BATCH in ("v3c", "v3d"):
        ok &= final_ingestion_section(m, mr)

    # ---------------------------------------------------------------- 3
    print("\n" + "=" * 78 + "\n3. STORE SIZE PER CELL\n" + "=" * 78)
    size = {}
    print(f"  {'role':5} {'condition':10} {'cell':38} {'store':38} {'period':6} {'memories':>8}")
    for r in mr.reports:
        e, s = r.info["entry"], r.info["summary"]
        size[e["cell_id"]] = s.get("memories_before_answering", s.get("memories_stored"))
        print(f"  {e['role']:5} {e['condition']:10} {short(e['cell_id']):38} "
              f"{short(e['store_namespace']):38} {e['store_period']:6} {size[e['cell_id']]!s:>8}")
    by = defaultdict(list)
    for r in mr.reports:
        e = r.info["entry"]
        if e["role"] == "main":
            by[(e["conv_id"], e["condition"])].append(size[e["cell_id"]])
    print("\n  mean store size, main cells, per conversation and condition:")
    print(f"  {'conversation':13} " + " ".join(f"{c:>10}" for c in MAIN))
    for conv in m["periods"]:
        print(f"  {conv + ' (' + m['periods'][conv] + ')':13} " + " ".join(
            f"{sum(by[(conv, c)]) / len(by[(conv, c)]):>10.1f}" if by[(conv, c)] else f"{'-':>10}"
            for c in MAIN))

    if BATCH in ("v3c", "v3d"):
        ok &= blocks_section(mr)

    # ---------------------------------------------------------------- 4
    print("\n" + "=" * 78 + "\n4. TOKEN OVERHEAD tau_d FROM THE MANIFEST\n" + "=" * 78)
    rep = {r.cell_id: r for r in mr.reports if r.info["entry"]["role"] == "main"}
    main_e = [r.info["entry"] for r in mr.reports if r.info["entry"]["role"] == "main"]

    def store_extract(e) -> int:
        b = e["store_built_by"]
        br = mr.builders[(b["root"], b["cell_id"])]
        return tokens(br.final_extract, lambda r: r["model_resolved"] == GEMINI)

    tau = {}
    for scope, keep in (("all", lambda e: True),
                        ("Oct", lambda e: e["store_period"] == "Oct"),
                        ("Aug", lambda e: e["store_period"] == "Aug")):
        es = [e for e in main_e if keep(e)]
        none = [e for e in es if e["condition"] == "none"]
        ext = sum(store_extract(e) for e in none)
        ans = sum(tokens(rep[e["cell_id"]].final_rows, is_answer) for e in none)
        base = ext + ans
        tau[scope] = {"baseline_base_tokens": base, "baseline_store_extraction": ext,
                      "baseline_answers": ans, "none_cells": len(none)}
        print(f"\n  [{scope}] baseline base tokens = {base:,} (store extraction {ext:,} + answers "
              f"{ans:,}; {len(none)} manifest none cells)")
        print(f"    {'condition':11} {'cells':>5} {'defense tokens':>15} {'tau':>8}")
        for cond in DEFENDED:
            cs = [e for e in es if e["condition"] == cond]
            d = sum(tokens(rep[e["cell_id"]].final_rows, is_defense) for e in cs)
            tau[scope][cond] = {"defense_tokens": d, "tau": d / base if base else None,
                                "cells": len(cs)}
            print(f"    {cond:11} {len(cs):>5} {d:>15,} "
                  f"{(f'{100 * d / base:7.2f}%') if base else '-':>8}")
    (RESULTS / "tau_manifest.json").write_text(json.dumps(tau, indent=2))
    print(f"\n  wrote {RESULTS / 'tau_manifest.json'}")

    # ---------------------------------------------------------------- 5
    print("\n" + "=" * 78 + "\n5. STORES UNCHANGED BY ANSWERING\n" + "=" * 78)
    now = fingerprint(QDRANT)
    if FP_BEFORE.exists():
        before = json.loads(FP_BEFORE.read_text())["namespaces"]
        changed = [ns for ns in before if now.get(ns) != before[ns]]
        print(f"  namespaces fingerprinted before the __{BATCH} batch: {len(before)}; changed since: "
              f"{len(changed)}" + (f"  {changed}" if changed else ""))
        ok &= not changed
    else:
        print(f"  MISSING {FP_BEFORE.name}: cannot compare")
        ok = False
    print(f"\n  {'cell':40} {'store':40} {'before':>6} {'after':>6} {'now':>6}  check")
    for r in mr.reports:
        e, s = r.info["entry"], r.info["summary"]
        n_now = (now.get(e["store_namespace"]) or {}).get("points", 0)
        if e["root"] == RESULTS.name:
            b, a = s.get("memories_before_answering"), s.get("memories_after_answering")
            same = b == a == n_now
        else:   # an old v2 cell: its store's copy must hold what it stored
            b = a = s.get("memories_stored")
            same = b == n_now
        ok &= same
        print(f"  {short(e['cell_id']):40} {short(e['store_namespace']):40} {b!s:>6} {a!s:>6} "
              f"{n_now:>6}  {'ok' if same else 'CHANGED'}")
    if CELLS_BEFORE.exists():
        before_files = json.loads(CELLS_BEFORE.read_text())
        now_files = {}
        for name in sorted({k.split("/")[0] for k in before_files}):
            now_files.update(cell_files_digest(CELLS / name))
        diff = sorted(k for k in before_files if now_files.get(k) != before_files[k])
        print(f"\n  cell directories that existed before the batch: "
              f"{len({k.split('/')[0] for k in before_files})}; files changed since: {len(diff)}"
              + (f" {diff[:5]}" if diff else ""))
        ok &= not diff
    else:
        print(f"\n  MISSING {CELLS_BEFORE.name}: cannot compare")
        ok = False

    # ---------------------------------------------------------------- 6
    print("\n" + "=" * 78 + "\n6. results/ UNTOUCHED\n" + "=" * 78)
    backup = REPO_ROOT / "results_backup_20261029.tar.gz"
    newer = subprocess.run(["find", str(OLD), "-newer", str(backup)], capture_output=True,
                           text=True).stdout.split()
    print(f"files under results/ newer than {backup.name}: {len(newer)}" + (f" {newer[:5]}" if newer else ""))
    ok &= not newer

    print(f"\nSTAGE {stage}: {'ALL CHECKS PASS' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


def final_ingestion_section(m, mr) -> bool:
    """Each main cell's store's final ingestion, measured from its
    builder's log, against the manifest's record and the cutoff."""
    from scripts.check_integrity import OCT_CUTOFF, OCT_PROMPT_DATE
    print(f"\n  final ingestion of each main cell's store (UTC) and its extraction-prompt date(s), "
          f"measured; October must end at or after {OCT_CUTOFF[:19]} with prompt date "
          f"{OCT_PROMPT_DATE}, August must be in August (its prompt dates are reported, not checked)")
    print(f"  {'cell':38} {'period':6} {'batch':5} {'start':19}   {'end':8}  {'recorded':8}  "
          f"{'prompt date(s)':22} check")
    good = True
    for e in m["cells"]:
        if e["role"] != "main":
            continue
        span = mr.store_ingestion.get(e["cell_id"])
        rec = e.get("store_final_ingestion")
        same = bool(rec) and span is not None and (rec["start"], rec["end"]) == span
        dates = mr.store_prompt_dates.get(e["cell_id"]) or []
        if span is None:
            fine = False
        elif e["store_period"] == "Oct":
            fine = span[1] >= OCT_CUTOFF and dates == [OCT_PROMPT_DATE]
        else:
            fine = span[0][:7] == span[1][:7] == "2026-08"
        same &= e.get("store_prompt_date") == dates
        good &= fine and same
        print(f"  {short(e['cell_id']):38} {e['store_period']:6} {e['batch']:5} "
              f"{(span or ('-', '-'))[0][:19]:19} .. {(span or ('-', '-'))[1][11:19]:8}  "
              f"{'same' if same else 'DIFFERS' if rec else 'MISSING':8}  "
              f"{'+'.join(dates) or '-':22} {'ok' if fine else 'FAIL'}")
    print(f"  -> {'all main stores in their batch window, October prompt date ' + OCT_PROMPT_DATE + ', manifest record matches' if good else 'FAILURES'}")
    return good


def blocks_section(mr) -> bool:
    """Write-time blocks in the main cells' stores. Blocks outside the
    known three are listed (never a failure: they are applied and
    recorded); a block without its raw reply is a failure."""
    from runner.v3 import KNOWN_BLOCKS
    print("\n" + "=" * 78 + "\n3b. WRITE-TIME BLOCKS in the main cells' stores (from each store "
          "builder's quarantine log)\n" + "=" * 78)
    good, new = True, []
    for r in mr.reports:
        e = r.info["entry"]
        if e["role"] != "main" or e["condition"] not in ("sanitize", "provenance", "anomaly", "stacked"):
            continue
        b = e["store_built_by"]
        rows = [json.loads(l) for l in (mf.root_dir(b["root"]) / "quarantine" / f"{b['cell_id']}.jsonl")
                .read_text().splitlines() if l.strip()] if (mf.root_dir(b["root"]) / "quarantine"
                                                            / f"{b['cell_id']}.jsonl").exists() else []
        blocks = [x for x in rows if x.get("phase") != "read"]
        for x in blocks:
            known = KNOWN_BLOCKS.get(e["conv_idx"]) == x.get("turn_id")
            if "reply=" not in (x.get("reason") or "") and x.get("defense") == "anomaly":
                good = False
                print(f"  NO RAW REPLY: {short(e['cell_id'])} {x.get('turn_id')}: {x.get('reason')}")
            if not known:
                new.append((e, x))
    print(f"  blocks outside the known three: {len(new)}")
    for e, x in new:
        print(f"    {short(e['cell_id']):38} turn {x.get('turn_id')} by {x.get('defense')}: "
              f"{x.get('reason')}\n      content: {(x.get('content') or '')[:160]!r}")
    return good


def snapshot() -> int:
    """Before the batch: fingerprint every namespace and hash every
    existing cell directory (read-only on both)."""
    fp = fingerprint(QDRANT)
    FP_BEFORE.write_text(json.dumps({"store": str(QDRANT), "namespaces": fp}, indent=2))
    files = {}
    for d in sorted(CELLS.iterdir()):
        if d.is_dir():
            files.update(cell_files_digest(d))
    CELLS_BEFORE.write_text(json.dumps(files, indent=1))
    print(f"wrote {FP_BEFORE.name} ({len(fp)} namespaces, {sum(v['points'] for v in fp.values())} "
          f"points) and {CELLS_BEFORE.name} ({len(files)} files)")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", action="store_true")
    ap.add_argument("--batch", default="v3b", choices=["v3b", "v3c", "v3d"])
    a = ap.parse_args()
    set_batch(a.batch)
    sys.exit(snapshot() if a.snapshot else main())
