# scripts/stage6_report.py
#
# STAGE 6, READ-ONLY. No API calls. Every number for the paper from the
# manifest cells (results_v3/final_cells.json), with the final hand labels:
#
#   MEMDEF_RESULTS=results_v3 python3 -m scripts.rescore_abstention   # adversarial_rescored.json
#   MEMDEF_RESULTS=results_v3 python3 -m scripts.stage6_report > results_v3/stage6_report.txt
#   (--json NAME: where under the root to write the numbers; default stage6.json)
#
# 1. Per condition: Delta_d (core) vs none with the stratified bootstrap
#    95% CI, exact McNemar (discordant b/c), exact sign-flip p; Delta_d
#    per replicate and per conversation.
# 2. The same Delta_d over the October (conv-26, 30, 41) and the August
#    (conv-42, 43) conversations separately.
# 3. Adversarial abstention change vs none, per replicate and pooled,
#    with the final labels; how many regex scores the labels changed.
# 4. Quarantine rates for anomaly, rerank and stacked.
# 5. The drop audit: J/R/I counts with exact (Clopper-Pearson) 95% CIs,
#    and the R drops that turned a baseline-correct answer wrong.
# 6. Token overhead tau_d (tau_manifest.json, Stage 4b).
# 7. The gpt-4o-mini split, (a) (b) (c), with CIs.
# 8. Ungraded questions per condition.
#
# Conventions are scripts/stats.py's: pooled deltas majority-vote the
# three replicates per question, then weight conversations equally;
# per-replicate deltas pair replicate k of a condition with replicate k
# of none.

from __future__ import annotations

import collections
import json
import math
import sys

from core import labels as lb
from core import manifest as mf
from core.paths import ADVERSARIAL_RESCORED, QUARANTINE, RESULTS
from scripts import stats as st
from scripts.check_integrity import require_integrity
from scripts.make_label_sheets import item_key

DEFENDED = ["sanitize", "provenance", "anomaly", "rerank", "stacked"]
MAIN = ["none"] + DEFENDED
BOOTS = 10000
RULE = "=" * 100


def jl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def pp(x, spec="+.2f"):
    return "n/a" if x is None else format(100 * x, spec)


def ci_str(ci):
    lo, hi = ci
    return "n/a" if lo is None else f"[{100 * lo:+.2f}, {100 * hi:+.2f}]"


# ------------------------------------------------------------------ data
def load(role: str):
    """entries, and rows[(cond, conv, seed)] with the final adversarial
    scores applied (st.load_cells' override, from adversarial_rescored.json)."""
    resc = st.load_rescored()
    ents = mf.entries((role,))
    rows = {}
    for e in ents:
        d = mf.entry_dir(e)
        rs = jl(d / "answers.jsonl")
        for r in rs:
            if r.get("category") == "adversarial":
                r["regex"] = r.get("correct")
                k = item_key(d.name, r["qa_id"])
                if k in resc:
                    r["correct"] = resc[k]
        rows[(e["condition"], e["conv_id"], e["seed"])] = rs
    return ents, rows


def pooled(rows, convs):
    """{(cond, conv): [replicate rows]} restricted to convs, for st.compare."""
    data = collections.defaultdict(list)
    for (cond, conv, seed), rs in sorted(rows.items()):
        if conv in convs:
            data[(cond, conv)].append(rs)
    return data


def per_replicate(rows, cond, base, metric, convs, seeds, key="correct"):
    """seed -> equal-weight mean over convs of the paired delta, replicate k vs k."""
    out = {}
    for s in seeds:
        ds = []
        for conv in convs:
            if (cond, conv, s) not in rows or (base, conv, s) not in rows:
                continue
            sel = lambda rs: st.select([dict(r, correct=r.get(key)) for r in rs], metric)
            d, _ = st.conv_delta(sel(rows[(base, conv, s)]), sel(rows[(cond, conv, s)]))
            if d is not None:
                ds.append(d)
        out[s] = sum(ds) / len(ds) if ds else None
    return out


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    """Exact binomial CI by bisection on the binomial CDF (stdlib only)."""
    if n == 0:
        return None, None

    def cdf(x, p):  # P(X <= x)
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(x + 1))

    def solve(f):
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(lambda p: cdf(k - 1, p) > 1 - alpha / 2)
    upper = 1.0 if k == n else solve(lambda p: cdf(k, p) > alpha / 2)
    return lower, upper


# -------------------------------------------------------------- sections
def delta_table(rows, convs, seeds, metric, title, per_conv=True):
    print(f"\n{title}")
    print(f"  {'condition':11} {'pairs':>5} {'Delta_d':>8} {'95% CI (pp)':>18} {'McNemar p':>10} "
          f"{'b/c':>7} {'sign-flip p':>11}   " + "  ".join(f"rep s{s}" for s in seeds))
    data = pooled(rows, convs)
    out = {}
    floor = None
    for cond in DEFENDED:
        r = st.compare(data, "none", cond, metric, BOOTS)
        if r is None:
            print(f"  {cond:11} (nothing paired)")
            continue
        reps = per_replicate(rows, cond, "none", metric, convs, seeds)
        floor = 2 / 2 ** r["n_conv"]
        r["per_replicate"] = reps
        out[cond] = r
        print(f"  {cond:11} {r['n_pairs']:>5} {pp(r['point']):>8} {ci_str(r['ci']):>18} "
              f"{r['mcnemar_p']:>10.4f} {str(r['mcnemar_b']) + '/' + str(r['mcnemar_c']):>7} "
              f"{r['signflip_p']:>11.4f}   " + "  ".join(f"{pp(reps[s]):>6}" for s in seeds))
    if per_conv and out:
        print(f"\n  Delta_d per conversation (majority vote over replicates), pp")
        print(f"  {'condition':11} " + " ".join(f"{c:>8}" for c in convs))
        for cond, r in out.items():
            print(f"  {cond:11} " + " ".join(f"{pp(r['per_conv'].get(c)):>8}" for c in convs))
    if floor is not None:
        print(f"  (sign-flip p floor at {len(convs)} conversations: {floor:.4f}; "
              f"b = none right & defense wrong, c = the reverse)")
    return out


def section_quarantine(ents):
    print(f"\n{RULE}\n4. QUARANTINE RATES (main cells)\n{RULE}")
    print("  write-time FQR = turns blocked / turns screened; read-time FQR = drops / memories "
          "judged by the reranker")
    out = {}
    for cond in ("anomaly", "rerank", "stacked"):
        cells = []
        for e in ents:
            if e["condition"] != cond:
                continue
            d = mf.entry_dir(e)
            s = json.loads((d / "summary.json").read_text())
            ws = s.get("write_screen") or {}
            blocked = len(ws.get("blocked") or [])
            turns = ws.get("turns_screened") or s.get("turns_ingested")
            rt = s.get("rerank_totals") or {}
            reads = [r for r in jl(QUARANTINE / f"{d.name}.jsonl") if r.get("phase") == "read"]
            ans = jl(d / "answers.jsonl")
            cells.append({"cell": d.name, "blocked": blocked, "turns": turns,
                          "drops": rt.get("dropped", 0), "judged": rt.get("retrieved", 0),
                          "log_drops": len(reads),
                          "distinct": len({r["dropped_memory"] for r in reads}),
                          "stored": s.get("memories_before_answering", s.get("memories_stored")),
                          "q_hit": sum((r.get("memories_dropped") or 0) > 0 for r in ans),
                          "q": len(ans)})
        assert all(c["log_drops"] == c["drops"] for c in cells), f"{cond}: log/summary drop mismatch"
        tot = lambda k: sum(c[k] for c in cells)
        rng = lambda f: (min(map(f, cells)), max(map(f, cells)))
        res = {"cells": len(cells)}
        print(f"\n  {cond} ({len(cells)} cells)")
        if cond in ("anomaly", "stacked"):
            lo, hi = rng(lambda c: c["blocked"] / c["turns"])
            res["write_fqr"] = tot("blocked") / tot("turns")
            res["write_fqr_range"] = [lo, hi]
            print(f"    write-time: {tot('blocked')} / {tot('turns')} turns blocked = "
                  f"{100 * res['write_fqr']:.2f}%   per cell {100 * lo:.2f}% - {100 * hi:.2f}%")
        if cond in ("rerank", "stacked"):
            lo, hi = rng(lambda c: c["drops"] / c["judged"])
            res.update({
                "read_fqr": tot("drops") / tot("judged"), "read_fqr_range": [lo, hi],
                "questions_with_drop": tot("q_hit") / tot("q"),
                "stored_dropped_once": tot("distinct") / tot("stored"),
                "drops_per_dropped_memory": tot("drops") / tot("distinct") if tot("distinct") else None})
            print(f"    read-time:  {tot('drops')} / {tot('judged')} judged = "
                  f"{100 * res['read_fqr']:.2f}%   per cell {100 * lo:.2f}% - {100 * hi:.2f}%")
            print(f"    questions with >=1 drop: {tot('q_hit')} / {tot('q')} = "
                  f"{100 * res['questions_with_drop']:.1f}%")
            print(f"    stored memories dropped at least once: {tot('distinct')} / {tot('stored')} = "
                  f"{100 * res['stored_dropped_once']:.1f}%   (distinct dropped texts per cell, summed)")
            dpm = res["drops_per_dropped_memory"]
            print(f"    mean drops per dropped memory: {tot('drops')} / {tot('distinct')} = "
                  + (f"{dpm:.2f}" if dpm is not None else "n/a"))
        out[cond] = res
    return out


def section_audit():
    print(f"\n{RULE}\n5. DROP AUDIT (50 sampled rerank drops, final labels)\n{RULE}")
    sample = {s["item"]: s for s in jl(lb.DROP.dir / "sample.jsonl")}
    fin = lb.final(lb.DROP)
    dec = lb.decided(lb.DROP, fin)
    unsure = sorted(k for k, v in fin.labels.items() if lb.is_unsure(lb.DROP, v))
    n = len(dec)
    print(f"  annotators: {', '.join(fin.annotators)}; owner: {fin.owner}; final-label sources: "
          f"{dict(collections.Counter(fin.source.values()))}")
    print(f"  decided: {n} of {len(sample)}; still unsure (left out): {len(unsure)} {unsure}; "
          f"unresolved disagreements: {len(fin.unresolved)}")
    c = collections.Counter(dec.values())
    out = {"n": n, "unsure": unsure, "counts": {}}
    for lab, letter in (("justified", "J"), ("relevant", "R"), ("irrelevant", "I")):
        lo, hi = clopper_pearson(c[lab], n)
        out["counts"][lab] = {"k": c[lab], "share": c[lab] / n if n else None, "ci": [lo, hi]}
        print(f"  {letter} {lab:11} {c[lab]:>3} / {n}  {100 * c[lab] / n:5.1f}%   exact 95% CI "
              f"[{100 * lo:.1f}%, {100 * hi:.1f}%]")
    rel = [sample[i] for i, v in dec.items() if v == "relevant"]
    flipped = [s for s in rel if s["baseline_correct"] and not s["rerank_correct"]]
    kept = [s for s in rel if s["baseline_correct"] and s["rerank_correct"]]
    base_wrong = [s for s in rel if not s["baseline_correct"]]
    out.update({"relevant_flipped_to_wrong": len(flipped), "relevant_baseline_right_still_right":
                len(kept), "relevant_baseline_wrong": len(base_wrong)})
    print(f"\n  among the {len(rel)} R drops (the answer to that question in the drop's rerank "
          f"cell vs its paired none cell, same conversation and replicate):")
    print(f"    none right -> rerank wrong: {len(flipped)}")
    print(f"    none right -> rerank right: {len(kept)}")
    print(f"    none wrong:                 {len(base_wrong)}")
    by_cat = collections.Counter(s["category"] for s in flipped)
    if flipped:
        print(f"    flipped, by category: {dict(by_cat)}")
    print("  note: a drop is one (question, memory) event; the same memory dropped for the same "
          "question in two replicates counts twice.")
    return out


def section_adversarial(rows, convs, seeds):
    print(f"\n{RULE}\n3. ADVERSARIAL ABSTENTION (final labels)\n{RULE}")
    resc = json.loads(ADVERSARIAL_RESCORED.read_text())
    fin = lb.decided(lb.ABSTENTION, lb.final(lb.ABSTENTION))
    # how the labels moved the regex scores, main cells
    changed = collections.Counter()
    det_vs_regex = 0
    keymap = {}
    for e in mf.entries(("main",)):
        d = mf.entry_dir(e)
        for r in jl(d / "answers.jsonl"):
            if r.get("category") == "adversarial" and r.get("predicted"):
                keymap[item_key(d.name, r["qa_id"])] = (e["condition"], bool(r["correct"]))
    per_item = resc["per_item"]
    for k, (cond, regex) in keymap.items():
        if k in fin:
            if (fin[k] == "a") != regex:
                changed[(cond, "failure->abstain" if fin[k] == "a" else "abstain->failure")] += 1
        elif bool(per_item[k]["abstained"]) != regex:
            det_vs_regex += 1
    n_reg_fail = sum(1 for c, rx in keymap.values() if not rx)
    lab_fail = sum(1 for k, (c, rx) in keymap.items() if k in fin and not rx)
    lab_abst = sum(1 for k, (c, rx) in keymap.items() if k in fin and rx)
    print(f"  main-cell adversarial answers: {len(keymap)}; regex failures {n_reg_fail} "
          f"(hand-labeled: {lab_fail}); regex abstentions hand-labeled: {lab_abst}")
    print(f"  regex scores changed by the labels: {sum(changed.values())}   " +
          ", ".join(f"{c} {w}: {v}" for (c, w), v in sorted(changed.items())))
    print(f"  unlabeled items where the score (expanded detector) differs from the regex: "
          f"{det_vs_regex}")
    cal = resc["calibration"]
    print(f"  calibration on {cal['n']} labels: original regex agreement "
          f"{cal['original_agreement']:.3f} (kappa {cal['original_kappa']:.3f}); "
          f"expanded detector {cal['expanded_agreement']:.3f} (kappa {cal['expanded_kappa']:.3f})")

    print(f"\n  abstention rate per condition and replicate (final scores; equal weight per "
          f"conversation), %")
    print(f"  {'condition':11} " + " ".join(f"{'s' + str(s):>7}" for s in seeds))
    rate = {}
    for cond in MAIN:
        vals = []
        for s in seeds:
            rs = [st.select(rows[(cond, c, s)], "adversarial") for c in convs if (cond, c, s) in rows]
            per = [sum(x.values()) / len(x) for x in rs if x]
            vals.append(sum(per) / len(per) if per else None)
        rate[cond] = vals
        print(f"  {cond:11} " + " ".join(f"{pp(v, '.2f'):>7}" for v in vals))
    out = delta_table(rows, convs, seeds, "adversarial",
                      "  change vs none, final scores (pooled = majority vote; then per replicate)",
                      per_conv=False)
    print("\n  for comparison, change vs none per replicate with the ORIGINAL regex scores, pp")
    for cond in DEFENDED:
        reps = per_replicate(rows, cond, "none", "adversarial", convs, seeds, key="regex")
        print(f"  {cond:11} " + "  ".join(f"s{s} {pp(v):>6}" for s, v in reps.items()))
    return {"changed": {f"{c} {w}": v for (c, w), v in changed.items()},
            "detector_vs_regex_unlabeled": det_vs_regex, "rate": rate, "delta": out}


def section_builds(ents, convs):
    """Which day built each store, its size, and how many answers two
    cells share word for word -- identical cells are not independent
    replicates, and stores built on different days differ in size."""
    print(f"\n{RULE}\n9. STORE BUILDS AND REPLICATE INDEPENDENCE (main cells)\n{RULE}")
    cell = {}
    for e in ents:
        d = mf.entry_dir(e)
        s = json.loads((d / "summary.json").read_text())
        b = e["store_built_by"]
        bs = json.loads((mf.root_dir(b["root"]) / "cells" / b["cell_id"] / "summary.json").read_text())
        # the batch that built the store (the first ran overnight across
        # midnight UTC, so the date alone does not tell them apart)
        started = ((bs.get("write_screen") or {}).get("started") or "?")[5:13].replace("T", " ")
        built = "Aug v2" if e["store_period"] == "Aug" else f"{e['batch']} {started}h"
        ans = {r["qa_id"]: r.get("predicted") for r in jl(d / "answers.jsonl")}
        cell[(e["conv_id"], e["condition"], e["seed"])] = {
            "built": built, "mem": s.get("memories_before_answering", s.get("memories_stored")),
            "ans": ans, "dates": e.get("store_prompt_date") or []}
    print("  store size by replicate, and the batch that built the store (with its write-time "
          "screening start, MM-DD HH UTC)")
    out = {"cells": {}, "identical": {}}
    for conv in convs:
        print(f"\n  {conv}")
        for cond in MAIN:
            cs = [cell.get((conv, cond, s)) for s in (1, 2, 3)]
            print(f"    {cond:11} " + "   ".join(f"s{s} {c['mem']:>3} ({c['built']:>12})"
                                               for s, c in zip((1, 2, 3), cs) if c))
            for s, c in zip((1, 2, 3), cs):
                out["cells"][f"{conv}/{cond}/s{s}"] = {"built": c["built"], "mem": c["mem"]}

    def same(a, b):
        A, B = cell[a]["ans"], cell[b]["ans"]
        q = set(A) & set(B)
        return sum(A[x] == B[x] for x in q), len(q)

    print("\n  answers identical word for word, summed over replicates s1-s3 "
          "(condition vs none, same replicate; and none s2 vs none s3)")
    print(f"  {'conversation':13} " + " ".join(f"{c:>11}" for c in DEFENDED) + f" {'none s2/s3':>11}")
    for conv in convs:
        vals = []
        for cond in DEFENDED:
            n = t = 0
            for s in (1, 2, 3):
                a, b = same((conv, cond, s), (conv, "none", s))
                n, t = n + a, t + b
            vals.append(f"{100 * n / t:>10.0f}%")
            out["identical"][f"{conv}/{cond}"] = [n, t]
        a, b = same((conv, "none", 2), (conv, "none", 3))
        out["identical"][f"{conv}/none_s2_s3"] = [a, b]
        print(f"  {conv:13} " + " ".join(vals) + f" {100 * a / b:>10.0f}%")
    # which batches built each conversation's main stores, with which
    # extraction-prompt date (Mem0 writes the date into every extraction
    # prompt), and how alike the replicates of none are
    print("\n  batches that built the main stores, and their extraction-prompt dates, per conversation:")
    out["batches"], out["prompt_dates"] = {}, {}
    for conv in convs:
        keys = [(conv, c, s) for c in MAIN for s in (1, 2, 3) if (conv, c, s) in cell]
        bs = collections.Counter(cell[k]["built"].split(" ")[0] for k in keys)
        ds = collections.Counter(d for k in keys for d in cell[k]["dates"])
        out["batches"][conv] = dict(bs)
        out["prompt_dates"][conv] = dict(ds)
        n, t = out["identical"][f"{conv}/none_s2_s3"]
        print(f"    {conv}: " + ", ".join(f"{b} {k}" for b, k in sorted(bs.items()))
              + f"   (prompt date{'s' if len(ds) > 1 else ''} "
              + ", ".join(f"{d} {k}" for d, k in sorted(ds.items()))
              + f"; none s2 vs s3 answers identical {100 * n / t:.0f}%)")
    print("  Stores extracted with the same prompt date by the same Gemini compare like with like "
          "(October: every main\n  store has 2026-10-30, whichever run built it); where replicate "
          "answers are near-identical\n  (deterministic extraction), its three replicates are not "
          "independent draws. August stores differ\n  between replicates even on the same prompt "
          "date.")
    return out


def main() -> int:
    if RESULTS.name != "results_v3":
        sys.exit("run with MEMDEF_RESULTS=results_v3")
    require_integrity()
    if not ADVERSARIAL_RESCORED.exists():
        sys.exit("run scripts.rescore_abstention first")
    m = mf.load()
    ents, rows = load("main")
    convs = list(m["periods"])
    oct_ = [c for c, p in m["periods"].items() if p == "Oct"]
    aug = [c for c, p in m["periods"].items() if p == "Aug"]
    seeds = sorted({e["seed"] for e in ents})
    report = {"manifest": {"created": m["created"], "git_commit": m["git_commit"]}}
    print(f"STAGE 6 REPORT  (read-only; manifest {mf.MANIFEST.name} written {m['created'][:19]} "
          f"at {m['git_commit'][:12]}; {len(ents)} main cells; bootstrap {BOOTS} resamples)")
    print("integrity check on the manifest cells: PASS (require_integrity)")

    print(f"\n{RULE}\n1. CORE ACCURACY: Delta_d vs none, all five conversations\n{RULE}")
    report["core"] = delta_table(rows, convs, seeds, "core4",
                                 "  pooled (majority vote over 3 replicates, equal weight per "
                                 "conversation); then Delta_d per replicate, pp")

    print(f"\n{RULE}\n2. CORE ACCURACY BY BUILD PERIOD\n{RULE}")
    report["core_oct"] = delta_table(rows, oct_, seeds, "core4",
                                     f"  October memory ({', '.join(oct_)})")
    report["core_aug"] = delta_table(rows, aug, seeds, "core4",
                                     f"  August memory ({', '.join(aug)})")

    report["adversarial"] = section_adversarial(rows, convs, seeds)
    report["quarantine"] = section_quarantine(ents)
    report["audit"] = section_audit()

    print(f"\n{RULE}\n6. TOKEN OVERHEAD tau_d (tau_manifest.json, Stage 4b)\n{RULE}")
    tau = json.loads((RESULTS / "tau_manifest.json").read_text())
    print(f"  {'condition':11} " + " ".join(f"{s:>9}" for s in tau))
    for cond in DEFENDED:
        print(f"  {cond:11} " + " ".join(f"{100 * tau[s][cond]['tau']:>8.2f}%" for s in tau))
    print("  baseline base tokens: " + ", ".join(f"{s} {tau[s]['baseline_base_tokens']:,}" for s in tau))
    report["tau"] = tau

    print(f"\n{RULE}\n7. MODEL-SWAP SPLIT (all five conversations; every split cell reads an "
          f"August-built v2 store)\n{RULE}")
    sents, srows = load("split")
    sconvs = sorted({e["conv_id"] for e in sents})
    sdata = pooled(srows, sconvs)
    print(f"  {len(sents)} cells; conversations {sconvs}")
    report["split"] = {}
    for metric in ("core4", "adversarial"):
        print(f"\n  metric = {metric}")
        print(f"  {'comparison':52} {'pairs':>5} {'delta':>8} {'95% CI (pp)':>18} {'McNemar p':>10} "
              f"{'b/c':>7} {'sign-flip p':>11}")
        for base, cond, label in st.SPLIT:
            r = st.compare(sdata, base, cond, metric, BOOTS)
            report["split"].setdefault(metric, {})[f"{cond}-{base}"] = r
            print(f"  {label:52} {r['n_pairs']:>5} {pp(r['point']):>8} {ci_str(r['ci']):>18} "
                  f"{r['mcnemar_p']:>10.4f} {str(r['mcnemar_b']) + '/' + str(r['mcnemar_c']):>7} "
                  f"{r['signflip_p']:>11.4f}")
    print(f"  (sign-flip p floor at {len(sconvs)} conversations: {2 / 2 ** len(sconvs):.2f})")
    n_split_fail = sum(1 for rs in srows.values() for r in rs
                       if r.get("category") == "adversarial" and r.get("regex") is False)
    print(f"  adversarial note: the {n_split_fail} regex failures in the split cells were not on "
          f"the labeling sheet; they are scored by the expanded detector.")

    print(f"\n{RULE}\n8. UNGRADED QUESTIONS (judge gave no usable grade; left out of core accuracy)\n{RULE}")
    report["ungraded"] = {}
    for role, es, rs_all in (("main", ents, rows), ("split", sents, srows)):
        by = collections.defaultdict(lambda: [0, 0])
        for (cond, conv, seed), rs in rs_all.items():
            core = [r for r in rs if r.get("category") != "adversarial"]
            by[cond][0] += sum(r.get("correct") is None for r in core)
            by[cond][1] += len(core)
        report["ungraded"][role] = dict(by)
        print(f"  {role:5} " + "   ".join(f"{c} {u}/{n}" for c, (u, n) in sorted(
            by.items(), key=lambda kv: (MAIN + ['mini']).index(kv[0]))))

    report["builds"] = section_builds(ents, convs)

    out = RESULTS / (sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else "stage6.json")
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
