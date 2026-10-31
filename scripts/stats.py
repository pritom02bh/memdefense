# scripts/stats.py
#
# PAIRED STATISTICS FOR DEFENSE DELTAS.
#
# The problem this solves: between-conversation spread (~11pp) is roughly
# ten times larger than the defense deltas we are trying to measure
# (~1-3pp). Comparing GROUP MEANS therefore says almost nothing -- the
# conversations drawn dominate the defense applied.
#
# The fix is to pair. Every condition asks the SAME questions of the SAME
# conversations, so each question has a matched baseline counterpart.
# Differencing within that pair cancels conversation difficulty entirely.
#
# Three tests are reported, deliberately:
#
#   1. STRATIFIED BOOTSTRAP CI -- resamples questions within each
#      conversation, recomputes the equal-weight-per-conversation delta
#      10k times, reports the 2.5/97.5 percentiles. This is the number
#      to put in the paper.
#
#   2. EXACT McNEMAR -- pooled over all paired questions, using only the
#      discordant pairs (baseline right/defense wrong, and vice versa).
#      Exact binomial, no chi-square approximation, so it stays valid
#      when discordant counts are small.
#
#   3. EXACT SIGN-FLIP PERMUTATION over the per-conversation deltas.
#      This is the conservative test: it treats the conversation as the
#      unit of analysis. NOTE ITS FLOOR -- with 5 conversations there are
#      only 2^5 = 32 sign assignments, so the smallest two-sided p-value
#      obtainable is 2/32 = 0.0625. At n=5 this test CANNOT reach p<0.05
#      no matter how large the effect. That is a property of the design,
#      not of the data, and it is why (1) is the headline.
#
# No third-party dependencies -- stdlib only, so it runs anywhere.
#
#   python3 -m scripts.stats
#   python3 -m scripts.stats --group anchor_gemini --metric core4
#   python3 -m scripts.stats --metric adversarial --boots 20000
#   python3 -m scripts.stats --json        # also write <root>/stats.json (read by the figures)
#
# CELLS. A root with a manifest (core/manifest.py) uses exactly its cells:
# the "main" cells for the per-condition table, and the "split" cells
# (old v2 none, old v2 rerank, baseline-mini; all on August memory) for
# the model-swap split, reported after it. A root without one (the
# original results/) uses every cell under it.

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import pathlib
from core import manifest as mf
from core.paths import ADVERSARIAL_RESCORED, CELLS, RESULTS as ROOT
from scripts.check_integrity import require_integrity
import random
from collections import defaultdict

RESULTS = CELLS
RESCORED = ADVERSARIAL_RESCORED
BASELINE = "none"

_RESCORE_CACHE = None


def load_rescored():
    """{item_key: abstained_bool} from rescore_abstention.py, or {}.

    item_key is sha256("<cell_dir_name>|<qa_id>")[:12], matching the
    key construction in label_abstention.py and rescore_abstention.py.
    """
    global _RESCORE_CACHE
    if _RESCORE_CACHE is not None:
        return _RESCORE_CACHE
    if not RESCORED.exists():
        print(f"!! WARNING: {RESCORED} not found.")
        print("!! Adversarial numbers below use the ORIGINAL biased regex.")
        print("!! Run: python3 -m scripts.rescore_abstention")
        _RESCORE_CACHE = {}
        return _RESCORE_CACHE
    blob = json.loads(RESCORED.read_text())
    _RESCORE_CACHE = {k: bool(v["abstained"])
                      for k, v in blob.get("per_item", {}).items()}
    return _RESCORE_CACHE


# ----------------------------------------------------------------- loading

def load_cells(group_filter=None, backbone_filter=None, role="main"):
    """Return {(group, backbone, condition, conv_id): [rows_per_replicate]}.

    Replicate runs (seed 2, 3, ...) of the same cell accumulate into the
    list rather than overwriting each other. With a manifest: its cells
    in `role` ("main" or "split"), under the manifest's condition names.
    """
    out = {}
    if mf.MANIFEST.exists() or mf.required():
        items = [(mf.entry_dir(e), e["condition"]) for e in mf.entries((role,))]
    elif role != "main" or not RESULTS.exists():
        return out
    else:
        items = [(d, None) for d in sorted(RESULTS.iterdir())]
    for d, cond in items:
        s_path, a_path = d / "summary.json", d / "answers.jsonl"
        if not (s_path.exists() and a_path.exists()):
            continue
        s = json.loads(s_path.read_text())
        if group_filter and s.get("group") != group_filter:
            continue
        if backbone_filter and s.get("backbone") != backbone_filter:
            continue
        key = (s.get("group"), s.get("backbone"), cond or s.get("condition"), s.get("conv_id"))
        rows = [json.loads(l) for l in a_path.read_text().splitlines() if l.strip()]
        # Override adversarial verdicts with the corrected labels.
        # Core4 rows are judge-graded and left exactly as run.
        resc = load_rescored()
        if resc:
            for r in rows:
                if r.get("category") != "adversarial":
                    continue
                ik = hashlib.sha256(
                    f"{d.name}|{r.get('qa_id')}".encode()).hexdigest()[:12]
                if ik in resc:
                    r["correct"] = resc[ik]
        out.setdefault(key, []).append(rows)
    return out


def select(rows, metric):
    """{qa_id: bool} for the requested metric, dropping ungraded items."""
    sel = {}
    for r in rows:
        if r.get("correct") is None:
            continue
        is_adv = r.get("category") == "adversarial"
        if metric == "core4" and is_adv:
            continue
        if metric == "adversarial" and not is_adv:
            continue
        sel[r["qa_id"]] = bool(r["correct"])
    return sel


def select_replicates(reps, metric):
    """Collapse replicate runs of one cell into a single {qa_id: bool}
    by majority vote across replicates.

    An item is scored correct only if it was correct in MORE than half
    the runs, so an even split resolves conservatively to wrong. With a
    single replicate this is the identity function, leaving single-seed
    conditions byte-identical to the pre-patch behaviour.

    Items missing from some replicates (e.g. a dropped judge call) are
    voted on by the replicates that do have them; pairing against the
    baseline still intersects on shared qa_ids downstream.
    """
    votes = defaultdict(list)
    for rows in reps:
        for q, ok in select(rows, metric).items():
            votes[q].append(ok)
    return {q: (sum(v) * 2 > len(v)) for q, v in votes.items()}


# ------------------------------------------------------------------- stats

def conv_delta(base, cond):
    """Mean paired delta within one conversation, and the paired items."""
    shared = sorted(set(base) & set(cond))
    if not shared:
        return None, []
    pairs = [(base[q], cond[q]) for q in shared]
    d = sum(int(c) - int(b) for b, c in pairs) / len(pairs)
    return d, pairs


def bootstrap_ci(per_conv_pairs, boots=10000, seed=12345):
    """Stratified paired bootstrap: resample questions WITHIN each
    conversation, average conversation-level deltas with equal weight
    (matching runner.analyse), repeat, take percentiles."""
    rng = random.Random(seed)
    if not per_conv_pairs:
        return None, None, None
    stats = []
    for _ in range(boots):
        conv_means = []
        for pairs in per_conv_pairs:
            n = len(pairs)
            if n == 0:
                continue
            tot = 0
            for _ in range(n):
                b, c = pairs[rng.randrange(n)]
                tot += int(c) - int(b)
            conv_means.append(tot / n)
        if conv_means:
            stats.append(sum(conv_means) / len(conv_means))
    if not stats:
        return None, None, None
    stats.sort()
    lo = stats[int(0.025 * len(stats))]
    hi = stats[min(int(0.975 * len(stats)), len(stats) - 1)]
    return lo, hi, sum(stats) / len(stats)


def mcnemar_exact(per_conv_pairs):
    """Exact two-sided McNemar over all paired questions pooled.
    b = baseline correct & defense wrong; c = the reverse."""
    b = c = 0
    for pairs in per_conv_pairs:
        for base_ok, cond_ok in pairs:
            if base_ok and not cond_ok:
                b += 1
            elif cond_ok and not base_ok:
                c += 1
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return b, c, min(1.0, 2 * tail)


def signflip_exact(deltas):
    """Exact two-sided sign-flip permutation test on per-conversation
    deltas. Conservative: n conversations = 2^n assignments."""
    n = len(deltas)
    if n == 0:
        return 1.0, None
    obs = abs(sum(deltas) / n)
    hits = 0
    for signs in itertools.product([1, -1], repeat=n):
        m = abs(sum(s * d for s, d in zip(signs, deltas)) / n)
        if m >= obs - 1e-12:
            hits += 1
    floor = 2 / (2 ** n)
    return hits / (2 ** n), floor


def compare(data, base_cond, cond, metric, boots):
    """Paired delta cond - base_cond over the conversations both have
    (majority vote over replicates), with the same three tests as the
    main table. None if nothing pairs."""
    per_conv_pairs, deltas, convs_used, n_pairs = [], [], [], 0
    for (c2, conv), rows in sorted(data.items()):
        if c2 != cond or (base_cond, conv) not in data:
            continue
        d, pairs = conv_delta(select_replicates(data[(base_cond, conv)], metric),
                              select_replicates(rows, metric))
        if d is None:
            continue
        deltas.append(d)
        convs_used.append(conv)
        per_conv_pairs.append(pairs)
        n_pairs += len(pairs)
    if not deltas:
        return None
    lo, hi, _ = bootstrap_ci(per_conv_pairs, boots)
    b, c, p_mc = mcnemar_exact(per_conv_pairs)
    p_perm, _ = signflip_exact(deltas)
    return {"point": sum(deltas) / len(deltas), "ci": [lo, hi], "n_conv": len(deltas),
            "n_pairs": n_pairs, "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": p_mc,
            "signflip_p": p_perm, "per_conv": dict(zip(convs_used, deltas))}


SPLIT = [("none", "mini", "(a) baseline-mini - old none   (model swap, no defense)"),
         ("mini", "rerank", "(b) old rerank - baseline-mini (rerank at equal model)"),
         ("none", "rerank", "(c) old rerank - old none      (the v2 delta = (a) + (b))")]


def report_split(args, metrics, out_json):
    """The model-swap split from the manifest's split cells."""
    split = load_cells(args.group, args.backbone, role="split")
    arms = defaultdict(dict)
    for (g, bb, cond, conv), rows in split.items():
        arms[(g, bb)][(cond, conv)] = rows
    for (g, bb), data in sorted(arms.items()):
        for metric in metrics:
            print("\n" + "=" * 92)
            print(f"MODEL-SWAP SPLIT  {g} / {bb}   metric = {metric}   (August memory only; "
                  f"old = v2 cells in results/)")
            print("=" * 92)
            print(f"{'comparison':<56} {'conv':>4} {'pairs':>6} {'delta':>9} {'95% CI':>22} "
                  f"{'McNemar':>9} {'b/c':>9} {'perm p':>8}")
            for base, cond, label in SPLIT:
                r = compare(data, base, cond, metric, args.boots)
                if r is None:
                    print(f"{label:<56}  (nothing paired)")
                    continue
                lo, hi = r["ci"]
                ci = f"[{lo:+.4f}, {hi:+.4f}]" if lo is not None else "n/a"
                print(f"{label:<56} {r['n_conv']:>4} {r['n_pairs']:>6} {r['point']:>+9.4f} "
                      f"{ci:>22} {r['mcnemar_p']:>9.4f} "
                      f"{str(r['mcnemar_b']) + '/' + str(r['mcnemar_c']):>9} "
                      f"{r['signflip_p']:>8.4f}")
                out_json.setdefault("split", {}).setdefault(f"{g}/{bb}", {}).setdefault(
                    metric, {})[f"{cond}-{base}"] = r


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default=None)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--metric", default="both",
                    choices=["core4", "adversarial", "both"])
    ap.add_argument("--boots", type=int, default=10000)
    ap.add_argument("--json", action="store_true",
                    help="also write every number printed to <root>/stats.json")
    args = ap.parse_args()
    require_integrity()

    cells = load_cells(args.group, args.backbone)
    if not cells:
        print(f"No cells found under {RESULTS}.")
        return 1

    metrics = ["core4", "adversarial"] if args.metric == "both" else [args.metric]

    # group by (group, backbone)
    arms = defaultdict(dict)   # (group,bb) -> {(cond,conv): rows}
    for (g, bb, cond, conv), rows in cells.items():
        arms[(g, bb)][(cond, conv)] = rows

    out_json = {}
    for (g, bb), data in sorted(arms.items()):
        conds = sorted({c for c, _ in data})
        if BASELINE not in conds:
            print(f"\n{g} / {bb}: no '{BASELINE}' baseline -- skipping.")
            continue

        for metric in metrics:
            print("\n" + "=" * 92)
            print(f"{g} / {bb}   metric = {metric}   "
                  f"(paired vs '{BASELINE}', equal weight per conversation)")
            print("=" * 92)
            print(f"{'condition':<12} {'conv':>4} {'pairs':>6} {'delta':>9} "
                  f"{'95% CI':>22} {'McNemar':>9} {'b/c':>9} {'perm p':>8}")
            reps_by_cond = {}
            for (c2, conv), reps in data.items():
                reps_by_cond.setdefault(c2, set()).add(len(reps))
            print("replicates/conv:  " + ",  ".join(
                f"{c}={'/'.join(str(x) for x in sorted(reps_by_cond[c]))}"
                for c in conds))
            print("-" * 92)

            deltas = []   # guard: baseline-only arms never enter the loop body
            for cond in conds:
                if cond == BASELINE:
                    continue
                per_conv_pairs, deltas, n_pairs, convs_used = [], [], 0, []
                for (c2, conv), rows in data.items():
                    if c2 != cond:
                        continue
                    base_rows = data.get((BASELINE, conv))
                    if base_rows is None:
                        continue
                    base = select_replicates(base_rows, metric)
                    cur = select_replicates(rows, metric)
                    d, pairs = conv_delta(base, cur)
                    if d is None:
                        continue
                    deltas.append(d)
                    convs_used.append(conv)
                    per_conv_pairs.append(pairs)
                    n_pairs += len(pairs)

                if not deltas:
                    print(f"{cond:<12} {'-':>4} {'-':>6}  (no paired baseline)")
                    continue

                point = sum(deltas) / len(deltas)
                lo, hi, _ = bootstrap_ci(per_conv_pairs, args.boots)
                b, c, p_mc = mcnemar_exact(per_conv_pairs)
                p_perm, floor = signflip_exact(deltas)

                ci = f"[{lo:+.4f}, {hi:+.4f}]" if lo is not None else "n/a"
                crosses = (lo is not None and lo <= 0 <= hi)
                mark = "" if not crosses else "  (CI includes 0)"

                print(f"{cond:<12} {len(deltas):>4} {n_pairs:>6} {point:>+9.4f} "
                      f"{ci:>22} {p_mc:>9.4f} {f'{b}/{c}':>9} {p_perm:>8.4f}{mark}")
                out_json.setdefault(f"{g}/{bb}", {}).setdefault(metric, {})[cond] = {
                    "point": point, "ci": [lo, hi], "n_conv": len(deltas), "n_pairs": n_pairs,
                    "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": p_mc, "signflip_p": p_perm,
                    "per_conv": dict(zip(convs_used, deltas))}

            if deltas:
                print("-" * 92)
                print(f"note: permutation p has a floor of {floor:.4f} at "
                      f"{len(deltas)} conversations -- it cannot go below this")
                print("      regardless of effect size. Use the bootstrap CI as the "
                      "headline number.")
                print("      b = baseline right / defense wrong;  c = defense right "
                      "/ baseline wrong.")
    if mf.MANIFEST.exists():
        report_split(args, metrics, out_json)
    if args.json:
        path = ROOT / "stats.json"
        path.write_text(json.dumps(out_json, indent=2))
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())