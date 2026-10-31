# scripts/final_labels.py
#
# FINAL HAND LABELS, AGREEMENT AND DISAGREEMENTS. No API calls.
#
#   MEMDEF_RESULTS=results_v3 python3 -m scripts.final_labels --owner "NAME"   # once
#   MEMDEF_RESULTS=results_v3 python3 -m scripts.final_labels                  # report
#   MEMDEF_RESULTS=results_v3 python3 -m scripts.final_labels --resolve FILE   # import
#
# For each label set (abstention, drop) it reports who labeled what, and
# the final label per item as core/labels.py defines it: the owner's
# resolution; else the shared label where the annotators agree (or the
# only annotator's label); else the owner's label.
#
# With two or more annotators it computes, per pair, percent agreement
# and Cohen's kappa -- over distinct shown texts (each text once, as on
# the sheets) and over sampled items -- and writes <dir>/agreement.json.
#
# Whatever is left for the owner -- items the annotators disagree on, and
# items whose final label is unsure -- goes to <dir>/disagreements.txt,
# one entry per distinct text, with every annotator's label and the
# current final label. Fill in a letter to change the final label, or
# leave "_" to keep it, and import it with --resolve; resolutions are
# appended to <dir>/resolutions.jsonl and override everything else.

from __future__ import annotations

import argparse
import datetime
import json
import sys

from core import labels as lb
from core import manifest as mf
from scripts.import_labels import SheetError, find_key, parse_answers
from scripts.make_label_sheets import (ADV_LETTERS, DROP_LETTERS, WIDTH, item_key, render,
                                       sheet_id, shown_gold)

LETTERS = {"abstention": ADV_LETTERS, "drop": DROP_LETTERS}
FIELDS = {"abstention": ("Question", "Agent's answer"),
          "drop": ("Question", "Gold answer", "Dropped memory")}


def letter(kind: lb.Kind, label) -> str:
    inv = {v: k for k, v in LETTERS[kind.name].items()}
    return inv.get(label, str(label))


def shown_texts(kind: lb.Kind) -> dict:
    """item -> the tuple a sheet shows for it."""
    if kind is lb.DROP:
        return {s["item"]: (s["question"], shown_gold(s), s["dropped_memory"])
                for s in lb.jl(kind.dir / "sample.jsonl")}
    return {item_key(d.name, r["qa_id"]): (r.get("question", ""), r["predicted"])
            for d in mf.cell_dirs() for r in lb.jl(d / "answers.jsonl")
            if r.get("category") == "adversarial" and r.get("predicted")}


def report(kind: lb.Kind) -> int:
    ann = lb.by_annotator(kind)
    print("=" * WIDTH + f"\n{kind.name.upper()} LABELS  ({kind.dir})\n" + "=" * WIDTH)
    if not ann and not lb.resolutions(kind):
        print("  no labels yet")
        return 0
    for a, d in sorted(ann.items()):
        c = {}
        for v in d.values():
            c[v] = c.get(v, 0) + 1
        print(f"  {a:24} {len(d):>4} items  " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))
    f = lb.final(kind, strict=False)
    texts = shown_texts(kind)
    group_of = texts

    if len(ann) >= 2 and any(set(ann[x]) & set(ann[y]) for x in ann for y in ann if x < y):
        names = sorted(ann)
        # only annotators who labeled some of the same items (a carried
        # label never shares an item with the labeler it was copied from)
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
                 if set(ann[a]) & set(ann[b])]
        out = []
        print("\n  agreement (unsure counted as its own category; '_sure' leaves out items "
              "either labeled unsure)")
        for a, b in pairs:
            for unit, g in (("distinct texts", group_of), ("sampled items", None)):
                s = lb.agreement(kind, a, b, g)
                s["unit"] = unit
                out.append(s)
                pct = f"{s['percent']:.1f}%" if s["percent"] is not None else "-"
                kap = f"{s['kappa']:.3f}" if s["kappa"] is not None else "-"
                pcs = f"{s['percent_sure']:.1f}%" if s["percent_sure"] is not None else "-"
                kas = f"{s['kappa_sure']:.3f}" if s["kappa_sure"] is not None else "-"
                inc = (f", {s['inconsistent_texts']} texts labeled inconsistently, left out"
                       if s.get("inconsistent_texts") else "")
                print(f"    {a} vs {b}, {unit}: n={s['n']}  {pct}  kappa {kap}   "
                      f"n_sure={s['n_sure']}  {pcs}  kappa_sure {kas}{inc}")
        (kind.dir / "agreement.json").write_text(json.dumps(out, indent=2))
        print(f"  wrote {kind.dir / 'agreement.json'}")

    src, lab = {}, {}
    for k, v in f.labels.items():
        lab[v] = lab.get(v, 0) + 1
        src[f.source[k]] = src.get(f.source[k], 0) + 1
    print(f"\n  final labels: {len(f.labels)} items  " +
          ", ".join(f"{k}={v}" for k, v in sorted(lab.items())))
    print("    from: " + ", ".join(f"{k} {v}" for k, v in sorted(src.items()))
          + (f"; unresolved {len(f.unresolved)}" if f.unresolved else ""))
    if f.unresolved and f.owner is None:
        print(f"  {len(f.unresolved)} items disagree and no owner is named; readers refuse "
              f"until an owner is named: python3 -m scripts.final_labels --owner NAME")
    write_disagreements(kind, f, ann, texts)
    return 0


def write_disagreements(kind: lb.Kind, f: lb.Final, ann: dict, texts: dict) -> None:
    txt, kp = kind.dir / "disagreements.txt", kind.dir / "disagreements.key.json"
    todo = [k for k in sorted(set(f.labels) | set(f.unresolved), key=str)
            if k in f.unresolved or f.source[k] in ("owner",)
            or (lb.is_unsure(kind, f.labels[k]) and f.source[k] != "resolved")]
    if not todo:
        for p in (txt, kp):
            if p.exists():
                p.unlink()
        print("  nothing to resolve")
        return
    groups, order = {}, []
    for k in todo:
        t = texts.get(k, (f"(item {k}: text not found)",))
        if t not in groups:
            groups[t] = []
            order.append(t)
        groups[t].append(k)
    shown, info = [], []
    for t in order:
        ks = groups[t]
        by = {}
        for a, d in sorted(ann.items()):
            got = sorted({letter(kind, d[k]) for k in ks if k in d})
            if got:
                by[a] = "/".join(got)
        cur = sorted({letter(kind, f.labels[k]) if k in f.labels else "none (unresolved)"
                      for k in ks})
        why = ("unresolved disagreement" if any(k in f.unresolved for k in ks) else
               "disagreement" if any(f.source.get(k) == "owner" for k in ks) else "unsure")
        status = (f"{why}; labels: " + ", ".join(f"{a} = {v}" for a, v in by.items())
                  + f"; current final label: {'/'.join(cur)}")
        shown.append(t + (status,))
        info.append({"items": ks, "why": why})
    sid = sheet_id(f"res-{kind.name}", shown)
    letters = LETTERS[kind.name]
    rubric = (f"Items the annotators disagree on, and items whose final label is unsure.\n"
              f"Write a letter ({', '.join(letters)}) to set the final label, or leave '_' to\n"
              f"keep the current one. Then:\n\n"
              f"  MEMDEF_RESULTS=... python3 -m scripts.final_labels --resolve {txt.name}")
    text = render(f"{kind.name} labels to resolve (owner only; shows annotators)", sid,
                  letters, rubric, "", shown, FIELDS[kind.name] + ("Status",))
    text = text.replace("4. Answer every item. A sheet with any item left blank is refused.",
                        "4. Leave '_' to keep an item's current final label.")
    txt.write_text(text)
    kp.write_text(json.dumps({"sheet": f"resolve_{kind.name}", "kind": kind.name,
                              "sheet_id": sid, "letters": letters, "n_display": len(shown),
                              "items": [{"n": n, **i} for n, i in enumerate(info, 1)]},
                             indent=2))
    print(f"  {len(shown)} to resolve ({len(todo)} items) -> {txt}")


def resolve(path) -> int:
    text = open(path).read()
    try:
        key = find_key(text, [k.dir for k in lb.KINDS.values()])
        if not key["sheet"].startswith("resolve_"):
            raise SheetError(f"{path} is a labeling sheet; use scripts.import_labels")
        answers = parse_answers(text, key, allow_blank=True)
    except SheetError as e:
        print(f"REFUSED {path}:\n{e}", file=sys.stderr)
        return 2
    kind = lb.KINDS[key["kind"]]
    own = lb.owner()
    if own is None:
        print("REFUSED: name the owner first (--owner NAME)", file=sys.stderr)
        return 2
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    rows = [{kind.key_field: k, "label": key["letters"][answers[it["n"]]], "by": own,
             "resolved": now, "note": f"{key['sheet_id']} item {it['n']} ({it['why']})"}
            for it in key["items"] if it["n"] in answers for k in it["items"]]
    with open(kind.resolutions_file, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"{len(answers)} of {key['n_display']} entries resolved ({len(rows)} items) "
          f"-> {kind.resolutions_file}; the rest keep their current final label")
    return report(kind)


def main() -> int:
    ap = argparse.ArgumentParser(description="final hand labels, agreement, disagreements")
    ap.add_argument("--owner", help="name the owner (whose label wins a disagreement)")
    ap.add_argument("--resolve", metavar="FILE", help="import a filled disagreements.txt")
    args = ap.parse_args()
    if args.owner:
        name = args.owner.strip()
        if not name or name == lb.UNATTRIBUTED:
            sys.exit("give a name")
        lb.OWNER_FILE.parent.mkdir(parents=True, exist_ok=True)
        lb.OWNER_FILE.write_text(name + "\n")
        print(f"owner: {name!r} ({lb.OWNER_FILE})")
        return 0
    if args.resolve:
        return resolve(args.resolve)
    print(f"owner: {lb.owner()!r}")
    for kind in lb.KINDS.values():
        report(kind)
    return 0


if __name__ == "__main__":
    sys.exit(main())
