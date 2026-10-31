# scripts/import_labels.py
#
# IMPORT A FILLED-IN LABELING SHEET (from scripts/make_label_sheets.py).
# No API calls.
#
#   MEMDEF_RESULTS=results_v3 python3 -m scripts.import_labels FILLED_SHEET.txt
#
# The sheet's "Sheet ID" line picks its key in <root>/labeling/keys/. The
# ANSWER SHEET must have every item exactly once with one valid letter;
# anything else (a blank, two letters, an unknown item, a missing item, a
# changed Sheet ID) refuses the whole sheet and writes nothing.
#
# It asks for the labeler's name and appends one row per item, through
# the key, to the normal label file, with the annotator recorded:
#
#   adversarial_labels -> abstention_labels/labels.jsonl
#       {"item_key", "label": a|b|s, "annotator", "sheet", "imported"}
#       (and merges the items into abstention_labels/keymap.json)
#   drop_audit         -> quarantine_audit/labels.jsonl
#       {"item", "label": justified|relevant|irrelevant|unsure, "annotator", ...}
#       (writing quarantine_audit/sample.jsonl from the key if it is absent;
#       an existing sample.jsonl must match the key)
#
# Rows are appended, never rewritten, and every annotator's labels stay
# separate; core/labels.py turns them into one final label per item
# (python3 -m scripts.final_labels for agreement and disagreements). A
# sheet already imported for the same annotator is refused.

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re
import sys

from core import labels as lb
from scripts.make_label_sheets import KEYS
SHEET_ID_RX = re.compile(r"^Sheet ID:\s*(\S+)\s*$", re.M)
LINE_RX = re.compile(r"^\s*(\d+)\s*:\s*(.*?)\s*$")


class SheetError(Exception):
    pass


def jl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def find_key(text: str, keys_dirs=(KEYS,)) -> dict:
    ids = SHEET_ID_RX.findall(text)
    if len(ids) != 1:
        raise SheetError(f"expected one 'Sheet ID:' line, found {len(ids)}")
    if isinstance(keys_dirs, pathlib.Path):
        keys_dirs = (keys_dirs,)
    for kp in sorted(p for d in keys_dirs for p in d.glob("*.key.json")):
        key = json.loads(kp.read_text())
        if key["sheet_id"] == ids[0]:
            return key
    raise SheetError(f"no key for sheet {ids[0]} in {', '.join(map(str, keys_dirs))} "
                     f"(right MEMDEF_RESULTS? was the Sheet ID line edited?)")


def parse_answers(text: str, key: dict, allow_blank: bool = False) -> dict[int, str]:
    """Item number -> letter, or SheetError listing every problem.
    allow_blank: a blank ('_') item is left out instead of refused."""
    if "\nANSWER SHEET\n" not in text:
        raise SheetError("no ANSWER SHEET section")
    section = text.split("\nANSWER SHEET\n", 1)[1]
    letters = set(key["letters"])
    n_items = key.get("n_display", len(key["items"]))
    got, problems = {}, []
    for line in section.splitlines():
        m = LINE_RX.match(line)
        if not m:
            continue
        n, val = int(m.group(1)), m.group(2).upper()
        if not 1 <= n <= n_items:
            problems.append(f"item {n}: no such item (sheet has 1-{n_items})")
        elif n in got:
            problems.append(f"item {n}: appears more than once")
        elif val in ("", "_"):
            if allow_blank:
                got[n] = None
            else:
                problems.append(f"item {n}: blank")
        elif val not in letters:
            problems.append(f"item {n}: {m.group(2)!r} is not one of {', '.join(sorted(letters))}")
        else:
            got[n] = val
    missing = [n for n in range(1, n_items + 1) if n not in got
               and not any(p.startswith(f"item {n}:") for p in problems)]
    if missing:
        problems.append(f"missing items: {missing}")
    if problems:
        raise SheetError("\n".join(problems))
    return {n: v for n, v in got.items() if v is not None}


def rows_for(key: dict, answers: dict[int, str], annotator: str, now: str) -> list[dict]:
    meta = {"annotator": annotator, "sheet": key["sheet_id"], "imported": now}
    if key["sheet"].startswith("adversarial_labels"):
        return [{"item_key": it["item_key"], "label": key["letters"][answers[it["n"]]], **meta}
                for it in key["items"]]
    if key["sheet"] == "drop_audit":
        return [{"item": it["sample"]["item"], "label": key["letters"][answers[it["n"]]], **meta}
                for it in key["items"]]
    raise SheetError(f"unknown sheet type {key['sheet']!r}")


def write_labels(key: dict, rows: list[dict], annotator: str,
                 abst_dir: pathlib.Path = lb.ABSTENTION.dir,
                 audit_dir: pathlib.Path = lb.DROP.dir) -> pathlib.Path:
    if key["sheet"].startswith("adversarial_labels"):
        labels = abst_dir / "labels.jsonl"
    else:
        labels = audit_dir / "labels.jsonl"
    if any(r.get("sheet") == key["sheet_id"] and r.get("annotator") == annotator
           for r in jl(labels)):
        raise SheetError(f"{annotator!r} already has labels from sheet {key['sheet_id']} "
                         f"in {labels}; nothing written")

    if key["sheet"].startswith("adversarial_labels"):
        abst_dir.mkdir(parents=True, exist_ok=True)
        kp = abst_dir / "keymap.json"
        keymap = json.loads(kp.read_text()) if kp.exists() else {}
        for it in key["items"]:
            keymap[it["item_key"]] = {"cell_id": it["cell_id"], "qa_id": it["qa_id"],
                                      "regex_said": it["regex_said"]}
        kp.write_text(json.dumps(keymap, indent=2))
    else:
        audit_dir.mkdir(parents=True, exist_ok=True)
        sp = audit_dir / "sample.jsonl"
        want = [it["sample"] for it in key["items"]]
        if sp.exists():
            have = jl(sp)
            if have != want:
                raise SheetError(f"{sp} exists and differs from the key's sample; nothing "
                                 f"written")
        else:
            sp.write_text("".join(json.dumps(s) + "\n" for s in want))

    with open(labels, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return labels


def main() -> int:
    ap = argparse.ArgumentParser(description="import a filled-in labeling sheet")
    ap.add_argument("sheet", type=pathlib.Path)
    args = ap.parse_args()
    text = args.sheet.read_text()
    try:
        key = find_key(text)
        answers = parse_answers(text, key)
    except SheetError as e:
        print(f"REFUSED {args.sheet}:\n{e}", file=sys.stderr)
        return 2
    counts = {}
    for v in answers.values():
        counts[v] = counts.get(v, 0) + 1
    print(f"sheet {key['sheet_id']} ({key['sheet']}): {len(answers)} items, all valid; "
          + ", ".join(f"{k}={counts.get(k, 0)}" for k in key["letters"]))
    annotator = ""
    while not annotator:
        try:
            annotator = input("Labeler's name: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nno name given; nothing written", file=sys.stderr)
            return 2
        if annotator == lb.UNATTRIBUTED:
            print(f"{annotator!r} is reserved; give a name", file=sys.stderr)
            annotator = ""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        out = write_labels(key, rows_for(key, answers, annotator, now), annotator)
    except SheetError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(f"wrote {len(key['items'])} labels ({len(answers)} items shown) by {annotator!r} "
          f"to {out}")
    print("next: MEMDEF_RESULTS=... python3 -m scripts.final_labels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
