# core/labels.py
#
# HAND LABELS FROM MORE THAN ONE ANNOTATOR, AND THE FINAL LABEL PER ITEM.
#
# Two label sets, each a directory under the results root:
#
#   abstention  abstention_labels/   item = item_key (sha256(cell|qa)[:12])
#                                    labels a (abstained), b (answered), s (unsure)
#   drop        quarantine_audit/    item = sample index (sample.jsonl)
#                                    labels justified, relevant, irrelevant, unsure
#
# In each directory:
#   labels.jsonl       every label ever given, append-only, one row per
#                      (item, annotator). Rows without an "annotator" (the
#                      interactive labeling in results/) count as one
#                      annotator, UNATTRIBUTED. Within one annotator a later
#                      row replaces an earlier one (their own correction);
#                      one annotator never replaces another's.
#   resolutions.jsonl  the owner's resolutions of disagreements and unsure
#                      items (scripts/final_labels.py --resolve, and
#                      adjudicate_abstention.py). Later rows win.
#
# The final label of an item, in this order:
#   1. the owner's resolution, if there is one;
#   2. the label every annotator who labeled it gave, if they agree
#      (one annotator alone counts as agreeing);
#   3. otherwise the owner's own label;
#   4. otherwise none (unresolved; excluded, and listed for the owner).
# An "unsure" final label is kept as such; readers exclude it.
#
# The owner is named once in <root>/labeling/owner.txt
# (python3 -m scripts.final_labels --owner NAME). It is needed only when
# annotators disagree.

from __future__ import annotations

import dataclasses
import json
import pathlib
from collections import Counter, defaultdict

from core.paths import ABSTENTION_LABELS, RESULTS

UNATTRIBUTED = "(unattributed)"
# labels copied to a new answer from an existing final label on the exact
# same question and answer text (scripts/make_label_sheets.py --v3c)
CARRIED = "(carried: exact text match)"
OWNER_FILE = RESULTS / "labeling" / "owner.txt"


@dataclasses.dataclass(frozen=True)
class Kind:
    name: str
    dir: pathlib.Path
    key_field: str
    labels: tuple
    unsure: str

    @property
    def labels_file(self) -> pathlib.Path:
        return self.dir / "labels.jsonl"

    @property
    def resolutions_file(self) -> pathlib.Path:
        return self.dir / "resolutions.jsonl"


ABSTENTION = Kind("abstention", ABSTENTION_LABELS, "item_key", ("a", "b", "s"), "s")
DROP = Kind("drop", RESULTS / "quarantine_audit", "item",
            ("justified", "relevant", "irrelevant", "unsure"), "unsure")
KINDS = {k.name: k for k in (ABSTENTION, DROP)}


def jl(p: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def owner() -> str | None:
    return (OWNER_FILE.read_text().strip() or None) if OWNER_FILE.exists() else None


def by_annotator(kind: Kind) -> dict[str, dict]:
    """annotator -> {item: label}; a later row of the same annotator wins."""
    out: dict[str, dict] = defaultdict(dict)
    for r in jl(kind.labels_file):
        out[r.get("annotator") or UNATTRIBUTED][r[kind.key_field]] = r["label"]
    return dict(out)


def resolutions(kind: Kind) -> dict:
    return {r[kind.key_field]: r["label"] for r in jl(kind.resolutions_file)}


@dataclasses.dataclass
class Final:
    labels: dict          # item -> final label (unsure included); unresolved items absent
    source: dict          # item -> "resolved" | "agreed" | "single" | "owner"
    unresolved: list      # items annotators disagree on, with no owner label or resolution
    annotators: list
    owner: str | None


def final(kind: Kind, strict: bool = True) -> Final:
    """The final labels (see the header). strict: raise SystemExit when
    annotators disagree and no owner is named, rather than leave those
    items silently unresolved."""
    ann = by_annotator(kind)
    res = resolutions(kind)
    own = owner()
    labels, source, unresolved = {}, {}, []
    items = set(res) | {k for d in ann.values() for k in d}
    for k in sorted(items, key=str):
        if k in res:
            labels[k], source[k] = res[k], "resolved"
            continue
        given = {a: d[k] for a, d in ann.items() if k in d}
        vals = set(given.values())
        if len(vals) == 1:
            labels[k] = vals.pop()
            source[k] = "single" if len(given) == 1 else "agreed"
        elif own is not None and own in given:
            labels[k], source[k] = given[own], "owner"
        else:
            unresolved.append(k)
    if unresolved and strict and own is None:
        raise SystemExit(f"{len(unresolved)} {kind.name} items have disagreeing labels and no "
                         f"owner is named: python3 -m scripts.final_labels --owner NAME")
    return Final(labels, source, unresolved, sorted(ann), own)


def is_unsure(kind: Kind, label) -> bool:
    return label == kind.unsure


def decided(kind: Kind, f: Final) -> dict:
    """item -> final label, unsure items excluded."""
    return {k: v for k, v in f.labels.items() if not is_unsure(kind, v)}


# ------------------------------------------------------------ agreement
def kappa(a_vals, b_vals) -> float:
    """Cohen's kappa for two raters over any set of categories."""
    n = len(a_vals)
    if n == 0:
        return float("nan")
    agree = sum(1 for x, y in zip(a_vals, b_vals) if x == y) / n
    pa, pb = Counter(a_vals), Counter(b_vals)
    exp = sum((pa[k] / n) * (pb[k] / n) for k in set(pa) | set(pb))
    if abs(1 - exp) < 1e-12:
        return float("nan")
    return (agree - exp) / (1 - exp)


def agreement(kind: Kind, a: str, b: str, group_of: dict | None = None) -> dict:
    """Percent agreement and Cohen's kappa between annotators a and b on
    the items both labeled (unsure counted as its own category).
    group_of maps item -> distinct shown text; with it, each distinct
    text counts once (an annotator who gave one text two different
    labels has it left out, and counted)."""
    ann = by_annotator(kind)
    da, db = ann.get(a, {}), ann.get(b, {})
    both = sorted(set(da) & set(db), key=str)
    units, inconsistent = [], 0
    if group_of is None:
        units = [(da[k], db[k]) for k in both]
    else:
        ga, gb = defaultdict(set), defaultdict(set)
        for k in both:
            g = group_of.get(k, ("item", k))
            ga[g].add(da[k])
            gb[g].add(db[k])
        for g in ga:
            if len(ga[g]) == 1 and len(gb[g]) == 1:
                units.append((next(iter(ga[g])), next(iter(gb[g]))))
            else:
                inconsistent += 1
    x, y = [u[0] for u in units], [u[1] for u in units]
    n = len(units)
    out = {"annotators": [a, b], "n": n,
           "percent": (100 * sum(p == q for p, q in units) / n) if n else None,
           "kappa": kappa(x, y) if n else None,
           "with_unsure_either_side": sum(kind.unsure in u for u in units)}
    if group_of is not None:
        out["inconsistent_texts"] = inconsistent
    sure = [(p, q) for p, q in units if kind.unsure not in (p, q)]
    out["n_sure"] = len(sure)
    out["percent_sure"] = (100 * sum(p == q for p, q in sure) / len(sure)) if sure else None
    out["kappa_sure"] = kappa([p for p, _ in sure], [q for _, q in sure]) if sure else None
    return out
