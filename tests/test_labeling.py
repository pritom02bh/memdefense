# tests/test_labeling.py -- the plain-text labeling sheets, their import,
# and the final label per item when several people label.
# Everything runs under the throwaway root set in tests/__init__.py.
import dataclasses
import json
import pathlib
import unittest

from tests import TMP
from core import labels as lb
from scripts import import_labels as il
from scripts.make_label_sheets import ADV_LETTERS, DROP_LETTERS, group, render

ROOT = pathlib.Path(TMP) / "labeling_test"


def adv_key(n=3, n_display=None):
    return {"sheet": "adversarial_labels", "sheet_id": "adv-0000000001", "letters": ADV_LETTERS,
            "n_display": n_display or n,
            "items": [{"n": min(i, n_display or n), "item_key": f"k{i}", "cell_id": f"cell{i}",
                       "qa_id": f"q{i}", "regex_said": False, "question": f"Q{i}?",
                       "predicted": f"ans {i}"} for i in range(1, n + 1)]}


def drop_key(n=2):
    return {"sheet": "drop_audit", "sheet_id": "drop-0000000001", "letters": DROP_LETTERS,
            "n_display": n,
            "items": [{"n": i + 1, "sample": {"item": i, "cell": "c", "question": f"Q{i}",
                                              "dropped_memory": f"m{i}"}} for i in range(n)]}


def sheet(key, answers):
    shown = [(f"Q{i}", f"text {i}") for i in range(key["n_display"])]
    text = render("t", key["sheet_id"], key["letters"], "rubric", "examples", shown,
                  ("Question", "Answer"))
    for n, a in answers.items():
        text = text.replace(f"\n{n}: _", f"\n{n}: {a}")
    return text


class TestParse(unittest.TestCase):
    def test_complete_sheet(self):
        k = adv_key()
        self.assertEqual(il.parse_answers(sheet(k, {1: "A", 2: "b", 3: "C"}), k),
                         {1: "A", 2: "B", 3: "C"})

    def test_blank_refused(self):
        k = adv_key()
        with self.assertRaisesRegex(il.SheetError, "item 3: blank"):
            il.parse_answers(sheet(k, {1: "A", 2: "B"}), k)

    def test_blank_allowed_for_resolutions(self):
        k = adv_key()
        self.assertEqual(il.parse_answers(sheet(k, {2: "B"}), k, allow_blank=True), {2: "B"})

    def test_invalid_letter_refused(self):
        k = drop_key()
        with self.assertRaisesRegex(il.SheetError, "item 2: 'A' is not one of"):
            il.parse_answers(sheet(k, {1: "J", 2: "A"}), k)

    def test_unsure_is_valid_on_drop_sheet(self):
        k = drop_key()
        self.assertEqual(il.parse_answers(sheet(k, {1: "U", 2: "r"}), k), {1: "U", 2: "R"})

    def test_two_letters_refused(self):
        k = adv_key()
        with self.assertRaisesRegex(il.SheetError, "item 1:"):
            il.parse_answers(sheet(k, {1: "A B", 2: "B", 3: "A"}), k)

    def test_missing_and_extra_refused(self):
        k = adv_key()
        text = sheet(k, {1: "A", 2: "B", 3: "A"}).replace("\n3: A", "\n4: A")
        with self.assertRaises(il.SheetError) as e:
            il.parse_answers(text, k)
        self.assertIn("item 4: no such item", str(e.exception))
        self.assertIn("missing items: [3]", str(e.exception))

    def test_sheet_id_must_match_a_key(self):
        d = ROOT / "keys_a"
        d.mkdir(parents=True, exist_ok=True)
        (d / "x.key.json").write_text(json.dumps(adv_key()))
        k = adv_key()
        self.assertEqual(il.find_key(sheet(k, {}), d)["sheet_id"], k["sheet_id"])
        with self.assertRaises(il.SheetError):
            il.find_key(sheet(k, {}).replace(k["sheet_id"], "adv-edited"), d)


class TestDuplicates(unittest.TestCase):
    def test_group_keeps_first_order_and_numbers_every_item(self):
        distinct, number = group([("q1", "x"), ("q2", "y"), ("q1", "x"), ("q1", "z")])
        self.assertEqual(distinct, [("q1", "x"), ("q2", "y"), ("q1", "z")])
        self.assertEqual(number, [1, 2, 1, 3])

    def test_one_letter_labels_every_duplicate(self):
        k = adv_key(n=4, n_display=3)          # items 3 and 4 share sheet item 3
        rows = il.rows_for(k, il.parse_answers(sheet(k, {1: "A", 2: "B", 3: "C"}), k), "Ann", "t")
        self.assertEqual([(r["item_key"], r["label"]) for r in rows],
                         [("k1", "a"), ("k2", "b"), ("k3", "s"), ("k4", "s")])


class TestWrite(unittest.TestCase):
    def test_adversarial_rows_and_refuse_reimport(self):
        k, d = adv_key(), ROOT / "w_adv"
        rows = il.rows_for(k, {1: "A", 2: "B", 3: "C"}, "Ann", "t")
        out = il.write_labels(k, rows, "Ann", abst_dir=d, audit_dir=d / "qa")
        got = [json.loads(l) for l in out.read_text().splitlines()]
        self.assertEqual([(r["item_key"], r["label"], r["annotator"]) for r in got],
                         [("k1", "a", "Ann"), ("k2", "b", "Ann"), ("k3", "s", "Ann")])
        self.assertEqual(set(json.loads((d / "keymap.json").read_text())), {"k1", "k2", "k3"})
        with self.assertRaisesRegex(il.SheetError, "already has labels"):
            il.write_labels(k, rows, "Ann", abst_dir=d, audit_dir=d / "qa")
        il.write_labels(k, il.rows_for(k, {1: "B", 2: "B", 3: "B"}, "Bo", "t"), "Bo",
                        abst_dir=d, audit_dir=d / "qa")
        self.assertEqual(len(out.read_text().splitlines()), 6)

    def test_drop_rows_write_sample_and_check_it(self):
        k, d = drop_key(), ROOT / "w_drop"
        out = il.write_labels(k, il.rows_for(k, {1: "J", 2: "U"}, "Ann", "t"), "Ann",
                              abst_dir=d / "ab", audit_dir=d)
        self.assertEqual([(r["item"], r["label"]) for r in map(json.loads, out.read_text().splitlines())],
                         [(0, "justified"), (1, "unsure")])
        self.assertEqual(len((d / "sample.jsonl").read_text().splitlines()), 2)
        other = drop_key()
        other["items"][0]["sample"]["dropped_memory"] = "different"
        with self.assertRaisesRegex(il.SheetError, "differs from the key"):
            il.write_labels(other, il.rows_for(other, {1: "R", 2: "R"}, "Bo", "t"), "Bo",
                            abst_dir=d / "ab", audit_dir=d)


class TestFinal(unittest.TestCase):
    """core/labels.final: resolution > shared label > owner's label."""

    def setUp(self):
        self.dir = ROOT / f"final_{self._testMethodName}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.kind = dataclasses.replace(lb.DROP, dir=self.dir)
        self.owner_file = self.dir / "owner.txt"
        self._old_owner = lb.OWNER_FILE
        lb.OWNER_FILE = self.owner_file

    def tearDown(self):
        lb.OWNER_FILE = self._old_owner

    def write(self, rows, name="labels.jsonl"):
        with open(self.dir / name, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def test_single_annotator_used_as_is_and_own_correction_wins(self):
        self.write([{"item": 0, "label": "relevant", "annotator": "Ann"},
                    {"item": 1, "label": "unsure", "annotator": "Ann"},
                    {"item": 0, "label": "irrelevant", "annotator": "Ann"}])
        f = lb.final(self.kind)
        self.assertEqual(f.labels, {0: "irrelevant", 1: "unsure"})
        self.assertEqual(lb.decided(self.kind, f), {0: "irrelevant"})
        self.assertEqual(f.source, {0: "single", 1: "single"})

    def test_legacy_rows_are_one_unattributed_annotator(self):
        self.write([{"item": 0, "label": "relevant"}, {"item": 0, "label": "justified"}])
        f = lb.final(self.kind)
        self.assertEqual((f.labels, f.annotators), ({0: "justified"}, [lb.UNATTRIBUTED]))

    def test_two_annotators_agree_shared_disagree_owner(self):
        self.owner_file.write_text("Ann\n")
        self.write([{"item": 0, "label": "relevant", "annotator": "Ann"},
                    {"item": 0, "label": "relevant", "annotator": "Jo"},
                    {"item": 1, "label": "relevant", "annotator": "Ann"},
                    {"item": 1, "label": "irrelevant", "annotator": "Jo"},
                    {"item": 2, "label": "justified", "annotator": "Jo"}])
        f = lb.final(self.kind)
        self.assertEqual(f.labels, {0: "relevant", 1: "relevant", 2: "justified"})
        self.assertEqual(f.source, {0: "agreed", 1: "owner", 2: "single"})
        # one annotator never overwrites another
        self.assertEqual(lb.by_annotator(self.kind)["Jo"][1], "irrelevant")

    def test_disagreement_without_owner_refuses_or_stays_unresolved(self):
        self.write([{"item": 0, "label": "relevant", "annotator": "Ann"},
                    {"item": 0, "label": "irrelevant", "annotator": "Jo"}])
        with self.assertRaises(SystemExit):
            lb.final(self.kind)
        f = lb.final(self.kind, strict=False)
        self.assertEqual((f.labels, f.unresolved), ({}, [0]))

    def test_resolution_overrides_everything(self):
        self.owner_file.write_text("Ann\n")
        self.write([{"item": 0, "label": "relevant", "annotator": "Ann"},
                    {"item": 0, "label": "irrelevant", "annotator": "Jo"},
                    {"item": 1, "label": "unsure", "annotator": "Ann"}])
        self.write([{"item": 0, "label": "justified"}, {"item": 1, "label": "irrelevant"}],
                   "resolutions.jsonl")
        f = lb.final(self.kind)
        self.assertEqual(f.labels, {0: "justified", 1: "irrelevant"})
        self.assertEqual(set(f.source.values()), {"resolved"})

    def test_agreement_percent_kappa_and_distinct_texts(self):
        self.owner_file.write_text("Ann\n")
        ann = ["relevant", "relevant", "irrelevant", "irrelevant", "justified"]
        jo = ["relevant", "irrelevant", "irrelevant", "irrelevant", "justified"]
        self.write([{"item": i, "label": l, "annotator": "Ann"} for i, l in enumerate(ann)]
                   + [{"item": i, "label": l, "annotator": "Jo"} for i, l in enumerate(jo)])
        s = lb.agreement(self.kind, "Ann", "Jo")
        self.assertEqual((s["n"], s["percent"]), (5, 80.0))
        self.assertAlmostEqual(s["kappa"], (0.8 - 0.36) / (1 - 0.36))
        # items 2 and 3 show the same text: counted once
        g = lb.agreement(self.kind, "Ann", "Jo", {0: "a", 1: "b", 2: "c", 3: "c", 4: "d"})
        self.assertEqual((g["n"], g["percent"]), (4, 75.0))


if __name__ == "__main__":
    unittest.main()
