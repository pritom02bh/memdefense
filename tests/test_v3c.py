# tests/test_v3c.py
#
# Offline tests for the third batch (__v3c): the plan, the manifest that
# replaces the night-of-Oct-29 stores, the October-cutoff integrity rule,
# the namespace guard, the budget, and the exact-text label carry-over.
# No proxy, no API, no Qdrant; the throwaway root from tests/__init__.py.
#
#   python3 -m unittest tests.test_v3c -v

import json
import shutil
import unittest

import tests  # noqa: F401  (sets MEMDEF_RESULTS before any project import)
from core import labels as lb
from core import manifest as mf
from core.paths import RESULTS
import runner.v3 as v3
from scripts import make_label_sheets as ms
from scripts.check_integrity import OCT_CUTOFF, manifest_violations
from tests.test_manifest import builder, entry, report, scenario

S = "anchor_gemini__gemini-backbone__"
EXPECTED_V3C = ({v3.old_id(c, i, s) for c in ("anomaly", "stacked") for i in (0, 1, 2)
                 for s in (1, 2, 3)}
                | {f"{S}none__conv0", f"{S}none__conv1", f"{S}sanitize__conv2__s3"})


class TestThirdBatchPlan(unittest.TestCase):

    def test_rebuild_list(self):
        self.assertEqual(v3.REBUILD_V3C, EXPECTED_V3C)
        self.assertEqual(len(v3.REBUILD_V3C), 21)
        self.assertEqual(v3.OCT_CUTOFF, OCT_CUTOFF)
        self.assertEqual(v3.KNOWN_BLOCKS, {0: "D1:12", 1: "D2:4", 2: "D4:11"})

    def test_namespace_guard(self):
        v3.assert_reingest_namespace(f"{S}anomaly__conv0__s2__v3c")
        for bad in (f"{S}none__conv0__mini__v3c", f"{S}anomaly__conv0__s2",
                    "repro__none__conv0__oct31"):
            with self.assertRaises(PermissionError, msg=bad):
                v3.assert_reingest_namespace(bad)


@unittest.skipUnless(v3.OLD_CELLS.exists(), "old results/cells not present")
class TestThirdBatchReal(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cells = v3.plan_cells()
        cls.m = v3.build_manifest(cls.cells)

    def test_v3c_cells(self):
        c3 = [c for c in self.cells if c.batch == "v3c"]
        self.assertEqual({c.old_id for c in c3}, EXPECTED_V3C)
        for c in c3:
            self.assertTrue(c.reingest)
            self.assertEqual(c.source_ns, c.cell_id)
            self.assertTrue(c.cell_id.endswith("__v3c"))
            self.assertEqual(c.answer_alias, "gemini-backbone")
            self.assertEqual(c.store_period, "Oct")
            self.assertEqual(c.kind, "v3")

    def test_none_first_within_replicate_and_conversation(self):
        seen = {}
        for c in self.cells:
            if c.batch == "v3c":
                seen.setdefault((c.seed, c.conv_idx), []).append(c.condition)
        for key, conds in seen.items():
            if "none" in conds:
                self.assertEqual(conds[0], "none", key)
            self.assertEqual(conds, sorted(conds, key=v3.CONDITION_ORDER.index), key)

    def test_manifest_uses_v3c_and_supersedes_the_night_batch(self):
        m = self.m
        self.assertEqual(m["counts"], {"main": 90, "split": 45})
        main = {(e["condition"], e["conv_idx"], e["seed"]): e for e in m["cells"]
                if e["role"] == "main"}
        for old in EXPECTED_V3C:
            cond, conv, seed = old.split("__")[2], int(old.split("__conv")[1][0]), v3._seed(old)
            e = main[(cond, conv, seed)]
            b = "v3d" if old in v3.REBUILD_V3D else "v3c"    # the fourth batch replaces five
            self.assertEqual(e["cell_id"], f"{old}__{b}")
            self.assertEqual(e["batch"], b)
            self.assertEqual(e["store_namespace"], f"{old}__{b}")
            self.assertIn(old + "__v3", m["superseded"])
        # every October main cell now comes from the Oct 30 or Oct 31 batches
        for e in main.values():
            if e["store_period"] == "Oct":
                self.assertIn(e["batch"], ("v3b", "v3c", "v3d"), e["cell_id"])
        self.assertNotIn("repro__none__conv0__oct31", json.dumps(m))

    def test_manifest_records_final_ingestion(self):
        some = next(e["cell_id"] for e in self.m["cells"] if e["role"] == "main")
        m = v3.build_manifest(self.cells, {some: {"start": "a", "end": "b"}})
        e = next(e for e in m["cells"] if e["cell_id"] == some)
        self.assertEqual(e["store_final_ingestion"], {"start": "a", "end": "b"})
        other = next(e for e in m["cells"] if e["role"] == "main" and e["cell_id"] != some)
        self.assertIsNone(other["store_final_ingestion"])
        self.assertTrue(all("store_final_ingestion" not in e for e in m["cells"]
                            if e["role"] == "split"))


class TestOctoberCutoff(unittest.TestCase):

    def test_night_batch_store_is_a_violation(self):
        m, es, builders = scenario()
        b = builder("n0__v3", "2026-10")
        b.final_extract = [{"ts": "2026-10-29T21:42:09+00:00", "model_resolved": v3.GEMINI},
                           {"ts": "2026-10-29T21:48:06+00:00", "model_resolved": v3.GEMINI}]
        builders[("results_v3", "n0__v3")] = b
        v = manifest_violations(m, [report(e) for e in es], builders)[0]
        self.assertTrue(any(x.startswith("OCTOBER BATCH: n0__v3") for x in v), v)

    def test_boundary_and_spans(self):
        m, es, builders = scenario()
        b = builder("n0__v3", "2026-10")
        b.final_extract = [{"ts": "2026-10-30T07:59:00+00:00", "model_resolved": v3.GEMINI},
                           {"ts": OCT_CUTOFF, "model_resolved": v3.GEMINI}]
        builders[("results_v3", "n0__v3")] = b
        v, _, _, spans, _ = manifest_violations(m, [report(e) for e in es], builders)
        self.assertEqual(v, [])      # ended exactly at the cutoff: allowed
        self.assertEqual(spans["n0__v3"], ("2026-10-30T07:59:00+00:00", OCT_CUTOFF))

    def test_recorded_ingestion_must_match_the_log(self):
        m, es, builders = scenario()
        es[0]["store_final_ingestion"] = {"start": "2026-10-30T12:00:00+00:00",
                                          "end": "2026-10-30T12:09:00+00:00"}
        v = manifest_violations(m, [report(e) for e in es], builders)[0]
        self.assertTrue(any("manifest records final ingestion" in x for x in v), v)

    def test_august_main_store_must_be_august(self):
        m, es, builders = scenario()
        builders[("results", "n3")] = builder("n3", "2026-10")
        v = manifest_violations(m, [report(e) for e in es], builders)[0]
        self.assertTrue(any(x.startswith("AUGUST STORE: n3__v3") for x in v), v)


class TestLabelCarry(unittest.TestCase):
    """make_label_sheets --v3c: exact-text carry-over, then a sheet for
    the regex failures still unlabeled."""

    def setUp(self):
        self.cells = RESULTS / "cells"
        def cell(name, rows):
            d = self.cells / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "answers.jsonl").write_text("".join(json.dumps(
                {"qa_id": q, "category": "adversarial", "question": f"Q {q}?", "predicted": t,
                 "correct": ok}) + "\n" for q, t, ok in rows))
        # an earlier (now superseded) cell whose answers were hand-labeled
        cell("old__v3", [("q1", "I don't know.", True), ("q2", "Tim surfs daily.", False),
                         ("q3", "Maybe red.", False)])
        cell("twin__v3", [("q3", "Maybe red.", False)])
        # the new cell
        cell("old__v3c", [("q1", "I don't know.", True), ("q2", "Tim surfs daily.", False),
                          ("q3", "Maybe red.", False), ("q4", "She went to Rome.", False),
                          ("q5", "No information.", True)])
        mf.MANIFEST.write_text(json.dumps({"cells": [
            dict(entry("main", "anomaly", "conv-0", 1, "old__v3c", "old__v3c", "old__v3c", "Oct"),
                 batch="v3c")]}))
        lb.ABSTENTION.dir.mkdir(parents=True, exist_ok=True)
        k = ms.item_key
        rows = [(k("old__v3", "q1"), "a"), (k("old__v3", "q2"), "b"),
                (k("old__v3", "q3"), "a"), (k("twin__v3", "q3"), "b")]   # q3: labels disagree
        lb.ABSTENTION.labels_file.write_text("".join(json.dumps(
            {"item_key": key, "label": lab, "annotator": "Hemel"}) + "\n" for key, lab in rows))

    def tearDown(self):
        mf.MANIFEST.unlink(missing_ok=True)
        shutil.rmtree(self.cells, ignore_errors=True)
        shutil.rmtree(lb.ABSTENTION.dir, ignore_errors=True)
        shutil.rmtree(ms.LABELING, ignore_errors=True)

    def test_carry_and_sheet(self):
        code = ms.carry_and_sheet({}, "v3c")
        self.assertEqual(code, 7)                      # q3 (conflict) and q4 (new) need labels
        fin = lb.decided(lb.ABSTENTION, lb.final(lb.ABSTENTION))
        k = ms.item_key
        self.assertEqual(fin[k("old__v3c", "q1")], "a")
        self.assertEqual(fin[k("old__v3c", "q2")], "b")
        self.assertNotIn(k("old__v3c", "q3"), fin)     # disagreeing labels: not carried
        self.assertNotIn(k("old__v3c", "q5"), fin)     # no match, regex abstention: not sheeted
        carried = [r for r in lb.jl(lb.ABSTENTION.labels_file) if r["annotator"] == lb.CARRIED]
        self.assertEqual({r["item_key"] for r in carried}, {k("old__v3c", "q1"), k("old__v3c", "q2")})
        self.assertEqual(carried[0]["source_item_keys"], [k("old__v3", "q1")])
        key = json.loads((ms.KEYS / "adversarial_labels_v3c.key.json").read_text())
        self.assertEqual(sorted(i["qa_id"] for i in key["items"]), ["q3", "q4"])
        text = (ms.LABELING / "adversarial_labels_v3c.txt").read_text()
        self.assertNotIn("old__v3c", text)
        self.assertIn("2: _", text)
        # running again carries nothing twice and gives the same sheet
        self.assertEqual(ms.carry_and_sheet({}, "v3c"), 7)
        self.assertEqual(len([r for r in lb.jl(lb.ABSTENTION.labels_file)
                              if r["annotator"] == lb.CARRIED]), 2)

    def test_everything_carried_needs_no_sheet(self):
        lb.ABSTENTION.labels_file.write_text("".join(json.dumps(
            {"item_key": ms.item_key(c, q), "label": lab, "annotator": "Hemel"}) + "\n"
            for c, q, lab in [("old__v3", "q2", "b"), ("old__v3", "q3", "a"), ("twin__v3", "q3", "a"),
                              ("old__v3c", "q4", "b")]))
        self.assertEqual(ms.carry_and_sheet({}, "v3c"), 0)
        self.assertFalse((ms.LABELING / "adversarial_labels_v3c.txt").exists())


if __name__ == "__main__":
    unittest.main()
