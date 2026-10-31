# tests/test_manifest.py
#
# Offline tests for the second batch: the cell manifest, the build-period
# check and the split's store rule, plus the second batch's plan, guard
# and namespace rules. No proxy, no API, no Qdrant; the throwaway root
# from tests/__init__.py.
#
#   python3 -m unittest tests.test_manifest -v

import json
import shutil
import unittest

import tests  # noqa: F401  (sets MEMDEF_RESULTS before any project import)
from core import manifest as mf
from core.paths import RESULTS
import runner.v3 as v3
import scripts.stats as stats
from scripts.check_integrity import CellReport, manifest_violations, period_of

S = "anchor_gemini__gemini-backbone__"


# ---------------------------------------------------------------------
# synthetic cells for build_manifest
# ---------------------------------------------------------------------
def synthetic_plan():
    """A complete plan obeying the rule: conv0-2 stores re-ingested in
    October, conv3-4 the old August stores, every mini on its old none."""
    cells = []
    for conv in range(5):
        for seed in (1, 2, 3):
            for cond in v3.MAIN_CONDITIONS:
                old = v3.old_id(cond, conv, seed)
                oct_ = v3.PERIOD_OF_CONV[conv] == "Oct"
                cells.append(v3.V3Cell(
                    cell_id=old + "__v3", old_id=old, kind="v3", condition=cond, defenses=[],
                    seed=seed, conv_idx=conv, conv_id=f"conv-{conv}", answer_alias="gemini-backbone",
                    source_ns=old + "__v3" if oct_ else old, reingest=oct_))
            none = v3.old_id("none", conv, seed)
            cells.append(v3.V3Cell(
                cell_id=none + "__mini", old_id=none, kind="mini", condition="mini", defenses=[],
                seed=seed, conv_idx=conv, conv_id=f"conv-{conv}", answer_alias="answering",
                source_ns=none, reingest=False))
    return cells


class TestBuildManifest(unittest.TestCase):

    def test_synthetic_plan(self):
        m = v3.build_manifest(synthetic_plan())
        self.assertEqual(m["counts"], {"main": 90, "split": 45})
        main = [e for e in m["cells"] if e["role"] == "main"]
        self.assertEqual(len({(e["condition"], e["conv_idx"], e["seed"]) for e in main}), 90)
        for e in main:
            self.assertEqual(e["store_period"], v3.PERIOD_OF_CONV[e["conv_idx"]])
        split = [e for e in m["cells"] if e["role"] == "split"]
        self.assertEqual(sorted({e["condition"] for e in split}), ["mini", "none", "rerank"])
        for e in split:
            self.assertEqual(e["store_period"], "Aug")
            if e["condition"] != "mini":
                self.assertEqual((e["root"], e["batch"]), ("results", "v2"))
                self.assertEqual(e["store_namespace"], e["cell_id"])
            else:
                self.assertEqual(e["store_namespace"], v3.old_id("none", e["conv_idx"], e["seed"]))
        self.assertEqual(m["superseded"], [])

    def test_second_batch_cell_wins(self):
        cells = synthetic_plan()
        old = v3.old_id("sanitize", 3, 3)
        cells.append(v3.V3Cell(cell_id=old + "__v3b", old_id=old, kind="v3", condition="sanitize",
                               defenses=[], seed=3, conv_idx=3, conv_id="conv-3",
                               answer_alias="gemini-backbone", source_ns=old, reingest=False,
                               batch="v3b"))
        m = v3.build_manifest(cells)
        e = next(e for e in m["cells"] if e["role"] == "main" and e["condition"] == "sanitize"
                 and e["conv_idx"] == 3 and e["seed"] == 3)
        self.assertEqual(e["cell_id"], old + "__v3b")
        self.assertEqual(m["superseded"], [old + "__v3"])

    def test_store_from_the_wrong_period_is_refused(self):
        cells = synthetic_plan()
        c = next(c for c in cells if c.condition == "rerank" and c.conv_idx == 0 and c.seed == 2)
        c.source_ns, c.reingest = c.old_id, False          # an August store in an October conv
        with self.assertRaisesRegex(AssertionError, "must be Oct"):
            v3.build_manifest(cells)
        cells = synthetic_plan()
        c = next(c for c in cells if c.condition == "none" and c.conv_idx == 4 and c.seed == 1)
        c.source_ns, c.reingest = c.cell_id, True           # an October store in an August conv
        with self.assertRaisesRegex(AssertionError, "must be Aug"):
            v3.build_manifest(cells)

    def test_mini_must_read_the_old_none_store(self):
        cells = synthetic_plan()
        m = next(c for c in cells if c.kind == "mini" and c.conv_idx == 0 and c.seed == 1)
        m.source_ns = m.old_id + "__v3"
        with self.assertRaisesRegex(AssertionError, "old August none store"):
            v3.build_manifest(cells)


@unittest.skipUnless(v3.OLD_CELLS.exists(), "old results/cells not present")
class TestRealManifest(unittest.TestCase):
    """The manifest the real plan produces (reads results/ read-only)."""

    @classmethod
    def setUpClass(cls):
        cls.cells = v3.plan_cells()
        cls.m = v3.build_manifest(cls.cells)

    def test_second_batch(self):
        b = [c for c in self.cells if c.batch == "v3b"]
        self.assertEqual(len(b), 36)
        self.assertEqual(sum(c.reingest for c in b), 33)
        self.assertEqual(len(v3.REBUILD_V3B), 33)
        rebuilt = {}
        for c in b:
            if c.reingest:
                self.assertEqual(c.source_ns, c.cell_id)
                self.assertTrue(c.cell_id.endswith("__v3b"))
                self.assertEqual(c.answer_alias, "gemini-backbone")
                rebuilt.setdefault(c.condition, []).append((c.conv_idx, c.seed))
        self.assertEqual({k: len(v) for k, v in rebuilt.items()},
                         {"none": 7, "sanitize": 8, "provenance": 9, "rerank": 9})
        self.assertNotIn((0, 1), rebuilt["none"])
        self.assertNotIn((1, 1), rebuilt["none"])
        self.assertNotIn((2, 3), rebuilt["sanitize"])
        ans = [c for c in b if not c.reingest and c.kind == "v3"]
        self.assertEqual([(c.cell_id, c.source_ns) for c in ans],
                         [(f"{S}sanitize__conv3__s3__v3b", f"{S}sanitize__conv3__s3")])
        minis = sorted((c.cell_id, c.source_ns, c.answer_alias) for c in b if c.kind == "mini")
        self.assertEqual(minis, [(f"{S}none__conv0__mini__v3b", f"{S}none__conv0", "answering"),
                                 (f"{S}none__conv1__mini__v3b", f"{S}none__conv1", "answering")])

    def test_none_is_rebuilt_first(self):
        pos = {}
        for i, c in enumerate(self.cells):
            pos.setdefault((c.seed, c.conv_idx), []).append((c.batch, c.condition, i))
        for key, rows in pos.items():
            v3b = [r for r in rows if r[0] == "v3b" and r[1] != "mini"]
            if any(r[1] == "none" for r in v3b):
                self.assertEqual(min(v3b, key=lambda r: r[2])[1], "none", key)

    def test_manifest(self):
        m = self.m
        self.assertEqual(m["counts"], {"main": 90, "split": 45})
        self.assertEqual(m["periods"], {"conv-26": "Oct", "conv-30": "Oct", "conv-41": "Oct",
                                        "conv-42": "Aug", "conv-43": "Aug"})
        for e in m["cells"]:
            if e["role"] == "main":
                self.assertEqual(e["store_period"], m["periods"][e["conv_id"]], e["cell_id"])
                self.assertEqual(e["store_namespace"].endswith(("__v3", "__v3b", "__v3c", "__v3d")),
                                 e["store_period"] == "Oct", e["cell_id"])
            else:
                self.assertEqual(e["store_period"], "Aug", e["cell_id"])
                self.assertFalse(e["store_namespace"].endswith(("__v3", "__v3b", "__v3c", "__v3d")))
        self.assertEqual(len(m["split"]["rerank"]), 15)
        self.assertEqual(len(m["split"]["none"]), 15)
        self.assertEqual(len(m["split"]["mini"]), 15)
        self.assertEqual(len(m["superseded"]), 36 + 21 + 5)
        used = {e["cell_id"] for e in m["cells"]}
        self.assertFalse(used & set(m["superseded"]))
        self.assertIn(f"{S}sanitize__conv3__s3__v3", m["superseded"])
        self.assertIn(f"{S}none__conv0__mini", m["superseded"])
        self.assertIn(f"{S}sanitize__conv3__s3__v3b", used)


# ---------------------------------------------------------------------
# the build-period check and the split's store rule (pure function)
# ---------------------------------------------------------------------
def entry(role, cond, conv, seed, cell, store, builder, period, root="results_v3",
          broot="results_v3"):
    return {"role": role, "condition": cond, "conv_idx": int(conv[-1]), "conv_id": conv,
            "seed": seed, "cell_id": cell, "root": root, "store_namespace": store,
            "store_built_by": {"cell_id": builder, "root": broot}, "store_period": period}


def report(e):
    r = CellReport(e["cell_id"], e["condition"])
    r.info.update(entry=e, summary={"source_namespace": e["store_namespace"]},
                  write_blocked=set())
    return r


def builder(cell_id, month, model=v3.GEMINI):
    r = CellReport(cell_id, "none")
    r.final_extract = [{"ts": f"{month}-30T12:00:00+00:00", "model_resolved": model}] * 3
    r.info.update(ingest_ok=3, lost_turns=[], write_blocked=set())
    return r


def scenario():
    """conv-0 October (two re-ingested main cells), conv-3 August (one
    answer-only main cell), and a split triple on conv-3."""
    es = [entry("main", "none", "conv-0", 1, "n0__v3", "n0__v3", "n0__v3", "Oct"),
          entry("main", "rerank", "conv-0", 1, "r0__v3b", "r0__v3b", "r0__v3b", "Oct"),
          entry("main", "none", "conv-3", 1, "n3__v3", "n3", "n3", "Aug", broot="results"),
          entry("split", "none", "conv-3", 1, "n3", "n3", "n3", "Aug", "results", "results"),
          entry("split", "rerank", "conv-3", 1, "r3", "r3", "r3", "Aug", "results", "results"),
          entry("split", "mini", "conv-3", 1, "n3__mini", "n3", "n3", "Aug", broot="results")]
    months = {"n0__v3": "2026-10", "r0__v3b": "2026-10", "n3": "2026-08", "r3": "2026-08"}
    builders = {(e["store_built_by"]["root"], e["store_built_by"]["cell_id"]):
                builder(e["store_built_by"]["cell_id"], months[e["store_built_by"]["cell_id"]])
                for e in es}
    m = {"periods": {"conv-0": "Oct", "conv-3": "Aug"}}
    return m, es, builders


class TestBuildPeriodCheck(unittest.TestCase):

    def run_check(self, m, es, builders):
        return manifest_violations(m, [report(e) for e in es], builders)[0]

    def test_consistent_manifest_passes(self):
        self.assertEqual(self.run_check(*scenario()), [])

    def test_period_of(self):
        self.assertEqual(period_of([{"ts": "2026-08-22T00:00:00"}] * 2), "Aug")
        self.assertEqual(period_of([{"ts": "2026-10-30T00:00:00"}]), "Oct")
        self.assertEqual(period_of([{"ts": "2026-08-31T23:59:00"}, {"ts": "2026-10-01T00:00:00"}]),
                         "mixed")
        self.assertEqual(period_of([]), "unknown")

    def test_mixed_periods_in_one_conversation(self):
        m, es, builders = scenario()
        # conv-0's rerank store declared AND built in August: each entry is
        # self-consistent, but the conversation now mixes periods
        es[1]["store_period"] = "Aug"
        builders[("results_v3", "r0__v3b")] = builder("r0__v3b", "2026-08")
        v = self.run_check(m, es, builders)
        self.assertTrue(any(x.startswith("BUILD PERIOD: main cells of conv-0") for x in v), v)

    def test_declared_period_must_match_the_measured_one(self):
        m, es, builders = scenario()
        builders[("results_v3", "n0__v3")] = builder("n0__v3", "2026-08")   # really August
        v = self.run_check(m, es, builders)
        self.assertTrue(any("store built in Aug" in x and "manifest says Oct" in x for x in v), v)

    def test_conversation_period_must_match_the_declared_one(self):
        m, es, builders = scenario()
        m["periods"]["conv-0"] = "Aug"
        v = self.run_check(m, es, builders)
        self.assertTrue(any("manifest declares Aug" in x for x in v), v)

    def test_cell_must_answer_from_the_declared_store(self):
        m, es, builders = scenario()
        rs = [report(e) for e in es]
        rs[2].info["summary"]["source_namespace"] = "n3__v3"
        v = manifest_violations(m, rs, builders)[0]
        self.assertTrue(any("answered from n3__v3" in x for x in v), v)

    def test_store_extracted_off_backbone(self):
        m, es, builders = scenario()
        builders[("results", "n3")] = builder("n3", "2026-08", model="gpt-4o-mini")
        v = self.run_check(m, es, builders)
        self.assertTrue(any("extracted by ['gpt-4o-mini']" in x for x in v), v)


class TestSplitStoreRule(unittest.TestCase):

    def test_split_must_use_august_stores(self):
        m, es, builders = scenario()
        es[5].update(store_namespace="n3x__v3", store_period="Oct",
                     store_built_by={"cell_id": "n3x__v3", "root": "results_v3"})
        builders[("results_v3", "n3x__v3")] = builder("n3x__v3", "2026-10")
        v = manifest_violations(m, [report(e) for e in es], builders)[0]
        self.assertTrue(any(x.startswith("SPLIT: n3__mini answers from a store built in Oct")
                            for x in v), v)

    def test_mini_must_read_its_old_none_store(self):
        m, es, builders = scenario()
        # an August store, but not the old none cell's
        es[5].update(store_namespace="other", store_built_by={"cell_id": "other", "root": "results"})
        builders[("results", "other")] = builder("other", "2026-08")
        v = manifest_violations(m, [report(e) for e in es], builders)[0]
        self.assertTrue(any("not the store of its old none cell n3" in x for x in v), v)


# ---------------------------------------------------------------------
# core.manifest and the scripts that read it
# ---------------------------------------------------------------------
class TestManifestReaders(unittest.TestCase):

    def setUp(self):
        self.cells = RESULTS / "cells"
        rows = {"none": [True, True, False, True], "rerank": [False, True, False, False],
                "mini": [True, False, False, True]}
        es = [entry("main", "none", "conv-0", 1, "n0__v3", "n0__v3", "n0__v3", "Oct"),
              entry("main", "rerank", "conv-0", 1, "r0__v3b", "r0__v3b", "r0__v3b", "Oct"),
              entry("split", "mini", "conv-0", 1, "n0__mini", "n0", "n0", "Aug",
                    broot="results")]
        for e in es:
            d = self.cells / e["cell_id"]
            d.mkdir(parents=True, exist_ok=True)
            (d / "summary.json").write_text(json.dumps(
                {"group": "g", "backbone": "b", "condition": "WRONG", "conv_id": "conv-0"}))
            (d / "answers.jsonl").write_text("".join(
                json.dumps({"qa_id": f"q{i}", "category": "single_hop", "correct": ok}) + "\n"
                for i, ok in enumerate(rows[e["condition"]])))
        # an old v2 split cell lives under results/: listed, but not local
        es.append(entry("split", "none", "conv-0", 1, "n0", "n0", "n0", "Aug", "results", "results"))
        mf.MANIFEST.write_text(json.dumps({"cells": es}))
        self.es = es

    def tearDown(self):
        mf.MANIFEST.unlink(missing_ok=True)
        shutil.rmtree(self.cells, ignore_errors=True)

    def test_cell_dirs(self):
        self.assertEqual([d.name for d in mf.cell_dirs(("main",))], ["n0__v3", "r0__v3b"])
        self.assertEqual([d.name for d in mf.cell_dirs(local_only=True)],
                         ["n0__v3", "r0__v3b", "n0__mini"])
        all_dirs = mf.cell_dirs()
        self.assertEqual(all_dirs[-1], mf.REPO_ROOT / "results" / "cells" / "n0")
        for d in all_dirs[:3]:
            self.assertTrue(str(d).startswith(str(RESULTS)), d)    # never the real results_v3
        self.assertEqual(mf.baseline_dir("r0__v3b").name, "n0__v3")

    def test_without_a_manifest_a_new_root_refuses(self):
        mf.MANIFEST.unlink()
        with self.assertRaises(SystemExit):
            mf.cell_dirs()

    def test_stats_reads_the_manifest_roles(self):
        main = stats.load_cells(role="main")
        self.assertEqual(sorted(k[2] for k in main), ["none", "rerank"])   # manifest names
        split = stats.load_cells(role="split")
        self.assertEqual([k[2] for k in split], ["mini"])                  # local split cells only
        data = {(k[2], k[3]): v for k, v in {**main, **split}.items()}
        r = stats.compare(data, "none", "rerank", "core4", boots=200)
        self.assertAlmostEqual(r["point"], -0.5)                            # 1/4 - 3/4
        r = stats.compare(data, "none", "mini", "core4", boots=200)
        self.assertAlmostEqual(r["point"], -0.25)


# ---------------------------------------------------------------------
# second-batch guard and namespace rules
# ---------------------------------------------------------------------
class TestSecondBatchGuards(unittest.TestCase):

    def test_mini_v3b_answers_on_gpt4o_mini_only(self):
        self.assertEqual(v3.allowed_model({"step": "answer", "cell_id": f"{S}none__conv0__mini__v3b"}),
                         v3.MINI)
        self.assertEqual(v3.allowed_model({"step": "answer", "cell_id": f"{S}none__conv0__s2__v3b"}),
                         v3.GEMINI)
        self.assertEqual(v3.allowed_model({"step": "answer", "cell_id": f"{S}none__conv0__mini"}),
                         v3.MINI)

    def test_reingest_namespaces(self):
        v3.assert_reingest_namespace(f"{S}none__conv0__s2__v3b")
        v3.assert_reingest_namespace(f"{S}none__conv0__v3")
        for bad in (f"{S}none__conv0", f"{S}none__conv0__mini__v3b", f"{S}none__conv0__mini",
                    f"{S}sanitize__conv3__s3"):
            with self.assertRaises(PermissionError, msg=bad):
                v3.assert_reingest_namespace(bad)

    def test_budget_counts_only_this_run(self):
        p = RESULTS.parent / "guard_since.jsonl"
        r = lambda ts, usd: {"ts": ts, "cell_id": "x__v3b", "step": "extract",
                             "model_resolved": v3.GEMINI, "status": "ok", "cost_usd": usd}
        p.write_text(json.dumps(r("2026-10-29T10:00:00+00:00", 9.0)) + "\n")   # first batch
        g = v3.LiveGuard(p, limits={"google": 1.0, "anthropic": 1.0, "openai": 1.0},
                         since="2026-10-30T12:00:00+00:00")
        self.assertEqual(g.spend["google"], 0)
        with open(p, "a") as f:
            f.write(json.dumps(r("2026-10-30T13:00:00+00:00", 0.6)) + "\n")
        g.check()
        self.assertAlmostEqual(g.spend["google"], 0.6)
        with open(p, "a") as f:
            f.write(json.dumps(r("2026-10-30T13:05:00+00:00", 0.5)) + "\n")
        with self.assertRaises(v3.BudgetExceeded):
            g.check()
        # a resumed run starts from the same timestamp and sees the same spend
        with self.assertRaises(v3.BudgetExceeded):
            v3.LiveGuard(p, limits={"google": 1.0, "anthropic": 1.0, "openai": 1.0},
                         since="2026-10-30T12:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
