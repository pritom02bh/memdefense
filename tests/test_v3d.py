# tests/test_v3d.py
#
# Offline tests for the fourth batch (__v3d) and the changes that came
# with it: the pinned extraction-prompt date (MEMDEF_PROMPT_DATE), the
# October prompt-date integrity rule, no retries in the proxy or the
# SDK clients, and the stacked conv0 v3c known exception. No proxy, no
# API; a real Mem0 Memory with stub HTTP clients in the throwaway root.
#
#   python3 -m unittest tests.test_v3d -v

import json
import os
import pathlib
import types
import unittest
from datetime import datetime, timezone

import httpx
import yaml

import tests  # noqa: F401  (sets MEMDEF_RESULTS before any project import)
from core import proxy_client as pc
from core.paths import REPO_ROOT
import runner.v3 as v3
from scripts import check_integrity as ci
from scripts.check_integrity import CellReport, manifest_violations
from tests.test_manifest import builder, report, scenario

S = "anchor_gemini__gemini-backbone__"
FIVE = {f"{S}anomaly__conv1__s3", f"{S}stacked__conv1__s3", f"{S}sanitize__conv2__s3",
        f"{S}anomaly__conv2__s3", f"{S}stacked__conv2__s3"}


class pinned:
    """Set (or clear, with None) MEMDEF_PROMPT_DATE for a block."""

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        self.old = os.environ.get("MEMDEF_PROMPT_DATE")
        os.environ.pop("MEMDEF_PROMPT_DATE", None)
        if self.value is not None:
            os.environ["MEMDEF_PROMPT_DATE"] = self.value

    def __exit__(self, *exc):
        os.environ.pop("MEMDEF_PROMPT_DATE", None)
        if self.old is not None:
            os.environ["MEMDEF_PROMPT_DATE"] = self.old


class TestFourthBatchPlan(unittest.TestCase):

    def test_rebuild_list_and_budget(self):
        self.assertEqual(v3.REBUILD_V3D, FIVE)
        self.assertEqual(v3.OCT_PROMPT_DATE, "2026-10-30")
        self.assertEqual(v3.OCT_PROMPT_DATE, ci.OCT_PROMPT_DATE)
        self.assertEqual(v3.BUDGET_LIMITS, {"google": 5.0, "anthropic": 3.0, "openai": 3.0})
        self.assertEqual(v3.RUN_START_FILE.name, "run_v3d_started.txt")
        v3.assert_reingest_namespace(f"{S}anomaly__conv1__s3__v3d")

    def test_v3d_cell_needs_the_pinned_date(self):
        cell = types.SimpleNamespace(cell_id="x__v3d", batch="v3d", reingest=True)
        for value in (None, "2026-10-31"):
            with pinned(value), self.assertRaises(RuntimeError):
                v3.require_prompt_date(cell)
        with pinned("2026-10-30"):
            v3.require_prompt_date(cell)
        with pinned(None):    # other batches are not pinned
            v3.require_prompt_date(types.SimpleNamespace(cell_id="y__v3c", batch="v3c",
                                                         reingest=True))


@unittest.skipUnless(v3.OLD_CELLS.exists(), "old results/cells not present")
class TestFourthBatchReal(unittest.TestCase):

    def test_v3d_replaces_its_v3c_versions(self):
        cells = v3.plan_cells()
        c4 = [c for c in cells if c.batch == "v3d"]
        self.assertEqual({c.old_id for c in c4}, FIVE)
        for c in c4:
            self.assertTrue(c.reingest)
            self.assertEqual(c.source_ns, c.cell_id)
            self.assertEqual(c.store_period, "Oct")
        m = v3.build_manifest(cells, prompt_dates={f"{old}__v3d": ["2026-10-30"] for old in FIVE})
        self.assertEqual(m["counts"], {"main": 90, "split": 45})
        used = {e["cell_id"]: e for e in m["cells"] if e["role"] == "main"}
        for old in FIVE:
            self.assertIn(old + "__v3d", used)
            self.assertEqual(used[old + "__v3d"]["store_prompt_date"], ["2026-10-30"])
            self.assertIn(old + "__v3c", m["superseded"])
        self.assertNotIn("repro__", json.dumps(m))


class TestPromptDatePin(unittest.TestCase):
    """The prompt Mem0 actually sends, captured at the HTTP client."""

    def capture(self):
        from adapters.mem0_adapter import Mem0Adapter
        sent = []

        def create(**kw):
            sent.append(kw)
            msg = types.SimpleNamespace(content='{"memory": []}', tool_calls=None)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=msg, finish_reason="stop")], usage=None)

        a = Mem0Adapter(run_id="test", cell_id="test_prompt_date", backbone="gemini-backbone")
        try:
            # the clients mem0 built: no SDK retries either
            self.assertEqual(a._mem.llm.client.max_retries, 0)
            self.assertEqual(a._mem.embedding_model.client.max_retries, 0)
            self.assertEqual(a._answer_client.max_retries, 0)
            a._mem.llm.client.chat.completions.create = create
            a._mem.embedding_model.client.embeddings.create = lambda **kw: types.SimpleNamespace(
                data=[types.SimpleNamespace(embedding=[0.0] * 1536)
                      for _ in (kw["input"] if isinstance(kw["input"], list) else [kw["input"]])])
            a.reset("test_prompt_date")
            a._mem.add([{"role": "user", "content": "[Conversation date: 1:56 pm on 8 May, 2023] "
                                                    "Caroline: I went to a support group."}],
                       user_id="test_prompt_date")
        finally:
            a.close()
        self.assertEqual(len(sent), 1)
        return a, sent[0]["messages"][-1]["content"]

    def test_pinned_date_is_in_the_sent_prompt(self):
        with pinned("2026-10-30"):
            a, user = self.capture()
        self.assertEqual(a.prompt_date, "2026-10-30")
        self.assertIn("## Observation Date\n2026-10-30", user)
        self.assertIn("## Current Date\n2026-10-30", user)

    def test_unpinned_is_mem0s_utc_today(self):
        with pinned(None):
            a, user = self.capture()
        today = datetime.now(timezone.utc).date().isoformat()
        self.assertIsNone(a.prompt_date)
        self.assertIn(f"## Current Date\n{today}", user)

    def test_bad_date_fails_before_any_call(self):
        from adapters.mem0_adapter import Mem0Adapter, prompt_date
        for bad in ("2026-10-3", "30-10-2026", "2026-02-30"):
            with pinned(bad):
                with self.assertRaises(ValueError):
                    prompt_date()
                with self.assertRaises(ValueError):
                    Mem0Adapter(run_id="t", cell_id="t", memory=types.SimpleNamespace(),
                                answer_client=object())


class TestOctoberPromptDate(unittest.TestCase):

    def test_measured_from_call_start_or_pinned(self):
        rows = [{"ts": "2026-10-31T00:00:01+00:00", "latency_ms": 2500.0},
                {"ts": "2026-10-31T00:10:00+00:00", "latency_ms": 900.0}]
        self.assertEqual(ci.prompt_dates(rows, {}), ["2026-10-30", "2026-10-31"])
        self.assertEqual(ci.prompt_dates(rows, {"prompt_date_pinned": "2026-10-30"}), ["2026-10-30"])

    def test_october_store_after_midnight_is_a_violation(self):
        m, es, builders = scenario()
        b = builder("n0__v3", "2026-10")
        b.final_extract = [{"ts": "2026-10-31T00:20:00+00:00", "model_resolved": v3.GEMINI}] * 3
        builders[("results_v3", "n0__v3")] = b
        v, _, _, _, dates = manifest_violations(m, [report(e) for e in es], builders)
        self.assertTrue(any(x.startswith("OCTOBER PROMPT DATE: n0__v3") for x in v), v)
        self.assertEqual(dates["n0__v3"], ["2026-10-31"])

    def test_pinned_october_store_passes(self):
        m, es, builders = scenario()
        b = builder("n0__v3", "2026-10")
        b.final_extract = [{"ts": "2026-10-31T00:20:00+00:00", "model_resolved": v3.GEMINI}] * 3
        b.info["summary"] = {"prompt_date_pinned": "2026-10-30"}
        builders[("results_v3", "n0__v3")] = b
        v, _, _, _, dates = manifest_violations(m, [report(e) for e in es], builders)
        self.assertEqual(v, [])
        self.assertEqual(dates["n0__v3"], ["2026-10-30"])

    def test_august_dates_are_reported_not_checked(self):
        m, es, builders = scenario()
        b = builder("n3", "2026-08")
        b.final_extract = [{"ts": "2026-08-22T23:59:00+00:00", "model_resolved": v3.GEMINI},
                           {"ts": "2026-08-23T00:01:00+00:00", "model_resolved": v3.GEMINI}]
        builders[("results", "n3")] = b
        v, _, _, _, dates = manifest_violations(m, [report(e) for e in es], builders)
        self.assertEqual(v, [])
        self.assertEqual(dates["n3__v3"], ["2026-08-22", "2026-08-23"])

    def test_recorded_prompt_date_must_match(self):
        m, es, builders = scenario()
        es[0]["store_prompt_date"] = ["2026-10-29"]
        v = manifest_violations(m, [report(e) for e in es], builders)[0]
        self.assertTrue(any("manifest records prompt date" in x for x in v), v)


class TestNoRetriesOutsideCallWithRetry(unittest.TestCase):

    def test_proxy_config(self):
        cfg = yaml.safe_load((REPO_ROOT / "proxy" / "config.yaml").read_text())
        self.assertEqual(cfg["litellm_settings"]["num_retries"], 0)
        self.assertEqual(cfg["router_settings"]["num_retries"], 0)
        t = cfg["litellm_settings"]["request_timeout"]
        self.assertEqual(cfg["router_settings"]["timeout"], t)
        # above the slowest ok call logged so far (110 s), below the clients'
        self.assertTrue(110 < t < pc.CLIENT_TIMEOUT, t)

    def test_client_makes_one_attempt(self):
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(503, json={"error": {"message": "overloaded"}})

        import openai
        c = openai.OpenAI(base_url="http://proxy.invalid/v1", api_key="k",
                          http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        pc.no_sdk_retries(c)
        with self.assertRaises(openai.InternalServerError):
            c.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
        self.assertEqual(len(seen), 1)
        p = pc.proxy_client()
        self.assertEqual((p.max_retries, p.timeout), (0, pc.CLIENT_TIMEOUT))

    def test_project_clients(self):
        from core import judge
        from defenses.defenses import wrap
        self.assertEqual(judge._client.max_retries, 0)
        base = types.SimpleNamespace(backbone="gemini-backbone")
        for name in ("anomaly", "rerank"):
            self.assertEqual(wrap([name], base, f"test_{name}")._client.max_retries, 0)

    def test_no_other_client_construction(self):
        hits = []
        for d in ("adapters", "core", "defenses", "runner", "scripts"):
            for p in (REPO_ROOT / d).rglob("*.py"):
                if p.name != "proxy_client.py" and "OpenAI(" in p.read_text():
                    hits.append(str(p.relative_to(REPO_ROOT)))
        self.assertEqual(hits, [])


class TestOrphanedProxyRetry(unittest.TestCase):
    CID = f"{S}stacked__conv0__v3c"

    def rows(self, with_timeout=True, late_ok=1):
        rs = [{"ts": "2026-10-30T20:52:05+00:00", "_step": "answer", "status": "ok"}]
        if with_timeout:
            rs.append({"ts": "2026-10-30T22:20:07+00:00", "_step": "rerank", "status": "error",
                       "error": "litellm.Timeout: Connection timed out. Timeout passed=6000.0"})
        rs += [{"ts": f"2026-10-30T22:20:0{8 + i}+00:00", "_step": "rerank", "status": "ok"}
               for i in range(late_ok)]
        return rs

    def settled(self, rs):
        r = CellReport(self.CID, "stacked")
        r.violations.append("rerank ok verdicts 751 != 750 memories judged")
        return r.settle(rs)

    def test_accepted_with_the_proxy_timeout(self):
        self.assertIsNone(ci.orphaned_proxy_retry(self.rows()))
        r = self.settled(self.rows())
        self.assertEqual(r.violations, [])
        self.assertEqual(len(r.known), 1)

    def test_not_accepted_otherwise(self):
        for rs in (self.rows(with_timeout=False), self.rows(late_ok=2), self.rows(late_ok=0)):
            self.assertIsNotNone(ci.orphaned_proxy_retry(rs))
            r = self.settled(rs)
            self.assertEqual(len(r.violations), 1)
            self.assertIn("known-exception condition not met", r.violations[0])
            self.assertEqual(r.known, [])

    def test_other_cells_and_counts_are_not_covered(self):
        r = CellReport(f"{S}stacked__conv0__s2__v3c", "stacked")
        r.violations.append("rerank ok verdicts 751 != 750 memories judged")
        self.assertEqual(len(r.settle(self.rows()).violations), 1)
        r = CellReport(self.CID, "stacked")
        r.violations.append("rerank ok verdicts 752 != 750 memories judged")
        self.assertEqual(len(r.settle(self.rows()).violations), 1)


if __name__ == "__main__":
    unittest.main()
