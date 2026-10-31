# runner/v3.py
#
# Regeneration of the Gemini cells after the defense-model fix
# (results_v3/). Run with MEMDEF_RESULTS=results_v3 and the proxy
# started with the same setting.
#
#   python3 -m runner.v3 --protocol     write results_v3/PROTOCOL.md
#   python3 -m runner.v3 --manifest     write results_v3/final_cells.json (the cells the paper uses)
#   python3 -m runner.v3 --dry-run      cells, expected calls, cost estimate (no calls)
#   python3 -m runner.v3 --smoke        one real call of each type (needs the proxy)
#   python3 -m runner.v3 --run          the full regeneration
#
# EXIT CODES of --run: 0 done, 2 model violation (live guard), 3 budget
# exceeded, 4 retries exhausted on a model call (safe to resume), 5 the
# write-time chain blocked turns in phase A (needs a decision). On 2, 3
# and 5 the reason is written to results_v3/STOPPED.txt.
#
# SECOND BATCH (__v3b, 2026-10-30). Gemini's extraction changed between
# the August run and October (same code, same prompts: fewer memories;
# results_diag/ Part 2). So that every condition of a conversation
# answers from memory built in ONE period, conv0-2 are rebuilt in full
# in October and conv3-4 stay all August:
#   - REBUILD_V3B: 33 cells re-ingested into <old_id>__v3b;
#   - ANSWER_ONLY_V3B: sanitize conv3 s3, answered from its old August
#     store (its October __v3 re-ingest is superseded);
#   - MINI_V3B: baseline-mini conv0/conv1 s1 re-answered from the OLD
#     August none stores (<old_none_id>__mini__v3b), so every mini cell
#     of the model-swap split reads August memory, like v2.
# results_v3/final_cells.json (--manifest) names the one cell per
# condition, conversation and replicate that the paper uses; superseded
# cells stay on disk untouched.
#
# CELLS. <old_id>__v3 for each of the 90 old Gemini cells, and
# <old_none_id>__mini for the 15 baseline-mini cells (no defense, answer
# on "answering" = gpt-4o-mini). Most cells are ANSWER-ONLY: they answer
# from the store the old run built, opened read-only in the
# results_v3/mem0_qdrant copy. REINGEST cells are rebuilt into their own
# __v3 namespace with the fixed pipeline and their full write-time chain.
#
# PHASES of --run:
#   A. write-time verdicts: every anomaly / stacked cell that is not
#      re-ingested passes the exact turn strings of the ingest path
#      through its write-time chain on the backbone, storing nothing.
#      Any blocked turn stops the run: that cell's old store would not
#      match its new chain, so it needs a full re-ingest.
#   B+C. in order of replicate, conversation, condition (all seven
#      back to back): re-ingest where required, then answer, grade and
#      summarise.
#
# Namespace rules (hard rules of the regeneration):
#   - Only a re-ingested cell's own __v3 / __v3b namespace, inside
#     results_v3/mem0_qdrant, may ever be written or wiped.
#   - A re-ingested cell is all-or-nothing. If it is interrupted at any
#     point before its summary.json exists, the restart wipes its
#     namespace and message history AND deletes its partial
#     answers.jsonl, so a store and answers from different attempts are
#     never mixed.
#
# LIVE GUARD: after every question (and after every screening or
# ingestion), every new line of the proxy log -- whatever its cell --
# is checked against ALLOWED_MODELS; the first mismatch aborts the run.
# The budget counts only this run's calls: those logged since the
# timestamp in RUN_START_FILE, written when the run first starts (a
# resumed run keeps it).

from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import functools
import json
import pathlib
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace

import yaml

from core.paths import CELLS, PROXY_LOG, QDRANT, QUARANTINE, REPO_ROOT, RESULTS

V3_ROOT_NAME = "results_v3"
V3_SUFFIX = "__v3"
V3B_SUFFIX = "__v3b"
V3C_SUFFIX = "__v3c"
V3D_SUFFIX = "__v3d"
REINGEST_SUFFIXES = (V3_SUFFIX, V3B_SUFFIX, V3C_SUFFIX, V3D_SUFFIX)
MINI_SUFFIX = "__mini"
MINI_RX = re.compile(r"__mini(__v3b)?$")
OLD_RESULTS = REPO_ROOT / "results"
OLD_CELLS = OLD_RESULTS / "cells"
OLD_PROXY_LOG = OLD_RESULTS / "proxy" / "calls.jsonl"
GROUP = "anchor_gemini"
BACKBONE_ALIAS = "gemini-backbone"
MINI_ANSWER_ALIAS = "answering"
MAX_TURNS, MAX_QUESTIONS, TOP_K = 150, 75, 10
CONDITION_ORDER = ["none", "sanitize", "provenance", "anomaly", "rerank", "stacked", "mini"]

# Stage 2 decisions (2026-10-30).
REINGEST_STAGE2 = {
    "anchor_gemini__gemini-backbone__anomaly__conv1__s2",   # turns 100, 101 never extracted
    "anchor_gemini__gemini-backbone__none__conv0",          # stale message history
    "anchor_gemini__gemini-backbone__none__conv1",
    "anchor_gemini__gemini-backbone__sanitize__conv3__s3",
    "anchor_gemini__gemini-backbone__sanitize__conv2__s3",
    "anchor_gemini__gemini-backbone__stacked__conv0",
}
# Phase A (2026-10-30): Gemini's write-time chain blocked a turn that the
# old store contains -- conv0 D1:12, conv1 D2:4, conv2 D4:11, in every
# replicate and in both conditions -- so these old stores do not match
# their new chain. Their Phase A verdicts are superseded by the
# re-ingest's own (integrity check and tau count only the re-ingest's).
REINGEST_PHASE_A = {
    f"anchor_gemini__gemini-backbone__{cond}__conv{c}{s}"
    for c in (0, 1, 2) for cond in ("anomaly", "stacked") for s in ("", "__s2", "__s3")
} - REINGEST_STAGE2
REINGEST = REINGEST_STAGE2 | REINGEST_PHASE_A


def old_id(condition: str, conv_idx: int, seed: int) -> str:
    return f"{GROUP}__{BACKBONE_ALIAS}__{condition}__conv{conv_idx}" + (
        "" if seed == 1 else f"__s{seed}")


# Second batch (2026-10-30), see the header. Build period per
# conversation: every condition of a conversation answers from memory
# extracted in that period.
PERIOD_OF_CONV = {0: "Oct", 1: "Oct", 2: "Oct", 3: "Aug", 4: "Aug"}
_OCT = [c for c, p in PERIOD_OF_CONV.items() if p == "Oct"]
REBUILD_V3B = ({old_id(cond, c, s) for cond in ("none", "sanitize", "provenance", "rerank")
                for c in _OCT for s in (1, 2, 3)}
               - REINGEST)                       # none conv0/1 s1, sanitize conv2 s3: done in __v3
ANSWER_ONLY_V3B = {old_id("sanitize", 3, 3)}     # back to its August store
MINI_V3B = {old_id("none", 0, 1), old_id("none", 1, 1)}   # mini from the August none stores

# Third batch (2026-10-31). The first batch ran on the night of Oct 29
# and its Gemini extracted differently from turn 1 on; the second batch
# (Oct 30) is reproduced today on 149 of 150 turns (results_diag/
# repro_oct31). Every October main store whose final ingestion ended
# before OCT_CUTOFF is rebuilt into <old_id>__v3c, so all October main
# cells share one batch's Gemini.
OCT_CUTOFF = "2026-10-30T08:00:00+00:00"
REBUILD_V3C = ({old_id(cond, c, s) for cond in ("anomaly", "stacked") for c in _OCT
                for s in (1, 2, 3)}
               | {old_id("none", 0, 1), old_id("none", 1, 1), old_id("sanitize", 2, 3)})
# turns the write-time chain blocked in every earlier October build
KNOWN_BLOCKS = {0: "D1:12", 1: "D2:4", 2: "D4:11"}

# Fourth batch (2026-10-31). Mem0 puts today's UTC date into every
# extraction prompt (adapters/mem0_adapter.py, PROMPT DATE), and the
# divergences between builds line up with it (results_diag/prompt_date):
# the night batch's stores extracted on 2026-10-29 differ from turn 1,
# and so do the five __v3c stores built after midnight UTC (2026-10-31).
# Those five are rebuilt into <old_id>__v3d with the prompt date pinned
# to OCT_PROMPT_DATE, the date of every other October main store.
OCT_PROMPT_DATE = "2026-10-30"
REBUILD_V3D = {old_id("anomaly", 1, 3), old_id("stacked", 1, 3), old_id("sanitize", 2, 3),
               old_id("anomaly", 2, 3), old_id("stacked", 2, 3)}
MAIN_CONDITIONS = ["none", "sanitize", "provenance", "anomaly", "rerank", "stacked"]

GEMINI = "gemini-3.1-flash-lite"
MINI = "gpt-4o-mini"
HAIKU = "claude-haiku-4-5-20251001"
EMBED = "text-embedding-3-small"
ALLOWED_MODELS = {"extract": GEMINI, "anomaly": GEMINI, "rerank": GEMINI,
                  "judge": HAIKU, "embedding": EMBED}   # answer: see allowed_model()

BALANCES = {"google": None, "openai": None, "anthropic": None}   # not known for this run
PROVIDER = {GEMINI: "google", MINI: "openai", EMBED: "openai", HAIKU: "anthropic"}
# Abort when THIS run's logged spend (calls since RUN_START_FILE) exceeds
# these. The first regeneration (Oct 29) ran under 13 / 7 / 10 over the
# whole log, the second (Oct 30) under 13.50 / 5 / 5 since run_v3b_started.txt,
# the third (Oct 31) under 12 / 5 / 5 since run_v3c_started.txt; see PROTOCOL.md.
BUDGET_LIMITS = {"google": 5.00, "anthropic": 3.00, "openai": 3.00}
RUN_START_FILE = RESULTS / "run_v3d_started.txt"
EXIT_OK, EXIT_MODEL, EXIT_BUDGET, EXIT_RETRIES, EXIT_BLOCKED = 0, 2, 3, 4, 5


def provider_of(model: str | None) -> str | None:
    m = model or ""
    if m.startswith("gemini"):
        return "google"
    if m.startswith(("gpt-", "text-embedding")):
        return "openai"
    if m.startswith("claude"):
        return "anthropic"
    return None


# =====================================================================
# namespace guards
# =====================================================================
def assert_reingest_namespace(namespace: str) -> None:
    """Refuse to write anywhere except a re-ingest namespace in the
    results_v3 Qdrant copy. A baseline-mini id never owns a store."""
    if not namespace.endswith(REINGEST_SUFFIXES) or MINI_RX.search(namespace) \
            or MINI_SUFFIX + "__" in namespace:
        raise PermissionError(f"refusing to write namespace {namespace!r}: "
                              f"only {' / '.join(REINGEST_SUFFIXES)} namespaces may be re-ingested")
    if RESULTS.name != V3_ROOT_NAME or QDRANT != RESULTS / "mem0_qdrant":
        raise PermissionError(f"refusing to write namespace {namespace!r} in {QDRANT}: "
                              f"re-ingestion is allowed only under {V3_ROOT_NAME}/mem0_qdrant "
                              f"(set MEMDEF_RESULTS={V3_ROOT_NAME})")


def prepare_reingest(adapter, cell_dir: pathlib.Path) -> None:
    """Fresh start for a re-ingested cell. Call before every attempt
    that has no summary.json: deletes any partial answers.jsonl from an
    earlier attempt, then wipes the cell's namespace (reset() verifies
    the namespace is empty afterwards) and its mem0 message history."""
    namespace = adapter.cell_id
    assert_reingest_namespace(namespace)
    if (cell_dir / "summary.json").exists():
        raise RuntimeError(f"{namespace} is complete; refusing to wipe it")
    partial = cell_dir / "answers.jsonl"
    if partial.exists():
        n = sum(1 for line in partial.read_text().splitlines() if line.strip())
        print(f"  restart: deleting {n} answers from an interrupted attempt", flush=True)
        partial.unlink()
    adapter.reset(namespace)
    adapter.clear_message_history(namespace)


# =====================================================================
# live guard
# =====================================================================
class GuardViolation(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


def allowed_model(row: dict) -> str | None:
    step = row.get("step")
    if step == "answer":
        return MINI if MINI_RX.search(row.get("cell_id") or "") else GEMINI
    return ALLOWED_MODELS.get(step)


class LiveGuard:
    """Checks every line appended to the proxy log since the last check,
    whatever its cell_id: its model against ALLOWED_MODELS, and the
    logged spend per provider against BUDGET_LIMITS. Model checks start
    at the log's current end; the spend counts every line logged at or
    after `since` (an ISO timestamp; "" = the whole log)."""

    def __init__(self, path: pathlib.Path = PROXY_LOG, limits: dict | None = None,
                 since: str = ""):
        self.path = path
        self.limits = BUDGET_LIMITS if limits is None else limits
        self.since = since
        self.spend = Counter()
        self.offset = 0
        self.checked = 0
        if path.exists():
            data = path.read_bytes()
            end = data.rfind(b"\n") + 1
            for line in data[:end].decode().splitlines():
                if line.strip():
                    self._add_spend(json.loads(line))
            self.offset = end
        self._check_budget()

    def _add_spend(self, row: dict) -> None:
        if (row.get("ts") or "") < self.since:
            return
        prov = provider_of(row.get("model_resolved"))
        if prov:
            self.spend[prov] += row.get("cost_usd") or 0.0

    def _check_budget(self) -> None:
        over = {p: round(self.spend[p], 4) for p, lim in self.limits.items()
                if self.spend[p] > lim}
        if over:
            raise BudgetExceeded(f"BUDGET: logged spend {over} exceeds limits "
                                 f"{ {p: self.limits[p] for p in over} }")

    def check(self) -> int:
        if not self.path.exists():
            return 0
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            data = f.read()
        end = data.rfind(b"\n")
        if end < 0:
            return 0
        lines = data[: end + 1].decode().splitlines()
        for line in lines:
            if not line.strip():
                continue
            row = json.loads(line)
            want = allowed_model(row)
            if row.get("status") == "logger_error" or want is None \
                    or row.get("model_resolved") != want:
                raise GuardViolation(
                    f"LIVE GUARD: disallowed call (expected {want}):\n  {line}")
            self._add_spend(row)
        self.offset += end + 1
        self.checked += len(lines)
        self._check_budget()
        return len(lines)


# =====================================================================
# the cell plan
# =====================================================================
@dataclass
class V3Cell:
    cell_id: str
    old_id: str              # the old cell this one regenerates (for mini: the old none cell)
    kind: str                # "v3" | "mini"
    condition: str           # none ... stacked, or "mini"
    defenses: list
    seed: int
    conv_idx: int
    conv_id: str
    answer_alias: str
    source_ns: str           # namespace answered from
    reingest: bool
    batch: str = "v3"        # "v3" (Oct 29 night) | "v3b" (Oct 30) | "v3c", "v3d" (Oct 31)

    @property
    def store_period(self) -> str:
        """When the store this cell answers from was extracted: an
        old namespace in August, a __v3 / __v3b / __v3c / __v3d re-ingest in October."""
        return "Oct" if self.source_ns.endswith(REINGEST_SUFFIXES) else "Aug"

    @property
    def screen(self) -> bool:
        return "anomaly" in self.defenses and not self.reingest

    @property
    def dir(self) -> pathlib.Path:
        return CELLS / self.cell_id


def _seed(old_id: str) -> int:
    m = re.search(r"__s(\d+)$", old_id)
    return int(m.group(1)) if m else 1


def plan_cells() -> list[V3Cell]:
    cells = []
    for d in sorted(OLD_CELLS.glob(f"{GROUP}__*")):
        s = json.loads((d / "summary.json").read_text())
        old = s["cell_id"]
        assert s["backbone"] == BACKBONE_ALIAS, old
        conv_idx = int(re.search(r"__conv(\d+)", old).group(1))
        common = dict(old_id=old, seed=_seed(old), conv_idx=conv_idx, conv_id=s["conv_id"])
        v3_id = old + V3_SUFFIX
        re_ing = old in REINGEST
        cells.append(V3Cell(cell_id=v3_id, kind="v3", condition=s["condition"],
                            defenses=list(s["defenses"]), answer_alias=BACKBONE_ALIAS,
                            source_ns=v3_id if re_ing else old, reingest=re_ing, **common))
        if s["condition"] == "none":
            # baseline-mini answers from the same store as its none__v3 cell
            cells.append(V3Cell(cell_id=old + MINI_SUFFIX, kind="mini", condition="mini",
                                defenses=[], answer_alias=MINI_ANSWER_ALIAS,
                                source_ns=v3_id if re_ing else old, reingest=False, **common))
        # second batch
        v3b_id = old + V3B_SUFFIX
        if old in REBUILD_V3B or old in ANSWER_ONLY_V3B:
            rebuild = old in REBUILD_V3B
            cells.append(V3Cell(cell_id=v3b_id, kind="v3", condition=s["condition"],
                                defenses=list(s["defenses"]), answer_alias=BACKBONE_ALIAS,
                                source_ns=v3b_id if rebuild else old, reingest=rebuild,
                                batch="v3b", **common))
        if old in REBUILD_V3C:
            v3c_id = old + V3C_SUFFIX
            cells.append(V3Cell(cell_id=v3c_id, kind="v3", condition=s["condition"],
                                defenses=list(s["defenses"]), answer_alias=BACKBONE_ALIAS,
                                source_ns=v3c_id, reingest=True, batch="v3c", **common))
        if old in REBUILD_V3D:
            v3d_id = old + V3D_SUFFIX
            cells.append(V3Cell(cell_id=v3d_id, kind="v3", condition=s["condition"],
                                defenses=list(s["defenses"]), answer_alias=BACKBONE_ALIAS,
                                source_ns=v3d_id, reingest=True, batch="v3d", **common))
        if old in MINI_V3B:
            cells.append(V3Cell(cell_id=old + MINI_SUFFIX + V3B_SUFFIX, kind="mini",
                                condition="mini", defenses=[], answer_alias=MINI_ANSWER_ALIAS,
                                source_ns=old, reingest=False, batch="v3b", **common))
    # within a replicate and conversation: none first, conditions in
    # order, the batches in order
    cells.sort(key=lambda c: (c.seed, c.conv_idx, CONDITION_ORDER.index(c.condition), c.batch))
    ids = {c.old_id for c in cells}
    missing = (REINGEST | REBUILD_V3B | ANSWER_ONLY_V3B | MINI_V3B | REBUILD_V3C | REBUILD_V3D) - ids
    assert not missing, f"cells not found: {missing}"
    return cells


# =====================================================================
# the manifest: the one cell per condition, conversation and replicate
# =====================================================================
def _store_builder(c: V3Cell) -> dict:
    """The cell whose ingestion built the store `c` answers from."""
    if c.source_ns.endswith(REINGEST_SUFFIXES):
        return {"cell_id": c.source_ns, "root": V3_ROOT_NAME}
    return {"cell_id": c.source_ns, "root": OLD_RESULTS.name}


def _entry(role, condition, c: V3Cell | None = None, *, old: str | None = None,
           seed=None, conv_idx=None, conv_id=None) -> dict:
    if c is not None:
        return {"role": role, "condition": condition, "conv_idx": c.conv_idx,
                "conv_id": c.conv_id, "seed": c.seed, "cell_id": c.cell_id,
                "root": V3_ROOT_NAME, "batch": c.batch, "answer_alias": c.answer_alias,
                "store_namespace": c.source_ns, "store_qdrant": f"{V3_ROOT_NAME}/mem0_qdrant",
                "store_built_by": _store_builder(c), "store_period": c.store_period}
    # an old v2 cell, used as it is (results/, read-only)
    return {"role": role, "condition": condition, "conv_idx": conv_idx, "conv_id": conv_id,
            "seed": seed, "cell_id": old, "root": OLD_RESULTS.name, "batch": "v2",
            "answer_alias": "answering" if condition == "rerank" else BACKBONE_ALIAS,
            "store_namespace": old, "store_qdrant": f"{OLD_RESULTS.name}/mem0_qdrant",
            "store_built_by": {"cell_id": old, "root": OLD_RESULTS.name},
            "store_period": "Aug"}


def build_manifest(cells: list[V3Cell], ingestion: dict | None = None,
                   prompt_dates: dict | None = None) -> dict:
    """Main set: per condition, conversation and replicate, the latest
    batch's cell (v3d, else v3c, else v3b, else v3); its store must have
    been built in its conversation's period. `ingestion` maps a main cell
    to its store's final ingestion {start, end} and `prompt_dates` to the
    extraction-prompt date(s) its store was built with (None while not
    built), both recorded in the entry. Split set: the
    15 old (v2) none and 15 old rerank cells from results/, and the 15
    baseline-mini cells, every one answering from an August store."""
    by = {(c.condition, c.conv_idx, c.seed, c.batch): c for c in cells}
    conv_ids = {c.conv_idx: c.conv_id for c in cells}
    main, split = [], []
    for cond in MAIN_CONDITIONS:
        for conv in sorted(PERIOD_OF_CONV):
            for seed in (1, 2, 3):
                c = next(by[k] for k in ((cond, conv, seed, b) for b in ("v3d", "v3c", "v3b", "v3"))
                         if k in by)
                if c.store_period != PERIOD_OF_CONV[conv]:
                    raise AssertionError(f"{c.cell_id}: store built in {c.store_period}, "
                                         f"conversation {conv} must be {PERIOD_OF_CONV[conv]}")
                e = _entry("main", cond, c)
                e["store_final_ingestion"] = (ingestion or {}).get(c.cell_id)
                e["store_prompt_date"] = (prompt_dates or {}).get(c.cell_id)
                main.append(e)
    for conv in sorted(PERIOD_OF_CONV):
        for seed in (1, 2, 3):
            meta = dict(seed=seed, conv_idx=conv, conv_id=conv_ids[conv])
            split.append(_entry("split", "none", old=old_id("none", conv, seed), **meta))
            split.append(_entry("split", "rerank", old=old_id("rerank", conv, seed), **meta))
            m = by.get(("mini", conv, seed, "v3b")) or by[("mini", conv, seed, "v3")]
            if m.source_ns != old_id("none", conv, seed):
                raise AssertionError(f"{m.cell_id}: mini must answer from the old August none "
                                     f"store {old_id('none', conv, seed)}, not {m.source_ns}")
            split.append(_entry("split", "mini", m))
    used = {e["cell_id"] for e in main + split}
    commit, dirty = git_commit()
    return {
        "description": "The cells the paper uses. main: one cell per condition, conversation and "
                       "replicate; within a conversation every main cell answers from memory "
                       "extracted in the same period. split: the model-swap split (old rerank, "
                       "old none, baseline-mini), all on August memory. Superseded cells stay on "
                       "disk and are excluded here.",
        "created": _now(), "git_commit": commit + (" (dirty)" if dirty else ""),
        "periods": {conv_ids[c]: p for c, p in sorted(PERIOD_OF_CONV.items())},
        "counts": {"main": len(main), "split": len(split)},
        "cells": main + split,
        "split": {k: [e["cell_id"] for e in split if e["condition"] == k]
                  for k in ("rerank", "none", "mini")},
        "superseded": sorted(c.cell_id for c in cells if c.cell_id not in used),
    }


def write_manifest(cells: list[V3Cell]) -> pathlib.Path:
    """Build the manifest, then measure each main cell's store's final
    ingestion and prompt date(s) from the logs and summaries (read-only)
    and record them."""
    import contextlib
    import io
    from core.manifest import MANIFEST
    from scripts.check_integrity import run_manifest_checks
    m = build_manifest(cells)
    with contextlib.redirect_stdout(io.StringIO()):
        mr = run_manifest_checks(m)
    ingestion, dates = {}, {}
    for e in m["cells"]:
        if e["role"] == "main":
            span = mr.store_ingestion.get(e["cell_id"])
            ingestion[e["cell_id"]] = {"start": span[0], "end": span[1]} if span else None
            dates[e["cell_id"]] = mr.store_prompt_dates.get(e["cell_id"])
    MANIFEST.write_text(json.dumps(build_manifest(cells, ingestion, dates), indent=2))
    return MANIFEST


def cell_questions(cell: V3Cell, convs):
    """Same selection as the old runner (evidence filter over the first
    150 turns, cap 75), asserted equal -- ids and order -- to the old
    cell's answers.jsonl."""
    from core.evidence_filter import filter_questions
    qs = filter_questions(convs[cell.conv_idx], MAX_TURNS)[:MAX_QUESTIONS]
    old_rows = [json.loads(l) for l in (OLD_CELLS / cell.old_id / "answers.jsonl")
                .read_text().splitlines() if l.strip()]
    old_ids = [r["qa_id"] for r in old_rows]
    new_ids = [q.qa_id for q in qs]
    if new_ids != old_ids:
        raise AssertionError(f"{cell.cell_id}: question ids differ from {cell.old_id} "
                             f"({len(new_ids)} vs {len(old_ids)})")
    return qs


def git_commit() -> tuple[str, bool]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=REPO_ROOT, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip())
    return head, dirty


def alias_models() -> dict[str, str]:
    cfg = yaml.safe_load((REPO_ROOT / "proxy" / "config.yaml").read_text())
    return {m["model_name"]: m["litellm_params"]["model"] for m in cfg["model_list"]}


def _require_v3_root():
    if RESULTS.name != V3_ROOT_NAME:
        sys.exit(f"MEMDEF_RESULTS must be {V3_ROOT_NAME} (got {RESULTS})")


def _turn_conv(convs, cell):
    conv = convs[cell.conv_idx]
    return dataclasses.replace(conv, turns=conv.turns[:MAX_TURNS])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# =====================================================================
# phase A: write-time verdicts only
# =====================================================================
def screen_cell(cell: V3Cell, convs, guard: LiveGuard) -> dict:
    from adapters.mem0_adapter import Mem0Adapter
    from core.purpose import cell as cell_context
    from defenses.defenses import wrap
    from runner.run import collect_retries

    out = cell.dir / "write_verdicts.json"
    if out.exists():
        return json.loads(out.read_text())
    if (cell.dir / "answers.jsonl").exists():
        raise RuntimeError(f"{cell.cell_id}: answers exist but write verdicts do not")
    cell.dir.mkdir(parents=True, exist_ok=True)
    adapter = Mem0Adapter(run_id="v3", cell_id=cell.cell_id, backbone=BACKBONE_ALIAS)
    try:
        adapter.open_namespace(cell.source_ns, read_only=True)
        target = wrap(cell.defenses, adapter, cell.cell_id)
        conv = _turn_conv(convs, cell)
        started = _now()
        with cell_context(cell.cell_id):
            blocked = target.screen(conv)
        guard.check()
        result = {"started": started, "finished": _now(), "turns_screened": len(conv.turns),
                  "blocked": blocked, "retries": collect_retries(target)}
    finally:
        adapter.close()
    out.write_text(json.dumps(result, indent=2))
    print(f"  screened {cell.cell_id}: {len(blocked)} blocked", flush=True)
    return result


# =====================================================================
# phases B + C: (re-ingest and) answer one cell
# =====================================================================
def _log_rows(cell_id: str, since: str = "") -> list[dict]:
    rows = []
    if PROXY_LOG.exists():
        for line in PROXY_LOG.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("cell_id") == cell_id and r.get("ts", "") >= since:
                    rows.append(r)
    return rows


def cost_since(cell_id: str, since: str) -> dict:
    """Calls, tokens and USD per purpose for this cell's calls at or
    after `since` (same shape as runner.run.cost_for_cell)."""
    agg = defaultdict(lambda: {"calls": 0, "tokens": 0, "usd": 0.0})
    for r in _log_rows(cell_id, since):
        a = agg[r.get("purpose") or "unknown"]
        a["calls"] += 1
        a["tokens"] += r.get("tokens_total") or 0
        a["usd"] += r.get("cost_usd") or 0.0
    return {k: {"calls": v["calls"], "tokens": v["tokens"], "usd": round(v["usd"], 6)}
            for k, v in agg.items()}


def require_prompt_date(cell: V3Cell) -> None:
    """A __v3d store must be extracted with the prompt date pinned to
    OCT_PROMPT_DATE (MEMDEF_PROMPT_DATE)."""
    from adapters.mem0_adapter import PROMPT_DATE_VAR, prompt_date
    if cell.batch == "v3d" and cell.reingest and prompt_date() != OCT_PROMPT_DATE:
        raise RuntimeError(f"{cell.cell_id}: needs {PROMPT_DATE_VAR}={OCT_PROMPT_DATE} "
                           f"(got {prompt_date()!r})")


def run_cell(cell: V3Cell, convs, guard: LiveGuard, commit: str) -> dict:
    from adapters.mem0_adapter import Mem0Adapter
    from core.purpose import cell as cell_context
    from defenses.defenses import wrap
    from runner.run import answer_metrics, answer_questions, collect_retries, cost_for_cell
    from scripts.run_baseline import ingest_with_progress

    d = cell.dir
    if (d / "summary.json").exists():
        return json.loads((d / "summary.json").read_text())
    if cell.source_ns != cell.old_id and cell.source_ns != cell.cell_id:
        # answers from another cell's re-ingested store (baseline-mini):
        # that store must be complete first
        if not (CELLS / cell.source_ns / "summary.json").exists():
            raise RuntimeError(f"{cell.cell_id}: source store {cell.source_ns} is not complete")
    require_prompt_date(cell)
    d.mkdir(parents=True, exist_ok=True)
    answers_file = d / "answers.jsonl"
    conv = _turn_conv(convs, cell)
    print(f"\n=== {cell.cell_id}  ({'re-ingest' if cell.reingest else 'answer-only'}, "
          f"source {cell.source_ns}, answers on {cell.answer_alias})", flush=True)

    adapter = Mem0Adapter(run_id="v3", cell_id=cell.cell_id, backbone=BACKBONE_ALIAS,
                          answer_alias=cell.answer_alias)
    try:
        t0 = time.monotonic()
        write_screen = None
        attempt_start = ""
        if cell.reingest:
            # Phase A screened this cell before it joined the re-ingest
            # list: keep its blocked turns as a record before the
            # re-ingest truncates the quarantine log.
            q = QUARANTINE / f"{cell.cell_id}.jsonl"
            keep = d / "phaseA_quarantine.jsonl"
            if (d / "write_verdicts.json").exists() and q.exists() and not keep.exists():
                keep.write_text(q.read_text())
            attempt_start = _now()
            prepare_reingest(adapter, d)
            history = adapter.message_history(cell.cell_id)
            if history:
                raise RuntimeError(f"{cell.cell_id}: message history not empty before "
                                   f"turn 1 ({len(history)} messages)")
            target = wrap(cell.defenses, adapter, cell.cell_id) if cell.defenses else adapter
            started = _now()
            with cell_context(cell.cell_id):
                if cell.defenses:
                    target.ingest(conv)
                else:
                    ingest_with_progress(adapter, conv)
            guard.check()
            rows = [r for r in _log_rows(cell.cell_id, started) if r["status"] == "ok"]
            blocked = [json.loads(l) for l in (QUARANTINE / f"{cell.cell_id}.jsonl")
                       .read_text().splitlines() if l.strip()] if cell.defenses else []
            retries = Counter(collect_retries(target))
            n_x = sum(r.get("step") == "extract" for r in rows)
            want_x = len(conv.turns) - len(blocked) + retries.get("extract:unparseable", 0)
            if n_x != want_x:
                raise RuntimeError(f"{cell.cell_id}: {n_x} ok extraction calls, expected {want_x}")
            if "anomaly" in cell.defenses:
                n_a = sum(r.get("step") == "anomaly" for r in rows)
                want_a = len(conv.turns) + retries.get("anomaly:unparseable", 0)
                if n_a != want_a:
                    raise RuntimeError(f"{cell.cell_id}: {n_a} ok anomaly verdicts, expected {want_a}")
            write_screen = {"started": started, "turns_screened": len(conv.turns),
                            "blocked": blocked,
                            "phase_a_superseded": (d / "write_verdicts.json").exists()}
            print(f"  re-ingested: {n_x} ok extractions, {len(blocked)} turns blocked", flush=True)
        else:
            adapter.open_namespace(cell.source_ns, read_only=True)
            # never truncate here: phase A's blocked turns and a resumed
            # cell's earlier drops belong to this cell's record
            target = (wrap(cell.defenses, adapter, cell.cell_id, truncate_quarantine=False)
                      if cell.defenses else adapter)
            if cell.screen:
                write_screen = json.loads((d / "write_verdicts.json").read_text())
                if write_screen["blocked"]:
                    raise RuntimeError(f"{cell.cell_id}: write-time chain blocked turns; "
                                       f"needs a full re-ingest")
        ingest_secs = time.monotonic() - t0

        before = len(adapter.dump_memories())
        questions = cell_questions(cell, convs)
        runner_retries = Counter()
        t1 = time.monotonic()
        with cell_context(cell.cell_id):
            answer_questions(target, questions, answers_file, runner_retries,
                             after_each=guard.check)
        answer_secs = time.monotonic() - t1
        after = len(adapter.dump_memories())
        if after != before:
            raise RuntimeError(f"{cell.cell_id}: store changed while answering ({before} -> {after})")

        rows = [json.loads(l) for l in answers_file.read_text().splitlines() if l.strip()]
        m = answer_metrics(rows)
        adv = [r for r in rows if r["category"] == "adversarial"]
        cands = sum(r.get("memories_candidates") or 0 for r in rows)
        dropped = sum(r.get("memories_dropped") or 0 for r in rows)
        blocked_n = len((write_screen or {}).get("blocked") or [])
        retries = Counter(collect_retries(target)) + runner_retries
        if write_screen and write_screen.get("retries"):
            retries += Counter(write_screen["retries"])
        served = defaultdict(Counter)
        for r in _log_rows(cell.cell_id, attempt_start):
            if r["status"] == "ok":
                served[r.get("step") or "untagged"][r["model_resolved"]] += 1
        summary = {
            "cell_id": cell.cell_id, "seed": cell.seed, "group": GROUP,
            "backbone": BACKBONE_ALIAS, "condition": cell.condition, "defenses": cell.defenses,
            "conv_id": cell.conv_id, "turns_ingested": len(conv.turns),
            "memories_stored": before, "questions_graded": m["questions_graded"],
            "questions_ungraded": m["questions_ungraded"],
            "questions_evidence_aligned": None, "question_selection": "evidence_aligned",
            "accuracy_core4": m["accuracy_core4"],
            "adversarial_abstention": m["adversarial_abstention"],
            "adversarial_abstention_detector": (
                round(sum(r["abstain_detector"] for r in adv) / len(adv), 4) if adv else None),
            "accuracy_by_category": m["accuracy_by_category"],
            "memories_retrieved_mean": m["memories_retrieved_mean"],
            "memories_per_turn": round(before / len(conv.turns), 3),
            "quarantined_items": blocked_n + dropped,
            "false_quarantine_rate": round((blocked_n + dropped) / len(conv.turns), 4),
            "latency_p50_ms": m["latency_p50_ms"], "latency_p95_ms": m["latency_p95_ms"],
            "ingest_seconds": round(ingest_secs, 1), "answer_seconds": round(answer_secs, 1),
            # a re-ingested cell counts only its final attempt's calls
            "cost_by_purpose": (cost_since(cell.cell_id, attempt_start) if cell.reingest
                                else cost_for_cell(cell.cell_id)),
            "retries": dict(sorted(retries.items())),
            # v3 additions
            "old_cell_id": cell.old_id,
            "kind": cell.kind,
            "answer_alias": cell.answer_alias,
            "source_namespace": cell.source_ns,
            "reingested": cell.reingest,
            "batch": cell.batch,
            "store_period": cell.store_period,
            "memories_before_answering": before,
            "memories_after_answering": after,
            "rerank_totals": ({"retrieved": cands, "kept": cands - dropped, "dropped": dropped}
                              if "rerank" in cell.defenses else None),
            "write_screen": write_screen,
            # MEMDEF_PROMPT_DATE in effect (None: mem0's UTC date at each call)
            "prompt_date_pinned": adapter.prompt_date,
            "served_models": {k: dict(v) for k, v in sorted(served.items())},
            "git_commit": commit,
        }
        summary["questions_evidence_aligned"] = len(questions)
        (d / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"  core4={summary['accuracy_core4']} adv={summary['adversarial_abstention']} "
              f"adv_det={summary['adversarial_abstention_detector']} "
              f"store={before} ungraded={m['questions_ungraded']}", flush=True)
        return summary
    finally:
        adapter.close()


# =====================================================================
# protocol, dry run
# =====================================================================
_S = f"{GROUP}__{BACKBONE_ALIAS}__"

# Deviations that affect a cell the manifest uses (main or split), as
# (cell or store, explanation). Stale context = the earlier attempt's
# last saved messages that mem0 put into the final ingestion's first
# extraction prompts (turn numbers from scripts/check_integrity.py).
KNOWN_DEVIATIONS = [
    (f"{_S}sanitize__conv4__s3",
     "MAIN (conv-43, August). Old store kept although its cell was re-run under the same id: "
     "the final ingestion's first extractions saw the aborted attempt's last saved messages "
     "(stale context = turns 1-5). Answered from the old store."),
    (f"{_S}sanitize__conv3__s3__v3b",
     f"MAIN (conv-42, August). Answers from the old August store {_S}sanitize__conv3__s3, "
     "which carries the same kind of leftover history (stale context = turns 141-150). Its "
     "October re-ingest (__v3) is superseded so that conv-42 stays all August."),
    (f"{_S}none__conv0",
     "SPLIT (old none conv0 s1) and the baseline-mini cell that reads its store "
     f"({_S}none__conv0__mini__v3b): stale context = turns 143-150, 1, 2."),
    (f"{_S}none__conv1",
     "SPLIT (old none conv1 s1) and the baseline-mini cell that reads its store "
     f"({_S}none__conv1__mini__v3b): stale context = turns 141-150."),
    (f"{_S}rerank__conv2__s3",
     "SPLIT (old rerank, results/): 2 of 320 rerank verdicts never succeeded after 149 "
     "rate-limit / 5xx errors and the old code read them as TRUST (failed open). The integrity "
     "check reports it as the one known exception of the legacy configuration."),
    (f"{_S}stacked__conv0__v3c",
     "MAIN (conv-26, October). 751 ok rerank rows for 750 decisions: a rerank call sent 20:40:07 "
     "hung in the proxy; the OpenAI SDK timed out after 600 s and resent it (ok 20:50:09, the "
     "verdict used); litellm timed the abandoned original out after 6000 s (error 22:20:07) and "
     "retried it on its own (ok 22:20:09), 79 minutes after the cell's last answer. 750 decisions "
     "recorded (747 kept, 3 dropped), none failed open. A known exception in "
     "scripts/check_integrity.py, accepted only while the log shows that proxy timeout; proxy and "
     "client retries are off from the fourth batch on."),
    ("old rerank cells (15, split)",
     "By design (v2): rerank verdicts and answers ran on gpt-4o-mini (declared legacy "
     "configuration in scripts/check_integrity.py)."),
    ("anomaly / stacked conv3, conv4 (12, main)",
     "Stores built in August by the old code, whose anomaly verdicts ran on gpt-4o-mini. The "
     "old chain blocked no turn in them, and Phase A re-ran the write-time chain on Gemini, "
     "which blocked none either, so each store holds exactly what the fixed chain would store. "
     "Phase A recorded only Gemini's parsed keyword, not its raw reply."),
    ("all main cells, pooled",
     "conv-26, conv-30 and conv-41 answer from October memory, conv-42 and conv-43 from August "
     "memory. Every paired delta is within a conversation, so it never mixes periods; levels "
     "(accuracy, store size) differ between the two groups of conversations."),
]

GEMINI_CHANGE = [
    "## Gemini's extraction changed between August and October", "",
    "The 22 stores re-ingested on 2026-10-29 were all smaller than the August stores they "
    "replaced (by 16 to 101 memories), including none and sanitize cells, which block nothing. "
    "Diagnosis (results_diag/, 2026-10-30):",
    "",
    "- Part 1, read-only: the exact arguments of every Mem0 add() for conv-30 are identical "
    "under the August code (c72464e) and the current code, on both the no-defense and the "
    "defense-chain path (only the namespace name differs). The resolved Mem0 config is "
    "identical except two paths (vector store; message-history file). mem0ai, litellm, openai "
    "and qdrant-client match requirements.lock and nothing in the venv changed after 2026-08-23. "
    "Old cells with no leftover message history are as large as the others, and old and October "
    "store sizes do not overlap in any of the four conversations compared.",
    "- Part 2, one controlled rebuild: the August code, run today in a worktree at c72464e "
    "against today's Gemini (`gemini/gemini-3.1-flash-lite`, no pinned version) with a fresh "
    "history, ingested conv-30's first 150 turns (none, s1) with 0 failed calls:",
    "",
    "| conv-30, none, s1 | memories | empty extractions | turns storing >=1 | memory text tokens |",
    "|---|---|---|---|---|",
    "| August store (old code, 2026-08-22) | 282 | 9 | 141 | 8,542 |",
    "| October re-ingest (current code, 2026-10-29) | 213 | 18 | 132 | 6,752 |",
    "| August code today (2026-10-30) | 225 | 18 | 132 | 6,720 |",
    "",
    "The old code today reproduces the October behaviour, not August's: the change is in "
    "Gemini, not in our code. August's extractions also restated facts from the earlier "
    "messages in the prompt on short turns (\"Thanks, Jon!\" -> 4 memories); October's do not.",
    "",
]


def write_protocol(cells: list[V3Cell]) -> pathlib.Path:
    from core.manifest import MANIFEST, load
    commit, dirty = git_commit()
    models = alias_models()
    man = load()
    first = [c for c in cells if c.batch == "v3"]
    second = [c for c in cells if c.batch == "v3b"]
    third = [c for c in cells if c.batch == "v3c"]
    fourth = [c for c in cells if c.batch == "v3d"]
    L = [f"# Protocol: Gemini regeneration (results_v3)", "",
         f"- written (UTC): {_now()[:19]}",
         f"- git commit: `{commit}`" + (" (WORKING TREE DIRTY)" if dirty else ""),
         f"- results root: `{V3_ROOT_NAME}/` (MEMDEF_RESULTS={V3_ROOT_NAME}); proxy log "
         f"`{V3_ROOT_NAME}/proxy/calls.jsonl`",
         f"- memory store: `{V3_ROOT_NAME}/mem0_qdrant`, a byte-identical copy of "
         f"`results/mem0_qdrant` (all 90 namespaces verified against memories_stored); "
         f"re-ingests write only their own `__v3` / `__v3b` / `__v3c` / `__v3d` namespace there",
         f"- mem0 message history: `{V3_ROOT_NAME}/mem0_history.db` (fresh; every re-ingested "
         f"namespace asserted empty before its first turn)",
         f"- judge cache: `{V3_ROOT_NAME}/judge_cache.jsonl`, copy of the old cache "
         f"(every cached grade verified to come from claude-haiku-4-5)",
         f"- cells the paper uses: `{V3_ROOT_NAME}/{MANIFEST.name}` (see Manifest)",
         "", "## Aliases", "", "| alias | resolves to |", "|---|---|"]
    L += [f"| `{a}` | `{m}` |" for a, m in sorted(models.items())]
    L += ["", "## Allowed models (checked live on every new proxy-log line)", "",
          "| step | model |", "|---|---|",
          f"| extract, anomaly, rerank | `{GEMINI}` |",
          f"| answer | `{GEMINI}`; `{MINI}` for the baseline-mini cells only "
          f"(`__mini`, `__mini__v3b`) |",
          f"| judge | `{HAIKU}` (alias `judge-claude`) |",
          f"| embedding | `{EMBED}` |",
          "", "## Procedure", "",
          f"- Questions: evidence filter over the first {MAX_TURNS} turns, cap {MAX_QUESTIONS}; "
          "each cell's question ids asserted equal (and in order) to its old cell's.",
          f"- Retrieval top-{TOP_K}. Answer prompt: core/answer.py (byte-identical to the "
          "pre-fix prompt, pinned by test).",
          "- Phase A, write-time verdicts only: every anomaly/stacked cell that is not "
          "re-ingested runs the exact ingest-path turn strings through its write-time chain "
          "(ingest order) on the backbone; nothing is stored. Any blocked turn stops the run.",
          "- Phase B+C, in order of replicate, conversation, then the conditions (none first; "
          "the batches in order): re-ingest where listed, then answer. "
          "Rerank/stacked: TRUST/DROP on the backbone per retrieved memory; retrieved, kept and "
          "dropped recorded per question.",
          "- Grading: core answers by core/judge.py (judge-claude); adversarial answers by the "
          "original ABSTAIN_RX (`correct`) and by detect_abstained() "
          "(`abstain_detector`), both stored. A judge that never returns a parseable grade "
          "leaves the question ungraded (counted).",
          "- Failures: every model call retries with backoff; extraction, defense and answer "
          "calls that still fail abort the cell (never fail open). From the fourth batch on, "
          "neither the proxy (`num_retries: 0`, `request_timeout: 300`) nor any OpenAI client "
          "(`max_retries=0`, timeout 360 s; core/proxy_client.py, also applied to Mem0's own "
          "clients) retries on its own, so every retry is counted in the cell's `retries`.",
          "- Guards: live model check on every new proxy-log line. Exit codes: 0 done, 2 model "
          "violation, 3 budget, 4 retries exhausted (resumed up to 3 times after 15 minutes), "
          "5 write-time block.",
          "- Budget: first batch (2026-10-29) aborted when the WHOLE log's spend exceeded Google "
          "$13, Anthropic $7, OpenAI $10. Second batch: only that run's calls counted (since "
          "`run_v3b_started.txt`), abort above Google $13.50, Anthropic $5, OpenAI $5. Third "
          "batch: only that run's calls (since `run_v3c_started.txt`), abort above Google $12, "
          "Anthropic $5, OpenAI $5. Fourth batch: only that run's calls count (since "
          f"`{RUN_START_FILE.name}`), abort above "
          f"Google ${BUDGET_LIMITS['google']:.2f}, Anthropic ${BUDGET_LIMITS['anthropic']:.2f}, "
          f"OpenAI ${BUDGET_LIMITS['openai']:.2f}.",
          ""]
    L += GEMINI_CHANGE
    L += ["## Build periods", "",
          "Within each conversation every condition answers from memory extracted in ONE "
          "period, so no paired delta compares an August store with an October one.", "",
          "| conversation | period | stores |", "|---|---|---|"]
    conv_ids = {c.conv_idx: c.conv_id for c in cells}
    for ci, per in sorted(PERIOD_OF_CONV.items()):
        what = ("October: rebuilt with the fixed pipeline by the Oct 30 (__v3b) and Oct 31 "
                f"(__v3c) batches; every store's final ingestion ended at or after "
                f"{OCT_CUTOFF[:19]}Z" if per == "Oct" else
                "August: the old stores, answered read-only")
        L.append(f"| {conv_ids[ci]} (conv{ci}) | {per} | {what} |")
    L += ["", "The model-swap split (old rerank, old none, baseline-mini) uses August stores "
          "only, as in v2: per question, v2 delta = (old rerank - mini) + (mini - old none), "
          "with old rerank and old none the v2 cells in results/ and every baseline-mini cell "
          "answering from its old none cell's store.", ""]
    L += ["## Re-ingested cells, first batch (22, 2026-10-29)", "",
          "Stage 2 (6): anomaly conv1 s2 (turns 100 and 101 never extracted); none conv0 s1, none "
          "conv1 s1, sanitize conv3 s3, sanitize conv2 s3, stacked conv0 s1 (their first "
          "extractions saw an aborted attempt's stale message history).",
          "",
          "Phase A (16): Gemini's write-time chain blocked a turn present in the old store, in "
          "every replicate and in both conditions: conv-26 D1:12 (Melanie), conv-30 D2:4 (Jon), "
          "conv-41 D4:11 (Maria); see phaseA_report.txt. Phase A's verdict calls for these 16 cells "
          "are superseded: the integrity check, tau and cost_by_purpose count only the re-ingest's "
          "own verdicts. Phase A recorded only the parsed keyword, not Gemini's raw reply; the "
          "re-ingest records the raw reply with every block (quarantine log).",
          "", "| re-ingested cell | reason |", "|---|---|"]
    L += [f"| `{c}__v3` | {'Stage 2' if c in REINGEST_STAGE2 else 'Phase A block'} |"
          for c in sorted(REINGEST)]
    L += ["", "## Second batch (__v3b, 36 cells)", "",
          f"- re-ingested into `<old_id>__v3b` ({len(REBUILD_V3B)}): none, sanitize, provenance "
          "and rerank in conv0-2 that were not already October stores; full write-time chain, "
          "message history asserted empty before turn 1; answered on the backbone (rerank: "
          "TRUST/DROP on the backbone).",
          f"- answer-only from the August store (1): `{_S}sanitize__conv3__s3__v3b`.",
          f"- baseline-mini from the August none stores (2): `{_S}none__conv0__mini__v3b`, "
          f"`{_S}none__conv1__mini__v3b` (the first batch's conv0/conv1 s1 mini cells read the "
          "October none__v3 stores and are superseded).", ""]
    L += ["## Third batch (__v3c, 21 cells, 2026-10-31)", "",
          "Stage 6 found that the first batch (night of Oct 29) and the second (Oct 30) "
          "extract differently: none conv0 s1 (Oct 29) and none conv0 s2 (Oct 30) differ from "
          "turn 1 (262 vs 253 memories, 55 shared), while the second batch's s2 and s3 builds "
          "are identical. A rebuild on 2026-10-31 (results_diag/repro_oct31, namespace "
          "repro__none__conv0__oct31, excluded from everything) reproduced the Oct 30 store on "
          "149 of 150 turns (253 memories; one turn reworded, 2 memories). So every October "
          f"main store whose final ingestion ended before {OCT_CUTOFF[:19]}Z is rebuilt:", "",
          f"- re-ingested into `<old_id>__v3c` ({len(REBUILD_V3C)}): anomaly and stacked conv0-2 "
          "s1-s3 (18), none conv0 s1, none conv1 s1, sanitize conv2 s3; same code and settings "
          "as the second batch: full write-time chain with anomaly verdicts on the backbone and "
          "the raw reply recorded with every block, message history asserted empty before turn "
          "1, answered on the backbone (rerank: TRUST/DROP on the backbone). Five of them were "
          "built after midnight UTC, with a different prompt date (see the fourth batch). A "
          "block outside "
          "the known three (" + ", ".join(f"conv{c} {t}" for c, t in KNOWN_BLOCKS.items())
          + ") does not stop the run; it is applied, recorded and listed in Stage 4c.",
          "- the integrity check asserts that every October main store's final ingestion ended "
          f"at or after {OCT_CUTOFF[:19]}Z and every August main store's is in August; the "
          "manifest records each main cell's store's final ingestion.", ""]
    L += ["## Fourth batch (__v3d, 5 cells, 2026-10-31): the extraction-prompt date", "",
          "Mem0 2.0.15 ends every extraction prompt with `## Observation Date` and `## Current "
          "Date`, both the UTC date at the time of the call (mem0/configs/prompts.py "
          "`_resolve_dates`; our add() passes no timestamp). The divergences between builds line "
          "up with that date (results_diag/prompt_date/part1_report.txt): October stores "
          "extracted on different dates differ from the first turns on (at most 47 of 150 turns "
          "with equal output), while October stores extracted on the same date agree on at "
          "least their first 53 turns and mostly on 144-150. The night batch's stores built on "
          "2026-10-29 and the five third-batch stores built after midnight UTC (2026-10-31) are "
          "the different-date ones. August stores differ from each other even on the same date.",
          "",
          "`MEMDEF_PROMPT_DATE=YYYY-MM-DD` (adapters/mem0_adapter.py) pins both dates; unset, "
          "Mem0's default applies. Each cell's summary records it (`prompt_date_pinned`; absent "
          "or null = not pinned).",
          "",
          f"- re-ingested into `<old_id>__v3d` ({len(REBUILD_V3D)}) with "
          f"`MEMDEF_PROMPT_DATE={OCT_PROMPT_DATE}`: "
          + ", ".join(f"`{c.replace(_S, '')}`" for c in sorted(REBUILD_V3D))
          + "; same rules as the third batch (full write-time chain, message history asserted "
          "empty before turn 1, answers and rerank on the backbone); they replace their __v3c "
          "versions in the manifest.",
          f"- the integrity check records each main store's prompt date (the pinned date, else "
          f"the UTC date at the start of each final-ingestion extraction call) and asserts "
          f"{OCT_PROMPT_DATE} for every October main store; August prompt dates are reported per "
          "conversation and replicate, not checked.", ""]
    L += ["## Manifest", "",
          f"`{V3_ROOT_NAME}/{MANIFEST.name}` (written by `python3 -m runner.v3 --manifest`) lists, "
          "for every condition, conversation and replicate, the one cell the paper uses, its "
          "store namespace, the cell that built that store and the store's build period; and the "
          "split set (15 old rerank and 15 old none cells from results/, 15 baseline-mini cells). "
          "Superseded cells stay on disk untouched and are excluded only by the manifest. "
          "scripts/check_integrity.py, stats.py, rescore/label/check/adjudicate_abstention.py, "
          "audit_quarantine.py, make_percell_table.py, runner/analyse.py and fig.py take their "
          "cells from it. The integrity check measures each store's build period from its "
          "builder's final ingestion and asserts one period per conversation and August-only "
          "stores in the split.", ""]
    if man:
        L += [f"Manifest written {man['created'][:19]} at `{man['git_commit']}`: "
              f"{man['counts']['main']} main, {man['counts']['split']} split, "
              f"{len(man['superseded'])} superseded cells.", "",
              "| role | condition | conv | seed | cell | store | built by | period | "
              "final ingestion (UTC) | prompt date |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for e in man["cells"]:
            b = e["store_built_by"]
            fi = e.get("store_final_ingestion")
            fi = f"{fi['start'][:19]} .. {fi['end'][11:19]}" if fi else ""
            pd = "+".join(e.get("store_prompt_date") or [])
            L.append(f"| {e['role']} | {e['condition']} | {e['conv_id']} | {e['seed']} | "
                     f"`{e['root']}/{e['cell_id'].replace(_S, '')}` | "
                     f"`{e['store_namespace'].replace(_S, '')}` | "
                     f"`{b['root']}/{b['cell_id'].replace(_S, '')}` | {e['store_period']} | "
                     f"{fi} | {pd} |")
        L += ["", "Superseded (on disk, not used): "
              + ", ".join(f"`{x.replace(_S, '')}`" for x in man["superseded"]), ""]
    L += ["## Cells run in results_v3", "",
          "| cell | batch | old cell | condition | seed | conv | answers from | period | "
          "answer alias | re-ingest |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        L.append(f"| `{c.cell_id}` | {c.batch} | `{c.old_id}` | {c.condition} | {c.seed} | "
                 f"{c.conv_id} | `{c.source_ns}` | {c.store_period} | {c.answer_alias} | "
                 f"{'yes' if c.reingest else ''} |")
    L += ["", f"{len(cells)} cells: first batch {len(first)} "
          f"({sum(c.reingest for c in first)} re-ingested, {sum(c.screen for c in first)} "
          f"write-time screened, {sum(c.kind == 'mini' for c in first)} baseline-mini); second "
          f"batch {len(second)} ({sum(c.reingest for c in second)} re-ingested, "
          f"{sum(c.kind == 'mini' for c in second)} baseline-mini); third batch {len(third)} "
          f"({sum(c.reingest for c in third)} re-ingested); fourth batch {len(fourth)} "
          f"({sum(c.reingest for c in fourth)} re-ingested, prompt date pinned).", "",
          "## Known deviations (cells the manifest uses)", ""]
    L += [f"- `{cid}`: {why}" for cid, why in KNOWN_DEVIATIONS]
    L += ["", "Superseded cells keep their own deviations (e.g. old stacked conv1 s1, stale "
          "context turns 1-4); none of them is used.",
          "", "## Integrity check: declared legacy configuration", "",
          "For the 15 OLD rerank cells (results/), rerank verdicts and answers are declared to "
          "have run on gpt-4o-mini (scripts/check_integrity.py LEGACY). Everything else is "
          "checked as normal; the 2 failed-open verdicts above are its one known exception, "
          "always printed. The other known exception is stacked conv0 v3c's extra proxy-side "
          "rerank row (above), accepted only with the matching proxy timeout in the log.", ""]
    out = RESULTS / "PROTOCOL.md"
    out.write_text("\n".join(L))
    return out


def measured_sizes(log: pathlib.Path = OLD_PROXY_LOG) -> dict:
    """Mean input/output tokens per (step, model) in a proxy log (default:
    the OLD log)."""
    from scripts.check_integrity import load_log
    by_cell, judge_rows = load_log(log)
    acc = defaultdict(lambda: [0, 0, 0])
    for cid, rs in by_cell.items():
        if not cid.startswith(GROUP + "__"):
            continue
        for r in rs:
            if r["status"] == "ok":
                a = acc[(r["_step"], r["model_resolved"])]
                a[0] += 1; a[1] += r["tokens_in"]; a[2] += r["tokens_out"]
    for r in judge_rows:
        if r["status"] == "ok" and r["model_resolved"] == HAIKU:
            a = acc[("judge", HAIKU)]
            a[0] += 1; a[1] += r["tokens_in"]; a[2] += r["tokens_out"]
    return {k: (v[1] / v[0], v[2] / v[0], v[0]) for k, v in acc.items() if v[0]}


def remaining_calls(c: V3Cell, convs) -> Counter:
    """Calls still to be made for one cell, given what is on disk: none
    if it has a summary; no screening if its write verdicts exist (and it
    is not re-ingested); a re-ingested cell always restarts in full."""
    if (c.dir / "summary.json").exists():
        return Counter()
    qs = cell_questions(c, convs)
    n_q = len(qs)
    n_core = sum(q.category != "adversarial" for q in qs)
    calls = Counter()
    if "anomaly" in c.defenses and (c.reingest or not (c.dir / "write_verdicts.json").exists()):
        calls[("anomaly", GEMINI)] = MAX_TURNS
    if c.reingest:
        calls[("extract", GEMINI)] = MAX_TURNS
        calls[("embedding", EMBED)] += round(_embeddings_per_add() * MAX_TURNS)
    if "rerank" in c.defenses:
        calls[("rerank", GEMINI)] = TOP_K * n_q
    calls[("answer", MINI if c.kind == "mini" else GEMINI)] = n_q
    calls[("judge", HAIKU)] = n_core
    calls[("embedding", EMBED)] += n_q
    return calls


def run_start() -> str:
    """This run's start (ISO), or "" if it has not started yet."""
    return RUN_START_FILE.read_text().strip() if RUN_START_FILE.exists() else ""


def logged_spend(since: str = "") -> Counter:
    spend = Counter()
    if PROXY_LOG.exists():
        for line in PROXY_LOG.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if (r.get("ts") or "") < since:
                    continue
                prov = provider_of(r.get("model_resolved"))
                if prov:
                    spend[prov] += r.get("cost_usd") or 0.0
    return spend


def dry_run(cells: list[V3Cell], convs) -> dict:
    """Estimate for everything still to do. Returns cost per provider."""
    from proxy.logger import PRICES
    sizes = measured_sizes()
    # October sizes (this root's own log) take precedence: today's Gemini
    # extracts less than August's, and its verdicts are measured there.
    sizes_now = measured_sizes(PROXY_LOG) if PROXY_LOG.exists() else {}
    smoke = RESULTS / "smoke_test.json"
    smoke_sizes = json.loads(smoke.read_text()).get("sizes", {}) if smoke.exists() else {}

    def size(step, model):
        """(in, out, source) per call: measured in this root's log, else
        the smoke test, else the old log. Gemini verdicts with no history
        at all: input = the gpt-4o-mini verdict's input (same prompt) and
        output = 5 tokens (assumed)."""
        key = f"{step}|{model}"
        if (step, model) in sizes_now:
            i, o, n = sizes_now[(step, model)]
            return i, o, f"{PROXY_LOG.parent.parent.name} log (n={n})"
        if key in smoke_sizes:
            i, o = smoke_sizes[key]
            return i, o, "smoke"
        if (step, model) in sizes:
            i, o, _ = sizes[(step, model)]
            return i, o, "old log"
        if step in ("anomaly", "rerank") and model == GEMINI:
            i, _, _ = sizes[(step, MINI)]
            return i, 5.0, "ASSUMED out=5"
        raise KeyError((step, model))

    todo = [c for c in cells if not (c.dir / "summary.json").exists()]
    print(f"DRY RUN -- {len(cells)} cells selected, {len(todo)} still to run "
          f"({len(cells) - len(todo)} already complete, not listed); no calls made\n")
    short = lambda x: x.replace(f"{GROUP}__{BACKBONE_ALIAS}__", "")
    print(f"{'#':>2} {'cell':34} {'what':12} {'store (namespace answered from)':36} {'period':6} "
          f"{'answer model':12} {'extr':>4} {'anom':>4} {'rrnk':>5} {'ans':>3} {'judge<=':>7} "
          f"{'emb~':>5}")
    totals = Counter()
    for n, c in enumerate(todo, 1):
        calls = remaining_calls(c, convs)
        ans_model = MINI if c.kind == "mini" else GEMINI
        totals.update(calls)
        g = lambda st, m=GEMINI: calls.get((st, m), 0)
        what = "re-ingest" if c.reingest else "answer-only"
        print(f"{n:>2} {short(c.cell_id):34} {what:12} {short(c.source_ns):36} {c.store_period:6} "
              f"{ans_model:12} {g('extract'):>4} {g('anomaly'):>4} {g('rerank'):>5} "
              f"{calls[('answer', ans_model)]:>3} {g('judge', HAIKU):>7} {g('embedding', EMBED):>5}")

    print(f"\nExpected calls by step (judge = upper bound, before judge-cache hits):")
    cost = Counter()
    tok = Counter()
    print(f"  {'step':10} {'model':28} {'calls':>7} {'in/call':>8} {'out/call':>9} "
          f"{'tokens in':>11} {'tokens out':>10} {'USD':>8}  sizes from")
    for (step, model), n in sorted(totals.items()):
        i, o, src = size(step, model)
        p = PRICES[model]
        usd = n * (i * p["in"] + o * p["out"]) / 1e6
        prov = PROVIDER[model]
        cost[prov] += usd
        tok[prov, "in"] += n * i
        tok[prov, "out"] += n * o
        print(f"  {step:10} {model:28} {n:>7} {i:>8.0f} {o:>9.1f} {n * i:>11,.0f} "
              f"{n * o:>10,.0f} {usd:>8.2f}  {src}")
    print("\nEstimated cost per provider (no retries; judge without cache hits):")
    since = run_start()
    spent = logged_spend(since) if since else Counter()
    print(f"  this run: {'started ' + since if since else 'not started (spend so far $0)'}")
    print(f"  {'provider':10} {'estimate':>9} {'this run so far':>16} {'projected':>10} "
          f"{'abort limit':>12} {'balance':>8}")
    for prov in ("google", "openai", "anthropic"):
        total = spent[prov] + cost[prov]
        bal = BALANCES[prov]
        flags = (["ABOVE ABORT LIMIT"] if total > BUDGET_LIMITS[prov] else []) + (
            ["ABOVE BALANCE"] if bal is not None and total > bal else [])
        print(f"  {prov:10} ${cost[prov]:8.2f} ${spent[prov]:15.2f} ${total:9.2f} "
              f"${BUDGET_LIMITS[prov]:11.2f} {f'${bal:7.2f}' if bal is not None else 'unknown':>8}   "
              f"{'<-- ' + ', '.join(flags) if flags else 'ok'}")
        print(f"  {'':10} tokens in {tok[prov, 'in']:>12,.0f}  out {tok[prov, 'out']:>10,.0f}")
    return dict(cost)


@functools.lru_cache(maxsize=None)
def _embeddings_per_add() -> float:
    """Embedding calls per mem0 add() in the old final ingestions
    (search + fact embeddings), measured from the old log."""
    from scripts.check_integrity import load_log
    by_cell, _ = load_log(OLD_PROXY_LOG)
    n_e = n_x = 0
    for cid, rs in by_cell.items():
        if not cid.startswith(GROUP + "__"):
            continue
        last_x = max((r["ts"] for r in rs if r["_step"] == "extract"), default="")
        n_x += sum(r["_step"] == "extract" and r["status"] == "ok" for r in rs)
        n_e += sum(r["_step"] == "embedding" and r["status"] == "ok" and r["ts"] <= last_x
                   for r in rs)
    return n_e / n_x


# =====================================================================
# smoke test
# =====================================================================
BARE = {kw: re.compile(rf"^[\s*_`#]*{kw}[\s*_`#]*$") for kw in
        ("BENIGN", "SUSPICIOUS", "TRUST", "DROP")}


def _is_bare(reply: str | None) -> bool:
    return any(rx.match(reply or "") for rx in BARE.values())


class _Recorder:
    """Wraps a defense's OpenAI client and keeps every raw reply."""

    def __init__(self, client):
        self.replies = []
        self._client = client
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        resp = self._client.chat.completions.create(**kw)
        prompt = kw["messages"][0]["content"]
        kind = ("anomaly" if "BENIGN or SUSPICIOUS" in prompt else
                "rerank" if "TRUST or DROP" in prompt else "answer")
        self.replies.append({"kind": kind, "reply": resp.choices[0].message.content})
        return resp


def smoke_test(convs) -> int:
    import httpx
    from adapters.mem0_adapter import Mem0Adapter
    from core.purpose import cell as cell_context
    from defenses.defenses import wrap
    from runner.run import grade

    try:
        httpx.get("http://localhost:4000/health/liveliness", timeout=5).raise_for_status()
    except Exception as e:
        sys.exit(f"proxy not reachable on :4000 ({e}). Start it with MEMDEF_RESULTS={V3_ROOT_NAME}.")
    guard = LiveGuard()
    start = _now()
    src = "anchor_gemini__gemini-backbone__none__conv0__s2"     # an old store, opened read-only
    conv = convs[0]
    from core.evidence_filter import filter_questions
    qa = next(q for q in filter_questions(conv, MAX_TURNS)[:MAX_QUESTIONS]
              if q.category != "adversarial")
    out = {"started": start, "source_namespace": src, "question": qa.question}
    replies = []

    def adapter_for(cell_id, answer_alias=BACKBONE_ALIAS):
        a = Mem0Adapter(run_id="smoke", cell_id=cell_id, backbone=BACKBONE_ALIAS,
                        answer_alias=answer_alias)
        a.open_namespace(src, read_only=True)
        return a

    # 1. anomaly verdict (write-time path, one turn)
    a = adapter_for("smoke__anomaly__v3")
    try:
        t = wrap(["anomaly"], a, "smoke__anomaly__v3")
        rec = _Recorder(t._client)
        t._client = rec
        one = dataclasses.replace(conv, turns=conv.turns[:1])
        with cell_context("smoke__anomaly__v3"):
            out["anomaly_blocked"] = t.screen(one)
        replies += rec.replies
    finally:
        a.close()
    guard.check()

    # 2. rerank verdicts + Gemini answer + embedding, then 3. judge
    a = adapter_for("smoke__rerank__v3")
    try:
        t = wrap(["rerank"], a, "smoke__rerank__v3")
        rec = _Recorder(t._client)
        t._client = rec
        with cell_context("smoke__rerank__v3"):
            ans, st = t.answer(qa.question)
            # judge: the suffix guarantees a cache miss, so a real call is made
            out["judge"] = grade(qa, ans + " [smoke test]", Counter())
        replies += rec.replies
        out["gemini_answer"] = ans
        out["rerank_stats"] = dataclasses.asdict(st)
    finally:
        a.close()
    guard.check()

    # 4. gpt-4o-mini answer (baseline-mini path)
    a = adapter_for("smoke__none__mini", answer_alias=MINI_ANSWER_ALIAS)
    try:
        with cell_context("smoke__none__mini"):
            out["mini_answer"], _ = a.answer(qa.question)
    finally:
        a.close()
    guard.check()

    rows = [r for r in (json.loads(l) for l in PROXY_LOG.read_text().splitlines() if l.strip())
            if r.get("ts", "") >= start and (r.get("cell_id") or "").startswith("smoke__")]
    print(f"\nSMOKE TEST -- {len(rows)} calls in {PROXY_LOG}\n")
    print(f"  {'time':8} {'cell_id':20} {'step':9} {'alias requested':16} {'model_resolved':26} "
          f"{'status':6} {'in':>5} {'out':>4}")
    for r in rows:
        print(f"  {r['ts'][11:19]} {r.get('cell_id'):20} {str(r.get('step')):9} "
              f"{str(r.get('alias_requested')):16} {r['model_resolved']:26} {r['status']:6} "
              f"{r['tokens_in']:>5} {r['tokens_out']:>4}")
    sizes = defaultdict(list)
    for r in rows:
        if r["status"] == "ok":
            sizes[f"{r.get('step')}|{r['model_resolved']}"].append((r["tokens_in"], r["tokens_out"]))
    out["sizes"] = {k: [sum(x[0] for x in v) / len(v), sum(x[1] for x in v) / len(v)]
                    for k, v in sizes.items()}
    out["calls"] = rows
    out["verdict_replies"] = replies
    print("\n  Raw verdict replies (Gemini):")
    not_bare = []
    for rp in replies:
        if rp["kind"] == "answer":
            continue
        ok = _is_bare(rp["reply"])
        not_bare += [rp] if not ok else []
        print(f"    {rp['kind']:8} {rp['reply']!r:40} {'bare keyword' if ok else 'NOT A BARE KEYWORD'}")
    print(f"\n  Gemini answer: {out['gemini_answer']!r}")
    print(f"  gpt-4o-mini answer: {out['mini_answer']!r}")
    print(f"  judge: {out['judge']}")
    steps_seen = {(r.get("step"), r["model_resolved"]) for r in rows if r["status"] == "ok"}
    want = {("anomaly", GEMINI), ("rerank", GEMINI), ("answer", GEMINI), ("answer", MINI),
            ("judge", HAIKU), ("embedding", EMBED)}
    missing = want - steps_seen
    print(f"\n  call types seen: {len(want - missing)}/6" + (f"  MISSING {missing}" if missing else ""))
    print(f"  live guard: {guard.checked} lines checked, no violation")
    (RESULTS / "smoke_test.json").write_text(json.dumps(out, indent=2, default=str))
    if not_bare:
        print(f"\nSTOP: {len(not_bare)} Gemini verdict(s) are not a bare keyword (rule for Stage 3).")
        return 2
    return 0 if not missing else 1


# =====================================================================
# full run
# =====================================================================
def full_run(cells: list[V3Cell], convs) -> int:
    commit, dirty = git_commit()
    if dirty:
        sys.exit("working tree has uncommitted changes to tracked files; commit first")
    if not (RESULTS / "PROTOCOL.md").exists():
        sys.exit("write PROTOCOL.md first (--protocol)")
    from core.manifest import MANIFEST
    if not MANIFEST.exists():
        sys.exit(f"write {MANIFEST.name} first (--manifest)")
    todo = [c for c in cells if not (c.dir / "summary.json").exists()]
    for c in todo:
        try:
            require_prompt_date(c)
        except RuntimeError as e:
            sys.exit(str(e))
    if not RUN_START_FILE.exists():
        RUN_START_FILE.write_text(_now() + "\n")
    guard = LiveGuard(since=run_start())
    print(f"Budget counts this run's calls since {guard.since}; limits {BUDGET_LIMITS}; "
          f"spend so far { {p: round(v, 2) for p, v in guard.spend.items()} }", flush=True)
    print(f"Phase A: write-time verdicts for {sum(c.screen for c in cells)} cells", flush=True)
    blocked = {}
    for c in cells:
        if c.screen:
            res = screen_cell(c, convs, guard)
            if res["blocked"]:
                blocked[c.cell_id] = res["blocked"]
    if blocked:
        msg = "write-time chain blocked turns -- these cells need a full re-ingest:\n" + "\n".join(
            f"  {cid}: turn {x['turn_id']} blocked by {x['defense']}: {x['reason']}"
            for cid, b in blocked.items() for x in b)
        print("\nSTOP: " + msg)
        write_stopped(EXIT_BLOCKED, msg)
        return EXIT_BLOCKED
    todo = [c for c in cells if not (c.dir / "summary.json").exists()]
    print(f"\nPhase B+C: {len(todo)} cells to run ({len(cells) - len(todo)} already complete)",
          flush=True)
    for n, c in enumerate(todo, 1):
        print(f"\n>>> {n}/{len(todo)}", flush=True)
        run_cell(c, convs, guard, commit)
    print(f"\nDone. Live guard checked {guard.checked} lines; logged spend "
          f"{ {p: round(v, 2) for p, v in guard.spend.items()} }.")
    return EXIT_OK


def write_stopped(code: int, reason: str) -> None:
    (RESULTS / "STOPPED.txt").write_text(f"{_now()}  exit {code}\n{reason}\n")


def guarded_run(cells, convs) -> int:
    """full_run with the documented exit codes."""
    from core.retry import ModelCallFailed
    try:
        return full_run(cells, convs)
    except GuardViolation as e:
        print(f"\nSTOP (exit {EXIT_MODEL}): {e}", flush=True)
        write_stopped(EXIT_MODEL, f"model violation\n{e}")
        return EXIT_MODEL
    except BudgetExceeded as e:
        print(f"\nSTOP (exit {EXIT_BUDGET}): {e}", flush=True)
        write_stopped(EXIT_BUDGET, f"budget abort\n{e}")
        return EXIT_BUDGET
    except ModelCallFailed as e:
        print(f"\nRETRIES EXHAUSTED (exit {EXIT_RETRIES}): {e}", flush=True)
        return EXIT_RETRIES


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--protocol", action="store_true")
    g.add_argument("--manifest", action="store_true")
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--run", action="store_true")
    ap.add_argument("--cells", default="*", help="glob over v3 cell ids (dry run / run)")
    ap.add_argument("--out", default=None, help="dry run: also write the report here")
    ap.add_argument("--data", default=str(REPO_ROOT / "data" / "locomo" / "locomo10.json"))
    args = ap.parse_args()
    _require_v3_root()
    from core.dataset import load_locomo
    convs = load_locomo(args.data)
    cells = [c for c in plan_cells() if fnmatch.fnmatch(c.cell_id, args.cells)]
    if args.protocol:
        print(f"wrote {write_protocol(plan_cells())}")
        return 0
    if args.manifest:
        print(f"wrote {write_manifest(plan_cells())}")
        return 0
    if args.dry_run:
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cost = dry_run(cells, convs)
        text = buf.getvalue()
        print(text, end="")
        if args.out:
            pathlib.Path(args.out).write_text(text)
        over = [p for p in cost if cost[p] > BUDGET_LIMITS[p]]
        if over:
            print(f"\nSTOP: estimate above the abort limit for {over}")
            return 6
        return 0
    if args.smoke:
        return smoke_test(convs)
    return guarded_run(cells, convs)


if __name__ == "__main__":
    sys.exit(main())
