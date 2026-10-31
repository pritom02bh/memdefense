# scripts/check_integrity.py
#
# READ-ONLY integrity check of every cell against the proxy log. Exits
# non-zero if any cell violates the protocol. scripts/stats.py and
# runner/analyse.py run it first and stop if it fails.
#
#   python3 -m scripts.check_integrity                  # the manifest's cells, if the root has one
#   python3 -m scripts.check_integrity --cells 'anchor_gemini__*'   # a glob over <root>/cells
#   python3 -m scripts.check_integrity --quiet          # violations + verdict only
#
# For each cell, using only its FINAL attempt:
#   - extraction: one ok extraction per turn that reached the store
#     (turns minus write-time quarantines), all on the backbone, and no
#     lost turn (a turn whose extraction never succeeded -- the old
#     adapter swallowed those);
#   - answers: every ok answer call on the intended answer model, and
#     as many ok answer calls as questions;
#   - defenses: every ok verdict on the backbone; anomaly verdicts ==
#     turns screened, rerank verdicts == memories judged (no decision
#     failed open);
#   - judge: every judge call served by claude-haiku-4-5-20251001;
#   - embeddings: text-embedding-3-small only.
#
# HOW THE FINAL ATTEMPT IS FOUND. A cell re-run under the same id leaves
# every attempt in the log. Each mem0 add() starts with a search
# embedding of the turn's message, whose logged token count depends
# only on the message text. Recomputing those counts from the dataset
# (tiktoken, exactly as mem0 builds the string) maps every extraction
# to its turn, so the final ingestion is found by aligning backwards
# from the cell's last extraction to turn 1. Verified 150/150 on a
# clean cell.
#
# MANIFEST MODE (a root with final_cells.json, see core/manifest.py).
# Every listed cell is checked against the log of the root it lives in
# (results_v3/ or the old results/), and so is every cell that BUILT a
# store a listed cell answers from. On top of the per-cell checks:
#   - store: the cell answered from the namespace the manifest says, and
#     the store's builder extracted every turn, on the backbone, in the
#     period the manifest declares (measured from the builder's final
#     ingestion timestamps: 2026-08 = Aug, 2026-10 = Oct);
#   - an answer-only cell whose write-time chain was re-run (Phase A)
#     blocks exactly the turns its store's builder blocked;
#   - build period: within each conversation every main cell's store
#     was built in one period, the one the manifest declares;
#   - split: every split cell answers from an August store, and each
#     baseline-mini cell from the very store of its old none cell.
#
# STEP CLASSIFICATION. Logs written after the Stage 1 fix carry a step
# tag. Older logs do not; there, base chat-completion calls with >= 2000 input
# tokens are extractions and smaller ones answers, and defense calls
# with >= 108 input tokens are anomaly verdicts and smaller ones rerank
# verdicts. Both cutoffs sit in empty gaps of the observed
# distributions; the histograms that justify them are printed.
# Exception: in an untagged cell that ran BOTH defenses (stacked), a
# rerank verdict on a long memory can exceed 108 tokens, so there the
# phase decides -- anomaly screening happens before the final
# ingestion's first extraction, rerank after its last. The report
# prints how often the size rule would have disagreed.

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import tiktoken
import yaml

from core import manifest as mf
from core.dataset import load_locomo
from core.paths import CELLS, PROXY_LOG, QUARANTINE, REPO_ROOT

BASE_CUTOFF = 2000     # tokens_in: extraction >= cutoff > answer
DEFENSE_CUTOFF = 108   # tokens_in: anomaly verdict >= cutoff > rerank verdict
EMBED_RETRY_WINDOW = 30.0   # seconds
TOP_K = 10
JUDGE_MODEL = "claude-haiku-4-5-20251001"
EMBED_MODEL = "text-embedding-3-small"
DATA = REPO_ROOT / "data" / "locomo" / "locomo10.json"

# DECLARED LEGACY CONFIGURATION. The 15 old rerank cells (results/) ran
# their rerank verdicts and answers on gpt-4o-mini. For them those two
# steps are checked against gpt-4o-mini; every other step is checked as
# normal. (Old anomaly and stacked cells have no such declaration: their
# gpt-4o-mini verdicts remain violations.)
LEGACY = [
    (re.compile(r"^anchor_gemini__gemini-backbone__rerank__conv\d+(__s\d+)?$"),
     {"rerank": "gpt-4o-mini", "answer": "gpt-4o-mini"}),
]

PROXY_TIMEOUT_TEXT = "litellm.Timeout"
ORPHAN_RETRY_WINDOW = 60.0   # seconds from the proxy's timeout error to its own retry


def orphaned_proxy_retry(rs: list[dict]) -> str | None:
    """The proxy timed out a request the client had already abandoned
    and resent (the OpenAI SDK's own timeout and retry, below
    call_with_retry), then retried it on its own after the cell had
    finished. Holds (returns None) when exactly one ok rerank row was
    logged after the cell's last ok answer, at most ORPHAN_RETRY_WINDOW
    seconds after a rerank error carrying litellm's timeout; otherwise
    returns why not."""
    last = max((r["ts"] for r in rs if r["_step"] == "answer" and r["status"] == "ok"),
               default=None)
    if last is None:
        return "no ok answer in the log"
    late = [r for r in rs if r["ts"] > last and r["_step"] in ("rerank", "defense?")]
    ok = [r for r in late if r["status"] == "ok"]
    timeouts = [r for r in late if r["status"] != "ok" and PROXY_TIMEOUT_TEXT in str(r.get("error"))]
    if len(ok) != 1:
        return f"{len(ok)} ok rerank rows after the cell's last answer, expected 1"
    if not any(0 <= _ts(ok[0]) - _ts(e) <= ORPHAN_RETRY_WINDOW for e in timeouts):
        return (f"no rerank error with {PROXY_TIMEOUT_TEXT} within {ORPHAN_RETRY_WINDOW:.0f} s "
                f"before the late ok row ({ok[0]['ts'][:19]})")
    return None


# KNOWN EXCEPTIONS: violations that are understood, recorded in
# results_v3/PROTOCOL.md, and accepted. They are always printed and never
# counted as failures. Matched by cell id and message prefix; an entry
# with a condition is accepted only when condition(the cell's log rows)
# returns None (else the violation stays, with the reason appended).
KNOWN_EXCEPTIONS = [
    ("anchor_gemini__gemini-backbone__rerank__conv2__s3", "rerank ok verdicts 318 != 320",
     "2 verdicts failed open to TRUST after 149 rate-limit / 5xx errors (old code)", None),
    ("anchor_gemini__gemini-backbone__stacked__conv0__v3c", "rerank ok verdicts 751 != 750",
     "one extra ok row from the proxy's own retry of an abandoned request: a rerank call sent "
     "20:40:07 hung; the client's SDK timed out after 600 s and resent it (ok 20:50:09, the "
     "verdict used); litellm timed the original out after 6000 s (error 22:20:07) and retried it "
     "(ok 22:20:09), 79 minutes after the cell's last answer. 750 decisions recorded",
     orphaned_proxy_retry),
]


# ---------------------------------------------------------------------
# models
# ---------------------------------------------------------------------
def alias_models() -> dict[str, str]:
    cfg = yaml.safe_load((REPO_ROOT / "proxy" / "config.yaml").read_text())
    return {m["model_name"]: m["litellm_params"]["model"].split("/")[-1]
            for m in cfg["model_list"]}


# ---------------------------------------------------------------------
# log rows
# ---------------------------------------------------------------------
def _ts(r) -> float:
    return datetime.fromisoformat(r["ts"]).timestamp()


def classify(rows: list[dict]) -> None:
    """Set r["_step"] on every row: extract, answer, anomaly, rerank,
    judge, embedding. Uses the step tag when present, else the size
    cutoffs. Failed calls log 0 tokens, so a failed untagged base call
    takes the step of the next ok base call in its cell (a turn's
    extraction retries, or an answer's retries); a failed untagged
    defense call is marked defense?."""
    for r in rows:
        m = r.get("model_resolved") or ""
        if r.get("step"):
            r["_step"] = r["step"]
        elif "embedding" in m:
            r["_step"] = "embedding"
        elif r.get("purpose") == "judge":
            r["_step"] = "judge"
        elif r.get("purpose") == "defense":
            r["_step"] = (("anomaly" if r["tokens_in"] >= DEFENSE_CUTOFF else "rerank")
                          if r["status"] == "ok" else "defense?")
        else:
            r["_step"] = (("extract" if r["tokens_in"] >= BASE_CUTOFF else "answer")
                          if r["status"] == "ok" else "base?")
    nxt = None
    for r in reversed(rows):
        if r["_step"] in ("extract", "answer") and r["status"] == "ok":
            nxt = r["_step"]
        elif r["_step"] == "base?":
            r["_step"] = nxt or "answer"


def load_log(path) -> tuple[dict[str, list[dict]], list[dict]]:
    by_cell = defaultdict(list)
    judge_rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "ts" not in r or r.get("status") == "logger_error":
                continue
            if r.get("purpose") == "judge":
                judge_rows.append(r)
            if r.get("cell_id"):
                by_cell[r["cell_id"]].append(r)
    for rs in by_cell.values():
        rs.sort(key=_ts)
        classify(rs)
    judge_rows.sort(key=_ts)
    return by_cell, judge_rows


# ---------------------------------------------------------------------
# turn signatures
# ---------------------------------------------------------------------
_ENC = None


def turn_signatures(turns) -> list[int]:
    """Token count of each turn's mem0 search embedding: mem0 2.0.15
    embeds parse_messages(messages) with newlines replaced by spaces."""
    global _ENC
    from mem0.memory.utils import parse_messages
    _ENC = _ENC or tiktoken.get_encoding("cl100k_base")
    out = []
    for t in turns:
        content = f"[Conversation date: {t.timestamp}] {t.speaker}: {t.text}"
        text = parse_messages([{"role": "user", "content": content}]).replace("\n", " ")
        out.append(len(_ENC.encode(text)))
    return out


@dataclass
class AddCall:
    """One mem0 add(): its search embedding and the extraction rows
    that follow it (the SDK's own retries share one search embedding)."""
    sig: int | None
    search: dict | None
    rows: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> list[dict]:
        return [r for r in self.rows if r["status"] == "ok"]


def add_calls(rs: list[dict]) -> list[AddCall]:
    calls, last_embed, cur = [], None, None
    for r in rs:
        if r["_step"] == "embedding":
            last_embed = r
            cur = None
        elif r["_step"] == "extract":
            if cur is None:
                ok_embed = last_embed if last_embed and last_embed["status"] == "ok" else None
                cur = AddCall(sig=ok_embed["tokens_in"] if ok_embed else None, search=last_embed)
                calls.append(cur)
            cur.rows.append(r)
        elif r["_step"] in ("answer", "rerank", "anomaly", "judge", "defense?"):
            cur = None
    return calls


def align_final_ingestion(calls: list[AddCall], expected: list[tuple[int, int]]):
    """Walk backwards from the last add() and match add() calls to the
    expected turns (turn index, signature), last to first.
    Returns (assignment: turn -> [AddCall], index of the first add() of
    the final ingestion, unmatched turns) or None if the log cannot be
    aligned."""
    assign = defaultdict(list)
    j = len(expected) - 1
    i = len(calls) - 1
    while i >= 0 and j >= 0:
        sig = calls[i].sig
        tj, sj = expected[j]
        if sig == sj:
            assign[tj].append(calls[i])
            # a failed add() retried by the runner shows the same
            # signature twice in a row; keep both on this turn
            while (i - 1 >= 0 and calls[i - 1].sig == sj and not calls[i - 1].ok
                   and (j == 0 or expected[j - 1][1] != sj)):
                i -= 1
                assign[tj].append(calls[i])
            i -= 1
            j -= 1
            continue
        # turn j produced no add() (failed before extraction): skip it if
        # the next expected turn matches
        k = next((k for k in range(j - 1, max(j - 4, -1), -1) if expected[k][1] == sig), None)
        if k is None:
            return None
        j = k
    if j >= 0:
        # remaining expected turns have no add() at all (e.g. lost turn 1)
        if i < 0 and all(not assign[t] for t, _ in expected[: j + 1]):
            pass
        else:
            return None
    first = i + 1
    return assign, first


def stale_context_turns(prior: list[AddCall], sigs: list[int]) -> list[int]:
    """Turns whose messages mem0 still held when the final attempt
    started: the last 10 turns that completed an add() in earlier
    attempts (mem0 saves a turn's messages only after a successful
    add(), and delete_all() never clears them). Earlier attempts are
    aligned forwards from each turn-1 start."""
    done, p = [], 0
    for k, c in enumerate(prior):
        if k + 1 < len(prior) and c.sig == sigs[0] and prior[k + 1].sig == sigs[1]:
            p = 0
        if not c.ok:
            continue
        q = next((q for q in range(p, len(sigs)) if sigs[q] == c.sig), None)
        if q is not None:
            done.append(q)
            p = q + 1
    return done[-10:]


# ---------------------------------------------------------------------
# per-cell check
# ---------------------------------------------------------------------
@dataclass
class CellReport:
    cell_id: str
    condition: str
    violations: list[str] = field(default_factory=list)
    known: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)
    # the final attempt's log rows (all steps) and its ok extraction rows,
    # for token accounting (Stage 4); not printed
    final_rows: list = field(default_factory=list, repr=False)
    final_extract: list = field(default_factory=list, repr=False)

    def settle(self, rs: list[dict] = ()) -> "CellReport":
        """Move declared known exceptions out of the violations (those
        with a condition only when it holds on the cell's log rows)."""
        for cid, prefix, why, condition in KNOWN_EXCEPTIONS:
            if cid != self.cell_id:
                continue
            for msg in [m for m in self.violations if m.startswith(prefix)]:
                not_met = condition(list(rs)) if condition else None
                i = self.violations.index(msg)
                if not_met:
                    self.violations[i] = f"{msg} (known-exception condition not met: {not_met})"
                    continue
                del self.violations[i]
                self.known.append(f"{msg} -- KNOWN EXCEPTION: {why}")
        return self


def _quarantine(cell_id, quarantine_dir=None):
    p = (quarantine_dir or QUARANTINE) / f"{cell_id}.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
    write = {r["turn_id"] for r in rows if r.get("phase") != "read" and "turn_id" in r}
    reads = [r for r in rows if r.get("phase") == "read"]
    return write, reads


def check_cell(cell_dir, rs, judge_rows, convs, models, quarantine_dir=None) -> CellReport:
    s = json.loads((cell_dir / "summary.json").read_text())
    cid = s["cell_id"]
    rep = CellReport(cid, s["condition"])
    v = rep.violations.append
    defenses = s.get("defenses") or []
    backbone = models[s["backbone"]]
    answer_alias = s.get("answer_alias") or s["backbone"]
    answer_model = models[answer_alias]
    expect = {"anomaly": backbone, "rerank": backbone}
    for rx, declared in LEGACY:
        if rx.match(cid):
            expect.update({k: v for k, v in declared.items() if k != "answer"})
            answer_model = declared.get("answer", answer_model)
            rep.info["legacy"] = declared
    source_ns = s.get("source_namespace", cid)
    answer_only = source_ns != cid
    retries = Counter(s.get("retries") or {})
    rows = [json.loads(l) for l in (cell_dir / "answers.jsonl").read_text().splitlines() if l.strip()]
    n_q = len(rows)

    conv = convs[s["conv_id"]]
    turns = conv.turns[: s["turns_ingested"]]
    sigs = turn_signatures(turns)
    blocked, read_drops = _quarantine(cid, quarantine_dir)
    expected = [(i, sigs[i]) for i, t in enumerate(turns) if t.turn_id not in blocked]

    # ---------------- ingestion -----------------
    calls = add_calls(rs)
    start_ts = rs[0]["ts"] if rs else None
    ing_end_ts = None
    if answer_only:
        if calls:
            v(f"answer-only cell made {sum(len(c.rows) for c in calls)} extraction calls")
        final_rows = rs
        rep.info.update(ingest_ok=None, ingest_expected=None, lost_turns=[], prior_attempts=0)
    else:
        al = align_final_ingestion(calls, expected) if calls else None
        if al is None:
            v("cannot align the log's extractions with the conversation's turns")
            rep.info.update(ingest_ok=None, ingest_expected=len(expected), lost_turns=[],
                            prior_attempts=None)
            return rep.settle(rs)
        assign, first = al
        final_calls = calls[first:]
        prior = calls[:first]
        lost = [t for t, _ in expected if not any(c.ok for c in assign.get(t, []))]
        okx = [r for c in final_calls for r in c.ok]
        rep.final_extract = okx
        unparseable = retries.get("extract:unparseable", 0)
        rep.info.update(
            ingest_ok=len(okx), ingest_expected=len(expected), lost_turns=lost,
            prior_attempts=sum(1 for k, c in enumerate(prior)
                               if c.sig == sigs[0] and k + 1 < len(prior)
                               and prior[k + 1].sig == sigs[1]) or (1 if prior else 0),
            prior_extraction_calls=len(prior),
            last_prior_ts=prior[-1].rows[-1]["ts"] if prior else None,
            stale_turns=stale_context_turns(prior, sigs) if prior else [],
        )
        if lost:
            v(f"{len(lost)} turn(s) never extracted (unrecovered failures): "
              f"turns {[t + 1 for t in lost]}")
        if len(okx) != len(expected) + unparseable:
            v(f"ok extraction calls {len(okx)} != {len(expected)} expected"
              + (f" + {unparseable} unparseable retries" if unparseable else ""))
        bad = Counter(r["model_resolved"] for r in okx if r["model_resolved"] != backbone)
        if bad:
            v(f"extraction served by {dict(bad)} instead of {backbone}")
        # the final ingestion window: from its first search embedding to
        # the first answer-phase call after its last extraction
        first_row = final_calls[0].search or final_calls[0].rows[0]
        start_ts = first_row["ts"]
        last_x = final_calls[-1].rows[-1]["ts"]
        after = [r for r in rs if r["ts"] > last_x and r["_step"] in ("answer", "rerank", "judge")]
        ing_end_ts = after[0]["ts"] if after else last_x
        # anything before the final ingestion belongs to earlier attempts,
        # except this attempt's own write-time anomaly screening
        prev_end = prior[-1].rows[-1]["ts"] if prior else ""
        final_rows = [r for r in rs if r["ts"] > prev_end]
        if prior:
            # earlier attempts may have answered too: drop their answer phase
            final_rows = [r for r in final_rows
                          if r["ts"] >= start_ts or r["_step"] in ("anomaly", "defense?")]

        if "anomaly" in defenses and "rerank" in defenses:
            flips = 0
            for r in final_rows:
                if r.get("step") or r["_step"] not in ("anomaly", "rerank"):
                    continue
                phase = "anomaly" if r["ts"] < start_ts else "rerank" if r["ts"] > last_x else r["_step"]
                flips += phase != r["_step"]
                r["_step"] = phase
            rep.info["size_rule_disagreements"] = flips

        # embedding failures during the final ingestion. A failure
        # between a turn's search embedding and the next turn's belongs
        # to that turn; if the turn is lost anyway, it costs nothing extra.
        win = [r for r in rs if start_ts <= r["ts"] <= ing_end_ts and r["_step"] == "embedding"]
        failed = [r for r in win if r["status"] != "ok"]
        recovered = sum(
            1 for f in failed
            if any(r["status"] == "ok" and 0 < _ts(r) - _ts(f) <= EMBED_RETRY_WINDOW
                   for r in rs if r["_step"] == "embedding"))
        turn_starts = sorted((c.search or c.rows[0])["ts"] for t in assign for c in assign[t])
        turn_of = {(c.search or c.rows[0])["ts"]: t for t in assign for c in assign[t]}
        in_lost = 0
        for f in failed:
            owner = max((ts for ts in turn_starts if ts <= f["ts"]), default=None)
            if owner is not None and turn_of[owner] in lost:
                in_lost += 1
        rep.info.update(embed_failed=len(failed), embed_followed_ok=recovered,
                        embed_not_followed=len(failed) - recovered, embed_in_lost_turn=in_lost)

    # Only the final write-time pass counts: a restarted screening, and
    # for a re-ingested cell the superseded Phase A screening, came earlier.
    screen_start = (s.get("write_screen") or {}).get("started")
    if screen_start:
        final_rows = [r for r in final_rows if r["_step"] != "anomaly" or r["ts"] >= screen_start]

    # ---------------- defenses -----------------
    an = [r for r in final_rows if r["_step"] == "anomaly"]
    rr = [r for r in final_rows if r["_step"] == "rerank"]
    dq = [r for r in final_rows if r["_step"] == "defense?"]
    for name, group in (("anomaly", an), ("rerank", rr)):
        bad = Counter(r["model_resolved"] for r in group
                      if r["status"] == "ok" and r["model_resolved"] != expect[name])
        if bad:
            v(f"{name} verdicts served by {dict(bad)} instead of {expect[name]}")
    an_ok = sum(r["status"] == "ok" for r in an)
    rr_ok = sum(r["status"] == "ok" for r in rr)
    if "anomaly" in defenses:
        screened = len(turns)
        want = screened + retries.get("anomaly:unparseable", 0)
        if an_ok != want:
            v(f"anomaly ok verdicts {an_ok} != {screened} turns screened"
              + (" (a missing verdict failed open)" if an_ok < screened else ""))
    elif an_ok:
        v(f"{an_ok} anomaly verdicts in a cell without the anomaly defense")
    if "rerank" in defenses:
        cands = [r.get("memories_candidates") for r in rows]
        if all(c is not None for c in cands):
            judged, how = sum(cands), "memories judged"
        else:
            judged, how = TOP_K * n_q, f"{TOP_K} x questions (inferred: old rows lack candidates)"
        want = judged + retries.get("rerank:unparseable", 0)
        if rr_ok != want:
            v(f"rerank ok verdicts {rr_ok} != {judged} {how}"
              + (" (a missing verdict failed open to TRUST)" if rr_ok < judged else ""))
    elif rr_ok:
        v(f"{rr_ok} rerank verdicts in a cell without the rerank defense")

    # ---------------- answers -----------------
    ans = [r for r in final_rows if r["_step"] == "answer"]
    ans_ok = [r for r in ans if r["status"] == "ok"]
    bad = Counter(r["model_resolved"] for r in ans_ok if r["model_resolved"] != answer_model)
    if bad:
        v(f"answers served by {dict(bad)} instead of {answer_model}")
    want = n_q + retries.get("answer:unparseable", 0)
    if len(ans_ok) != want:
        v(f"ok answer calls {len(ans_ok)} != {n_q} questions"
          + (" (answers.jsonl mixes attempts)" if len(ans_ok) < n_q else ""))

    # ---------------- judge -----------------
    tagged = [r for r in final_rows if r["_step"] == "judge"]
    if tagged:
        jr = tagged
    else:   # untagged old logs: judge calls inside this cell's answer window
        lo = ing_end_ts or (ans[0]["ts"] if ans else None)
        hi = ans[-1]["ts"] if ans else None
        jr = [r for r in judge_rows if lo and hi and lo <= r["ts"] <= hi] if lo and hi else []
    bad = Counter(r["model_resolved"] for r in jr if r["model_resolved"] != JUDGE_MODEL)
    if bad:
        v(f"judge calls served by {dict(bad)} instead of {JUDGE_MODEL}")

    # ---------------- embeddings -----------------
    bad = Counter(r["model_resolved"] for r in final_rows
                  if r["_step"] == "embedding" and r["status"] == "ok"
                  and r["model_resolved"] != EMBED_MODEL)
    if bad:
        v(f"embeddings served by {dict(bad)} instead of {EMBED_MODEL}")

    rep.info.update(anomaly_ok=an_ok, rerank_ok=rr_ok, answer_ok=len(ans_ok), questions=n_q,
                    defense_failed=len(dq), judge_calls=len(jr),
                    window=(start_ts, rs[-1]["ts"] if rs else None))
    rep.final_rows = final_rows
    return rep.settle(rs)


# ---------------------------------------------------------------------
# histograms
# ---------------------------------------------------------------------
def print_histograms(by_cell: dict[str, list[dict]], cells: list[str]) -> None:
    base, dfn = [], defaultdict(list)
    for c in cells:
        for r in by_cell.get(c, []):
            if r.get("step") or r["status"] != "ok":
                continue
            if r["_step"] in ("extract", "answer"):
                base.append(r["tokens_in"])
            elif r["_step"] in ("anomaly", "rerank"):
                cond = c.split("__")[2]
                dfn[cond].append(r["tokens_in"])
    if not base and not dfn:
        print("(no untagged rows: size cutoffs not used)")
        return
    edges = [0, 250, 500, 750, 1000, 1500, BASE_CUTOFF, 3000, 4000, 6000, 8000, 10000, 12000, 10**9]
    print(f"\nBase chat-completion calls, untagged, ok -- input-token histogram (cutoff {BASE_CUTOFF}):")
    for a, b in zip(edges, edges[1:]):
        n = sum(a <= x < b for x in base)
        mark = "  <-- cutoff" if b == BASE_CUTOFF else ""
        print(f"  {a:>6}-{(b - 1) if b < 10**9 else '':<6} {n:>6}{mark}")
    below = [x for x in base if x < BASE_CUTOFF]
    above = [x for x in base if x >= BASE_CUTOFF]
    print(f"  largest answer {max(below) if below else '-'}, smallest extraction "
          f"{min(above) if above else '-'}: gap contains the cutoff")
    print(f"\nDefense calls, untagged, ok -- input tokens by condition (cutoff {DEFENSE_CUTOFF}):")
    for cond in sorted(dfn):
        xs = dfn[cond]
        h = Counter((x // 10) * 10 for x in xs)
        print(f"  {cond:9} n={len(xs):>6} min={min(xs)} max={max(xs)}   "
              + " ".join(f"{k}:{h[k]}" for k in sorted(h)))
    an = dfn.get("anomaly", [])
    rr = dfn.get("rerank", [])
    if an and rr:
        print(f"  anomaly-only cells min {min(an)} > {DEFENSE_CUTOFF} > {max(rr)} rerank-only cells max: "
              f"cutoff in the gap; stacked cells split by it")


# ---------------------------------------------------------------------
def run_checks(cell_glob: str = "*", quiet: bool = False, log_path=None,
               cells_dir=None, quarantine_dir=None) -> list[CellReport]:
    """Check every cell under cells_dir (default: this root's) against
    log_path (default: this root's proxy log)."""
    models = alias_models()
    convs = {c.conv_id: c for c in load_locomo(str(DATA))}
    by_cell, judge_rows = load_log(log_path or PROXY_LOG)
    dirs = sorted(d for d in (cells_dir or CELLS).iterdir()
                  if d.is_dir() and fnmatch.fnmatch(d.name, cell_glob)
                  and (d / "summary.json").exists())
    reports = [check_cell(d, by_cell.get(d.name, []), judge_rows, convs, models, quarantine_dir)
               for d in dirs]
    if not quiet:
        print_report(reports, by_cell)
    return reports


# ---------------------------------------------------------------------
# manifest mode
# ---------------------------------------------------------------------
PERIOD_OF_MONTH = {"2026-08": "Aug", "2026-10": "Oct"}
BACKBONE_MODEL = "gemini-3.1-flash-lite"
# October main stores must come from the Oct 30 / Oct 31 Gemini: a store
# whose final ingestion ended before this (the night-of-Oct-29 batch)
# extracts differently (runner/v3.py REBUILD_V3C).
OCT_CUTOFF = "2026-10-30T08:00:00+00:00"
# Mem0 puts the UTC date of each call into every extraction prompt unless
# MEMDEF_PROMPT_DATE pins it (adapters/mem0_adapter.py, PROMPT DATE).
# Every October main store must have been extracted with this date.
OCT_PROMPT_DATE = "2026-10-30"


def prompt_dates(rows: list[dict], summary: dict) -> list[str]:
    """The extraction-prompt date(s) a store was built with: the pinned
    date its builder's summary records, else the UTC date at the start
    of each final-ingestion extraction call (logged time minus latency)."""
    pinned = (summary or {}).get("prompt_date_pinned")
    if pinned:
        return [pinned]
    return sorted({(datetime.fromisoformat(r["ts"])
                    - timedelta(milliseconds=r.get("latency_ms") or 0)).date().isoformat()
                   for r in rows})


def period_of(rows: list[dict]) -> str:
    """Aug / Oct from the extraction timestamps; "mixed" or "unknown"
    when they do not sit in one known month."""
    months = {r["ts"][:7] for r in rows}
    if len(months) > 1:
        return "mixed"
    return PERIOD_OF_MONTH.get(months.pop(), "unknown") if months else "unknown"


@dataclass
class ManifestReport:
    reports: list            # CellReport per manifest cell, in manifest order
    builders: dict           # (root, cell_id) -> CellReport of each store's builder
    store_period: dict       # cell_id -> measured period of the store it answers from
    conv_periods: dict       # conv_id -> sorted measured periods of its main cells
    violations: list         # manifest-level violations
    by_cell: dict            # merged log rows, for the histograms
    store_ingestion: dict    # cell_id -> (start, end) of its store's final ingestion, or None
    store_prompt_dates: dict = field(default_factory=dict)  # cell_id -> its store's prompt date(s)


def manifest_violations(m: dict, reports: list, builders: dict) -> tuple[list, dict, dict, dict, dict]:
    """The store, build-period, October-batch and split rules over
    checked reports. Pure: everything it needs is in the reports and the
    manifest."""
    v = []
    store_period = {}
    store_ingestion = {}
    store_dates = {}
    for r in reports:
        e = r.info["entry"]
        b = e["store_built_by"]
        br = builders.get((b["root"], b["cell_id"]))
        if br is None:
            v.append(f"{e['cell_id']}: store builder {b['root']}/{b['cell_id']} not checked")
            continue
        s = r.info.get("summary") or {}
        used = s.get("source_namespace", e["cell_id"])
        if s and used != e["store_namespace"]:
            v.append(f"{e['cell_id']}: answered from {used}, manifest says {e['store_namespace']}")
        if e["store_namespace"] != b["cell_id"]:
            v.append(f"{e['cell_id']}: store {e['store_namespace']} is not the namespace of its "
                     f"builder {b['cell_id']}")
        x = br.final_extract
        per = period_of(x)
        store_period[e["cell_id"]] = per
        span = (min(r["ts"] for r in x), max(r["ts"] for r in x)) if x else None
        store_ingestion[e["cell_id"]] = span
        if e["role"] == "main" and e["store_period"] == "Oct" and span and span[1] < OCT_CUTOFF:
            v.append(f"OCTOBER BATCH: {e['cell_id']}: its store's final ingestion ended "
                     f"{span[1][:19]}, before {OCT_CUTOFF[:19]} (the night-of-Oct-29 batch)")
        if e["role"] == "main" and e["store_period"] == "Aug" and per != "Aug":
            v.append(f"AUGUST STORE: {e['cell_id']}: its store's final ingestion is not in "
                     f"August ({per})")
        dates = prompt_dates(x, br.info.get("summary")) if x else []
        store_dates[e["cell_id"]] = dates
        if e["role"] == "main" and e["store_period"] == "Oct" and x and dates != [OCT_PROMPT_DATE]:
            v.append(f"OCTOBER PROMPT DATE: {e['cell_id']}: its store was extracted with prompt "
                     f"date(s) {dates}, must be {OCT_PROMPT_DATE}")
        rec_dates = e.get("store_prompt_date")
        if rec_dates is not None and x and rec_dates != dates:
            v.append(f"{e['cell_id']}: manifest records prompt date(s) {rec_dates}, measured {dates}")
        rec = e.get("store_final_ingestion")
        if rec and span and (rec["start"], rec["end"]) != span:
            v.append(f"{e['cell_id']}: manifest records final ingestion {rec['start'][:19]}.."
                     f"{rec['end'][:19]}, the log gives {span[0][:19]}..{span[1][:19]}")
        if per != e["store_period"]:
            v.append(f"{e['cell_id']}: store built in {per} (measured from {b['cell_id']}'s final "
                     f"ingestion), manifest says {e['store_period']}")
        bad = {row["model_resolved"] for row in x} - {BACKBONE_MODEL}
        if bad:
            v.append(f"{e['cell_id']}: store extracted by {sorted(bad)}")
        if br.info.get("lost_turns"):
            v.append(f"{e['cell_id']}: its store lost turns {[t + 1 for t in br.info['lost_turns']]}")
        if br.info.get("ingest_ok") is None:
            v.append(f"{e['cell_id']}: its store's ingestion could not be aligned "
                     f"({b['root']}/{b['cell_id']})")
        # an answer-only cell whose write-time chain was re-run must block
        # exactly what its store's builder blocked
        ws = s.get("write_screen")
        if ws and not s.get("reingested") and br is not r:
            mine = {x["turn_id"] for x in ws.get("blocked") or []}
            theirs = br.info.get("write_blocked", set())
            if mine != theirs:
                v.append(f"{e['cell_id']}: write-time chain blocks {sorted(mine)}, its store's "
                         f"builder blocked {sorted(theirs)}")
    conv_periods = defaultdict(set)
    for r in reports:
        e = r.info["entry"]
        if e["role"] == "main" and e["cell_id"] in store_period:
            conv_periods[e["conv_id"]].add(store_period[e["cell_id"]])
    for conv, ps in sorted(conv_periods.items()):
        if len(ps) != 1:
            v.append(f"BUILD PERIOD: main cells of {conv} answer from stores built in "
                     f"{sorted(ps)} -- must be one period")
        elif m.get("periods") and ps != {m["periods"].get(conv)}:
            v.append(f"BUILD PERIOD: {conv} stores built in {sorted(ps)}, manifest declares "
                     f"{m['periods'].get(conv)}")
    split = [r.info["entry"] for r in reports if r.info["entry"]["role"] == "split"]
    for e in split:
        if store_period.get(e["cell_id"]) != "Aug" or e["store_period"] != "Aug":
            v.append(f"SPLIT: {e['cell_id']} answers from a store built in "
                     f"{store_period.get(e['cell_id'])}; the split must use August stores only")
    for e in split:
        if e["condition"] != "mini":
            continue
        none = [x for x in split if x["condition"] == "none" and x["conv_idx"] == e["conv_idx"]
                and x["seed"] == e["seed"]]
        if len(none) != 1 or e["store_namespace"] != none[0]["store_namespace"]:
            v.append(f"SPLIT: {e['cell_id']} answers from {e['store_namespace']}, not the store of "
                     f"its old none cell {none[0]['cell_id'] if none else '?'}")
    keys = Counter((e["condition"], e["conv_idx"], e["seed"])
                   for e in (r.info["entry"] for r in reports) if e["role"] == "main")
    dup = [k for k, n in keys.items() if n > 1]
    if dup:
        v.append(f"manifest lists more than one main cell for {dup}")
    return (v, store_period, {c: sorted(p) for c, p in conv_periods.items()}, store_ingestion,
            store_dates)


def run_manifest_checks(manifest: dict | None = None) -> ManifestReport:
    """Check every manifest cell, and every store builder, each against
    the log and quarantine of its own root."""
    m = manifest or mf.load()
    models = alias_models()
    convs = {c.conv_id: c for c in load_locomo(str(DATA))}
    logs = {}

    def log(root):
        if root not in logs:
            logs[root] = load_log(mf.root_dir(root) / "proxy" / "calls.jsonl")
        return logs[root]

    def check(root, cell_id, entry=None):
        by_cell, judge_rows = log(root)
        d = mf.root_dir(root) / "cells" / cell_id
        if not (d / "summary.json").exists():
            rep = CellReport(cell_id, entry["condition"] if entry else "?")
            rep.violations.append("not run: no summary.json")
            rep.info.update(summary={}, write_blocked=set(), entry=entry)
            return rep
        rep = check_cell(d, by_cell.get(cell_id, []), judge_rows, convs, models,
                         mf.root_dir(root) / "quarantine")
        rep.info["summary"] = json.loads((d / "summary.json").read_text())
        rep.info["write_blocked"] = _quarantine(cell_id, mf.root_dir(root) / "quarantine")[0]
        if entry is not None:
            rep.info["entry"] = entry
        return rep

    reports = [check(e["root"], e["cell_id"], e) for e in m["cells"]]
    builders = {(r.info["entry"]["root"], r.cell_id): r for r in reports}
    for e in m["cells"]:
        b = e["store_built_by"]
        if (b["root"], b["cell_id"]) not in builders:
            builders[(b["root"], b["cell_id"])] = check(b["root"], b["cell_id"])
    viol, store_period, conv_periods, store_ingestion, store_dates = manifest_violations(
        m, reports, builders)
    by_cell = {}
    for root in logs:
        by_cell.update(logs[root][0])
    return ManifestReport(reports, builders, store_period, conv_periods, viol, by_cell,
                          store_ingestion, store_dates)


def print_manifest_report(mr: ManifestReport) -> None:
    print("\nMANIFEST: store, build-period and split checks")
    print(f"  cells checked: {len(mr.reports)} "
          f"({sum(r.info['entry']['role'] == 'main' for r in mr.reports)} main, "
          f"{sum(r.info['entry']['role'] == 'split' for r in mr.reports)} split); "
          f"store builders checked: {len(mr.builders)}")
    print("  build period of the main cells' stores, per conversation (measured):")
    for conv, ps in mr.conv_periods.items():
        print(f"    {conv}: {', '.join(ps)}")
    split = [r for r in mr.reports if r.info["entry"]["role"] == "split"]
    sp = Counter(mr.store_period.get(r.cell_id) for r in split)
    print(f"  split cells' stores (measured): {dict(sp)}")
    print_prompt_dates(mr)
    print(f"  manifest-level violations: {len(mr.violations)}")
    for msg in mr.violations:
        print(f"    {msg}")


def print_prompt_dates(mr: ManifestReport) -> None:
    """Extraction-prompt dates of the main cells' stores: October must be
    OCT_PROMPT_DATE (a violation otherwise, see manifest_violations);
    August is reported per conversation and replicate, never a failure."""
    main = [r.info["entry"] for r in mr.reports if r.info["entry"]["role"] == "main"]
    oct_dates = Counter(tuple(mr.store_prompt_dates.get(e["cell_id"]) or ["?"])
                        for e in main if e["store_period"] == "Oct")
    print(f"  prompt date(s) of the October main stores (must be {OCT_PROMPT_DATE}): "
          + ", ".join(f"{'+'.join(d)} x{n}" for d, n in sorted(oct_dates.items())))
    print("  prompt dates of the August main stores, per conversation and replicate "
          "(reported, not checked):")
    by = defaultdict(dict)
    for e in main:
        if e["store_period"] == "Aug":
            by[(e["conv_id"], e["seed"])][e["condition"]] = mr.store_prompt_dates.get(e["cell_id"]) or ["?"]
    for (conv, seed), d in sorted(by.items()):
        allds = sorted({x for ds in d.values() for x in ds})
        print(f"    {conv} s{seed}: {'one date' if len(allds) == 1 else f'{len(allds)} dates'}  "
              + "  ".join(f"{c}={'+'.join(ds)}" for c, ds in sorted(d.items())))


def print_report(reports, by_cell) -> None:
    print(f"INTEGRITY CHECK  cells={CELLS}  log={PROXY_LOG}")
    print_histograms(by_cell, [r.cell_id for r in reports])

    short = lambda c: c.replace("anchor_gemini__gemini-backbone__", "")
    print("\nPer cell (final attempt):")
    print(f"  {'cell':30} {'ingest ok/exp':>13} {'lost':>4} {'prior':>5} "
          f"{'anom ok':>7} {'rerank ok':>9} {'ans ok/q':>8} {'viol':>4}")
    for r in reports:
        i = r.info
        ing = (f"{i['ingest_ok']}/{i['ingest_expected']}" if i.get("ingest_ok") is not None
               else "-" if i.get("ingest_expected") is None else f"?/{i['ingest_expected']}")
        print(f"  {short(r.cell_id):30} {ing:>13} {len(i.get('lost_turns') or []):>4} "
              f"{i.get('prior_attempts') if i.get('prior_attempts') is not None else '?':>5} "
              f"{i.get('anomaly_ok', '-'):>7} {i.get('rerank_ok', '-'):>9} "
              f"{str(i.get('answer_ok', '-')) + '/' + str(i.get('questions', '-')):>8} "
              f"{len(r.violations):>4}")

    print("\nUnrecovered extraction failures (expected turns - ok extractions, final ingestion):")
    any_lost = False
    for r in reports:
        lost = r.info.get("lost_turns") or []
        if lost:
            any_lost = True
            print(f"  {short(r.cell_id):30} {len(lost)} lost: turns {[t + 1 for t in lost]}")
    if not any_lost:
        print("  none")

    print(f"\nEmbedding failures during the final ingestion "
          f"(followed = an ok embedding in the same cell within {EMBED_RETRY_WINDOW:.0f}s):")
    print(f"  {'cell':30} {'failed':>6} {'followed':>8} {'not followed':>12}"
          f" {'of which in a lost turn':>24}")
    for r in reports:
        i = r.info
        if "embed_failed" in i:
            print(f"  {short(r.cell_id):30} {i['embed_failed']:>6} {i['embed_followed_ok']:>8} "
                  f"{i['embed_not_followed']:>12} {i['embed_in_lost_turn']:>24}")
    print("  (a call that never reached the proxy leaves no row, so failures of that kind are"
          " not countable here; lost turns are, because they lack an ok extraction)")

    prior = [r for r in reports if r.info.get("prior_attempts")]
    print("\nCells re-ingested under the same id (mem0 kept the earlier attempt's last 10 "
          "messages, which entered the final attempt's first extractions):")
    for r in prior:
        i = r.info
        st = i.get("stale_turns") or []
        print(f"  {short(r.cell_id):30} prior attempts {i['prior_attempts']}, "
              f"{i['prior_extraction_calls']} earlier add() calls, last at {i['last_prior_ts'][:19]}; "
              f"stale context = turns {[t + 1 for t in st]}")
    if not prior:
        print("  none")

    dis = {r.cell_id: r.info["size_rule_disagreements"] for r in reports
           if r.info.get("size_rule_disagreements")}
    print(f"\nStacked cells where the {DEFENSE_CUTOFF}-token rule disagrees with the phase "
          f"(phase used): {sum(dis.values())} calls in {len(dis)} cells "
          + ", ".join(f"{short(c)}={n}" for c, n in dis.items()))

    legacy = [r for r in reports if r.info.get("legacy")]
    print(f"\nDeclared legacy configuration applied to {len(legacy)} cells "
          f"(rerank verdicts and answers checked against gpt-4o-mini)"
          + (": " + ", ".join(short(r.cell_id) for r in legacy) if legacy else ""))
    print_known(reports)

    bad = [r for r in reports if r.violations]
    print(f"\nVIOLATIONS: {sum(len(r.violations) for r in bad)} in {len(bad)} of {len(reports)} cells")
    by_cond = defaultdict(list)
    for r in bad:
        by_cond[r.condition].append(r)
    for cond in sorted(by_cond):
        print(f"\n  [{cond}]")
        for r in by_cond[cond]:
            for msg in r.violations:
                print(f"    {short(r.cell_id):30} {msg}")


def print_known(reports) -> None:
    known = [(r.cell_id, m) for r in reports for m in r.known]
    print(f"\nKNOWN EXCEPTIONS ({len(known)}, accepted, recorded in PROTOCOL.md):")
    for cid, m in known:
        print(f"  {cid}: {m}")
    if not known:
        print("  none in these cells")


def require_integrity(cell_glob: str = "*") -> None:
    """For analysis scripts: run the check quietly; on any violation,
    print them and exit. Known exceptions are always printed. A root
    with a manifest is checked in manifest mode."""
    mviol = []
    if mf.MANIFEST.exists():
        mr = run_manifest_checks()
        reports, mviol = mr.reports, mr.violations
    elif mf.required():
        sys.exit(f"no manifest at {mf.MANIFEST}; stopping")
    else:
        reports = run_checks(cell_glob, quiet=True)
    if any(r.known for r in reports):
        print_known(reports)
    bad = [r for r in reports if r.violations]
    if mviol:
        print(f"INTEGRITY CHECK FAILED: {len(mviol)} manifest violations")
        for msg in mviol:
            print(f"  {msg}")
    if bad or mviol:
        print(f"INTEGRITY CHECK FAILED: {sum(len(r.violations) for r in bad)} violations in "
              f"{len(bad)} of {len(reports)} cells under {CELLS}.")
        for r in bad:
            for msg in r.violations:
                print(f"  {r.cell_id}: {msg}")
        print("Run python3 -m scripts.check_integrity for the full report. Stopping.")
        sys.exit(1)
    print(f"integrity check passed: {len(reports)} cells"
          + (f" in {mf.MANIFEST}" if mf.MANIFEST.exists() else f" under {CELLS}"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=None,
                    help="glob over the cell ids under <root>/cells (default: the manifest's "
                         "cells if the root has one, else every cell)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    mviol = []
    if args.cells is None and mf.MANIFEST.exists():
        mr = run_manifest_checks()
        reports, mviol = mr.reports, mr.violations
        if not args.quiet:
            print_report(reports, mr.by_cell)
        print_manifest_report(mr)
    else:
        reports = run_checks(args.cells or "*", quiet=args.quiet)
    bad = [r for r in reports if r.violations]
    if args.quiet:
        print_known(reports)
        for r in bad:
            for msg in r.violations:
                print(f"{r.cell_id}: {msg}")
    fail = bool(bad or mviol)
    print(f"\nRESULT: {'FAIL' if fail else 'PASS'} ({len(bad)} of {len(reports)} cells with "
          f"violations" + (f"; {len(mviol)} manifest violations" if mviol else "") + ")")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
