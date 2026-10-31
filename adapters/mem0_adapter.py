# adapters/mem0_adapter.py
#
# Wraps Mem0 behind the MemoryAdapter interface (adapters/base.py).
#
# THE POINT OF THIS FILE: Mem0 makes its own hidden LLM calls internally
# (fact extraction on every .add(), memory-update decisions, etc). Those
# calls must go through the metering proxy at http://localhost:4000 so
# they show up, tagged, in results/proxy/calls.jsonl. If they don't,
# Component 3's blocking test fails and none of the paper's cost numbers
# are trustworthy.
#
# WHY WE DON'T USE MEM0's `extra_headers` CONFIG KEY:
# Mem0 supports a static `extra_headers` dict in its LLM config, but it
# is set ONCE at construction and never changes. Our purpose tag has to
# flip between "base" and "defense" *within the same adapter instance*,
# call to call (core.purpose is a contextvar for exactly this reason).
# So instead we reach into the underlying openai.OpenAI client that Mem0
# builds internally and attach an httpx event hook that reads the
# CURRENT purpose at the moment each request goes out. This works no
# matter what Mem0's config schema does or doesn't support, and survives
# Mem0 version bumps -- it doesn't depend on Mem0 shipping a feature.
#
# The attribute paths below (`self._mem.llm.client`,
# `self._mem.embedding_model.client`) are Mem0 internals, verified against
# mem0ai 2.0.15. If a Mem0 upgrade moves them, _build_memory() raises
# instead of running with unmetered calls: an unpatched client would
# bypass the proxy and silently zero the cost numbers.

from __future__ import annotations

import json
import os
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import TYPE_CHECKING

# mem0 sends usage telemetry to PostHog unless this is off, and reads it
# once at import. Off by default here: a measurement run should make no
# outbound calls except through the metering proxy. Set
# MEM0_TELEMETRY=True explicitly to opt back in.
os.environ.setdefault("MEM0_TELEMETRY", "False")

from mem0 import Memory
from mem0.memory.utils import extract_json, remove_code_blocks

from core.answer import generate_answer
from core.paths import QDRANT, REPO_ROOT, RESULTS
from core.proxy_client import PROXY_BASE_URL, no_sdk_retries, proxy_client
from core.purpose import current_purpose, current_step, purpose_header, step
from core.retry import ModelCallFailed, UnparseableResponse, call_with_retry
from adapters.base import MemoryAdapter, MemoryRecord, IngestStats, AnswerStats

if TYPE_CHECKING:
    from core.dataset import Conversation

TOP_K = 10  # fixed across every framework and every cell -- do not vary this per-run


class ExtractionFailed(ModelCallFailed):
    """A turn's extraction did not complete. Aborts the cell: a missing
    turn is a missing memory, which looks exactly like a real result."""


def _inject_headers_hook(run_id: str, cell_id: str, fixed_step: str | None = None):
    """
    An httpx request hook: stamps every outgoing request with the
    CURRENT purpose and step (read live off the contextvars, so they
    reflect whatever `with purpose(...)` / `with step(...)` block is
    active at call time) plus this adapter instance's fixed
    run_id/cell_id. The embedder passes fixed_step="embedding".
    """
    def hook(request):
        request.headers["x-purpose"] = current_purpose()
        request.headers["x-run-id"] = run_id
        request.headers["x-cell-id"] = cell_id
        s = fixed_step or current_step()
        if s:
            request.headers["x-step"] = s
    return hook


def _patch_client_headers(openai_client, run_id: str, cell_id: str,
                          fixed_step: str | None = None) -> bool:
    """
    Reaches into an openai.OpenAI instance's private httpx.Client and
    appends our header hook to its request event hooks. Returns False
    if this client's internals don't match the expected shape, so the
    caller can fail loudly instead of silently leaving calls unmetered.
    """
    httpx_client = getattr(openai_client, "_client", None)
    if httpx_client is None or not hasattr(httpx_client, "event_hooks"):
        return False
    httpx_client.event_hooks.setdefault("request", []).append(
        _inject_headers_hook(run_id, cell_id, fixed_step)
    )
    return True


def _find_and_patch(mem: Memory, run_id: str, cell_id: str, candidate_paths: list[str],
                    fixed_step: str | None = None) -> bool:
    """Try a list of dotted attribute paths (e.g. 'llm.client') until one
    resolves to a patchable openai.OpenAI instance."""
    for path in candidate_paths:
        obj = mem
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if obj is not None and _patch_client_headers(obj, run_id, cell_id, fixed_step):
            return True
    return False


def _check_extraction_json(response) -> None:
    """Parse the extraction response the way mem0 2.0.15 does
    (memory/main.py, _add_to_vector_store). mem0 turns a parse failure
    -- or an empty completion, e.g. Gemini spending its token budget on
    reasoning -- into "no facts extracted" and carries on. Here it is a
    failed call, retried and then fatal. A genuine "nothing to remember"
    is {"memory": []}, which passes."""
    text = remove_code_blocks(response or "")
    if not text.strip():
        raise UnparseableResponse("empty extraction response")
    try:
        try:
            parsed = json.loads(text, strict=False)
        except json.JSONDecodeError:
            parsed = json.loads(extract_json(text), strict=False)
    except Exception as e:
        raise UnparseableResponse(f"extraction JSON unparseable: {e}") from e
    if not isinstance(parsed, dict) or not isinstance(parsed.get("memory"), list):
        raise UnparseableResponse("extraction JSON has no 'memory' list")


class Mem0Adapter:
    """MemoryAdapter implementation backed by Mem0 (mem0ai>=2.0)."""

    def __init__(self, run_id: str = "dev", cell_id: str = "mem0_dev",
                 backbone: str = "answering", answer_alias: str | None = None,
                 memory: Memory | None = None, answer_client=None):
        # `backbone` is the proxy ALIAS this cell's agent runs on. It
        # drives BOTH memory extraction (Mem0's internal LLM) and
        # answering, because a "backbone" that only answers is not
        # really the system's backbone -- extraction quality is a large
        # part of what a memory system is. Defaults to "answering" so
        # every existing call site keeps its previous behaviour.
        #
        # `answer_alias` overrides the answer model only (default: the
        # backbone). Used by the baseline-mini cells, which answer with
        # gpt-4o-mini from a store the backbone extracted.
        #
        # `memory` / `answer_client` are for tests: a stub store and a
        # stub client, so no Qdrant folder or proxy is touched.
        self.run_id = run_id
        self.cell_id = cell_id
        self.backbone = backbone
        self.answer_alias = answer_alias or backbone
        self._namespace: str | None = None
        self.retries: Counter = Counter()
        # Failures mem0 caught and swallowed internally (see
        # _install_guards). Checked after every add().
        self._swallowed: list[str] = []
        self._mem: Memory = memory if memory is not None else self._build_memory()
        self._install_guards(self._mem)
        # Separate client for answer() generation -- deliberately NOT
        # Mem0's internal one, so answer-time calls are simple to reason
        # about and always tagged "base" (the default) unless answer()
        # is itself called inside a `with purpose(...):` block.
        if answer_client is None:
            answer_client = proxy_client()
            _patch_client_headers(answer_client, run_id, cell_id)
        self._answer_client = answer_client

    def build_config(self) -> dict:
        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": self.backbone,
                    "openai_base_url": PROXY_BASE_URL,
                    "api_key": "proxy-dummy-key",
                    "temperature": 0.0,
                    # Mem0 defaults to a low max_tokens. Gemini emits
                    # reasoning tokens before its JSON, so the default
                    # budget gets exhausted mid-object and the fact
                    # extraction is silently dropped for that turn.
                    "max_tokens": 4000,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "embedding",
                    "openai_base_url": PROXY_BASE_URL,
                    "api_key": "proxy-dummy-key",
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "path": str(QDRANT),
                    "collection_name": "memdefense",
                },
            },
        }
        # mem0 keeps each namespace's last 10 messages in a SQLite file
        # (default ~/.mem0/history.db, shared by every run on the machine)
        # and puts them into every extraction prompt; delete_all() never
        # clears them. Any root other than the original "results" gets
        # its own file inside that root, so a run is self-contained and
        # every new namespace starts with an empty history.
        if RESULTS != REPO_ROOT / "results":
            config["history_db_path"] = str(RESULTS / "mem0_history.db")
        return config

    def _build_memory(self) -> Memory:
        mem = Memory.from_config(self.build_config())

        patched_llm = _find_and_patch(
            mem, self.run_id, self.cell_id,
            ["llm.client", "llm_provider.client"],
        )
        patched_embed = _find_and_patch(
            mem, self.run_id, self.cell_id,
            ["embedding_model.client", "embedder.client"],
            fixed_step="embedding",
        )

        if not (patched_llm and patched_embed):
            raise RuntimeError(
                "Could not attach header injection to Mem0's internal OpenAI "
                f"clients (llm patched={patched_llm}, embedder patched={patched_embed}).\n"
                "Mem0's internal attribute names likely differ from the ones this "
                "adapter expects (verified on mem0ai 2.0.15).\n"
                "List the real attribute names and update the paths in "
                "_build_memory():\n\n"
                "    print([a for a in dir(mem) if not a.startswith('_')])\n"
                "    print([a for a in dir(mem.llm) if not a.startswith('_')])\n\n"
                "Do NOT proceed past this error -- an unpatched client means "
                "Mem0's hidden calls bypass the metering proxy and every cost "
                "number downstream is silently wrong."
            )
        # mem0 builds these clients itself: no SDK retries either, so
        # every retry is the guards' (counted in self.retries)
        no_sdk_retries(mem.llm.client)
        no_sdk_retries(mem.embedding_model.client)
        return mem

    def _install_guards(self, mem) -> None:
        """Wrap mem0's own LLM and embedder so every call it makes is
        retried with backoff, and a call that still fails cannot vanish.

        mem0 2.0.15 raises when the extraction call itself fails, but
        quietly drops data in two other places: an unparseable
        extraction becomes "no facts", and a failed embedding of an
        extracted fact is logged and skipped. The guards make the first
        a retried failure, and record the second in self._swallowed so
        ingest() can abort even though mem0 caught the exception.

        embed_batch failures are not recorded: mem0 falls back to
        per-text embed(), which is guarded on its own."""
        llm_generate = mem.llm.generate_response
        embed = mem.embedding_model.embed
        embed_batch = mem.embedding_model.embed_batch

        def extract(*args, **kwargs):
            def once():
                out = llm_generate(*args, **kwargs)
                _check_extraction_json(out)
                return out
            return self._guarded(once, "extract")

        mem.llm.generate_response = extract
        mem.embedding_model.embed = (
            lambda *a, **kw: self._guarded(lambda: embed(*a, **kw), "embedding"))
        mem.embedding_model.embed_batch = (
            lambda *a, **kw: self._guarded(lambda: embed_batch(*a, **kw), "embedding",
                                           record=False))

    def _guarded(self, fn, step_name: str, record: bool = True):
        try:
            return call_with_retry(fn, f"{step_name} ({self.cell_id})",
                                   step=step_name, retries=self.retries)
        except ModelCallFailed as e:
            if record:
                self._swallowed.append(str(e))
            raise

    def open_namespace(self, namespace: str, *, read_only: bool) -> None:
        """Point the adapter at an EXISTING namespace without clearing it.
        With read_only=True every mem0 write path (add, update, delete,
        delete_all, reset) raises, so answering from an already-built
        store cannot change it."""
        self._namespace = namespace
        if read_only:
            def refuse(*args, **kwargs):
                raise PermissionError(f"namespace {namespace!r} is opened read-only")
            for name in ("add", "update", "delete", "delete_all", "reset"):
                if hasattr(self._mem, name):
                    setattr(self._mem, name, refuse)

    def message_history(self, namespace: str) -> list:
        """The messages mem0 will prepend to this namespace's next
        extraction (its last 10 saved messages)."""
        return self._mem.db.get_last_messages(f"user_id={namespace}", limit=10)

    def clear_message_history(self, namespace: str) -> None:
        """Delete this namespace's saved messages. mem0 2.0.15 has no API
        for it (delete_all() leaves them), so this is one SQL statement
        on mem0's own table, under mem0's own lock. Only ever called on a
        re-ingested __v3 namespace (runner/v3.py:prepare_reingest)."""
        db = self._mem.db
        with db._lock:
            db.connection.execute("DELETE FROM messages WHERE session_scope = ?",
                                  (f"user_id={namespace}",))
            db.connection.commit()

    def close(self) -> None:
        """Release the local Qdrant folder and mem0's SQLite file. Local
        Qdrant allows one client per folder per process, so a runner that
        goes cell by cell must close each adapter before the next."""
        client = getattr(getattr(self._mem, "vector_store", None), "client", None)
        if client is not None:
            client.close()
        db = getattr(self._mem, "db", None)
        if db is not None and hasattr(db, "close"):
            db.close()

    def reset(self, namespace: str) -> None:
        self._namespace = namespace
        # NOTE: delete_all kept the OLD signature (user_id=...), unlike
        # search/get_all which moved to filters={...}. mem0 2.0.15 renamed
        # the param on two of three methods. Verified via inspect.signature.
        self._mem.delete_all(user_id=namespace)
        # Verify the clear actually happened. delete_all failed SILENTLY for
        # the entire project until 2026-08-08 (wrong param shape swallowed by
        # a bare except), so reset() must prove itself every time.
        remaining = len(self.dump_memories())
        if remaining != 0:
            raise RuntimeError(
                f"reset('{namespace}') left {remaining} memories behind. "
                "delete_all is not clearing -- check its parameter shape "
                "against inspect.signature(Memory.delete_all)."
            )

    def ingest(self, conversation: "Conversation") -> IngestStats:
        if self._namespace is None:
            self.reset(conversation.conv_id)
        # No try/except here, on purpose. This loop used to count a
        # failed add() as an "error" and move on, which silently dropped
        # the turn. Every model call inside add() is already retried by
        # the guards; whatever still fails aborts the cell.
        written = 0
        with step("extract"):
            for turn in conversation.turns:
                self._swallowed.clear()
                self._mem.add(
                    [{"role": "user", "content": f"[Conversation date: {turn.timestamp}] {turn.speaker}: {turn.text}"}],
                    user_id=self._namespace,
                    metadata={"turn_id": turn.turn_id, "session_id": turn.session_id},
                )
                if self._swallowed:
                    raise ExtractionFailed(
                        f"turn {turn.turn_id}: mem0 swallowed a failed call: "
                        f"{self._swallowed[0]}")
                written += 1
        return IngestStats(
            turns_ingested=len(conversation.turns),
            memories_written=written,
            errors=0,
        )

    def retrieve(self, question: str) -> list[str]:
        """Top-k memory texts for a question. Shared by answer() and
        Rerank.answer(), so both see the same retrieval."""
        hits = self._mem.search(query=question, filters={"user_id": self._namespace}, top_k=TOP_K)
        results = hits.get("results", []) if isinstance(hits, dict) else (hits or [])
        return [r.get("memory", r.get("text", "")) for r in results]

    def answer(self, question: str) -> tuple[str, AnswerStats]:
        start = time.monotonic()
        memories = self.retrieve(question)
        # cell/run ids are sent explicitly as well as by the client hook,
        # so attribution does not depend on the hook having attached.
        answer_text = generate_answer(
            self._answer_client, self.answer_alias, memories, question,
            extra_headers={**purpose_header(), "x-cell-id": self.cell_id,
                           "x-run-id": self.run_id},
            retries=self.retries, label=f"answer ({self.cell_id})")
        latency_ms = (time.monotonic() - start) * 1000
        return answer_text, AnswerStats(latency_ms=latency_ms, memories_retrieved=len(memories),
                                        memories_candidates=len(memories))

    def dump_memories(self) -> list[MemoryRecord]:
        # top_k, not limit: mem0 2.0 silently ignores unknown kwargs, and the
        # default page of 20 masked the real store size for two days.
        raw = self._mem.get_all(filters={"user_id": self._namespace}, top_k=100000)
        items = raw.get("results", []) if isinstance(raw, dict) else (raw or [])
        out = []
        for item in items:
            out.append(MemoryRecord(
                id=str(item.get("id", uuid.uuid4())),
                text=item.get("memory", item.get("text", "")),
                origin="unknown",
                created_at=item.get("created_at") or datetime.now(timezone.utc).isoformat(),
            ))
        return out


if __name__ == "__main__":
    # Component 3 blocking test. Run this with the proxy already running
    # in another terminal:
    #
    #   python3 -m adapters.mem0_adapter data/locomo/locomo10.json
    #
    # PASS means:
    #   1. no exception during ingest
    #   2. answer text prints and looks plausible
    #   3. dump_memories() is non-empty
    #   4. (check separately, see instructions after this runs)
    #      results/proxy/calls.jsonl has MANY more lines than 1 --
    #      Mem0's hidden fact-extraction/update calls must appear.
    import sys
    from core.dataset import load_locomo

    path = sys.argv[1] if len(sys.argv) > 1 else "data/locomo/locomo10.json"
    convs = load_locomo(path)
    conv = convs[0]
    print(f"Ingesting conversation {conv.conv_id}: {len(conv.turns)} turns...")

    adapter = Mem0Adapter(run_id="dev", cell_id="mem0_component3_test")
    stats = adapter.ingest(conv)
    print(f"Ingest stats: {stats}")

    question = conv.qa_pairs[0].question
    gold = conv.qa_pairs[0].answer
    print(f"\nQuestion: {question}")
    print(f"Gold answer: {gold}")
    answer_text, answer_stats = adapter.answer(question)
    print(f"Mem0 answer: {answer_text}")
    print(f"Answer stats: {answer_stats}")

    memories = adapter.dump_memories()
    print(f"\ndump_memories() returned {len(memories)} records.")
    if memories:
        print(f"Sample: {memories[0]}")

    print(
        "\nNow check: wc -l results/proxy/calls.jsonl\n"
        "It must show substantially more than 1 line. If it shows only 1,\n"
        "Mem0 is bypassing the proxy -- stop and report back before continuing."
    )