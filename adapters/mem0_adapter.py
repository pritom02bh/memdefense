# adapters/mem0_adapter.py
#
# Wraps Mem0 behind the MemoryAdapter interface (adapters/base.py).
#
# THE POINT OF THIS FILE: Mem0 makes its own hidden LLM calls internally
# (fact extraction when you .add(), memory-update decisions, etc). Those
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
# IMPORTANT CAVEAT FOR YOU (Pritom): I could not run this against a live
# Mem0 install (my sandbox has no network access to your machine's proxy
# and can't install mem0ai's full dependency chain here). The attribute
# names below (`self._mem.llm`, `.client`, etc.) are my best read of
# Mem0's public source, but Mem0 2.0.x internals could differ slightly
# from what I inspected. If _build_memory() raises the RuntimeError
# below, it means the patch didn't attach -- paste me the error and the
# `dir(self._mem)` / `dir(self._mem.llm)` output it prints, and I'll fix
# the attribute path immediately. This is the same "fail loud, don't
# fail silent" approach that caught the logger.py bug earlier.

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from mem0 import Memory
from openai import OpenAI

from core.purpose import current_purpose, purpose_header
from adapters.base import MemoryAdapter, MemoryRecord, IngestStats, AnswerStats

if TYPE_CHECKING:
    from core.dataset import Conversation

PROXY_BASE_URL = "http://localhost:4000/v1"
TOP_K = 10  # fixed across every framework and every cell -- do not vary this per-run


def _inject_headers_hook(run_id: str, cell_id: str):
    """
    An httpx request hook: stamps every outgoing request with the
    CURRENT purpose (read live off the contextvar, so it reflects
    whatever `with purpose(...):` block is active at call time) plus
    this adapter instance's fixed run_id/cell_id.
    """
    def hook(request):
        request.headers["x-purpose"] = current_purpose()
        request.headers["x-run-id"] = run_id
        request.headers["x-cell-id"] = cell_id
    return hook


def _patch_client_headers(openai_client, run_id: str, cell_id: str) -> bool:
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
        _inject_headers_hook(run_id, cell_id)
    )
    return True


def _find_and_patch(mem: Memory, run_id: str, cell_id: str, candidate_paths: list[str]) -> bool:
    """Try a list of dotted attribute paths (e.g. 'llm.client') until one
    resolves to a patchable openai.OpenAI instance."""
    for path in candidate_paths:
        obj = mem
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if obj is not None and _patch_client_headers(obj, run_id, cell_id):
            return True
    return False


class Mem0Adapter:
    """MemoryAdapter implementation backed by Mem0 (mem0ai>=2.0)."""

    def __init__(self, run_id: str = "dev", cell_id: str = "mem0_dev"):
        self.run_id = run_id
        self.cell_id = cell_id
        self._namespace: str | None = None
        self._mem: Memory = self._build_memory()
        # Separate client for answer() generation -- deliberately NOT
        # Mem0's internal one, so answer-time calls are simple to reason
        # about and always tagged "base" (the default) unless answer()
        # is itself called inside a `with purpose(...):` block.
        self._answer_client = OpenAI(base_url=PROXY_BASE_URL, api_key="proxy-dummy-key")
        _patch_client_headers(self._answer_client, run_id, cell_id)

    def _build_memory(self) -> Memory:
        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "answering",
                    "openai_base_url": PROXY_BASE_URL,
                    "api_key": "proxy-dummy-key",
                    "temperature": 0.0,
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
                    "path": "results/mem0_qdrant",
                    "collection_name": "memdefense",
                },
            },
        }
        mem = Memory.from_config(config)

        patched_llm = _find_and_patch(
            mem, self.run_id, self.cell_id,
            ["llm.client", "llm_provider.client"],
        )
        patched_embed = _find_and_patch(
            mem, self.run_id, self.cell_id,
            ["embedding_model.client", "embedder.client"],
        )

        if not (patched_llm and patched_embed):
            raise RuntimeError(
                "Could not attach header injection to Mem0's internal OpenAI "
                f"clients (llm patched={patched_llm}, embedder patched={patched_embed}).\n"
                "Mem0's internal attribute names likely differ from what this "
                "adapter expects in your installed version (mem0ai 2.0.15).\n"
                "Run this to find the real attribute names, then tell me what "
                "you see so I can fix the paths above:\n\n"
                "    print([a for a in dir(mem) if not a.startswith('_')])\n"
                "    print([a for a in dir(mem.llm) if not a.startswith('_')])\n\n"
                "Do NOT proceed past this error -- an unpatched client means "
                "Mem0's hidden calls bypass the metering proxy and every cost "
                "number downstream is silently wrong."
            )
        return mem

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
        written = 0
        errors = 0
        for turn in conversation.turns:
            try:
                self._mem.add(
                    [{"role": "user", "content": f"[Conversation date: {turn.timestamp}] {turn.speaker}: {turn.text}"}],
                    user_id=self._namespace,
                    metadata={"turn_id": turn.turn_id, "session_id": turn.session_id},
                )
                written += 1
            except Exception:
                errors += 1
        return IngestStats(
            turns_ingested=len(conversation.turns),
            memories_written=written,
            errors=errors,
        )

    def answer(self, question: str) -> tuple[str, AnswerStats]:
        start = time.monotonic()
        hits = self._mem.search(query=question, filters={"user_id": self._namespace}, top_k=TOP_K)
        results = hits.get("results", []) if isinstance(hits, dict) else (hits or [])
        context = "\n".join(f"- {r.get('memory', r.get('text', ''))}" for r in results)

        prompt = (
            "Answer the question using only the memories below. "
            "If the memories don't contain the answer, say you don't know.\n\n"
            f"Memories:\n{context}\n\nQuestion: {question}"
        )
        resp = self._answer_client.chat.completions.create(
            model="answering",
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
            extra_headers=purpose_header(),
        )
        answer_text = resp.choices[0].message.content
        latency_ms = (time.monotonic() - start) * 1000
        return answer_text, AnswerStats(latency_ms=latency_ms, memories_retrieved=len(results))

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