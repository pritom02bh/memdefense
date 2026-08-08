# adapters/base.py
#
# The uniform interface every memory framework (Mem0, Letta) and every
# defense wrapper implements. This is what makes them interchangeable:
# the experiment runner never needs to know which one it's talking to.

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from core.dataset import Conversation


@dataclass
class MemoryRecord:
    """One stored memory, as returned by dump_memories().

    `origin` is required now even though nothing reads it until the
    provenance defense (Component 6). Retrofitting it onto every
    already-written adapter later would mean touching every framework
    integration a second time -- easier to require it from day one.
    Valid values: "user" | "tool" | "web" | "unknown".
    """
    id: str
    text: str
    origin: str
    created_at: str


@dataclass
class IngestStats:
    turns_ingested: int
    memories_written: int
    errors: int = 0


@dataclass
class AnswerStats:
    latency_ms: float
    memories_retrieved: int


class MemoryAdapter(Protocol):
    """Every framework adapter and every defense wrapper implements this.

    Defenses wrap an adapter and expose the identical interface
    (Component 6): `Sanitize(Mem0Adapter())` looks, to the runner,
    exactly like a plain `Mem0Adapter()`.
    """

    def reset(self, namespace: str) -> None:
        """Fully clear this namespace's stored memories before a fresh
        conversation. Must leave nothing behind -- leftover state
        silently corrupts every subsequent result for that namespace."""
        ...

    def ingest(self, conversation: "Conversation") -> IngestStats:
        """Write a conversation into memory, turn by turn (not as one
        blob) -- that's what generates the hidden internal calls this
        whole project measures."""
        ...

    def answer(self, question: str) -> tuple[str, AnswerStats]:
        """Retrieve relevant memories and answer the question from them."""
        ...

    def dump_memories(self) -> list[MemoryRecord]:
        """Return every memory currently stored in this namespace.
        Not optional -- this is how false-quarantine rate gets computed
        and how a defense's blocked items get inspected later."""
        ...
