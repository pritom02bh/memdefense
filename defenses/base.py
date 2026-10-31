# defenses/base.py
#
# The base class every defense wrapper extends.
#
# KEY IDEA (Component 6): a defense IS a MemoryAdapter. It wraps another
# adapter (a framework, or another defense) and exposes the identical
# four methods. So to the runner, Sanitize(Mem0Adapter()) is just a
# MemoryAdapter -- it cannot tell a defended cell from an undefended one.
# That's what makes "defense on vs off" a controlled comparison: the ONLY
# thing that changes is this wrapper.
#
# Two kinds of defense, both handled here:
#   WRITE-TIME (sanitize, provenance, anomaly): inspect each turn before
#     it reaches the inner adapter's ingest(). Override should_write().
#   READ-TIME (rerank): inspect retrieved memories before answering.
#     Override filter_retrieved() -- hooked in the mem0 layer for now;
#     see rerank.py for why it lives where it does.
#
# Every blocked item is logged to results/quarantine/{cell_id}.jsonl.
# That log is the raw material for false-quarantine rate (FQR): in a
# BENIGN run, everything a defense blocks is by definition a legitimate
# memory wrongly discarded, so the quarantine log IS the FQR numerator.

from __future__ import annotations

import json
import dataclasses
from collections import Counter
from typing import TYPE_CHECKING

from core.paths import QUARANTINE
from core.purpose import purpose
from adapters.base import MemoryAdapter, MemoryRecord, IngestStats, AnswerStats

if TYPE_CHECKING:
    from core.dataset import Conversation, Turn


class Defense:
    """Base defense wrapper. Delegates everything to the wrapped adapter;
    subclasses override the hooks they need.

    Not declared as `implements MemoryAdapter` in code (Python Protocols
    are structural), but it satisfies the interface: reset/ingest/answer/
    dump_memories are all present with matching signatures.
    """

    name = "base"  # subclasses override; used in cell IDs and logs

    def __init__(self, inner: MemoryAdapter, cell_id: str = "defense_dev",
                 truncate_quarantine: bool = True):
        self.inner = inner
        self.cell_id = cell_id
        self.retries: Counter = Counter()
        self._quarantine_path = QUARANTINE / f"{cell_id}.jsonl"
        self._quarantine_path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate any stale quarantine log for this cell at construction,
        # so a re-run doesn't append to a previous run's blocked items.
        # A RESUMED answer-only cell passes False: its answers.jsonl keeps
        # the earlier rows, so the drops logged for them must stay too.
        if truncate_quarantine:
            self._quarantine_path.write_text("")

    # ---- the framework adapter at the bottom of the chain --------------

    def base_adapter(self):
        """Walk the wrapper chain to the framework adapter. With one
        defense that is self.inner; in the stacked condition it is
        several wrappers down."""
        inner = self.inner
        while isinstance(inner, Defense):
            inner = inner.inner
        return inner

    @property
    def backbone(self) -> str:
        """The cell's backbone alias. Every defense model call uses it,
        so a defense runs on the same model as the system it protects."""
        return self.base_adapter().backbone

    # ---- hooks a subclass overrides -----------------------------------

    def should_write(self, turn: "Turn", content: str) -> tuple[bool, str]:
        """WRITE-TIME decision. Return (allow, reason).
        Default: allow everything (this base class is a no-op defense,
        useful as a control that isolates wrapper overhead from filtering
        effect). reason is logged only when allow is False."""
        return True, ""

    # ---- MemoryAdapter interface --------------------------------------

    def reset(self, namespace: str) -> None:
        self.inner.reset(namespace)

    def _filter(self, conversation: "Conversation") -> tuple["Conversation", list[dict]]:
        """Run should_write() on every turn; return the conversation of
        survivors and the blocked turns. Shared by ingest() and screen(),
        so both see exactly the same turn strings."""
        allowed = []
        blocked = []
        for turn in conversation.turns:
            # MUST match the adapter's ingest format exactly, including the
            # timestamp prefix. Without it the extractor resolves relative
            # dates against today, and every defended cell loses temporal
            # accuracy that would be misattributed to the defense.
            content = f"[Conversation date: {turn.timestamp}] {turn.speaker}: {turn.text}"
            with purpose("defense"):
                ok, reason = self.should_write(turn, content)
            if ok:
                allowed.append(turn)
            else:
                blocked.append({"defense": self.name, "turn_id": turn.turn_id,
                                "reason": reason})
                self._log_quarantine(turn, content, reason)
        return dataclasses.replace(conversation, turns=tuple(allowed)), blocked

    def ingest(self, conversation: "Conversation") -> IngestStats:
        """Filter turns through should_write(), then ingest only the
        survivors. Blocked turns are logged and never reach the framework.

        We ingest survivors one conversation-slice at a time by rebuilding
        a Conversation with the allowed turns, so the inner adapter's own
        ingest logic (and its metering) runs unchanged."""
        filtered_conv, blocked = self._filter(conversation)
        stats = self.inner.ingest(filtered_conv)
        # Surface how many this defense blocked, on top of the inner stats.
        return IngestStats(
            turns_ingested=len(conversation.turns),
            memories_written=stats.memories_written,
            errors=stats.errors,
        )

    def screen(self, conversation: "Conversation") -> list[dict]:
        """WRITE-TIME VERDICTS ONLY: run the conversation through this
        wrapper and every wrapper inside it, in the same order ingest()
        would, but stop before the framework -- nothing is extracted or
        stored. Returns every blocked turn."""
        filtered_conv, blocked = self._filter(conversation)
        if isinstance(self.inner, Defense):
            blocked += self.inner.screen(filtered_conv)
        return blocked

    def answer(self, question: str) -> tuple[str, AnswerStats]:
        return self.inner.answer(question)

    def dump_memories(self) -> list[MemoryRecord]:
        return self.inner.dump_memories()

    # ---- quarantine logging -------------------------------------------

    def _log_quarantine(self, turn: "Turn", content: str, reason: str) -> None:
        with open(self._quarantine_path, "a") as f:
            f.write(json.dumps({
                "defense": self.name,
                "turn_id": turn.turn_id,
                "session_id": turn.session_id,
                "content": content,
                "reason": reason,
            }) + "\n")

    def quarantine_count(self) -> int:
        if not self._quarantine_path.exists():
            return 0
        return sum(1 for line in self._quarantine_path.read_text().splitlines() if line.strip())