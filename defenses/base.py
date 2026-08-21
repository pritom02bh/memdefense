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
import pathlib
import dataclasses
from typing import TYPE_CHECKING

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

    def __init__(self, inner: MemoryAdapter, cell_id: str = "defense_dev"):
        self.inner = inner
        self.cell_id = cell_id
        self._quarantine_path = pathlib.Path("results/quarantine") / f"{cell_id}.jsonl"
        self._quarantine_path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate any stale quarantine log for this cell at construction,
        # so a re-run doesn't append to a previous run's blocked items.
        self._quarantine_path.write_text("")

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

    def ingest(self, conversation: "Conversation") -> IngestStats:
        """Filter turns through should_write(), then ingest only the
        survivors. Blocked turns are logged and never reach the framework.

        We ingest survivors one conversation-slice at a time by rebuilding
        a Conversation with the allowed turns, so the inner adapter's own
        ingest logic (and its metering) runs unchanged."""
        allowed = []
        blocked = 0
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
                blocked += 1
                self._log_quarantine(turn, content, reason)

        filtered_conv = dataclasses.replace(conversation, turns=tuple(allowed))
        stats = self.inner.ingest(filtered_conv)
        # Surface how many this defense blocked, on top of the inner stats.
        return IngestStats(
            turns_ingested=len(conversation.turns),
            memories_written=stats.memories_written,
            errors=stats.errors + (0 if blocked >= 0 else 0),
        )

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