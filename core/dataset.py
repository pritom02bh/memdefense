# core/dataset.py
#
# Loads LoCoMo into a uniform shape that every adapter consumes.
#
# Structure verified against data/locomo/locomo10.json
# (sha256 79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4):
#
#   top level      : list of 10 conversations
#   conversation[] : sample_id, conversation, qa, observation,
#                    session_summary, event_summary
#   conversation   : speaker_a, speaker_b,
#                    session_<n>          -> list of turns
#                    session_<n>_date_time -> e.g. "1:56 pm on 8 May, 2023"
#   turn           : speaker, dia_id ("D1:1"), text
#                    (+ optional img_url, blip_caption, query, re-download)
#   qa             : question, evidence (list of dia_ids), category (int),
#                    answer  -- EXCEPT category 5, which uses
#                    adversarial_answer instead (444 of 446 have no
#                    "answer" key at all)
#
# Dataset totals: 10 conversations, 5,882 turns, 1,986 QA pairs.

from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Category mapping
# ---------------------------------------------------------------------------
# LoCoMo labels categories as integers with no key in the release. Mapping
# below was inferred empirically from the data itself, not assumed:
#
#   cat 1: mean 3.13 evidence ids, 98% cite multiple turns  -> multi_hop
#   cat 2: "When did X ...", answers are dates              -> temporal
#   cat 3: "Would X likely ...", requires world knowledge   -> open_domain
#   cat 4: mean 1.07 evidence ids, direct factual lookup    -> single_hop
#   cat 5: only category carrying `adversarial_answer`      -> adversarial
#
# Counts: cat1=282, cat2=321, cat3=96, cat4=841, cat5=446 (total 1,986).

CATEGORY_MAP = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}

CATEGORIES = tuple(CATEGORY_MAP.values())

_SESSION_RE = re.compile(r"^session_(\d+)$")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Turn:
    turn_id: str            # LoCoMo dia_id, e.g. "D1:1"
    session_id: str         # "session_1"
    session_num: int        # 1
    speaker: str
    text: str
    timestamp: str | None   # session-level, e.g. "1:56 pm on 8 May, 2023"
    has_image: bool = False
    blip_caption: str | None = None

    def as_message(self) -> str:
        """Plain rendering used when feeding a turn to a memory framework."""
        if self.blip_caption:
            return f"{self.speaker}: {self.text} [shares an image: {self.blip_caption}]"
        return f"{self.speaker}: {self.text}"


@dataclass(frozen=True)
class QAPair:
    qa_id: str              # "<conv_id>::q<index>" -- LoCoMo has no native id
    conv_id: str
    question: str
    answer: str
    category: str           # normalised label from CATEGORY_MAP
    raw_category: int       # original integer, kept for auditing
    evidence: tuple[str, ...]  # dia_ids containing the answer
    is_adversarial: bool


@dataclass(frozen=True)
class Conversation:
    conv_id: str            # LoCoMo sample_id, e.g. "conv-26"
    speaker_a: str
    speaker_b: str
    turns: tuple[Turn, ...]
    qa_pairs: tuple[QAPair, ...]
    session_count: int = 0

    def __post_init__(self):
        object.__setattr__(
            self, "session_count", len({t.session_id for t in self.turns})
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _parse_turns(conv_block: dict) -> list[Turn]:
    """Extract turns from all session_<n> keys, in true chronological order."""
    session_nums = sorted(
        int(m.group(1))
        for k in conv_block
        if (m := _SESSION_RE.match(k))
    )
    # NOTE: sorting by int, not by string. String order would place
    # session_10 before session_2 and silently scramble chronology.

    turns: list[Turn] = []
    for n in session_nums:
        session_key = f"session_{n}"
        timestamp = conv_block.get(f"{session_key}_date_time")
        for raw in conv_block[session_key]:
            turns.append(
                Turn(
                    turn_id=raw["dia_id"],
                    session_id=session_key,
                    session_num=n,
                    speaker=raw["speaker"],
                    text=raw.get("text", ""),
                    timestamp=timestamp,
                    has_image="img_url" in raw,
                    blip_caption=raw.get("blip_caption"),
                )
            )
    return turns


def _parse_qa(raw_qa: list[dict], conv_id: str) -> list[QAPair]:
    """
    Extract QA pairs.

    Critical: category-5 (adversarial) items store the gold answer under
    `adversarial_answer`; 444 of 446 have no `answer` key at all. Reading
    q["answer"] unconditionally raises KeyError on ~22% of the benchmark.
    """
    pairs: list[QAPair] = []
    for i, q in enumerate(raw_qa):
        raw_cat = q.get("category")
        is_adv = "adversarial_answer" in q

        answer = q.get("answer")
        if answer is None:
            answer = q.get("adversarial_answer")

        # 6 of 1,986 answers are ints (e.g. a bare year "2022"). Normalise
        # to str so downstream judging never sees a mixed type.
        answer = "" if answer is None else str(answer)

        pairs.append(
            QAPair(
                qa_id=f"{conv_id}::q{i:04d}",
                conv_id=conv_id,
                question=q.get("question", ""),
                answer=answer,
                category=CATEGORY_MAP.get(raw_cat, f"unknown_{raw_cat}"),
                raw_category=raw_cat,
                evidence=tuple(q.get("evidence") or ()),
                is_adversarial=is_adv,
            )
        )
    return pairs


def load_locomo(path: str | Path) -> list[Conversation]:
    """Load locomo10.json into Conversation objects, deterministically ordered."""
    path = Path(path)
    with open(path) as f:
        raw = json.load(f)

    conversations: list[Conversation] = []
    for sample in raw:
        conv_id = sample["sample_id"]
        conv_block = sample["conversation"]
        conversations.append(
            Conversation(
                conv_id=conv_id,
                speaker_a=conv_block.get("speaker_a", ""),
                speaker_b=conv_block.get("speaker_b", ""),
                turns=tuple(_parse_turns(conv_block)),
                qa_pairs=tuple(_parse_qa(sample.get("qa", []), conv_id)),
            )
        )

    # Deterministic ordering so runs are comparable across machines.
    conversations.sort(key=lambda c: c.conv_id)
    return conversations


def dataset_checksum(path: str | Path) -> str:
    """SHA256 of the dataset file -- record this in versions.md."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def summarise(conversations: list[Conversation]) -> dict:
    from collections import Counter
    cats = Counter(q.category for c in conversations for q in c.qa_pairs)
    return {
        "conversations": len(conversations),
        "turns": sum(len(c.turns) for c in conversations),
        "qa_pairs": sum(len(c.qa_pairs) for c in conversations),
        "sessions": sum(c.session_count for c in conversations),
        "categories": dict(sorted(cats.items())),
    }


if __name__ == "__main__":
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/locomo/locomo10.json"
    convs = load_locomo(data_path)
    stats = summarise(convs)

    print(f"file      : {data_path}")
    print(f"sha256    : {dataset_checksum(data_path)}")
    print()
    print(f"conversations : {stats['conversations']}")
    print(f"sessions      : {stats['sessions']}")
    print(f"turns         : {stats['turns']}")
    print(f"qa pairs      : {stats['qa_pairs']}")
    print()
    print("category distribution:")
    for cat, n in stats["categories"].items():
        print(f"  {cat:12} {n:5}")
    print()
    print("per conversation:")
    for c in convs:
        print(f"  {c.conv_id:10} sessions={c.session_count:3} "
              f"turns={len(c.turns):5} qa={len(c.qa_pairs):4} "
              f"speakers={c.speaker_a}/{c.speaker_b}")

    # integrity checks
    print()
    missing = [q.qa_id for c in convs for q in c.qa_pairs if not q.answer]
    unknown = [q.qa_id for c in convs for q in c.qa_pairs
               if q.category.startswith("unknown_")]
    assert not unknown, f"unmapped categories: {unknown[:5]}"
    print(f"unmapped categories : {len(unknown)}")
    print(f"empty answers       : {len(missing)}")

    # chronology check: session numbers must be non-decreasing within a conv
    for c in convs:
        nums = [t.session_num for t in c.turns]
        assert nums == sorted(nums), f"{c.conv_id}: sessions out of order"
    print("chronology          : ok (sessions non-decreasing in every conversation)")