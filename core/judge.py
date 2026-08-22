# core/judge.py
#
# Grades a predicted answer as correct/incorrect against a gold answer.
# Uses the "judge" model alias (gpt-4o) -- deliberately stronger than and
# different from the "answering" model (gpt-4o-mini) to avoid the judge
# rating its own kind of output favorably.
#
# BEFORE THIS IS TRUSTED: run scripts/generate_judge_calibration_set.py,
# hand-grade the output, then scripts/compute_kappa.py must show kappa >= 0.8
# against this exact prompt. If it doesn't, revise prompts/judge.txt and
# re-run calibration -- do not proceed to Component 5 until it passes.
#
# Once approved, prompts/judge.txt is frozen. Editing it after real runs
# begin means cells graded before and after the edit are not comparable.

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass

from openai import OpenAI

from core.purpose import purpose, purpose_header

PROXY_BASE_URL = "http://localhost:4000/v1"
PROMPT_PATH = pathlib.Path(__file__).resolve().parent.parent / "prompts" / "judge.txt"
CACHE_PATH = pathlib.Path(__file__).resolve().parent.parent / "results" / "judge_cache.jsonl"

_client = OpenAI(base_url=PROXY_BASE_URL, api_key="proxy-dummy-key")
_prompt_template = PROMPT_PATH.read_text()

# Hash of the prompt itself. Included in every cache key so that
# editing prompts/judge.txt invalidates cached grades automatically
# instead of silently returning judgments made under the old prompt.
_PROMPT_HASH = hashlib.sha256(_prompt_template.encode("utf-8")).hexdigest()[:16]


@dataclass
class JudgeResult:
    correct: bool
    reason: str
    from_cache: bool


def _cache_key(question: str, gold: str, predicted: str) -> str:
    raw = f"{_PROMPT_HASH}|{question}|{gold}|{predicted}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_cache() -> dict:
    cache = {}
    if CACHE_PATH.exists():
        for line in CACHE_PATH.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cache[row["key"]] = row
    return cache


def _append_cache(key: str, question: str, gold: str, predicted: str, result: JudgeResult) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "a") as f:
        f.write(json.dumps({
            "key": key,
            "question": question,
            "gold": gold,
            "predicted": predicted,
            "correct": result.correct,
            "reason": result.reason,
            "prompt_hash": _PROMPT_HASH,
        }) + "\n")


def judge(question: str, gold: str, predicted: str) -> JudgeResult:
    """
    Grades one (question, gold, predicted) triple. Cached by content hash --
    you will re-grade the same items often (kappa calibration, re-runs after
    a bug fix), and re-paying for identical judgments wastes money and adds
    non-determinism risk from re-sampling the model.
    """
    key = _cache_key(question, gold, predicted)
    cache = _load_cache()
    if key in cache:
        row = cache[key]
        return JudgeResult(correct=row["correct"], reason=row["reason"], from_cache=True)

    prompt = _prompt_template.format(question=question, gold=gold, predicted=predicted)

    with purpose("judge"):
        resp = _client.chat.completions.create(
            # Cross-PROVIDER judge (Anthropic) grading an OpenAI answering
            # model -- a stronger control against self-preference than
            # same-provider judging. Recalibrated 2026-08-08: kappa=0.909
            # on 24 core-4 hand-graded items, vs 0.841 for gpt-4o.
            model="judge-claude",
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
            extra_headers=purpose_header(),
        )

    raw = resp.choices[0].message.content.strip()
    # Models often wrap JSON in markdown fences despite instructions not to.
    # Strip them before parsing; this is formatting noise, not a grading signal.
    if raw.startswith("```"):
        lines = raw.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        raw = "\n".join(lines).strip()
    # Strict parsing on purpose: a parse failure must raise, never silently
    # count as "incorrect" -- that would corrupt accuracy numbers invisibly.
    try:
        parsed = json.loads(raw)
        grade = parsed["grade"].strip().lower()
        reason = parsed.get("reason", "")
    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        raise ValueError(
            f"Judge returned unparseable output, refusing to guess a grade.\n"
            f"Raw response: {raw!r}\n"
            f"Error: {e}"
        )

    if grade not in ("correct", "incorrect"):
        raise ValueError(f"Judge returned unexpected grade value: {grade!r} (raw: {raw!r})")

    result = JudgeResult(correct=(grade == "correct"), reason=reason, from_cache=False)
    _append_cache(key, question, gold, predicted, result)
    return result


if __name__ == "__main__":
    # Quick sanity check -- run with the proxy running.
    r = judge(
        question="When did Caroline go to the LGBTQ support group?",
        gold="7 May 2023",
        predicted="Caroline attended the LGBTQ support group on August 7, 2026.",
    )
    print(f"correct={r.correct}  reason={r.reason}  from_cache={r.from_cache}")
    assert r.correct is False, "This example should be graded incorrect -- wrong date entirely."
    print("Sanity check passed.")