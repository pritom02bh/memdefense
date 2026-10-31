# tests/test_check_integrity.py
#
# Synthetic-log tests for the final-attempt alignment in
# scripts/check_integrity.py. No files are read.
#
#   python3 -m unittest tests.test_check_integrity -v

import unittest

from scripts.check_integrity import add_calls, align_final_ingestion, classify, stale_context_turns

SIGS = [36, 50, 37, 45, 41]          # per-turn search-embedding token counts
EXPECTED = list(enumerate(SIGS))


class Log:
    """Builds a cell's log rows in order, mem0 add() by add()."""

    def __init__(self):
        self.rows, self.t = [], 0

    def _row(self, **kw):
        self.t += 1
        row = {"ts": f"2026-08-22T00:{self.t // 60:02d}:{self.t % 60:02d}+00:00",
               "purpose": "base", "model_resolved": "gemini-3.1-flash-lite",
               "status": "ok", "tokens_in": 0, **kw}
        self.rows.append(row)

    def turn(self, i, ok=True, fails=0):
        self._row(model_resolved="text-embedding-3-small", tokens_in=SIGS[i])   # search
        for _ in range(fails):
            self._row(status="error")                                            # SDK retries
        if ok:
            self._row(tokens_in=8229 + 60 * i)                                   # extraction
            self._row(model_resolved="text-embedding-3-small", tokens_in=20)     # fact embedding
        return self

    def answer(self):
        self._row(model_resolved="text-embedding-3-small", tokens_in=12)
        self._row(tokens_in=300)
        return self

    def done(self):
        classify(self.rows)
        return self.rows


def _align(rows):
    calls = add_calls(rows)
    out = align_final_ingestion(calls, EXPECTED)
    assert out is not None
    assign, first = out
    lost = [t for t, _ in EXPECTED if not any(c.ok for c in assign.get(t, []))]
    return calls, first, lost


class TestAlignment(unittest.TestCase):

    def test_single_clean_attempt(self):
        log = Log()
        for i in range(5):
            log.turn(i)
        calls, first, lost = _align(log.answer().done())
        self.assertEqual((first, lost), (0, []))
        self.assertEqual(sum(len(c.ok) for c in calls[first:]), 5)

    def test_restart_after_partial_attempt(self):
        log = Log()
        for i in range(3):          # aborted after turn 3, answers nothing
            log.turn(i)
        for i in range(5):          # final attempt
            log.turn(i)
        calls, first, lost = _align(log.answer().done())
        self.assertEqual((first, lost), (3, []))
        self.assertEqual(stale_context_turns(calls[:first], SIGS), [0, 1, 2])

    def test_restart_after_complete_attempt_that_answered(self):
        log = Log()
        for i in range(5):
            log.turn(i)
        log.answer().answer()
        for i in range(5):
            log.turn(i)
        calls, first, lost = _align(log.answer().done())
        self.assertEqual((first, lost), (5, []))

    def test_lost_turn_is_found(self):
        log = Log()
        for i in range(5):
            log.turn(i, ok=(i != 2), fails=3 if i == 2 else 0)
        calls, first, lost = _align(log.answer().done())
        self.assertEqual((first, lost), (0, [2]))

    def test_failed_turn_retried_by_runner_is_not_lost(self):
        log = Log()
        log.turn(0).turn(1)
        log.turn(2, ok=False, fails=3)   # whole-turn retry (old with_backoff)
        log.turn(2).turn(3).turn(4)
        calls, first, lost = _align(log.answer().done())
        self.assertEqual((first, lost), (0, []))

    def test_failed_rows_take_the_step_of_the_next_ok_call(self):
        rows = Log().turn(0, fails=2).answer().done()
        self.assertEqual([r["_step"] for r in rows if r["status"] == "error"],
                         ["extract", "extract"])

    def test_unalignable_log_returns_none(self):
        log = Log()
        for i in (0, 1, 4, 3, 2):     # turns out of order
            log.turn(i)
        self.assertIsNone(align_final_ingestion(add_calls(log.answer().done()), EXPECTED))


if __name__ == "__main__":
    unittest.main()
