# core/retry.py
#
# One retry policy for every model call in a measurement run.
#
# RULE: a model call either succeeds or aborts the cell. It never
# degrades into a default answer. Before this existed, a failed
# extraction was counted as an "error" and the turn silently vanished
# from the store, and a failed defense call returned BENIGN / TRUST --
# both look exactly like real results.
#
# Transient failures (rate limits, 5xx, timeouts, connection drops, and
# responses we cannot parse) are retried with exponential backoff.
# Anything else, or running out of attempts, raises ModelCallFailed,
# which the runner lets propagate so the whole cell aborts.

from __future__ import annotations

import time
from collections import Counter

import openai

MAX_ATTEMPTS = 8
BASE_BACKOFF = 2.0    # seconds; doubles each attempt
MAX_BACKOFF = 60.0

_sleep = time.sleep   # tests replace this


class ModelCallFailed(RuntimeError):
    """A model call could not be completed. Aborts the cell."""


class UnparseableResponse(Exception):
    """The call returned, but not in the format the caller requires.
    Retried like a transient error, because the alternative -- guessing
    a default -- is exactly the silent failure this module forbids."""


_TRANSIENT_TYPES = (
    openai.RateLimitError,
    openai.APIConnectionError,     # includes APITimeoutError
    openai.InternalServerError,
    UnparseableResponse,
)
_TRANSIENT_TEXT = ("rate limit", "ratelimit", "429", "503", "overloaded",
                   "unavailable", "timeout", "timed out", "connection")


def is_transient(e: BaseException) -> bool:
    if isinstance(e, ModelCallFailed):
        return False
    if isinstance(e, _TRANSIENT_TYPES):
        return True
    if isinstance(e, openai.APIStatusError):
        return e.status_code in (408, 409, 429) or e.status_code >= 500
    msg = str(e).lower()
    return any(t in msg for t in _TRANSIENT_TEXT)


def call_with_retry(fn, label: str, *, step: str = "",
                    retries: Counter | None = None,
                    max_attempts: int = MAX_ATTEMPTS):
    """Run fn() until it succeeds. Each retry is counted in
    retries[step], or retries[step + ":unparseable"] when the call
    returned but could not be parsed -- the proxy logs those as ok, so
    the integrity check needs them to reconcile ok-call counts. Raises
    ModelCallFailed when a non-transient error occurs or the attempts
    run out."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except ModelCallFailed:
            raise
        except Exception as e:
            if not is_transient(e):
                raise ModelCallFailed(
                    f"{label}: non-retryable {type(e).__name__}: {e}") from e
            if attempt == max_attempts:
                raise ModelCallFailed(
                    f"{label}: gave up after {max_attempts} attempts; "
                    f"last error {type(e).__name__}: {e}") from e
            if retries is not None:
                unparseable = isinstance(e, UnparseableResponse)
                retries[f"{step}:unparseable" if unparseable else step] += 1
            wait = min(BASE_BACKOFF * (2 ** (attempt - 1)), MAX_BACKOFF)
            print(f"    {type(e).__name__} on {label}; retry "
                  f"{attempt}/{max_attempts - 1} in {wait:.0f}s", flush=True)
            _sleep(wait)
    raise RuntimeError("unreachable")
