# core/purpose.py
#
# Tags every outgoing LLM call with WHY it was made: "base", "defense",
# "attack", or "judge". Uses contextvars because we cannot pass an argument
# into calls that a framework (Mem0, Letta) makes internally -- but a
# contextvar is visible anywhere in the call stack, including inside
# a framework's own code, as long as the call happens while the
# context manager below is active.
#
# Default is "base" -- an unwrapped call still gets counted, just
# conservatively attributed, rather than silently dropped.
#
# Two finer tags ride along the same way:
#   step    -- WHICH model step: extract, answer, anomaly, rerank, judge,
#              embedding. Lets the integrity check verify the served
#              model per step instead of guessing from token counts.
#   cell_id -- the cell the call belongs to, for callers (the judge)
#              that have no per-cell client of their own.

import contextvars
import contextlib

_purpose_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "purpose", default="base"
)
_step_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "step", default=None
)
_cell_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "cell_id", default=None
)

VALID_PURPOSES = {"base", "defense", "attack", "judge"}
VALID_STEPS = {"extract", "answer", "anomaly", "rerank", "judge", "embedding"}


@contextlib.contextmanager
def purpose(tag: str):
    """
    Usage:
        with purpose("defense"):
            result = sanitizer.check(memory_text)   # any LLM call inside
                                                      # this block is tagged
                                                      # "defense"
    """
    if tag not in VALID_PURPOSES:
        raise ValueError(f"Unknown purpose '{tag}'. Must be one of {VALID_PURPOSES}")
    token = _purpose_var.set(tag)
    try:
        yield
    finally:
        _purpose_var.reset(token)


@contextlib.contextmanager
def step(tag: str):
    if tag not in VALID_STEPS:
        raise ValueError(f"Unknown step '{tag}'. Must be one of {VALID_STEPS}")
    token = _step_var.set(tag)
    try:
        yield
    finally:
        _step_var.reset(token)


@contextlib.contextmanager
def cell(cell_id: str):
    token = _cell_var.set(cell_id)
    try:
        yield
    finally:
        _cell_var.reset(token)


def current_purpose() -> str:
    return _purpose_var.get()


def current_step() -> str | None:
    return _step_var.get()


def current_cell() -> str | None:
    return _cell_var.get()


def purpose_header() -> dict:
    """
    Returns an HTTP header dict to attach to any outgoing request so the
    proxy-side logger can read the current purpose off the wire.
    Pass this into extra_headers= wherever the OpenAI/Anthropic client,
    or a framework's client config, allows custom headers.

    Also carries x-step and x-cell-id when a step() / cell() block is
    active.
    """
    headers = {"x-purpose": current_purpose()}
    if current_step():
        headers["x-step"] = current_step()
    if current_cell():
        headers["x-cell-id"] = current_cell()
    return headers
