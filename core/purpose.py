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

import contextvars
import contextlib

_purpose_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "purpose", default="base"
)

VALID_PURPOSES = {"base", "defense", "attack", "judge"}


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


def current_purpose() -> str:
    return _purpose_var.get()


def purpose_header() -> dict:
    """
    Returns an HTTP header dict to attach to any outgoing request so the
    proxy-side logger can read the current purpose off the wire.
    Pass this into extra_headers= wherever the OpenAI/Anthropic client,
    or a framework's client config, allows custom headers.
    """
    return {"x-purpose": current_purpose()}