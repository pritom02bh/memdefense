# proxy/logger.py
#
# LiteLLM proxy callback (targets litellm 1.79.x's CustomLogger interface).
# Fires on every completed / failed call and writes ONE JSON line to a log
# file, tagged with tokens, latency, cost, and WHY the call was made
# (read from the x-purpose header set by core/purpose.py).
#
# This file is imported by proxy/config.yaml via:
#     callbacks: ["proxy.logger.metering_logger"]

import json
import time
import pathlib
from datetime import datetime, timezone

from litellm.integrations.custom_logger import CustomLogger

LOG_DIR = pathlib.Path(__file__).resolve().parent.parent / "results" / "proxy"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "calls.jsonl"

# Frozen price table -- USD per 1M tokens. Update the date if you re-price.
PRICE_DATE = "2026-08-02"
PRICES = {
    # PLACEHOLDER price -- verify on Google AI pricing page.
    "gemini-3.6-flash": {"in": 0.30, "out": 2.50},   # PLACEHOLDER -- VERIFY

    # PLACEHOLDER price -- verify on Anthropic pricing page.
    "claude-haiku-4-5-20251001": {"in": 1.00, "out": 5.00},

    # Cross-lab backbone (Anthropic). Same units as the rest of this
    # table: USD per MILLION tokens, dict form. PLACEHOLDER -- verify
    # against Anthropic's pricing page and record the date in versions.md.
    "claude-sonnet-4-5-20250929": {"in": 3.00, "out": 15.00},

    "gpt-4o-mini":            {"in": 0.15,  "out": 0.60},
    "gpt-4o":                 {"in": 2.50,  "out": 10.00},
    "text-embedding-3-small": {"in": 0.02,  "out": 0.00},
}


def _extract_headers(kwargs: dict) -> dict:
    """
    Custom headers sent by the client live at kwargs["litellm_params"]
    ["proxy_server_request"]["headers"] in litellm 1.79.x -- NOT at the
    top level of kwargs. (Verified against litellm/litellm_core_utils/
    litellm_logging.py, which does:
        proxy_server_request = litellm_params.get("proxy_server_request") or {}
    )
    """
    try:
        litellm_params = kwargs.get("litellm_params", {}) or {}
        proxy_req = litellm_params.get("proxy_server_request", {}) or {}
        return proxy_req.get("headers", {}) or {}
    except Exception:
        return {}


def _get_header(headers: dict, name: str, default=None):
    """Case-insensitive header lookup -- HTTP header casing isn't guaranteed."""
    name_lower = name.lower()
    for k, v in headers.items():
        if k.lower() == name_lower:
            return v
    return default


def _base_model_name(model_field: str) -> str:
    """'openai/gpt-4o-mini' -> 'gpt-4o-mini' for the price table lookup."""
    return model_field.split("/")[-1] if model_field else "unknown"


def _cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    price = PRICES.get(model)
    if price is None:
        return 0.0
    return (tokens_in / 1_000_000) * price["in"] + (tokens_out / 1_000_000) * price["out"]


def _write_line(record: dict) -> None:
    # Flush immediately -- a killed run must leave a valid, readable file.
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def _extract_usage(response_obj) -> dict:
    """
    Defensive usage extraction. response_obj.usage may be a pydantic model,
    a dict, or an object with plain attributes depending on the provider and
    litellm version -- try each, never raise.
    """
    if response_obj is None:
        return {}
    usage = getattr(response_obj, "usage", None)
    if usage is None and isinstance(response_obj, dict):
        usage = response_obj.get("usage")
    if usage is None:
        return {}

    # pydantic model
    if hasattr(usage, "model_dump"):
        try:
            return usage.model_dump()
        except Exception:
            pass
    # already a dict
    if isinstance(usage, dict):
        return usage
    # plain attributes
    out = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        val = getattr(usage, field, None)
        if val is not None:
            out[field] = val
    return out


def _build_record(kwargs: dict, response_obj, start_time, end_time, status: str, error: str | None = None) -> dict:
    headers = _extract_headers(kwargs)
    model_alias = kwargs.get("model", "unknown")          # e.g. "answering"
    resolved_model = (kwargs.get("litellm_params") or {}).get("model", model_alias)
    base_model = _base_model_name(resolved_model)

    usage = _extract_usage(response_obj)

    tokens_in = usage.get("prompt_tokens", 0) or 0
    tokens_out = usage.get("completion_tokens", 0) or 0

    latency_ms = None
    try:
        latency_ms = round((end_time - start_time).total_seconds() * 1000, 1)
    except Exception:
        try:
            latency_ms = round((end_time - start_time) * 1000, 1)
        except Exception:
            pass

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "purpose": _get_header(headers, "x-purpose", "base"),
        "run_id": _get_header(headers, "x-run-id"),
        "cell_id": _get_header(headers, "x-cell-id"),
        "model_alias": model_alias,
        "model_resolved": base_model,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_total": tokens_in + tokens_out,
        "latency_ms": latency_ms,
        "cost_usd": round(_cost_usd(base_model, tokens_in, tokens_out), 8),
        "price_date": PRICE_DATE,
        "status": status,
        "error": error,
    }


def _safe_log(kwargs, response_obj, start_time, end_time, status: str, error: str | None = None) -> None:
    """
    Never let an exception escape into litellm (it swallows them silently,
    which produces an empty log file and no explanation). If record-building
    fails, write a diagnostic line instead so the failure is visible.
    """
    try:
        record = _build_record(kwargs, response_obj, start_time, end_time, status=status, error=error)
        _write_line(record)
    except Exception as e:
        import traceback
        _write_line({
            "ts": datetime.now(timezone.utc).isoformat(),
            "status": "logger_error",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-1500:],
            "kwargs_keys": sorted(list(kwargs.keys())) if isinstance(kwargs, dict) else str(type(kwargs)),
        })


class MeteringLogger(CustomLogger):
    """Registered in proxy/config.yaml. Async hooks are what the proxy
    server actually invokes; the sync ones are kept as a fallback."""

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        _safe_log(kwargs, response_obj, start_time, end_time, status="ok")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        _safe_log(kwargs, response_obj, start_time, end_time, status="error",
                  error=str(kwargs.get("exception", "unknown error")) if isinstance(kwargs, dict) else "unknown")

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        _safe_log(kwargs, response_obj, start_time, end_time, status="ok")

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        _safe_log(kwargs, response_obj, start_time, end_time, status="error",
                  error=str(kwargs.get("exception", "unknown error")) if isinstance(kwargs, dict) else "unknown")


# LiteLLM's callback loader instantiates this from the string
# "proxy.logger.metering_logger" given in config.yaml.
metering_logger = MeteringLogger()