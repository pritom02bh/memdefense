# core/proxy_client.py
#
# The one way to build an OpenAI client for the metering proxy, and to
# fix up the clients Mem0 builds for itself.
#
# NO SDK RETRIES (2026-10-31). The OpenAI SDK retries twice on its own
# (timeouts, 5xx) below core/retry.call_with_retry, so those retries
# were never counted. In stacked conv0 v3c a rerank call hung, the SDK
# gave up after its default 600 s and resent it, and the proxy later
# retried the abandoned original: an extra ok row nothing recorded.
# With max_retries=0 every retry goes through call_with_retry, which
# counts it per step. CLIENT_TIMEOUT is above the proxy's
# request_timeout (proxy/config.yaml), so the proxy answers first -- with
# an error the caller retries -- and never keeps working on a request
# the client has abandoned.

from __future__ import annotations

from openai import OpenAI

PROXY_BASE_URL = "http://localhost:4000/v1"
CLIENT_TIMEOUT = 360.0   # seconds; proxy request_timeout is 300
MAX_RETRIES = 0


def proxy_client(base_url: str = PROXY_BASE_URL) -> OpenAI:
    return OpenAI(base_url=base_url, api_key="proxy-dummy-key",
                  max_retries=MAX_RETRIES, timeout=CLIENT_TIMEOUT)


def no_sdk_retries(client) -> None:
    """Same settings for a client built elsewhere (Mem0's LLM and
    embedder). The SDK reads both attributes on every request."""
    client.max_retries = MAX_RETRIES
    client.timeout = CLIENT_TIMEOUT
