# scripts/check_backbones.py
#
# PREFLIGHT. Sends one tiny request to every backbone alias and confirms
# (a) it routes, (b) the proxy logged it, (c) the cost is non-zero.
#
# Run this BEFORE launching the matrix. A backbone that fails here fails
# fast for free; the same failure discovered eight hours into a run costs
# the whole run. A backbone that routes but logs cost_usd = 0.0 is worse --
# it produces a complete-looking result with a silently broken cost axis.
#
#   python3 -m scripts.check_backbones

import json
import pathlib
from core.paths import RESULTS
import sys
import time

from core.proxy_client import proxy_client

PROXY = "http://localhost:4000/v1"
LOG = RESULTS / "proxy" / "calls.jsonl"

ALIASES = ["answering", "judge", "claude-backbone", "gemini-backbone", "deepseek-backbone"]


def log_len() -> int:
    if not LOG.exists():
        return 0
    return len([l for l in LOG.read_text().splitlines() if l.strip()])


def main():
    client = proxy_client(PROXY)
    results = {}

    for alias in ALIASES:
        before = log_len()
        try:
            r = client.chat.completions.create(
                model=alias,
                temperature=0.0,
                max_tokens=10,
                messages=[{"role": "user", "content": "Reply with the single word: ok"}],
                extra_headers={"x-purpose": "base",
                               "x-run-id": "preflight",
                               "x-cell-id": f"preflight_{alias}"},
            )
            text = (r.choices[0].message.content or "").strip()
        except Exception as e:
            results[alias] = ("FAIL", f"{type(e).__name__}: {str(e)[:110]}")
            continue

        time.sleep(0.8)  # let the callback flush
        rows = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()][before:]
        mine = [x for x in rows if x.get("cell_id") == f"preflight_{alias}"]
        if not mine:
            results[alias] = ("WARN", f"responded {text!r} but nothing logged")
            continue
        cost = sum(x.get("cost_usd") or 0 for x in mine)
        toks = sum(x.get("tokens_total") or 0 for x in mine)
        if cost == 0:
            results[alias] = ("WARN", f"logged {toks} tokens but cost_usd=0 "
                                      "-- add it to PRICES in proxy/logger.py")
        else:
            results[alias] = ("OK", f"{toks} tokens, ${cost:.6f}")

    print("\n" + "=" * 62)
    for alias, (status, detail) in results.items():
        print(f"  {status:4}  {alias:20} {detail}")
    print("=" * 62)

    bad = [a for a, (s, _) in results.items() if s == "FAIL"]
    warn = [a for a, (s, _) in results.items() if s == "WARN"]
    if bad:
        print(f"\n{len(bad)} backbone(s) unreachable: {', '.join(bad)}")
        print("Missing API key in .env, or the proxy needs restarting after")
        print("config changes. Backbones that fail here should be removed")
        print("from runner/cells.yaml before launching.")
    if warn:
        print(f"\n{len(warn)} backbone(s) route but have a cost problem:")
        print("fix PRICES in proxy/logger.py or their cost axis reads zero.")
    if not bad and not warn:
        print("\nAll backbones ready.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())