# scripts/add_backbones.py
#
# Adds cross-lab backbones to proxy/config.yaml AND to PRICES in
# proxy/logger.py. Both are required: a model in the config but not in
# PRICES routes fine and silently logs cost_usd = 0.0, which zeroes out
# the cost axis for that backbone without any error.
#
# PRICES BELOW ARE PLACEHOLDERS -- verify against the provider's pricing
# page and record the check date in versions.md before any real run.

import pathlib
import sys

CONFIG = pathlib.Path("proxy/config.yaml")
LOGGER = pathlib.Path("proxy/logger.py")

MODEL_BLOCK = """
  - model_name: claude-backbone
    litellm_params:
      model: anthropic/claude-sonnet-4-20250514
      api_key: os.environ/ANTHROPIC_API_KEY
"""

NEW_PRICES = '''    # Cross-lab backbone. PLACEHOLDER PRICE -- verify before use.
    "claude-sonnet-4-20250514": (0.003, 0.015),
'''


def patch_config() -> bool:
    if not CONFIG.exists():
        print(f"  ! {CONFIG} not found")
        return False
    s = CONFIG.read_text()
    if "claude-backbone" in s:
        print("  config.yaml already has claude-backbone")
        return True
    if "model_list:" not in s:
        print("  ! no 'model_list:' in config.yaml")
        return False
    CONFIG.write_text(s.rstrip() + "\n" + MODEL_BLOCK)
    print("  config.yaml: added claude-backbone")
    return True


def patch_prices() -> bool:
    if not LOGGER.exists():
        print(f"  ! {LOGGER} not found")
        return False
    s = LOGGER.read_text()
    if "claude-sonnet-4" in s:
        print("  logger.py PRICES already has claude")
        return True
    idx = s.find("PRICES")
    brace = s.find("{", idx) if idx != -1 else -1
    if brace == -1:
        print("  ! could not locate the PRICES dict")
        return False
    LOGGER.write_text(s[:brace + 1] + "\n" + NEW_PRICES + s[brace + 1:])
    print("  logger.py: added placeholder Claude price")
    return True


def main():
    print("Adding cross-lab backbone...")
    ok1 = patch_config()
    ok2 = patch_prices()
    print()
    if ok1 and ok2:
        print("Done. NEXT:")
        print("  1. VERIFY the Claude price in proxy/logger.py")
        print("  2. Restart the proxy (Ctrl+C, then):")
        print("       set -a; source .env; set +a")
        print("       litellm --config proxy/config.yaml --port 4000")
        print("  3. python3 -m scripts.check_backbones")
        return 0
    print("Some patches failed -- inspect manually.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
