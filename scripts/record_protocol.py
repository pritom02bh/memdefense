#!/usr/bin/env python3
# scripts/record_protocol.py
#
# Writes results/PROTOCOL.md -- a complete, timestamped record of the
# experimental configuration, produced BEFORE the run starts.
#
# WHY BEFORE: a protocol reconstructed after the fact is a description of
# what you remember doing. Written first, it is a commitment you can be
# held to -- and it catches configuration you did not realise was
# unspecified. It also fixes the exact model strings and code state, both
# of which drift: this project has already had one model deprecated
# (claude-sonnet-5 rejecting `temperature`) and one closed to new users
# (gemini-2.5-flash) mid-project.
#
# The manifest captures: parameters, resolved model identifiers, the
# frozen judge prompt hash, the dataset checksum, git commit, and the
# hash of every file that defines experimental behaviour. If any of these
# change, results from before and after are not comparable, and the
# manifest is what lets you prove which is which.
#
#   python3 -m scripts.record_protocol --turns 150 --questions 75 \
#       --backbone gemini-backbone --group anchor_gemini

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys

import yaml

OUT = pathlib.Path("results/PROTOCOL.md")

# Files whose contents define experimental behaviour. A change to any of
# these invalidates comparability with earlier cells.
BEHAVIOUR_FILES = [
    "runner/run.py",
    "runner/cells.yaml",
    "adapters/mem0_adapter.py",
    "defenses/defenses.py",
    "defenses/base.py",
    "core/judge.py",
    "core/evidence_filter.py",
    "core/dataset.py",
    "prompts/judge.txt",
    "proxy/config.yaml",
    "proxy/logger.py",
]


def sha256_file(p: pathlib.Path) -> str:
    if not p.exists():
        return "MISSING"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(args, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return "unavailable"
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def resolved_models() -> dict:
    """Map each proxy alias to the concrete model string it resolves to.
    Aliases are stable; the models behind them are not."""
    cfg_path = pathlib.Path("proxy/config.yaml")
    if not cfg_path.exists():
        return {}
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    out = {}
    for entry in cfg.get("model_list", []) or []:
        alias = entry.get("model_name")
        model = (entry.get("litellm_params") or {}).get("model")
        if alias:
            out[alias] = model
    return out


def prices() -> dict:
    """Extract the PRICES table so the manifest records what conversion
    was in force. Dollar figures are only meaningful with a dated table;
    token counts are the durable measurement."""
    p = pathlib.Path("proxy/logger.py")
    if not p.exists():
        return {}
    ns: dict = {}
    src = p.read_text()
    start = src.find("PRICES")
    if start == -1:
        return {}
    brace = src.find("{", start)
    depth, i = 0, brace
    while i < len(src):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    try:
        exec("PRICES = " + src[brace:i + 1], {}, ns)
        return ns.get("PRICES", {})
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, required=True)
    ap.add_argument("--questions", type=int, required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--seeds", type=int, default=1,
                    help="runs per cell; 1 means no within-cell variance "
                         "estimate, which must be disclosed")
    ap.add_argument("--data", default="data/locomo/locomo10.json")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    now = datetime.datetime.now(datetime.timezone.utc)
    models = resolved_models()

    lines = []
    lines.append("# Experimental Protocol")
    lines.append("")
    lines.append(f"Recorded **before** the run at "
                 f"`{now.isoformat(timespec='seconds')}`.")
    lines.append("")
    lines.append("This file is written prior to execution so that the")
    lines.append("configuration is a commitment rather than a reconstruction.")
    lines.append("If any hash below differs from the hash recorded for another")
    lines.append("set of cells, those cells are not directly comparable.")
    if args.note:
        lines.append("")
        lines.append(f"**Note:** {args.note}")
    lines.append("")

    lines.append("## Run parameters")
    lines.append("")
    lines.append("| parameter | value |")
    lines.append("|---|---|")
    lines.append(f"| group | `{args.group}` |")
    lines.append(f"| backbone alias | `{args.backbone}` |")
    lines.append(f"| turns ingested per conversation | {args.turns} |")
    lines.append(f"| question cap per cell | {args.questions} |")
    lines.append("| question selection | evidence-aligned "
                 "(all evidence turns must survive truncation) |")
    lines.append(f"| runs per cell | {args.seeds} |")
    lines.append("| temperature | 0.0 (all roles) |")
    lines.append("| adversarial scoring | deterministic abstention regex, "
                 "no judge |")
    lines.append("")

    lines.append("## Resolved models")
    lines.append("")
    lines.append("Aliases are stable across the project; the models behind them")
    lines.append("are not. This project has already lost two model strings")
    lines.append("mid-flight (one deprecated `temperature`, one closed to new")
    lines.append("users), so the concrete identifiers are recorded here.")
    lines.append("")
    lines.append("| alias | resolves to |")
    lines.append("|---|---|")
    for alias, model in sorted(models.items()):
        lines.append(f"| `{alias}` | `{model}` |")
    lines.append("")

    lines.append("## Price table in force")
    lines.append("")
    lines.append("USD per million tokens. **Token counts are the durable")
    lines.append("measurement**; dollar figures are a dated conversion and")
    lines.append("should be reported as such. Costs can be recomputed exactly")
    lines.append("from the raw token counts in `results/proxy/calls.jsonl`.")
    lines.append("")
    pr = prices()
    if pr:
        lines.append("| model | input | output |")
        lines.append("|---|---|---|")
        for m, v in sorted(pr.items()):
            if isinstance(v, dict):
                lines.append(f"| `{m}` | {v.get('in')} | {v.get('out')} |")
    else:
        lines.append("_PRICES table could not be parsed._")
    lines.append("")

    lines.append("## Dataset")
    lines.append("")
    data_path = pathlib.Path(args.data)
    lines.append(f"- path: `{args.data}`")
    lines.append(f"- sha256 (first 16): `{sha256_file(data_path)}`")
    lines.append("- conversations used: as listed in `runner/cells.yaml`")
    lines.append("")

    lines.append("## Code state")
    lines.append("")
    g = git_state()
    lines.append(f"- git commit: `{g['commit']}`")
    lines.append(f"- branch: `{g['branch']}`")
    if g["dirty"]:
        lines.append("- **working tree DIRTY at run time** -- uncommitted "
                     "changes were in effect. Commit before running to make "
                     "this record exact.")
    lines.append("")
    lines.append("Behaviour-defining files (sha256, first 16):")
    lines.append("")
    lines.append("| file | hash |")
    lines.append("|---|---|")
    for f in BEHAVIOUR_FILES:
        lines.append(f"| `{f}` | `{sha256_file(pathlib.Path(f))}` |")
    lines.append("")

    lines.append("## Known limitations of this run")
    lines.append("")
    if args.seeds == 1:
        lines.append("- **Single run per cell.** Variance reported across")
        lines.append("  conversations only; model nondeterminism is not")
        lines.append("  separated from between-conversation variance.")
    lines.append("- **Truncation.** Conversations are cut to")
    lines.append(f"  {args.turns} turns for cost. Questions are filtered to")
    lines.append("  those answerable within that window, so absolute accuracy")
    lines.append("  is not comparable to full-conversation LoCoMo results.")
    lines.append("  Filtering changes question composition (multi-hop is")
    lines.append("  disproportionately removed); report the retention table.")
    lines.append("- **No attack.** This run measures benign-case cost only.")
    lines.append("  No adversarial injection was performed, so no protection")
    lines.append("  claim (ASR / dASR) can be made from these results.")
    lines.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT}")
    if g["dirty"]:
        print("\nWARNING: uncommitted changes. Commit first so the recorded")
        print("commit hash actually identifies the code that ran.")
    return 0


if __name__ == "__main__":
    sys.exit(main())