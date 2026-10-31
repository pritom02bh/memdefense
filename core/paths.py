# core/paths.py
#
# ONE output root for every reader and writer in the project. Set
# MEMDEF_RESULTS to point a whole run (proxy log, judge cache, cells,
# quarantine logs, Qdrant store, labels, analysis) somewhere else, e.g.
#
#     MEMDEF_RESULTS=results_v3 python3 -m runner.run ...
#
# Default "results" keeps the original layout. A relative value is
# resolved against the repo root, not the current directory, so the
# proxy (started from anywhere) and the runner always agree.
#
# proxy/logger.py repeats this resolution inline because litellm loads
# it relative to proxy/ and the repo root may not be importable there.

from __future__ import annotations

import os
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
ENV_VAR = "MEMDEF_RESULTS"


def results_root() -> pathlib.Path:
    root = pathlib.Path(os.environ.get(ENV_VAR, "results"))
    return root if root.is_absolute() else REPO_ROOT / root


RESULTS = results_root()
CELLS = RESULTS / "cells"
PROXY_LOG = RESULTS / "proxy" / "calls.jsonl"
QUARANTINE = RESULTS / "quarantine"
JUDGE_CACHE = RESULTS / "judge_cache.jsonl"
QDRANT = RESULTS / "mem0_qdrant"
ABSTENTION_LABELS = RESULTS / "abstention_labels"
ADVERSARIAL_RESCORED = RESULTS / "adversarial_rescored.json"
