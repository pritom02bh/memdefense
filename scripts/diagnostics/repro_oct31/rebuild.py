# scripts/diagnostics/repro_oct31/rebuild.py
#
# Rebuilds none conv0 (conv-26, first 150 turns) once, today, with exactly
# the pipeline the Oct 30 batch (__v3b) used for a no-defense re-ingest
# (runner/v3.py run_cell, re-ingest branch): Mem0Adapter(run_id="v3",
# backbone gemini-backbone), reset + clear the message history, assert the
# history is empty, then ingest_with_progress turn by turn inside
# cell_context(cell_id). No answering.
#
# Isolation: run with MEMDEF_RESULTS pointing at this folder's root/, so
# the adapter's Qdrant folder and mem0 history DB live here, not in
# results_v3/ (no existing namespace or file is touched); MEM0_DIR keeps
# mem0's home here too. The proxy (MEMDEF_RESULTS=results_v3) logs the
# calls to results_v3/proxy/calls.jsonl under TEST_CELL, which no manifest
# or report reads. A watchdog kills the run if this cell's logged spend
# passes LIMIT_USD.
#
#   MEMDEF_RESULTS=results_diag/repro_oct31/root MEM0_DIR=results_diag/repro_oct31/mem0_home \
#     .venv/bin/python scripts/diagnostics/repro_oct31/rebuild.py

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import sys
import threading
import time

sys.path.insert(0, ".")
REPO = pathlib.Path(".").resolve()
HERE = REPO / "results_diag" / "repro_oct31"
LOG = REPO / "results_v3" / "proxy" / "calls.jsonl"
TEST_CELL = "repro__none__conv0__oct31"
LIMIT_USD = 0.90
ALLOWED = {"extract": "gemini-3.1-flash-lite", "embedding": "text-embedding-3-small"}

from core.paths import QDRANT, RESULTS  # noqa: E402

assert RESULTS == HERE / "root", f"MEMDEF_RESULTS must be {HERE / 'root'} (got {RESULTS})"
assert pathlib.Path(os.environ.get("MEM0_DIR", "")).resolve() == HERE / "mem0_home", "set MEM0_DIR"


def rows():
    if not LOG.exists():
        return []
    return [r for r in map(json.loads, LOG.read_text().splitlines())
            if r.get("cell_id") == TEST_CELL]


def watchdog():
    while True:
        time.sleep(5)
        rs = rows()
        spend = sum(r.get("cost_usd") or 0 for r in rs)
        bad = [r for r in rs if ALLOWED.get(r.get("step")) != r.get("model_resolved")]
        if spend > LIMIT_USD or bad:
            print(f"\nWATCHDOG: spend ${spend:.3f}, off-model rows {len(bad)}; killing", flush=True)
            os._exit(9)


from adapters.mem0_adapter import Mem0Adapter  # noqa: E402
from core.dataset import load_locomo  # noqa: E402
from core.purpose import cell as cell_context  # noqa: E402
from runner.v3 import BACKBONE_ALIAS, MAX_TURNS  # noqa: E402
from scripts.run_baseline import ingest_with_progress  # noqa: E402

adapter = Mem0Adapter(run_id="v3", cell_id=TEST_CELL, backbone=BACKBONE_ALIAS,
                      answer_alias=BACKBONE_ALIAS)
cfg = adapter.build_config()
print(f"qdrant {cfg['vector_store']['config']['path']}\nhistory {cfg.get('history_db_path')}")
print(f"llm {cfg['llm']['config']['model']} temperature {cfg['llm']['config']['temperature']} "
      f"max_tokens {cfg['llm']['config']['max_tokens']}; embedder {cfg['embedder']['config']['model']}")
for p in (cfg["vector_store"]["config"]["path"], cfg["history_db_path"]):
    assert pathlib.Path(p).resolve().is_relative_to(HERE), p
assert QDRANT.resolve().is_relative_to(HERE)

conv_full = load_locomo("data/locomo/locomo10.json")[0]
assert conv_full.conv_id == "conv-26"
conv = dataclasses.replace(conv_full, turns=conv_full.turns[:MAX_TURNS])
threading.Thread(target=watchdog, daemon=True).start()
started = time.monotonic()
try:
    adapter.reset(TEST_CELL)
    adapter.clear_message_history(TEST_CELL)
    history = adapter.message_history(TEST_CELL)
    assert not history, f"message history not empty before turn 1 ({len(history)})"
    print(f"\n{TEST_CELL}: {conv.conv_id}, {len(conv.turns)} turns; history empty", flush=True)
    with cell_context(TEST_CELL):
        ingest_with_progress(adapter, conv)
    mems = adapter.dump_memories()
    print(f"memories stored: {len(mems)} ({(time.monotonic() - started) / 60:.1f} min)", flush=True)
finally:
    adapter.close()
rs = rows()
print(f"logged: {len(rs)} rows, {sum(r['status'] == 'ok' for r in rs)} ok; "
      f"spend ${sum(r.get('cost_usd') or 0 for r in rs):.4f}")
(HERE / "rebuild_done.json").write_text(json.dumps({"cell": TEST_CELL, "memories": len(mems)}))
