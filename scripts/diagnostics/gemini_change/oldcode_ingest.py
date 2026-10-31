# scripts/diagnostics/gemini_change/oldcode_ingest.py
#
# PART 2. Ingests the first 150 turns of conv-30 with the c72464e code,
# the way runner/run.py:run_cell did for a no-defense cell at that
# commit: Mem0Adapter(...), reset(namespace), ingest_with_progress().
# No answering, no judge. Run with cwd = a worktree of c72464e:
#
#   git worktree add ../memdefense_oldcode c72464e
#   cd ../memdefense_oldcode
#   MEM0_DIR=<repo>/results_diag/mem0_home MEM0_TELEMETRY=False \
#     <repo>/.venv/bin/python <repo>/scripts/diagnostics/gemini_change/oldcode_ingest.py DATA LOG OUT_JSON
#
# A watchdog thread reads the proxy log and kills the process if the
# Google spend logged for this cell passes LIMIT_USD.

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import sys
import threading
import time

WT = pathlib.Path.cwd().resolve()
assert WT.name == "memdefense_oldcode", f"run from the c72464e worktree, not {WT}"
sys.path.insert(0, str(WT))
DATA, LOG, OUT = (pathlib.Path(a).resolve() for a in sys.argv[1:4])
CELL = "diag__none__conv1__oldcode"
LIMIT_USD = 0.90
REPO = DATA.parent.parent.parent
DIAG = REPO / "results_diag"


def google_spend() -> float:
    if not LOG.exists():
        return 0.0
    usd = 0.0
    for line in LOG.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("cell_id") == CELL and "gemini" in (r.get("model_resolved") or ""):
                usd += r.get("cost_usd") or 0.0
    return usd


def watchdog():
    while True:
        time.sleep(5)
        s = google_spend()
        if s > LIMIT_USD:
            print(f"\nWATCHDOG: Google spend ${s:.3f} > ${LIMIT_USD}; killing the run", flush=True)
            os._exit(9)


def inside(p, root) -> bool:
    return pathlib.Path(p).resolve().is_relative_to(root)


import mem0.configs.base as mcb
import mem0.memory.setup as mms

from adapters.mem0_adapter import Mem0Adapter
from core.dataset import load_locomo
import core.judge as judge
import runner.run as old_run
from scripts.run_baseline import ingest_with_progress

# ---- where everything resolves -----------------------------------------
adapter = Mem0Adapter(run_id="diag", cell_id=CELL, backbone="gemini-backbone")
cfg = adapter._mem.config
paths = {
    "code (adapters/mem0_adapter.py)": pathlib.Path(sys.modules["adapters.mem0_adapter"].__file__),
    "qdrant path": pathlib.Path(cfg.vector_store.config.path),
    "cell outputs (runner/run.py RESULTS_ROOT)": old_run.RESULTS_ROOT,
    "proxy log constant (runner/run.py PROXY_LOG, unused here)": old_run.PROXY_LOG,
    "judge cache (core/judge.py CACHE_PATH)": judge.CACHE_PATH,
    "mem0 home (configs/base.py mem0_dir)": pathlib.Path(mcb.mem0_dir),
    "mem0 home (memory/setup.py mem0_dir)": pathlib.Path(mms.mem0_dir),
    "mem0 history db (config.history_db_path)": pathlib.Path(cfg.history_db_path),
    "data file": DATA,
}
print("RESOLVED PATHS")
for k, p in paths.items():
    print(f"  {k:58} {p.resolve()}")
for k in ("code (adapters/mem0_adapter.py)", "qdrant path", "cell outputs (runner/run.py RESULTS_ROOT)",
          "judge cache (core/judge.py CACHE_PATH)"):
    assert inside(paths[k], WT), f"{k} is not inside the worktree"
for k in ("mem0 home (configs/base.py mem0_dir)", "mem0 home (memory/setup.py mem0_dir)",
          "mem0 history db (config.history_db_path)"):
    assert inside(paths[k], DIAG), f"{k} is not under results_diag/"
for p in paths.values():
    for bad in (REPO / "results", REPO / "results_v3", pathlib.Path.home() / ".mem0"):
        assert not inside(p, bad) or p == DATA, f"{p} is under {bad}"
db_file = adapter._mem.db.connection.execute("PRAGMA database_list").fetchall()
print(f"  sqlite file actually open: {db_file}")
print(f"  model alias: {cfg.llm.config.get('model')}  max_tokens: {cfg.llm.config.get('max_tokens')}  "
      f"temperature: {cfg.llm.config.get('temperature')}")

# ---- run_cell, ingest part (c72464e runner/run.py lines 143-159) ---------
convs = load_locomo(str(DATA))
conv_full = convs[1]
assert conv_full.conv_id == "conv-30", conv_full.conv_id
conv = dataclasses.replace(conv_full, turns=conv_full.turns[:150])
print(f"\nCELL {CELL}  conv={conv.conv_id}  turns={len(conv.turns)}", flush=True)
threading.Thread(target=watchdog, daemon=True).start()
t0 = time.monotonic()
adapter.reset(CELL)
assert not adapter._mem.db.get_last_messages(f"user_id={CELL}", limit=10), "history not empty"
written, errors = ingest_with_progress(adapter, conv)
print(f"  ingest done: {written} written, {errors} errors", flush=True)
stored = len(adapter.dump_memories())
print(f"  memories stored: {stored}  ({(time.monotonic() - t0) / 60:.1f} min)", flush=True)
spend = google_spend()
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"cell": CELL, "memories_stored": stored, "written": written,
                           "errors": errors, "google_usd": spend,
                           "paths": {k: str(p.resolve()) for k, p in paths.items()}}, indent=2))
adapter._mem.vector_store.client.close()
print(f"  Google spend for this cell: ${spend:.3f}")
