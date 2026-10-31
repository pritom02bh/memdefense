# scripts/diagnostics/gemini_change/record_add_args.py
#
# DIAGNOSTIC, NO API CALLS. Drives one code version's real ingest path
# for conv-30 (conv_idx 1, first 150 turns) with mem0's Memory replaced
# by a recorder, and writes every add() call -- positional args and all
# kwargs -- as one JSON line. Model calls made by defenses (anomaly) are
# answered "BENIGN" by a local stub; nothing leaves the machine.
#
#   cd <code root>; python scripts/diagnostics/gemini_change/record_add_args.py old|new none|chain OUT.jsonl CFG.json
#
# "old" runs from the c72464e snapshot, so cwd must be that snapshot.
# "new" runs from the repo with MEMDEF_RESULTS pointing under results_diag.

from __future__ import annotations

import json
import sqlite3
import sys
import threading
from types import SimpleNamespace

version, path_kind, out_path, cfg_path = sys.argv[1:5]
sys.path.insert(0, ".")

calls = []
configs = []


class _Db:
    def __init__(self):
        self._lock = threading.Lock()
        self.connection = sqlite3.connect(":memory:")
        self.connection.execute("CREATE TABLE messages (session_scope TEXT, role TEXT, content TEXT)")

    def get_last_messages(self, scope, limit=10):
        return []

    def close(self):
        self.connection.close()


class Recorder:
    """Stands in for mem0.Memory. Records add(); everything else is inert."""

    def __init__(self):
        self.llm = SimpleNamespace(generate_response=lambda *a, **k: '{"memory": []}')
        self.embedding_model = SimpleNamespace(embed=lambda *a, **k: [0.0],
                                               embed_batch=lambda *a, **k: [[0.0]])
        self.db = _Db()
        self.vector_store = SimpleNamespace(client=None)

    def add(self, *args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return {"results": []}

    def delete_all(self, *args, **kwargs):
        calls.append({"delete_all": True, "args": args, "kwargs": kwargs})

    def get_all(self, *args, **kwargs):
        return {"results": []}


# ---- no network: stub chat completions (anomaly verdicts) ------------
import openai.resources.chat.completions as cc


def _fake_create(self, *args, **kwargs):
    msg = SimpleNamespace(content="BENIGN", role="assistant")
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                           usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0,
                                                 total_tokens=0))


cc.Completions.create = _fake_create

import mem0


def _from_config(cfg):
    configs.append(cfg)
    return Recorder()


mem0.Memory.from_config = staticmethod(_from_config)

import adapters.mem0_adapter as ma

ma._find_and_patch = lambda *a, **k: True
from core.dataset import load_locomo

DATA = sys.argv[5]
convs = load_locomo(DATA)
DEFS = ["sanitize", "provenance", "anomaly", "rerank"] if path_kind == "chain" else []
COND = "stacked" if DEFS else "none"

if version == "old":
    # runner/run.py:run_cell at c72464e, ingest part
    import dataclasses
    from defenses.defenses import wrap
    from scripts.run_baseline import ingest_with_progress

    cell_id = f"anchor_gemini__gemini-backbone__{COND}__conv1"
    conv_full = convs[1]
    conv = dataclasses.replace(conv_full, turns=conv_full.turns[:150])
    adapter = ma.Mem0Adapter(run_id="anchor_gemini", cell_id=cell_id, backbone="gemini-backbone")
    target = wrap(DEFS, adapter, cell_id) if DEFS else adapter
    target.reset(cell_id)
    if DEFS:
        target.ingest(conv)
    else:
        ingest_with_progress(adapter, conv)
else:
    # runner/v3.py:run_cell, re-ingest part (prepare_reingest's reset and
    # history clear called directly: its path assertion requires results_v3)
    from core.purpose import cell as cell_context
    from defenses.defenses import wrap
    from runner.v3 import BACKBONE_ALIAS, _turn_conv, plan_cells
    from scripts.run_baseline import ingest_with_progress

    cell_id = f"anchor_gemini__gemini-backbone__{COND}__conv1__v3"
    cell = next(c for c in plan_cells() if c.cell_id == cell_id)
    conv = _turn_conv(convs, cell)
    adapter = ma.Mem0Adapter(run_id="v3", cell_id=cell.cell_id, backbone=BACKBONE_ALIAS,
                             answer_alias=cell.answer_alias)
    adapter.reset(cell.cell_id)
    adapter.clear_message_history(cell.cell_id)
    assert not adapter.message_history(cell.cell_id)
    target = wrap(cell.defenses, adapter, cell.cell_id) if cell.defenses else adapter
    with cell_context(cell.cell_id):
        if cell.defenses:
            target.ingest(conv)
        else:
            ingest_with_progress(adapter, conv)
    cfg = adapter.build_config()
    if cfg not in configs:
        configs.append(cfg)

with open(out_path, "w") as f:
    for c in calls:
        f.write(json.dumps(c, sort_keys=True, ensure_ascii=False) + "\n")
with open(cfg_path, "w") as f:
    json.dump(configs, f, indent=2, sort_keys=True)
print(f"{version}/{path_kind}: {sum('delete_all' not in c for c in calls)} add() calls, "
      f"{len(configs)} config(s) -> {out_path}")
