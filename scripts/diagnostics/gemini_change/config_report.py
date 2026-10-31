# scripts/diagnostics/gemini_change/config_report.py -- full resolved mem0 config, old vs new (no API, no store opened)
import json, sys, difflib, os
os.environ["MEM0_TELEMETRY"] = "False"
import mem0
from mem0.configs.base import MemoryConfig
from mem0.configs.llms.openai import OpenAIConfig
from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.configs.prompts import ADDITIVE_EXTRACTION_PROMPT
import hashlib
def resolve(cfg):
    mc = MemoryConfig(**cfg).model_dump()
    llm = OpenAIConfig(**cfg["llm"]["config"]).__dict__
    emb = BaseEmbedderConfig(**cfg["embedder"]["config"]).__dict__
    return {"memory_config": mc,
            "llm_resolved": {k: (v if isinstance(v, (str, int, float, bool, type(None), list, dict)) else repr(v)) for k, v in llm.items()},
            "embedder_resolved": {k: (v if isinstance(v, (str, int, float, bool, type(None), list, dict)) else repr(v)) for k, v in emb.items()},
            "mem0ai_version": mem0.__version__ if hasattr(mem0, "__version__") else "?",
            "extraction_prompt_sha256": hashlib.sha256(ADDITIVE_EXTRACTION_PROMPT.encode()).hexdigest(),
            "MEM0_DIR": os.environ.get("MEM0_DIR")}
out = {}
for v in ("old", "new"):
    cfg = json.load(open(f"results_diag/step1/{v}_cfg_none.json"))
    assert len(cfg) == 1
    chain = json.load(open(f"results_diag/step1/{v}_cfg_chain.json"))
    assert chain[0] == cfg[0] or v == "new", "old chain/none configs differ"
    out[v] = {"passed_to_from_config": cfg[0], **resolve(cfg[0])}
    if v == "new":
        out[v]["chain_config_equals_none_config"] = chain[0] == cfg[0]
txt = {v: json.dumps(out[v], indent=2, sort_keys=True, default=str).splitlines() for v in out}
json.dump(out, open("results_diag/step2_config.json", "w"), indent=2, sort_keys=True, default=str)
print("\n".join(txt["new"]))
print("\n===== DIFF old (c72464e) -> new =====")
print("\n".join(difflib.unified_diff(txt["old"], txt["new"], "old", "new", n=1, lineterm="")))
