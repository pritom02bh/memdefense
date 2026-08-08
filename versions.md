
## Dataset
- LoCoMo: data/locomo/locomo10.json
- Downloaded: 2026-08-06 from snap-research/locomo (main branch)
- sha256: 79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4
- Totals: 10 conversations, 272 sessions, 5882 turns, 1986 QA pairs
- Category map (1=multi_hop, 2=temporal, 3=open_domain, 4=single_hop,
  5=adversarial) is INFERRED, not documented in the release. Reasoning
  in core/dataset.py docstring.

## Models (frozen)
- answering: gpt-4o-mini -> gpt-4o-mini-2024-07-18
- judge: gpt-4o
- embedding: text-embedding-3-small
- price freeze date: 2026-08-02

## Environment notes
- fastapi pinned <0.120 (litellm 1.79 imports get_flat_dependant, removed in 0.120)
- proxy callback path is "logger.metering_logger", relative to the folder
  containing config.yaml

## Component 3 notes (Mem0 adapter)
- mem0ai 2.0.15 API: search/get_all/delete_all require filters={"user_id": ...},
  and search uses top_k (not limit). Older docs show user_id/limit — those fail.
- Header injection attaches an httpx request hook to Mem0's internal OpenAI
  clients (llm.client, embedding_model.client). Static extra_headers config
  would not work: purpose must change per-call, not per-instance.
- OBSERVED: Mem0 does not preserve session timestamps in extracted memories.
  Smoke test asked "When did Caroline go to the LGBTQ support group?" (gold:
  7 May 2023); Mem0 answered with the current date. Expect low temporal-category
  accuracy at baseline. Must be reported as a framework property, not a
  defense effect.
- COST NOTE: each ingest turn sends the full accumulated memory context
  (~8.5k tokens/call by turn 400). Hit OpenAI 200k TPM limit on a full
  419-turn conversation. Runner (Component 8) needs rate-limit backoff.
