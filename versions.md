
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

## Component 4 notes (judge)
- Judge model: gpt-4o (alias "judge"). Answering model gpt-4o-mini.
  Deliberately different and stronger to avoid self-preference.
- Judge prompt frozen at prompts/judge.txt after calibration. DO NOT EDIT.
- Calibration: 30 hand-graded items, 6 per category, from conv-26/conv-30
  first 60 turns. Cohen's kappa = 0.841, percent agreement 93.3%.
- Prompt rules tightened during calibration:
  (a) relative time refs ("last year") where gold is absolute = INCORRECT;
      judge must not infer what they resolve to.
  (b) "I don't know" is INCORRECT whenever gold states any substantive
      fact, including hedged ones.
- Judge strips markdown code fences before JSON parse (model wraps output
  in ```json despite instructions).
- LoCoMo category 5 ("adversarial") questions DO have supporting evidence
  and substantive gold answers; the adversarial element is a misleading
  premise (wrong speaker attributed), not unanswerability. "I don't know"
  is therefore a genuine failure on these.
- KNOWN ISSUE: judge cache key hashes (question, gold, predicted) but NOT
  the prompt. Editing the prompt returns stale grades silently. Must
  `rm results/judge_cache.jsonl` after any prompt change. Fix pending.
