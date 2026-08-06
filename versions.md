
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
