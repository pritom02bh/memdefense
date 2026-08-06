# The Price of Safety

**Quantifying the Utility and Cost Overhead of Memory Defenses in LLM Agents**

Research code for an empirical study of what memory-poisoning defenses *cost* — not just whether they work.

---

## Motivation

LLM agents with persistent memory are vulnerable to memory poisoning: adversarial content injected in one session corrupts stored memories and steers behavior in later sessions. A wave of defenses has followed — memory sanitization, provenance tracking, trust-aware retrieval, anomaly detection — and OWASP's 2026 Agentic AI Top 10 (ASI06) now prescribes layered defenses as standard practice.

Every published defense is evaluated on **attack success rate**. Almost none report what the defense costs when there is *no* attack:

- **Utility** — legitimate memories wrongly filtered, lowering recall accuracy
- **Latency** — added round-trips on every memory read and write
- **Money** — extra LLM calls for sanitization and judging

This repository measures all three at once, alongside protection, producing a **security–utility–cost frontier** that indicates which defenses are worth their overhead.

## Research questions

| | Question |
|---|---|
| **RQ1** | What is the benign-condition overhead of published memory defenses (accuracy, latency, cost)? |
| **RQ2** | How much do the same defenses reduce attack success rate? |
| **RQ3** | Which defenses are Pareto-optimal? Is the full "OWASP stack" dominated by cheaper subsets? |
| **RQ4** | Does overhead depend on memory architecture (vector vs. graph vs. self-editing context)? |

## Method in one diagram

```
                    ┌──────────────────────┐
   benign runs ───► │  memory framework    │ ◄─── adversarial runs
                    │  (Mem0 / Letta)      │
                    └──────────┬───────────┘
                               │ wrapped by
                    ┌──────────▼───────────┐
                    │  defense wrapper(s)  │  sanitize / provenance /
                    └──────────┬───────────┘  anomaly / rerank / stack
                               │ every LLM call
                    ┌──────────▼───────────┐
                    │   metering proxy     │  tokens, latency, cost,
                    └──────────┬───────────┘  purpose tag
                               │
                          calls.jsonl  ──►  Δaccuracy, ΔASR, Δ$, Δlatency,
                                            false-quarantine rate
```

Defenses are implemented as **wrappers** around a uniform adapter interface. No framework source is modified, so every comparison isolates the defense rather than a modified framework.

## Metrics

| Metric | Definition |
|---|---|
| **BUD** (Benign Utility Delta) | accuracy(defense on) − accuracy(off), benign conditions |
| **ΔASR** | attack success(off) − attack success(on) |
| **Δ$** | cost per conversation, split into *base* vs. *defense-attributed* tokens |
| **Δlatency** | p50 and p95, per query and per ingestion |
| **False-quarantine rate** | legitimate memories blocked ÷ total memories written |

Cost *attribution* — separating a framework's own token spend from the defense's — is possible because every LLM call carries a purpose tag through the proxy.

## Status

| # | Component | State |
|---|---|---|
| 0 | Environment | ✅ Complete |
| 1 | Metering proxy | ✅ Complete — verified against API `usage` |
| 2 | Data loader (LoCoMo) | ⬜ Next |
| 3 | Mem0 adapter | ⬜ |
| 4 | LLM judge | ⬜ |
| 5 | Letta adapter (Docker) | ⬜ |
| 6 | Defense wrappers | ⬜ |
| 7 | Attack module | ⬜ |
| 8 | Experiment runner | ⬜ |
| 9 | Analysis + figures | ⬜ |

## Setup

Requires Python 3.11 and an OpenAI API key.

```bash
git clone <this-repo>
cd memdefense

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # or requirements.lock for exact pins

echo "OPENAI_API_KEY=sk-..." > .env
```

Start the metering proxy:

```bash
set -a; source .env; set +a
litellm --config proxy/config.yaml --port 4000
```

Verify metering works:

```bash
curl -s -X POST http://localhost:4000/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-purpose: base" \
  -d '{"model":"answering","messages":[{"role":"user","content":"say hi"}]}'

cat results/proxy/calls.jsonl
```

The logged `tokens_in` / `tokens_out` must match the response's `usage` block exactly.

## Layout

```
memdefense/
├── proxy/          metering proxy: config.yaml, logger.py
├── core/           purpose.py, dataset.py, judge.py, prices.py
├── adapters/       uniform framework interface + Mem0, Letta
├── defenses/       sanitize, provenance, anomaly, rerank, stack
├── attacks/        memory-poisoning attack integration
├── experiments/    cell matrix + runner
├── analysis/       metrics, frontier figures
├── results/        proxy logs, run logs, quarantine logs
└── versions.md     pinned models, datasets, prices, known constraints
```

## Design invariants

These are asserted in code, not left to discipline:

- One `answering` model, one `judge` model, one `embedding` model, temperature 0, across every experimental cell
- Identical `top_k` across frameworks
- Defenses are wrappers; framework source is never modified
- Every LLM call carries a purpose tag (`base` / `defense` / `attack` / `judge`)
- Prices frozen to a single date; raw token counts stored alongside dollar figures
- No reported number without a bootstrap confidence interval

## Known environment constraints

| Constraint | Reason |
|---|---|
| `fastapi < 0.120` | litellm's proxy imports `get_flat_dependant`, removed in 0.120 |
| `litellm 1.79.x` | Resolves against the FastAPI pin above |
| Letta runs in Docker, not pip | Letta 0.6.x pins `pytz<2024.0`, conflicting with pandas |
| Callback path is relative to `config.yaml`'s directory | Use `logger.metering_logger`, not `proxy.logger.metering_logger` |

## Reproducibility

`versions.md` pins model version strings, dataset checksums, framework commits, and the price-freeze date. `requirements.lock` pins the full resolved dependency tree. Total API spend for the study will be reported in the paper.

## Ethics

This repository integrates published memory-poisoning attacks solely to evaluate published defenses, on public benchmark datasets. No live or third-party systems are targeted. Attack code is included for reproducibility of the security axis and is not novel offensive research.

## Citation

Paper in preparation. Citation details will be added on preprint release.

## License

TBD before public release.
