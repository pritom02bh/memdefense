# The Price of Safety

**Benign-Case Utility and Token Overhead of Memory-Poisoning Defenses in LLM Agents**

Pritom Bhowmik · `pritombhowmik@ieee.org`

Code, harness, and per-cell outputs for the paper.

---

## TL;DR

Memory-poisoning defenses are evaluated on attack success rate, yet almost all traffic they process is benign. We measure what four defenses cost on **entirely attack-free** LoCoMo conversations, holding the memory backend, retrieval, backbone, and judge fixed so that only the defense varies. Each condition is replicated three times across five conversations (90 cells).

- **Write-time defenses** (sanitize, provenance, LLM anomaly detection): no utility cost resolvable at our ±4.5 pp resolution; zero false quarantines.
- **Read-time reranker:** −4.4 pp core accuracy (95% CI [−9.0, −0.05]; McNemar p = 0.064), replicated but borderline. Its clearer cost: it quarantines **33.6%** of legitimate memories it adjudicates (up to 106 in one conversation) at 2.7% token overhead.
- **Stacking all four** does not compound the cost: −3.1 pp, CI includes zero.
- **Where a defense intercepts the pipeline, not whether it uses an LLM, predicts its benign-case price.**

## Results

| Defense    | Stage        | Utility Δ (pp), 95% CI     | Token overhead | FQR   |
|------------|--------------|----------------------------|----------------|-------|
| No defense | —            | —                          | —              | 0.000 |
| Sanitize   | write        | +1.8  [−1.7, +5.6]         | 0.0%           | 0.000 |
| Provenance | write        | +1.9  [−2.2, +6.0]         | 0.0%           | 0.000 |
| Anomaly    | write        | +3.1  [−0.9, +7.1]         | 1.2%           | 0.000 |
| Rerank     | read         | **−4.4\***  [−9.0, −0.05]  | 2.7%           | 0.336 |
| Stacked    | write + read | −3.1  [−7.4, +1.0]         | 4.1%           | 0.303 |

Paired change in core accuracy vs. the undefended baseline, majority vote over 3 replicates × 5 conversations; stratified bootstrap CIs (10⁴ resamples). \*McNemar p = 0.064.

<p align="center">
  <img src="assets/fig_forest.png" width="75%" alt="Forest plot of paired accuracy deltas"><br>
  <em>Paired core-accuracy deltas with 95% bootstrap CIs.</em>
</p>

<p align="center">
  <img src="assets/fig_fqr.png" width="48%" alt="False quarantine rate by condition">
  <img src="assets/fig_cost_utility.png" width="48%" alt="Token overhead vs utility change"><br>
  <em>Left: false quarantine rate on benign traffic (every quarantine is a false positive). Right: token overhead vs. utility change.</em>
</p>

**Reading the nulls:** CIs span roughly ±4.5 pp. Write-time results mean *no effect detectable at this resolution*, not zero cost. The stacked result shows no evidence of compounding; it does not prove the defenses interact protectively.

---

## Setup

| Role                | Model                         | Provider  |
|---------------------|-------------------------------|-----------|
| Memory extraction   | `gemini-3.1-flash-lite`       | Google    |
| Question answering  | `gemini-3.1-flash-lite`       | Google    |
| Anomaly detection   | `gemini-3.1-flash-lite`       | Google    |
| Retrieval embedding | `text-embedding-3-small`      | OpenAI    |
| Grading judge       | `claude-haiku-4-5-20251001`   | Anthropic |

- **Memory backend:** Mem0 with local vector storage; each cell writes to an isolated namespace.
- **Data:** LoCoMo conversations conv-26, conv-30, conv-41, conv-42, conv-43, truncated to 150 turns, ≤ 75 evidence-aligned questions each.
- **Design:** 6 conditions × 5 conversations × 3 replicates = **90 cells**, temperature 0, run sequentially.
- **Judge:** cross-provider, κ = 0.909 on 24 hand-graded core items; prompt frozen.
- **Adversarial abstention:** calibrated detector, κ = 1.000 on 58 blind hand labels (original regex: 86.2%, κ = 0.663; original scores preserved).

### Defenses

| Defense      | Stage | Mechanism                                                         |
|--------------|-------|-------------------------------------------------------------------|
| `sanitize`   | write | Lexical filter for injection patterns; no model call              |
| `provenance` | write | Source/trust tagging stored as metadata; no model call            |
| `anomaly`    | write | LLM checks each candidate memory for consistency before storage   |
| `rerank`     | read  | LLM re-scores retrieved memories; inconsistent ones quarantined   |
| `stacked`    | both  | All four, applied in sequence                                      |

Defenses are wrappers around a uniform adapter; framework source is never modified.

### Metrics

| Metric                      | Definition                                                                |
|-----------------------------|---------------------------------------------------------------------------|
| **BUD** (Benign Utility Δ)  | Paired per-question accuracy delta vs. baseline, equal weight per conversation |
| **Token overhead τ**        | Defense-attributed tokens ÷ framework base tokens                         |
| **FQR**                     | Quarantined items ÷ items the defense adjudicated (all false by construction) |

Every LLM call routes through a LiteLLM proxy with a purpose tag (`base` / `defense` / `judge`) and cell ID, so defense spend is measured, not estimated. Latency and dollar cost are logged but not headline metrics: latency is contaminated by rate-limit backoff, and prices change.

### Statistics (`scripts/stats.py`)

1. **Stratified bootstrap CI** — headline; resamples questions within conversations.
2. **Exact McNemar** — pooled discordant pairs, exact binomial.
3. **Exact sign-flip permutation** — conversation as unit; floor p = 0.0625 at n = 5, so it cannot reach p < 0.05 by design.

Replicates are collapsed per question by majority vote before any statistic is computed.

---

## Reproduce

Requires Python 3.11 and OpenAI, Anthropic, and Google API keys.

```bash
git clone https://github.com/pritom02bh/memdefense.git
cd memdefense
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock

cat > .env <<EOF
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
EOF
```

Place LoCoMo (`locomo10.json`, from `snap-research/locomo`) at `data/locomo/locomo10.json`:

```bash
shasum -a 256 data/locomo/locomo10.json
# 79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4
```

**Terminal 1 — metering proxy**

```bash
set -a; source .env; set +a
litellm --config proxy/config.yaml --port 4000
```

**Terminal 2 — experiment**

```bash
python3 -m runner.run --plan                                                # list cells
python3 -m runner.run --only anchor_gemini --max-turns 150 --max-questions 75
python3 -m runner.run --only anchor_gemini --condition rerank --max-turns 150 --max-questions 75

python3 -m runner.analyse --csv results/main.csv                            # tables, BUD, τ, FQR
python3 -m scripts.stats --group anchor_gemini --metric core4               # CIs and tests
```

Runs checkpoint per question and resume automatically; completed cells are skipped.

**Calibration:** `scripts/generate_judge_calibration_set.py` → `scripts/grade_calibration.py` → `scripts/compute_kappa.py` (judge); `scripts/label_abstention.py` (blind abstention labels).

---

## Layout

```
memdefense/
├── proxy/        LiteLLM config (model aliases) + metering logger
├── core/         LoCoMo loader, evidence filter, judge, purpose tagging
├── adapters/     uniform adapter interface + Mem0 adapter
├── defenses/     wrapper base + sanitize / provenance / anomaly / rerank
├── runner/       cells.yaml (matrix), run.py, analyse.py
├── prompts/      judge.txt (frozen)
├── scripts/      calibration, kappa, abstention labeling, statistics
├── versions.md   pinned models, dataset checksum, prices, constraints
└── requirements.lock
```

## Pitfalls that silently corrupt measurement

Each produced a plausible number instead of an error (paper, Appendix D):

- **Silent truncation.** No output-token cap on extraction → a reasoning model exhausted its budget and ~75% of memories were silently dropped. Fixed with an explicit `max_tokens`.
- **Composition break.** In the stack, the reranker could not reach the store through the wrapper chain. Fixed by walking the chain.
- **Narrow detector as ground truth.** The original abstention regex misgraded correct refusals. Replaced with a calibrated detector.
- **Stale price entry.** A duplicate price silently overrode the correct one (5× error), caught by per-purpose logging.
- **Evidence truncation.** Unfiltered questions against 150-turn transcripts gave ~25% accuracy. Fixed by evidence alignment, applied to every condition.

## Limitations

Five conversations and three replicates bound resolution to ~±4.5 pp. One backbone throughout, so the write/read split is untested on other models. Act-time defenses are not evaluated. Token overhead is backbone-specific.

## Ethics

All conditions are benign, on a public benchmark. No attacks are run and no live or third-party systems are targeted.

## Citation

```bibtex
@misc{bhowmik2026priceofsafety,
  title  = {The Price of Safety: Benign-Case Utility and Token Overhead of Memory-Poisoning Defenses in LLM Agents},
  author = {Bhowmik, Pritom},
  year   = {2026},
  note   = {Preprint},
  url    = {https://github.com/pritom02bh/memdefense}
}
```

## License

MIT. See [LICENSE](LICENSE).
