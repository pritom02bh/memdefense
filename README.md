# The Price of Safety

**Benign-Case Utility and Token Overhead of Memory-Poisoning Defenses in LLM Agents**

Pritom Bhowmik · `pritombhowmik@ieee.org` · [arXiv:2609.22818](https://arxiv.org/abs/2609.22818)

Harness, labels, and per-run outputs for the paper.

## Summary

Memory-poisoning defenses are judged by attack success rate, yet almost all traffic they see is benign. We measure what four defenses cost when no attack is present. The memory backend, retrieval, answer prompt, and judge stay fixed; only the defense changes. Design: 6 conditions × 5 LoCoMo conversations × 3 replicates = 90 runs.

- **Write-time defenses** (sanitize, provenance, anomaly) never fired on benign traffic. No accuracy cost is detectable at about ±4 points.
- **The read-time reranker** fired on 9.9% of its memory decisions, withheld memory from 45% of questions, and lowered core accuracy by 4.4 points (95% CI [−9.0, −0.05]; McNemar p = 0.064). The direction held in all six runs that used it. A blind audit found only 2 of 50 sampled drops defensible.
- **Stacking** all four did not compound the accuracy cost. It only added tokens.

Stage and implementation vary together in this design, so the write/read pattern is an association, not a cause.

## Results

| Defense    | Stage | Δ core accuracy (pts), 95% CI | Token overhead τ | FQR  |
|------------|-------|-------------------------------|------------------|------|
| Sanitize   | write | +1.8 [−1.7, +5.6]             | 0.0%             | 0.0% |
| Provenance | write | +1.9 [−2.2, +6.0]             | 0.0%             | 0.0% |
| Anomaly    | write | +3.1 [−0.9, +7.1]             | 1.2%             | 0.0% |
| Rerank     | read  | −4.4 [−9.0, −0.05]            | 2.7%             | 9.9% |
| Stacked    | both  | −3.1 [−7.4, +1.0]             | 4.1%             | 8.9% |

Δ is the paired change against the undefended baseline: majority vote over three replicates, conversations weighted equally, stratified bootstrap CI (10⁴ resamples). τ is defense tokens over baseline tokens. FQR is quarantines per decision; no content is planted, so every quarantine is a false alarm. A null means no effect detectable at this resolution, not zero cost.

## Setup

| Role                           | Model                       |
|--------------------------------|-----------------------------|
| Agent (extraction, answering)  | `gemini-3.1-flash-lite`     |
| LLM defenses (anomaly, rerank) | `gemini-3.1-flash-lite`     |
| Embedding                      | `text-embedding-3-small`    |
| Judge                          | `claude-haiku-4-5-20251001` |

- **Memory:** Mem0 2.0.15 with local Qdrant, one namespace per run. Top-10 retrieval, no threshold. Temperature 0. Extraction `max_tokens=4000`.
- **Data:** LoCoMo conv-26, 30, 41, 42, 43, first 150 turns. Only questions whose evidence survives truncation, capped at 75 per conversation: 255 per run (200 core, 55 adversarial).
- **Scoring:** core answers by a cross-provider judge (κ = 0.909 on 24 hand-graded items). Adversarial answers by abstention; hand labels override the detector.
- **Cost:** every LLM call passes a LiteLLM proxy tagged `base`, `defense`, or `judge`, so defense tokens are measured, not estimated.

| Defense      | Stage | What it does                                                                 |
|--------------|-------|------------------------------------------------------------------------------|
| `sanitize`   | write | 8 regexes for injection phrasing on each raw turn; no model call             |
| `provenance` | write | Blocks turns not from the user; LoCoMo has no origin field, so it never fires |
| `anomaly`    | write | One LLM call per turn: `BENIGN` or `SUSPICIOUS`; fails open                  |
| `rerank`     | read  | One LLM call per retrieved memory, judged alone: `TRUST` or `DROP`; fails open |
| `stacked`    | both  | All four, each wrapping the next                                             |

## Reproduce

Requires Python 3.11 and OpenAI, Anthropic, and Google API keys.

```bash
git clone https://github.com/pritom02bh/memdefense.git && cd memdefense
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock
printf 'OPENAI_API_KEY=...\nANTHROPIC_API_KEY=...\nGOOGLE_API_KEY=...\n' > .env
```

Place LoCoMo (`locomo10.json`, from `snap-research/locomo`) at `data/locomo/locomo10.json`. SHA-256: `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`.

```bash
# terminal 1: metering proxy
set -a; source .env; set +a
litellm --config proxy/config.yaml --port 4000

# terminal 2: three replicates, then scoring and statistics
for s in 1 2 3; do
  python3 -m runner.run --only anchor_gemini --seed $s --max-turns 150 --max-questions 75
done
python3 -m scripts.rescore_abstention
python3 -m scripts.stats --group anchor_gemini --metric core4
```

Runs checkpoint per question and resume; finished cells are skipped. Rerunning hosted models will not reproduce the exact numbers, which is why the outputs below are released.

## Released outputs

`results_v2.tar.gz` unpacks to `results/`:

- `cells/`: all 90 runs, with per-question answers and grades
- `quarantine/`: every block and drop
- `proxy/calls.jsonl`: tokens for every LLM call, by purpose and run
- `abstention_labels/`: blind hand labels, including the 8 corrections
- `adversarial_rescored.json`: original and corrected adversarial scores
- `quarantine_audit/`: the 50-drop blind audit
- `judge_calibration/`: the hand-graded judge items

## Pitfalls

Each of these produced a plausible result or nearly did (paper, Appendix C):

- **Silent extraction failure.** A larger Gemini reasoning model, under Mem0's default 2,000-token limit, stored 82 and 67 memories on conv-26 and conv-30. The final backbone with `max_tokens=4000` stored 309 and 282. Only 4 of 300 extraction calls reached the limit, so most of the gap came from the model change or discarded output. No error was raised.
- **Composition break.** In the stack, the reranker could not reach the store through the wrapper chain and crashed. Fixed by walking the chain.
- **Narrow abstention regex.** It misread 31 of the 51 answers it scored as failures. Replaced by a detector; hand labels override it.
- **Stale price entry.** A duplicate price inflated dollar cost fivefold. Token metrics were unaffected.

## Limitations

Five conversations and three replicates give about ±4-point resolution. One backbone, memory framework, and judge. No attacks are run, and no act-time defense is tested.

## Citation

```bibtex
@misc{bhowmik2026priceofsafety,
  title         = {The Price of Safety: Benign-Case Utility and Token Overhead of Memory-Poisoning Defenses in LLM Agents},
  author        = {Bhowmik, Pritom},
  year          = {2026},
  eprint        = {2609.22818},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR}
}
```

## License

MIT. See [LICENSE](LICENSE).
