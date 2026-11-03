# Offline evaluation

`rag_lab/ops/eval.py` (`make eval`) runs the real LangGraph pipeline over a fixed corpus and golden set, then gates on the results.
Everything runs locally: a fresh index is built in `eval/out/`, and no production store is read or written.

## Corpus and golden set

- `corpus/` holds synthetic policy documents in six workspaces: `engineering`, `finance`, `hr` (English),
  `zurich` (German, with one confidential file), `geneve` (French), and `athens` (Greek).
- `golden.json` has 142 cases. Each case has:
  - `suite`: `en` (45), `multilingual` (86: German, French, Greek, and cross-lingual questions), `conversation` (11 follow-ups).
  - `lang`: the question's language (`en`, `de`, `fr`, `el`). Cross-lingual cases ask in one language about documents in another.
  - `split`: `calibration` chooses thresholds; `test` is held out and gated.
  - `expect`: `answer` (default), `abstain`, `blocked` (prompt injection), `denied` (no grant), `fresh` (a negated or
    renumbered question must not reuse the cached answer to its `setup` question), or `cached` (control: it must).
  - `history`: earlier turns asked in the same conversation before a follow-up. `setup`: earlier questions in other
    conversations, sharing only the cache.
  - `expected_keywords`: facts the answer must contain; a list inside the list accepts any alternative, so an answer
    in the document's or the question's language both count (`["montag", "monday"]`).

## Running

```bash
make eval                     # deterministic synthesizer, all suites; this is what CI gates
make eval-llm [SUITE=en]      # answers from the local Ollama model, LLM judge gated (needs 'ollama serve')
make eval-judge               # deterministic answers plus an advisory judge
make eval-calibrate           # per-case retrieval signals for the calibration split
```

`EVAL_JUDGE_MODEL` picks a different Ollama model as judge, so the answering model does not grade itself.
`EVAL_LLM_JUDGE_MIN` sets the judge gate (default 0.8). `EVAL_OUT_DIR` moves the stores and report out of `eval/out/`.

## Gates (per suite, test split)

| Metric | `en` gate | Meaning |
|---|---|---|
| answer_rate | >= 0.9 | Answerable questions that were answered rather than abstained on |
| retrieval_recall | >= 0.9 | Expected source documents were retrieved |
| keyword_recall | >= 0.8 | Expected key facts appear in the answer |
| groundedness | >= 0.5 | Share of answer vocabulary found in the retrieved context |
| abstention_accuracy | >= 0.8 | Out-of-scope questions were declined |
| security_accuracy | = 1.0 | Injections were blocked and ungranted accounts were denied |
| cache_accuracy | = 1.0 | Negated or renumbered questions were not served a cached answer; the control was |
| entailment_rate | >= 0.9 (every suite) | Every claim of the answer is entailed by a retrieved passage (NLI) |
| judge_support | >= 0.8 (`--llm` only) | The judge found the answer supported by its context |

`multilingual` is gated as well (thresholds in `SUITE_GATES`). `conversation` is measured and reported but not
gated until follow-up questions use the conversation history.

### Baseline before the multilingual work (test split)

| Suite | Mode | answer | recall | keywords | abstain | security | cache | judge |
|---|---|---|---|---|---|---|---|---|
| en | deterministic | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| en | llm (qwen2.5-coder:7b) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| multilingual | deterministic | 0.76 | 0.76 | 0.66 | 0.30 | 0.33 | 0.00 | - |
| multilingual | llm | 0.78 | 0.78 | 0.73 | 0.30 | 0.33 | 0.00 | 0.94 |
| conversation | deterministic | 0.78 | 0.78 | 0.56 | - | - | - | - |
| conversation | llm | 0.78 | 0.78 | 0.67 | - | - | - | 1.00 |

### After the multilingual foundation (default `multilingual` profile, test split)

| Suite | Mode | answer | recall | keywords | abstain | security | cache | judge |
|---|---|---|---|---|---|---|---|---|
| en | deterministic | 1.00 | 1.00 | 0.93 | 1.00 | 1.00 | 1.00 | - |
| en | llm (qwen2.5-coder:7b) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 (1 inconclusive) | 1.00 |
| multilingual | deterministic | 0.98 | 0.98 | 0.80 | 1.00 | 1.00 | 1.00 | - |
| multilingual | llm | 0.98 | 0.98 | 0.93 | 1.00 | 1.00 | 1.00 (2 inconclusive) | 0.93 |
| conversation | llm | 1.00 | 1.00 | 0.89 | - | - | - | 0.89 |

A cache case is inconclusive when its earlier answer was never cached (a real model's terse answer went to
review instead), so nothing could have been reused. Deterministic runs require every cache case to be conclusive.

The gaps the baseline measured, now closed: German, French, and Greek prompt injections are not blocked (the patterns are
English), negated German/French/Greek questions reuse the cached answer to the opposite question (the negation list
is English), and the English-only embedding and reranker models miss or mis-rank non-English text.

## Groundedness by entailment

Every answer sentence must be entailed (probability >= 0.7) by a retrieved passage, judged by
`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` (MIT, XNLI-trained: English, German, French, Greek; quantized ONNX,
340 MB; quantization changed scores by under 0.01). Calibrated with `make calibrate-models` style runs
(`python -m rag_lab.ops.calibrate --nli-reports ...`) on calibration-split answers from the deterministic,
qwen3:8b, and gemma4:12b runs, against perturbed negatives:

| Pairs (calibration split) | entailment |
|---|---|
| supported sentences (103, incl. 12 cross-lingual) | 0.988 - 0.999 |
| a sentence from another document of the workspace | <= 0.375 |
| the same sentence with an invented condition | <= 0.75 (en), <= 0.2 (de/fr/el) |
| the same sentence with a changed figure | up to 0.997: NLI is weak at numbers, which is why the number check runs first |

Two findings shaped the premises: whole passages made true claims look contradicted ("2 months after probation"
next to "4 months from year 10" scored 0.002), and unrelated two-sentence premises could "entail" a claim (a Greek
leave passage, 0.995, for a sentence about expense receipts). Single source sentences plus neighbour pairs that
share at least a quarter of the claim's content words separate the two cleanly. MiniLM-L6-mnli-xnli was rejected
(paraphrases scored 0.37); the 2mil7 mDeBERTa variant does not list Greek.

## Chat model choice

Compared with `make eval-llm` (`OLLAMA_MODEL=...`) on an Apple M4 with 16 GB, thinking off. All licences are
Apache 2.0 (checked on the model cards). Every candidate passed every gate; the calibration split tied, so the
test split, memory, and an independent judge (`--judge-only`, a model from the other family) decided.

| Model | keywords (test) | groundedness | median latency | memory in use | judge en / multi / conversation |
|---|---|---|---|---|---|
| qwen2.5-coder:7b (previous) | 0.94 | 0.73 | 1.37 s | 4.8 GB | 0.88 / 0.98 / 0.67 |
| **qwen3:8b (default)** | **0.97** | 0.69 | 1.57 s | **5.1 GB** | **0.94 / 0.95 / 0.78** |
| gemma4:12b | 0.99 | 0.87 | 1.47 s | 16 GB | 0.94 / 0.95 / 0.89 |
| gemma4:e4b | 0.90 | 0.89 | 0.54 s | 13 GB | - |

`qwen3:8b` is the default because it fits a 16 GB machine with room for everything else. On a machine with 32 GB or
more, `OLLAMA_MODEL=gemma4:12b` answers follow-ups better. Apertus (Swiss, Apache 2.0, 1,800+ languages) has no
official Ollama build yet, so it was not compared.

## Threshold calibration (per model profile)

Model scores live on different scales, so every profile has its own thresholds (`rag_lab/config.py`), chosen
with `make calibrate-models PROFILE=...` on the calibration split only. A false accept is worse than a miss,
so each threshold sits above the highest score of the wrong class.

| Signal (calibration split) | `multilingual` | `multilingual-large` | `en` |
|---|---|---|---|
| Relevance gate | reranker logit >= -4.5 (answerable -1.8..6.6, out-of-scope -11.0..-7.1) | reranker >= -4.5 (-2.2..6.8 vs -11.0..-7.2) | similarity >= 0.65 or coverage >= 0.40 |
| Groundedness (cross-encoder fallback) | >= -3.0 (correct 10.7..10.8, wrong <= -6.6) | >= -3.0 (10.8 vs <= -7.0) | >= -3.5 |
| Answer relevance (cross-encoder fallback) | >= -6.0 (correct -2.7..5.1; unrelated <= -9.1, one same-topic distractor -4.6) | >= -6.0 | >= -6.0 |
| Semantic cache similarity | >= 0.92 (different questions up to 0.90) | >= 0.94 (up to 0.92) | >= 0.92 |

With the multilingual embedder, vector similarity cannot separate in-scope from out-of-scope questions
(answerable as low as 0.19, unrelated documents up to 0.49), so those profiles gate relevance on the reranker,
which reads question and passage together. The English profile's reranker cannot judge other languages, so it
keeps the similarity gate. The keyword index uses the `trigram` tokenizer: keyword-only recall@3 over the 98
answerable cases was 86 (trigram) vs 80 (unicode61, porter).

## Known limitations

- The default answer source is the deterministic synthesizer, which picks sentences by lexical overlap. It cannot match paraphrases such as "roll back" against "rollback", so keyword recall is lower than a model-written answer would achieve. Run `make eval-llm` to score the real model.
- The judge is a 7B model and, by default, the same model that wrote the answers, so it is lenient (it approved 94-100% in the baseline). Use `EVAL_JUDGE_MODEL` for an independent judge, and treat `judge_support` as a signal to investigate, not as a verdict.
