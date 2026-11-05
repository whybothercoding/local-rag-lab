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
| judge_support | >= 0.8 (`--llm` only) | The judge found the answer supported by its context |

`multilingual` and `conversation` are measured and reported but not gated yet: they get gates once the multilingual
foundation (models, injection patterns, language data) and history-aware follow-ups land.

### Baseline before the multilingual work (test split)

| Suite | Mode | answer | recall | keywords | abstain | security | cache | judge |
|---|---|---|---|---|---|---|---|---|
| en | deterministic | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| en | llm (qwen2.5-coder:7b) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| multilingual | deterministic | 0.76 | 0.76 | 0.66 | 0.30 | 0.33 | 0.00 | - |
| multilingual | llm | 0.78 | 0.78 | 0.73 | 0.30 | 0.33 | 0.00 | 0.94 |
| conversation | deterministic | 0.78 | 0.78 | 0.56 | - | - | - | - |
| conversation | llm | 0.78 | 0.78 | 0.67 | - | - | - | 1.00 |

The multilingual gaps this measures: German, French, and Greek prompt injections are not blocked (the patterns are
English), negated German/French/Greek questions reuse the cached answer to the opposite question (the negation list
is English), and the English-only embedding and reranker models miss or mis-rank non-English text.

## Relevance gate calibration

Retrieval abstains unless the best chunk has cosine similarity >= 0.65 or covers >= 40% of the query's content words.
On the calibration split, answerable questions scored similarity 0.72-0.91 and coverage 0.50-1.00.
Out-of-scope questions scored similarity 0.42-0.54 and coverage 0.00-0.25.

The margin is narrow and the calibration set is small. Revisit the thresholds whenever the corpus or the embedding model changes.

## Known limitations

- The default answer source is the deterministic synthesizer, which picks sentences by lexical overlap. It cannot match paraphrases such as "roll back" against "rollback", so keyword recall is lower than a model-written answer would achieve. Run `make eval-llm` to score the real model.
- The judge is a 7B model and, by default, the same model that wrote the answers, so it is lenient (it approved 94-100% in the baseline). Use `EVAL_JUDGE_MODEL` for an independent judge, and treat `judge_support` as a signal to investigate, not as a verdict.
