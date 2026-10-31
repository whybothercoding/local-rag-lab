# Offline evaluation

`eval.py` runs the real LangGraph pipeline over a fixed corpus and golden set, then gates on the results.
Everything runs locally: a fresh index is built in `eval/out/`, and no production store is read or written.

## Corpus and golden set

- `corpus/` is a small synthetic set of policy documents across three workspaces (engineering, finance, hr).
- `golden.json` has 40 cases. Each case has a `split`:
  - `calibration` (21 cases) is used to choose thresholds and relevance settings.
  - `test` (19 cases) is held out. Gates are measured on this split only.
- `expect` is `answer` (default), `abstain`, `blocked` (prompt injection), or `denied` (no grant).

## Running

```bash
make eval                         # gated run: prints the test split and fails below thresholds
./venv/bin/python eval.py --calibrate   # per-case retrieval signals for the calibration split
make eval-judge                   # also asks the local Ollama model whether each answer is supported (reported, not gated)
```

## Gates (test split)

| Metric | Gate | Meaning |
|---|---|---|
| answer_rate | >= 0.9 | Answerable questions that were answered rather than abstained on |
| retrieval_recall | >= 0.9 | Expected source documents were retrieved |
| keyword_recall | >= 0.8 | Expected key facts appear in the answer |
| groundedness | >= 0.5 | Share of answer vocabulary found in the retrieved context |
| abstention_accuracy | >= 0.8 | Out-of-scope questions were declined |
| security_accuracy | = 1.0 | Injections were blocked and ungranted accounts were denied |

Reported but not gated: `context_precision` (capped by the small corpus and top-3 retrieval) and `judge_support`.

## Relevance gate calibration

Retrieval abstains unless the best chunk has cosine similarity >= 0.65 or covers >= 40% of the query's content words.
On the calibration split, answerable questions scored similarity 0.72-0.91 and coverage 0.50-1.00.
Out-of-scope questions scored similarity 0.42-0.54 and coverage 0.00-0.25.

The margin is narrow and the calibration set is small. Revisit the thresholds whenever the corpus or the embedding model changes.

## Known limitations

- The default answer source is the deterministic synthesizer, which picks sentences by lexical overlap. It cannot match paraphrases such as "roll back" against "rollback", so keyword recall is lower than a model-written answer would achieve. Run with `LLM_PROVIDER=ollama` to score the real model.
- The judge is a 7B model and disagrees with human reading on some cases. Treat `judge_support` as a signal to investigate, not as a verdict.
