# Offline evaluation

`rag_lab/ops/eval.py` (`make eval`) runs the real LangGraph pipeline over a fixed corpus and golden set, then gates on the results.
Everything runs locally: a fresh index is built in `eval/out/`, and no production store is read or written.

## Corpus and golden set

- `corpus/` holds synthetic policy documents in six workspaces: `engineering`, `finance`, `hr` (English),
  `zurich` (German, with one confidential file), `geneve` (French), and `athens` (Greek).
- `golden.json` has 159 cases. Each case has:
  - `suite`: `en` (50), `multilingual` (98: German, French, Greek, and cross-lingual questions), `conversation` (11 follow-ups).
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

`multilingual` and `conversation` are gated as well (thresholds in `SUITE_GATES`; `conversation` gates answer rate,
retrieval recall, keyword recall >= 0.7, groundedness and entailment).

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

The tables above are the record of their milestones, measured on an earlier, smaller `golden.json`. A deterministic
`make eval` on the current 159 cases (test split, 2026-11-05) gives:

| Suite | Mode | answer | recall | keywords | abstain | security | cache | judge |
|---|---|---|---|---|---|---|---|---|
| en | deterministic | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| multilingual | deterministic | 0.98 | 0.98 | 0.84 | 1.00 | 1.00 | 1.00 | - |
| conversation | deterministic | 1.00 | 1.00 | 0.78 | - | - | - | - |

A cache case is inconclusive when its earlier answer was never cached (a real model's terse answer went to
review instead), so nothing could have been reused. Deterministic runs require every cache case to be conclusive.

The gaps the baseline measured, now closed: German, French, and Greek prompt injections are not blocked (the patterns are
English), negated German/French/Greek questions reuse the cached answer to the opposite question (the negation list
is English), and the English-only embedding and reranker models miss or mis-rank non-English text.

## Groundedness by entailment

Every answer sentence must be entailed (probability >= 0.7) by a retrieved passage, judged by
`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (MIT; full-precision ONNX, 1.1 GB; see *Why this
model* below). Phase 4 first shipped `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` in int8; the table below was measured
with that model on Apple silicon, where int8 and fp32 agree to under 0.01, and is kept as the record of the method.
Calibrated with `make calibrate-models` style runs
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
share at least a quarter of the claim's content words separate the two cleanly. With qwen3:8b (test split) every suite clears the 0.9 entailment gate (en 0.94, multilingual 1.0,
conversation 1.0). The one rejected answer added a condition the source does not state ("... sign-off before
reimbursement") and went to review, as intended. Attribution openers ("Selon les documents, ...") are stripped
before the check and "the documents do not provide ..." sentences are not claims; both phrase lists live in the
language data.

### Why this model (and why not int8)

CI was the first run on another CPU, and the int8 file failed there: on an AMD EPYC (x86, AVX-512 VNNI) the same
sentence a Mac scored 0.93 scored 0.12, and a false claim scored 0.64 instead of 0.01. Across a 42-claim battery 25
claims differed by more than 0.1 between the two machines (largest 0.81). A session option did not change it
(`session.x64quantprecision`), and switching graph optimizations off changed the numbers on both machines, so the
quantized graph itself is the unstable part. A grader whose verdict depends on the CPU is not usable in production.
The full-precision weights gave **identical scores on both machines** (largest difference under 0.0001), for every
candidate tried.

Candidates, all fp32, on the 42-claim labelled battery (English, German, French, Greek; true claims, wrong figures,
wrong documents), identical on arm64 and x86:

| Model | AUC | lowest true / highest false | wrong at the 0.7 gate | size | ms per claim (M-series / 2-vCPU x86) |
|---|---|---|---|---|---|
| `mDeBERTa-v3-base-mnli-xnli` (the Phase 4 model) | 0.977 | 0.93 / 1.00 | 2 of 42 | 1.1 GB | 99 / 269 |
| `mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | **1.000** | 0.87 / 0.35 | **0 of 42** | 1.1 GB | 106 / 272 |
| `multilingual-MiniLMv2-L6-mnli-xnli` | 0.973 | 0.43 / 0.99 | 2 of 42 | 0.4 GB | 10 / 32 |

On every sentence of the eval corpus (174 claims: the sentence itself, the same sentence against another document,
the sentence with an invented condition appended) the chosen model rejected every unrelated sentence
(highest unrelated score 0.11; the Phase 4 model reached 0.91 for a French one) and rejected invented conditions in
German, French and Greek. Its one miss, an invented English condition on the change-freeze sentence, is missed by all
three. On 52 real qwen3:8b answer sentences, each grounded in the right document, it rejected 3 (0.21 for a German
sentence hedged with "normalerweise", 0.48 for a French "Non, ... ne sont pas perdus", 0.67 for a Greek paraphrase),
where the Phase 4 model rejected none. That is the price: a few more correct answers go to human review (the safe
side), in exchange for far fewer wrong claims released. Greek is **not** in this model's NLI fine-tuning data (its
card lists Greek as a zero-shot test language, which is why it was passed over in Phase 4); on every measurement above
it still separated Greek claims cleanly (true claims 0.67 and up, in the corpus checks false claims 0.09 and below), and the gate stays 0.7.
If review volume on Greek answers turns out to be high, that is the first thing to look at.

The int8 reranker (`bge-reranker-v2-m3-int8`, default profile) was checked the same way: scores differed by at most
0.56 logits between the machines on a 17-logit range, all 10 queries ranked the right document first on both, and the
smallest margin was 2.46 against 2.51. That is the size of ordinary quantization error (int8 against fp32 on one
machine: 0.62), not a swing, so it stays.

`tests/test_nli.py` keeps 17 labelled claims with wide margins (true at or above 0.8, false at or below 0.5) and refuses
quantized NLI weights, so a model swap or a numerics change shows up in the suite on any machine.

## Why there is no Ollama embedding path

An Ollama embedding path (`EMBEDDING_PROVIDER=ollama`, `nomic-embed-text`) was removed after it was measured on the eval
corpus. With it the multilingual gates fail, because every threshold is calibrated for the ONNX model (retrieval recall
0.75, abstention accuracy 0.27, both required >= 0.9); its model cannot be hash-pinned; and it was about 2.5x slower per
query (11 ms vs 4.5 ms). Retrieval quality was close (recall@1 0.90 vs 0.93, a three-case difference of 91; recall@3 1.00
for both) and was not the reason: as integrated it never added the task prefixes `nomic-embed-text` expects. A leftover
`EMBEDDING_PROVIDER=ollama` now fails with instructions; indexes built with it need `make sync FORCE=1`.

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
