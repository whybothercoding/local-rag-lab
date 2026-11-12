# Design decisions

For each decision: what was chosen, what was rejected, and the evidence. Every figure comes from [`eval/README.md`](../eval/README.md) or a commit message cited here. Each entry says whether the choice was **measured** or only **reasoned**; where the repository records no rationale, that is stated rather than filled in. All measurements are on the synthetic eval corpus written by the author, so they show a choice beating its alternatives there, not on other data.

## 1. fp32 NLI weights, not int8 — measured

- **Chosen:** the groundedness grader (`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`) is pinned in full precision (1.1 GB).
- **Rejected:** int8 quantized weights, which the first version shipped (the int8 build of `mDeBERTa-v3-base-mnli-xnli`).
- **Evidence:** CI was the first run on another CPU. On an AMD EPYC (x86, AVX-512 VNNI) a sentence a Mac scored 0.93 scored 0.12, and a false claim scored 0.64 instead of 0.01. Across a 42-claim battery 25 claims differed by more than 0.1 between the two machines (largest 0.81). A session option did not help and switching graph optimizations off changed the numbers on both machines, so the quantized graph itself is unstable. fp32 scores matched on both machines to under 0.0001 for every candidate tried.
- **Cost:** on 52 real qwen3:8b answer sentences, each grounded in the right document, the chosen model rejected 3 where the previous model rejected none; those answers go to human review.
- **Guard:** `tests/test_nli.py` refuses quantized NLI weights.
- **Not generalised:** the int8 *reranker* stays. The same cross-machine check gave a largest score difference of 0.56 logits on a 17-logit range and the same top document for all 10 queries on both machines.

## 2. No Ollama embedding path — measured, one confound

- **Chosen:** in-process ONNX embeddings are the only provider.
- **Rejected:** `EMBEDDING_PROVIDER=ollama` with `nomic-embed-text`.
- **Evidence:** with it the multilingual gates failed (retrieval recall 0.75, abstention accuracy 0.27; both gated at 0.9) because every threshold is calibrated for the ONNX model; its model cannot be hash-pinned; it was about 2.5x slower per query (11 ms vs 4.5 ms).
- **Confound, stated in `eval/README.md`:** retrieval quality alone was close (recall@1 0.90 vs 0.93, a three-case difference out of 91; recall@3 1.00 for both) and was not the reason. As integrated, the path never added the task prefixes `nomic-embed-text` expects, so the gate failure partly reflects thresholds tuned to the other model, not only embedding quality.
- Commits: `fd7e252` (remove the silent Ollama fallback, fingerprint indexes), `957bed8` (drop the path).

## 3. Reranker-gated relevance in the multilingual profile — measured on the calibration split

- **Chosen:** in the `multilingual` and `multilingual-large` profiles an answerable question must have a best reranker logit of at least -4.5. The `en` profile keeps a cosine-similarity gate (0.65) because its reranker cannot judge other languages.
- **Rejected:** a cosine-similarity gate for the multilingual profiles.
- **Evidence (calibration split):** with the multilingual embedder, similarity cannot separate in-scope from out-of-scope questions (answerable as low as 0.19, unrelated documents up to 0.49), while the reranker, which reads question and passage together, does (answerable -1.8 to 6.6; out-of-scope -11.0 to -7.1).
- **Not measured:** the margin was calibrated on one author-written split; there is no run on a third-party corpus, so the -4.5 threshold is untested outside it. See the external slice in `eval/README.md`.

## 4. Trigram FTS5 tokenizer — measured, small difference

- **Chosen:** `SPARSE_TOKENIZER=trigram`, plus query words also tried without their inflection ending.
- **Rejected:** `unicode61` and `porter`. Porter's English stemming mangles other languages.
- **Evidence:** keyword-only recall@3 over the 98 answerable multilingual golden questions: porter 80, unicode61 80, trigram 86 (EN 34 to 36, DE 16 to 18, EL 13 to 15). The end-to-end eval cannot separate the three on this corpus, which is why the keyword index was measured on its own (commit `e6dd898`).
- **Weakness:** six questions on one author-written set, no interval, no significance test. Trigram also needs query words of at least three characters.

## 5. Two separate circuit breakers — reasoned, not measured

- **Chosen:** `MAX_RETRIEVAL_RETRIES` (default 2, query rewrites) and `MAX_GENERATION_RETRIES` (default 1, regenerations with grader feedback), each counted separately.
- **Rejected:** one shared retry budget (the original design capped retries at 2 with a single breaker).
- **Recorded reason (commit `2cbe4e3`):** "so one loop cannot spend the other's attempts." A regeneration with an identical prompt would repeat the same answer at temperature 0, so the grader's reason is added to the prompt. The deterministic extractive answerer escalates straight to review instead of retrying. The worst retry path is 26 graph steps, which is why the `langgraph` floor was raised to 1.1.0 (the old default recursion limit was 25).
- **Not measured:** there is no sweep over the two defaults (2 and 1) and no comparison of one shared budget against two. The values are judgement calls; larger models may justify higher ones (`.env`).

## 6. No OCR — reasoned, no measurement recorded

- **Chosen:** PDF extraction is plain text through `pypdf`; a PDF with no text layer is refused at ingestion with a note, and images are not read.
- **Rejected:** bundling an OCR engine.
- **Recorded reason:** `docs/usage.md` calls it "by design, to stay simple and offline". No commit or document records a measurement or a comparison of OCR engines, so there is no evidence here that OCR would not be worth its cost.
- **Known gap:** scanned documents cannot be searched; the messy-document suite does not cover them (`eval/README.md`).
