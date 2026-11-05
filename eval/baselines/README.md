# Evaluation Baselines & Reference Benchmarks

This directory contains versioned reference runs of the offline evaluation suite (`rag_lab/ops/eval.py`) across the 159 golden cases in `eval/golden.json`.

## Reference Artifacts

* **`reference_eval_run.json`**:
  * **Engine**: LangGraph pipeline with deterministic context synthesizer and in-process ONNX embeddings (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`), cross-encoder reranker (`BAAI/bge-reranker-v2-m3-int8`), and fp32 multilingual NLI (`mDeBERTa-v3-base-xnli`).
  * **Test Split Cases**: 108 cases across English (28), Multilingual (71), and Conversation (9).
  * **Results**:
    * **English**: 100% retrieval recall, 100% keyword recall, 100% groundedness, 100% security accuracy, 100% cache accuracy.
    * **Multilingual**: 97.7% retrieval recall, 84.1% keyword recall, 91.1% groundedness, 97.7% entailment, 100% security accuracy.
    * **Conversation**: 100% retrieval recall, 77.8% keyword recall, 93.6% groundedness, 100% entailment.

## Reproducing

Run the evaluation suite locally:
```bash
make eval
```
To run with a live local LLM and LLM judge:
```bash
make eval-llm
```
