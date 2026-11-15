# Evaluation and measured findings

`make eval` runs the real LangGraph pipeline over a fixed synthetic corpus and a golden set of 201 cases, then gates on the results. CI enforces the gates.
The corpus has six workspaces: engineering, finance, hr (English), zurich (German), geneve (French) and athens (Greek).
The suites are en, multilingual, conversation, messy (DOCX, XLSX and PDF), graph (multi-hop) and stress (near-duplicate files).
Each case is either `calibration`, which chooses thresholds, or `test`, which is held out and gated.
Expected outcomes are answer, abstain, blocked (prompt injection), denied (no grant), fresh and cached.

## Honest limits

Most synthetic suites score 1.00 on their held-out split, so they are saturated: they catch regressions but cannot rank improvements.
The author wrote both the corpus and the questions. The gated run uses a deterministic extractive answerer, not a model.
`make eval-external` runs 895 cases from the public XQuAD benchmark. It is reported, not gated.
`make eval-llm` scores the real model and runs weekly and on demand. It never gates.

## Measured findings

1. int8 NLI weights give different verdicts on different CPUs. On an AMD EPYC x86 machine the int8 model scored one sentence 0.12 where an Apple-silicon Mac scored it 0.93. The fp32 weights matched to under 0.0001, so the NLI model is pinned in full precision and a test refuses quantized NLI weights.
2. Candidate-pool size only matters at the extreme. On the 72-chunk stress suite, recall was 0.92 at a pool of 1+1 and 1.00 from 2+2 up. The default pool of 5+5 stays.
3. Running the real model found a grader defect the deterministic run could not: an office name that appeared only in a heading was never offered as an NLI premise. Titled premises fixed it.
4. The Ollama embedding path was removed after measurement. Against the thresholds calibrated for the ONNX model it scored retrieval recall 0.75 and abstention accuracy 0.27, and it could not be hash-pinned.

## Other eval commands

`make eval-graph` runs the GraphRAG ablation, with graph retrieval off and then on.
`make eval-judge` adds an LLM-as-judge from local Ollama.
`make eval-calibrate` prints retrieval calibration signals.
