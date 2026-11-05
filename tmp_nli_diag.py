"""Diagnostic: do the NLI scores depend on the CPU? Prints CPU flags, then scores a few claims with variants of the session."""
import os
import platform
import time

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from rag_lab.config import settings
from rag_lab.retrieval import nli
from rag_lab.security.model_integrity import PINNED_MODELS, verify_model

print("platform", platform.platform(), platform.machine(), "| onnxruntime", ort.__version__, "| cpus", os.cpu_count())
try:
    flags = next(line for line in open("/proc/cpuinfo") if line.startswith("flags")).split()
    print("cpu flags of interest:", [f for f in flags if f in ("avx", "avx2", "avx512f", "avx512_vnni", "avx_vnni", "avx512bw")])
    print("model name:", next(line for line in open("/proc/cpuinfo") if line.startswith("model name")).strip())
except OSError:
    print("no /proc/cpuinfo")

TRAVEL = open("eval/corpus/finance/travel.md", encoding="utf-8").read()
NOTICE_DE = open("eval/corpus/zurich/kuendigung.md", encoding="utf-8").read()
CLAIMS = [
    ("economy (true)", "Flights shorter than six hours are booked in economy.", TRAVEL),
    ("notice DE (true)", "Nach der Probezeit gilt eine Kündigungsfrist von 2 Monaten auf das Monatsende.", NOTICE_DE),
    ("business (FALSE)", "Business class is the default for flights under six hours.", TRAVEL),
]

name = settings.nli_model
snapshot = verify_model(name, settings.models_dir)
spec = PINNED_MODELS[name]
weights = next(f for f in spec["files"] if f.endswith(".onnx"))
print("pinned weights:", weights, round((snapshot / weights).stat().st_size / 1e6), "MB")
tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
tokenizer.enable_truncation(max_length=512)
import json
labels = json.loads((snapshot / "config.json").read_text())["id2label"]
entailment = next(int(i) for i, n in labels.items() if n.lower() == "entailment")


def session(path, **options):
    so = ort.SessionOptions()
    for key, value in options.pop("config", {}).items():
        so.add_session_config_entry(key, value)
    if "level" in options:
        so.graph_optimization_level = options["level"]
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def report(label, sess):
    nli._SESSIONS[(name, settings.models_dir)] = (sess, tokenizer, entailment)
    values = []
    started = time.perf_counter()
    for _, claim, passage in CLAIMS:
        values.append(nli.sentence_support(claim, [passage]))
    elapsed = time.perf_counter() - started
    print(f"{label:<38}", "  ".join(f"{n}={v:.3f}" for (n, _, _), v in zip(CLAIMS, values)), f"| {elapsed:.1f}s")


quantized = snapshot / weights
report("pinned int8 (as shipped)", session(quantized))
report("int8 + session.x64quantprecision=1", session(quantized, config={"session.x64quantprecision": "1"}))
report("int8 + optimizations disabled", session(quantized, level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL))

try:
    from huggingface_hub import hf_hub_download

    full = hf_hub_download(spec["repo"], "onnx/model.onnx", revision=spec["revision"], cache_dir=settings.models_dir + "/_diag")
    print("fp32 file:", round(os.path.getsize(full) / 1e6), "MB (plus external data if any)")
    report("fp32 model.onnx (reference)", session(full))
except Exception as exc:  # the repo may not ship one
    print("fp32 reference unavailable:", type(exc).__name__, str(exc)[:160])
