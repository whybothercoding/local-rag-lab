"""Embeddings of fixed texts from the pinned default embedding model; run on two machines and compare the JSON.
Prints the vectors' fingerprint so differences show, plus the similarities retrieval actually uses. Run from the repo root."""
import glob
import json
import platform
import sys
from pathlib import Path

from rag_lab.config import settings

import numpy as np

from rag_lab.retrieval import embedding_provider as ep

provider = ep.LocalFastEmbeddings()

texts = []
for path in sorted(glob.glob("eval/corpus/*/*.md")):
    texts += [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith(("#", "---"))]
questions = [c["question"] for c in json.load(open("eval/golden.json"))][:60]

docs = np.array(provider.embed_documents(texts), dtype=np.float64)
qs = np.array([provider.embed_query(q) for q in questions], dtype=np.float64)
docs /= np.linalg.norm(docs, axis=1, keepdims=True)
qs /= np.linalg.norm(qs, axis=1, keepdims=True)
sims = qs @ docs.T  # what retrieval ranks by
out = {"platform": f"{platform.machine()} {platform.system()}", "model": settings.fastembed_model,
       "docs": docs.round(5).tolist(), "queries": qs.round(5).tolist(), "n_docs": len(texts), "n_queries": len(questions)}
print(out["platform"], settings.fastembed_model, len(texts), "passages,", len(questions), "questions", flush=True)
Path(sys.argv[1]).write_text(json.dumps(out))
import base64, gzip
print("EMBED_B64", base64.b64encode(gzip.compress(json.dumps(out).encode())).decode())
print("EMBED_SIMS_SAMPLE", np.round(sims[0, :5], 4).tolist())
