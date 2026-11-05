#!/usr/bin/env python3
"""Suggest the model-dependent thresholds for the active model profile, from the calibration split only.

Every model scores on its own scale, so these four values must be re-chosen whenever MODEL_PROFILE (or
FASTEMBED_MODEL / RERANKER_MODEL) changes:

- relevance_min_rerank_score (the primary relevance gate) and relevance_min_similarity (its fallback when
  the reranker is off): best passage scores of answerable vs out-of-scope questions, read from an eval
  report (`make eval` first; its calibration-split rows carry best_rerank_score and best_similarity).
- groundedness_cross_encoder_threshold: reranker logit of (answer, sources) for a correct answer vs an
  answer taken from another workspace.
- relevance_cross_encoder_threshold: reranker logit of (question, answer), correct vs unrelated.
- semantic_cache_similarity_threshold: embedding similarity of paraphrase pairs vs different questions.

A false accept is worse than a miss (an unsupported answer released, or another question's cached answer
served), so each suggestion sits above the highest negative score, with a margin, and the overlap between
the two distributions is reported. Run: `make calibrate-models [PROFILE=...]`.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
from pathlib import Path

from rag_lab import ROOT

GOLDEN = ROOT / "eval" / "golden.json"
CORPUS = ROOT / "eval" / "corpus"


def _sentences(text: str) -> list[str]:
    body = re.sub(r"(?s)\A---.*?---\s*", "", text)
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+|\n+", body) if s.strip() and not s.startswith("#")]


def _answer_sentence(case: dict) -> str | None:
    """The source sentence that holds the case's first expected fact: a correct, grounded answer."""
    doc = (CORPUS / case["workspace"] / case["expected_sources"][0]).read_text(encoding="utf-8")
    first = case["expected_keywords"][0]
    options = [first] if isinstance(first, str) else first
    for sentence in _sentences(doc):
        if any(option.lower() in sentence.lower() for option in options):
            return sentence
    return None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def _summary(name: str, positives: list[float], negatives: list[float], margin: float) -> dict:
    lowest_positive, highest_negative = min(positives), max(negatives)
    suggestion = round(highest_negative + margin, 2)
    accepted = sum(1 for p in positives if p >= suggestion) / len(positives)
    return {
        "threshold": name,
        "positives": [round(min(positives), 3), round(max(positives), 3)],
        "negatives": [round(min(negatives), 3), round(highest_negative, 3)],
        "separated": lowest_positive > highest_negative,
        "suggested": suggestion,
        "positives_accepted_at_suggestion": round(accepted, 3),
    }


def calibrate(report_path: Path | None) -> list[dict]:
    from rag_lab.config import settings
    from rag_lab.retrieval.embedding_provider import get_embedding_model
    from rag_lab.retrieval.reranker import score_pairs

    cases = [c for c in json.loads(GOLDEN.read_text(encoding="utf-8")) if c["split"] == "calibration"]
    answerable = [c for c in cases if c.get("expect", "answer") == "answer" and not c.get("history")]
    pairs = [(c, s) for c in answerable if (s := _answer_sentence(c))]
    results: list[dict] = []

    # Cross-encoder: correct answer vs an answer from another workspace, for both graders.
    grounded, ungrounded, relevant, irrelevant = [], [], [], []
    for (case, answer), (other, wrong) in zip(pairs, pairs[1:] + pairs[:1]):
        if other["workspace"] == case["workspace"]:
            continue
        docs = _sentences((CORPUS / case["workspace"] / case["expected_sources"][0]).read_text(encoding="utf-8"))
        grounded += [max(score_pairs(answer, docs) or [float("-inf")])]
        ungrounded += [max(score_pairs(wrong, docs) or [float("-inf")])]
        relevant += score_pairs(case["question"], [answer])
        irrelevant += score_pairs(case["question"], [wrong])
    results.append(_summary("groundedness_cross_encoder_threshold", grounded, ungrounded, margin=0.5))
    results.append(_summary("relevance_cross_encoder_threshold", relevant, irrelevant, margin=0.5))

    # Cache: paraphrases (same sources and facts) vs different questions in the same workspace.
    model = get_embedding_model()
    vectors = {c["id"]: model.embed_query(c["question"]) for c in answerable}
    same, different = [], []
    for a, b in itertools.combinations(answerable, 2):
        if a["workspace"] != b["workspace"]:
            continue
        sim = _cosine(vectors[a["id"]], vectors[b["id"]])
        if a["expected_sources"] == b["expected_sources"] and a["expected_keywords"] == b["expected_keywords"]:
            same.append(sim)
        else:
            different.append(sim)
    if same and different:
        results.append(_summary("semantic_cache_similarity_threshold", same, different, margin=0.02))

    # Relevance gate: needs retrieval, so it reads an eval report run under the same profile.
    if report_path and report_path.is_file():
        rows = [r for r in json.loads(report_path.read_text(encoding="utf-8"))["results"]
                if r["split"] == "calibration" and r.get("best_similarity") is not None and not r.get("history")]
        answer_sims = [r["best_similarity"] for r in rows if r["expect"] == "answer"]
        abstain_sims = [r["best_similarity"] for r in rows if r["expect"] == "abstain"]
        if answer_sims and abstain_sims:
            results.append(_summary("relevance_min_similarity", answer_sims, abstain_sims, margin=0.02))
        scored = [r for r in rows if r.get("best_rerank_score") is not None]
        answer_scores = [r["best_rerank_score"] for r in scored if r["expect"] == "answer"]
        abstain_scores = [r["best_rerank_score"] for r in scored if r["expect"] == "abstain"]
        if answer_scores and abstain_scores:
            results.append(_summary("relevance_min_rerank_score", answer_scores, abstain_scores, margin=0.5))

    for row in results:
        row["current"] = getattr(settings, row["threshold"])
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Suggest model-dependent thresholds from the calibration split.")
    parser.add_argument("--report", type=Path, default=ROOT / "eval" / "out" / "report.json",
                        help="eval report from the same profile, for the retrieval similarity gate")
    args = parser.parse_args(argv)
    from rag_lab.config import settings

    print(f"Profile {settings.model_profile}: {settings.fastembed_model} + {settings.reranker_model}")
    for row in calibrate(args.report):
        flag = "" if row["separated"] else "  (distributions overlap)"
        print(f"{row['threshold']:<40} current={row['current']:<7} suggested={row['suggested']:<7} "
              f"pos={row['positives']} neg={row['negatives']} accepted={row['positives_accepted_at_suggestion']}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
