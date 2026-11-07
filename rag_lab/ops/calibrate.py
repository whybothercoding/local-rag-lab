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
    if not positives or not negatives:
        # A calibration split that produced no usable pairs cannot set a threshold: report that
        # instead of dying on min()/max() of an empty list.
        return {
            "threshold": name,
            "error": (
                f"not enough data: {len(positives)} positive and {len(negatives)} negative pairs. "
                "Calibrate with a larger or more varied report."
            ),
        }
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


_CLAUSE = {
    "en": ", but only with written approval from the CEO.",
    "de": ", aber nur mit schriftlicher Genehmigung des CEO.",
    "fr": ", mais uniquement avec l'accord écrit du PDG.",
    "el": ", αλλά μόνο με γραπτή έγκριση του CEO.",
}


def _perturbations(sentence: str, workspace: str, documents: list[str]) -> list[tuple[str, str]]:
    """Unsupported variants of a supported sentence: a changed figure, an invented condition, and a true
    sentence from another document of the same workspace."""
    from rag_lab.lang import detect_language

    variants: list[tuple[str, str]] = []
    if re.search(r"\d", sentence):
        variants.append(("number", re.sub(r"\d+", lambda m: str(int(m.group(0)) + 7), sentence)))
    lang = detect_language(sentence)
    variants.append(("condition", sentence.rstrip(".!;") + _CLAUSE.get(lang, _CLAUSE["en"])))
    retrieved = " ".join(" ".join(d.split()) for d in documents)
    for path in sorted((CORPUS / workspace).glob("*.md")):
        for other in _sentences(path.read_text(encoding="utf-8")):
            if " ".join(other.split()) not in retrieved and len(other) > 25:
                variants.append(("other_document", other))
                break
        if variants[-1][0] == "other_document":
            break
    return variants


def calibrate_nli(reports: list[Path]) -> dict:
    """Entailment of supported answer sentences vs perturbed ones, calibration split only.

    Supported: sentences of answers that found every expected fact and, in a model run, that an
    independent judge approved. Cross-lingual pairs (answer language != document language) are reported
    on their own, since that is where XNLI-trained models are weakest.
    """
    from rag_lab.lang import detect_language, is_no_info_sentence
    from rag_lab.pipeline.text import split_sentences
    from rag_lab.retrieval.nli import sentence_support

    positives: list[tuple[float, bool]] = []
    negatives: dict[str, list[float]] = {}
    for path in reports:
        for row in json.loads(path.read_text(encoding="utf-8"))["results"]:
            if (row["split"] != "calibration" or row["expect"] != "answer" or not row.get("passed")
                    or row.get("judge_supported") is False or not row.get("documents")):
                continue
            answer = re.sub(r"^Based on available documentation:\s*", "", row["answer"])
            doc_lang = detect_language(" ".join(row["documents"]))
            for sentence in split_sentences(answer):
                if is_no_info_sentence(sentence):
                    continue
                cross = detect_language(sentence) not in (doc_lang, "und")
                positives.append((sentence_support(sentence, row["documents"]), cross))
                for kind, variant in _perturbations(sentence, row["workspace"], row["documents"]):
                    negatives.setdefault(kind, []).append(sentence_support(variant, row["documents"]))
    # Changed figures are the number check's job (it runs first and is deterministic); NLI models are weak
    # at figures, so those pairs are reported but do not set the entailment threshold.
    all_negatives = [v for kind, values in negatives.items() if kind != "number" for v in values]
    summary = _summary("groundedness_min_entailment", [p for p, _ in positives], all_negatives, margin=0.05)
    if "error" in summary:
        return summary
    summary["suggested_midpoint"] = round((min(p for p, _ in positives) + max(all_negatives)) / 2, 2)
    cross_scores = [p for p, c in positives if c]
    summary["cross_lingual_positives"] = (
        [round(min(cross_scores), 3), round(max(cross_scores), 3), len(cross_scores)] if cross_scores else None)
    summary["negatives_by_kind"] = {k: [round(max(v), 3), len(v)] for k, v in negatives.items()}
    summary["positives"] = summary["positives"] + [len(positives)]
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Suggest model-dependent thresholds from the calibration split.")
    parser.add_argument("--report", type=Path, default=ROOT / "eval" / "out" / "report.json",
                        help="eval report from the same profile, for the retrieval similarity gate")
    parser.add_argument("--nli-reports", type=Path, nargs="*", default=[],
                        help="eval reports (deterministic and/or --llm, ideally judged) for the NLI threshold")
    args = parser.parse_args(argv)
    from rag_lab.config import settings

    print(f"Profile {settings.model_profile}: {settings.fastembed_model} + {settings.reranker_model}")
    if args.nli_reports:
        row = calibrate_nli(args.nli_reports)
        print(json.dumps(row, indent=1, ensure_ascii=False))
        return 0
    for row in calibrate(args.report):
        flag = "" if row["separated"] else "  (distributions overlap)"
        print(f"{row['threshold']:<40} current={row['current']:<7} suggested={row['suggested']:<7} "
              f"pos={row['positives']} neg={row['negatives']} accepted={row['positives_accepted_at_suggestion']}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
