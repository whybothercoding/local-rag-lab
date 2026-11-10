"""Multilingual natural-language inference: is a sentence stated by a source passage?

Runs a pinned ONNX model (XNLI-trained, so English, German, French, and Greek are in its training data)
in-process with onnxruntime and the model's own tokenizer.json. The premise is always the source and
the hypothesis the answer sentence: swapping them inverts the meaning. Label order is read from the
model's config.json, never assumed.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from rag_lab.config import settings
from rag_lab.security.model_integrity import verify_model

_MAX_LENGTH = 512
_SESSIONS: dict[tuple[str, str], tuple[Any, Any, int]] = {}
_LOCK = threading.Lock()


class NLIUnavailableError(RuntimeError):
    """The NLI model cannot be loaded; callers fall back to the lexical check with a labelled note."""


def _load(model_name: str, cache_dir: str) -> tuple[Any, Any, int]:
    key = (model_name, cache_dir)
    if key in _SESSIONS:
        return _SESSIONS[key]
    with _LOCK:
        if key in _SESSIONS:
            return _SESSIONS[key]
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            from rag_lab.security.model_integrity import PINNED_MODELS

            snapshot = verify_model(model_name, cache_dir)
            weights = next(f for f in PINNED_MODELS[model_name]["files"] if f.endswith(".onnx"))
            labels = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))["id2label"]
            entailment = next(int(i) for i, name in labels.items() if name.lower() == "entailment")
            tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
            tokenizer.enable_truncation(max_length=_MAX_LENGTH)
            session = ort.InferenceSession(str(Path(snapshot) / weights), providers=["CPUExecutionProvider"])
        except Exception as exc:
            raise NLIUnavailableError(f"NLI model {model_name} could not be loaded: {exc}") from exc
        _SESSIONS[key] = (session, tokenizer, entailment)
        return _SESSIONS[key]


def entailment_scores(
    premises: list[str],
    hypothesis: str,
    model_name: str | None = None,
    cache_dir: str | None = None,
    batch_size: int = 16,
) -> list[float]:
    """Probability that each premise entails the hypothesis (softmax over the model's three labels)."""
    if not premises or not hypothesis.strip():
        return []
    model_name = settings.nli_model if model_name is None else model_name
    cache_dir = settings.models_dir if cache_dir is None else cache_dir
    session, tokenizer, entailment = _load(model_name, cache_dir)

    all_scores: list[float] = []
    # Sub-batch inference to prevent exceeding ONNX Runtime's 4GB single-tensor Tile limit in DeBERTa attention
    for i in range(0, len(premises), batch_size):
        chunk = premises[i : i + batch_size]
        encodings = tokenizer.encode_batch([(premise, hypothesis) for premise in chunk])
        width = max(len(e.ids) for e in encodings)
        ids = np.array([e.ids + [0] * (width - len(e.ids)) for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask + [0] * (width - len(e.attention_mask)) for e in encodings], dtype=np.int64)
        logits = session.run(None, {"input_ids": ids, "attention_mask": mask})[0]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        all_scores.extend(float(p[entailment]) for p in probs)

    return all_scores


def _vocabulary_overlap(hypothesis: str, premise: str) -> float:
    from rag_lab.lang import same_word, words
    from rag_lab.pipeline.text import _content_terms

    terms = _content_terms(hypothesis)
    if not terms:
        return 1.0
    premise_words = set(words(premise))
    return sum(1 for t in terms if any(same_word(t, w) for w in premise_words)) / len(terms)


def _titled(passage: str, hypothesis: str) -> list[str]:
    """Each sentence prefixed with the headings above it: "Alder office policy. Hotel cap. The hotel cap is 100 EUR."

    Headings are dropped from the plain premises, but a claim may name what only a heading states ("the Alder
    office"). Only headings written in the passage count, the plain sentence stays a premise too, and a titled
    one is offered only when the heading supplies words of the claim that the sentence lacks: otherwise it is
    noise (a contradicted German notice claim scored 0.72 against a premise carrying its heading).
    """
    from rag_lab.pipeline.text import split_sentences

    chain: list[tuple[int, str]] = []
    titled: list[str] = []
    block: list[str] = []

    def flush() -> None:
        if chain and block:
            prefix = ". ".join(title for _, title in chain)
            for sentence in split_sentences("\n".join(block)):
                if _vocabulary_overlap(hypothesis, f"{prefix}. {sentence}") > _vocabulary_overlap(hypothesis, sentence):
                    titled.append(f"{prefix}. {sentence}")
        block.clear()

    for line in passage.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            flush()
            level = len(stripped) - len(stripped.lstrip("#"))
            while chain and chain[-1][0] >= level:
                chain.pop()
            chain.append((level, stripped.lstrip("#").strip()))
        else:
            block.append(line)
    flush()
    return titled


def _premises(passages: list[str], hypothesis: str) -> list[str]:
    """Premises for one answer sentence: every source sentence, plus neighbouring pairs that share the
    claim's vocabulary.

    NLI models judge long premises poorly in both directions. A passage saying "2 months after probation"
    and "4 months from the 10th year" makes "the notice period after probation is 2 months" look
    contradicted, so single sentences are the base. Pairs keep claims that join two sentences, but a pair
    unrelated to the claim can be read as entailing it (calibration: a Greek leave passage "entailed"
    "receipts are due within 30 days" at 0.99), so a pair counts only if it shares settings.
    nli_pair_min_overlap of the claim's content words.
    """
    from rag_lab.pipeline.text import split_sentences

    singles: list[str] = []
    pairs: list[str] = []
    for passage in passages:
        body = "\n".join(line for line in passage.splitlines() if not line.lstrip().startswith("#"))
        sentences = split_sentences(body)
        singles += sentences
        singles += _titled(passage, hypothesis)
        pairs += [f"{a} {b}" for a, b in zip(sentences, sentences[1:])]
    pairs = [p for p in pairs if _vocabulary_overlap(hypothesis, p) >= settings.nli_pair_min_overlap]
    return list(dict.fromkeys(p for p in singles + pairs if p.strip()))


_VERBATIM_MIN_CHARS = 20  # a short fragment ("yes", "5 days") is not evidence by being found somewhere


def sentence_support(sentence: str, passages: list[str]) -> float:
    """Best entailment of one answer sentence by any premise built from the passages (see _premises)."""
    # A sentence quoted word for word from a source is stated by that source. The NLI model is not asked: it scored
    # "A single taxi ride may cost at most 141 EUR." against itself 0.55, below the 0.7 bar.
    folded = " ".join(sentence.lower().split())
    if len(folded) >= _VERBATIM_MIN_CHARS and any(folded in " ".join(p.lower().split()) for p in passages):
        return 1.0
    scores = entailment_scores(_premises(passages, sentence), sentence)
    return max(scores) if scores else 0.0
