"""Language heuristics shared by retrieval and grading: content words, coverage, and stated numbers.

Everything compares folded words (rag_lab.lang): any script, case- and accent-insensitive.
"""
from __future__ import annotations

from typing import List
import re


from rag_lab.lang import ARTICLE_NUMBERS, NUMBER_WORDS, STEM_SUFFIXES, STOPWORDS, fold, words

# Folded words in any script (see rag_lab.lang): kept under this name for the graders and the router.
_WORD = re.compile(r"[^\W_]+")
_NUMBER_WORDS = NUMBER_WORDS


_NUMBER = re.compile(r"\d+(?:[.,:]\d+)*")


_LIST_MARKER = re.compile(r"(?m)^\s*\d+[.)]\s")


def _numbers(text: str, *, include_one: bool = True) -> set[str]:
    """Numbers in a text, with spelled-out ones folded to digits ("five" == "5", "1,000" == "1000")."""
    lowered = re.sub(r"(?<=\d),(?=\d{3}\b)", "", fold(text))
    found = {m.rstrip(".,:") for m in _NUMBER.findall(lowered)}
    found |= {
        NUMBER_WORDS[w]
        for w in words(lowered)
        if w in NUMBER_WORDS and (include_one or w not in ARTICLE_NUMBERS)
    }
    return found


def _unsupported_numbers(generation: str, documents: List[str], question: str) -> set[str]:
    """Numbers an answer states that neither the retrieved context nor the question contains."""
    stated = _numbers(_LIST_MARKER.sub("", generation), include_one=False)
    return stated - _numbers(" ".join(documents)) - _numbers(question)


def _keywords(text: str) -> list[str]:
    """Distinct content words in order, dropping filler. Used to turn a question into a search query."""
    seen: dict[str, None] = {}
    for word in words(text):
        if word not in STOPWORDS and (len(word) >= 3 or word.isdigit() or re.fullmatch(r"[^\W\d_]\d+", word)):
            seen.setdefault(word)
    return list(seen)


def _stem_variants(words: list[str]) -> list[str]:
    variants: dict[str, None] = {}
    for word in words:
        for suffix in STEM_SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                variants.setdefault(word[: -len(suffix)])
                break
    return [v for v in variants if v not in words]


def _content_terms(text: str) -> set[str]:
    return {word for word in words(text) if len(word) >= 4 and word not in STOPWORDS}


def _coverage(query_terms: set[str], chunk: str) -> float:
    """Share of the query's content words that appear in one chunk."""
    if not query_terms:
        return 0.0
    return len(query_terms & set(words(chunk))) / len(query_terms)


# Sentence ends: . ! ? followed by whitespace, and ; only after a Greek letter (the Greek question mark; in
# German and French ";" joins clauses). Not after a lone one- or two-digit number ("2. Klasse",
# "1. Januar"); times and decimals ("02:00", "4.5") never meet whitespace.
_SENTENCE_END = re.compile(r"(?:(?<=[.!?])(?<!\b\d\.)(?<!\b\d\d\.)|(?<=[\u0370-\u03ff\u1f00-\u1fff];))\s+")
_BULLET = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+")


def split_sentences(text: str) -> list[str]:
    """Sentences of an answer, one per bullet or line, for checking each claim on its own."""
    sentences: list[str] = []
    for line in _BULLET.sub("", text).splitlines():
        sentences += [s.strip() for s in _SENTENCE_END.split(line) if s.strip()]
    return sentences
