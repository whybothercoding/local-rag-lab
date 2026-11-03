"""Language heuristics shared by retrieval and grading: content words, coverage, and stated numbers."""
from __future__ import annotations

from typing import List
import re


STOPWORDS = frozenset({
    "what", "which", "when", "where", "who", "whom", "whose", "why", "how", "does", "did", "have",
    "has", "had", "with", "from", "into", "about", "there", "their", "your", "this", "that", "these",
    "those", "must", "should", "would", "could", "under", "over", "between", "after", "before", "tell",
    "give", "show", "list", "they", "them", "were", "been", "being", "will", "shall", "each", "every",
    "some", "any", "details", "overview",
})


_WORD = re.compile(r"[a-z0-9]+")


_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60",
    "ninety": "90", "hundred": "100", "thousand": "1000",
}


_NUMBER = re.compile(r"\d+(?:[.,:]\d+)*")


_LIST_MARKER = re.compile(r"(?m)^\s*\d+[.)]\s")


def _numbers(text: str, *, include_one: bool = True) -> set[str]:
    """Numbers in a text, with spelled-out ones folded to digits ("five" == "5", "1,000" == "1000")."""
    lowered = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text.lower())
    found = {m.rstrip(".,:") for m in _NUMBER.findall(lowered)}
    found |= {
        _NUMBER_WORDS[w]
        for w in _WORD.findall(lowered)
        if w in _NUMBER_WORDS and (include_one or w != "one")
    }
    return found


def _unsupported_numbers(generation: str, documents: List[str], question: str) -> set[str]:
    """Numbers an answer states that neither the retrieved context nor the question contains."""
    stated = _numbers(_LIST_MARKER.sub("", generation), include_one=False)
    return stated - _numbers(" ".join(documents)) - _numbers(question)


def _keywords(text: str) -> list[str]:
    """Distinct content words in order, dropping filler. Used to turn a question into a search query."""
    seen: dict[str, None] = {}
    for word in _WORD.findall(text.lower()):
        if word not in STOPWORDS and (len(word) >= 3 or word.isdigit() or re.fullmatch(r"[a-z]\d+", word)):
            seen.setdefault(word)
    return list(seen)


def _stem_variants(words: list[str]) -> list[str]:
    variants: dict[str, None] = {}
    for word in words:
        for suffix in ("ing", "ed", "es", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                variants.setdefault(word[: -len(suffix)])
                break
    return [v for v in variants if v not in words]


def _content_terms(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.lower()) if len(word) >= 4 and word not in STOPWORDS}


def _coverage(query_terms: set[str], chunk: str) -> float:
    """Share of the query's content words that appear in one chunk."""
    if not query_terms:
        return 0.0
    return len(query_terms & set(_WORD.findall(chunk.lower()))) / len(query_terms)
