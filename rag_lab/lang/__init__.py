"""Language data and text folding shared by retrieval, grading, the cache, and the extractive fallback.

Every comparison runs on folded text: case-folded and stripped of diacritics, so "Τετάρτη", "τετάρτη",
and "τεταρτη" match, as do "Kündigung" and "kundigung". Word lists are the union over all supported
languages (data/<lang>.json): the heuristics need no language detection, and a mixed-language
workspace is handled the same way as a single-language one.
"""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

LANGUAGES = ("en", "de", "fr", "el")
_DATA = Path(__file__).resolve().parent / "data"
_WORD = re.compile(r"[^\W_]+")


def fold(text: str) -> str:
    """Case-fold and strip diacritics (NFKD, drop combining marks). ß -> ss, final ς -> σ."""
    decomposed = unicodedata.normalize("NFKD", (text or "").casefold())
    return unicodedata.normalize("NFC", "".join(c for c in decomposed if not unicodedata.combining(c)))


def words(text: str) -> list[str]:
    """Folded words: letters and digits in any script, never underscores."""
    return _WORD.findall(fold(text))


@lru_cache(maxsize=None)
def _load(lang: str) -> dict:
    data: dict = json.loads((_DATA / f"{lang}.json").read_text(encoding="utf-8"))
    return data


def _folded_set(key: str) -> frozenset[str]:
    return frozenset(fold(w) for lang in LANGUAGES for w in _load(lang)[key])


STOPWORDS: frozenset[str] = _folded_set("stopwords")
NEGATIONS: frozenset[str] = _folded_set("negations")
CHITCHAT_WORDS: frozenset[str] = _folded_set("chitchat_words")
CHITCHAT_PHRASES: tuple[str, ...] = tuple(sorted(" ".join(words(p)) for lang in LANGUAGES for p in _load(lang)["chitchat_phrases"]))
NUMBER_WORDS: dict[str, str] = {fold(k): v for lang in LANGUAGES for k, v in _load(lang)["number_words"].items()}
# Words that are both the number one and an article ("un", "μια"); counting them as a stated "1" would
# make every French or Greek answer look like it invents a number.
ARTICLE_NUMBERS: frozenset[str] = _folded_set("article_numbers")
# Longest first, so "ungen" is tried before "en" and "n".
STEM_SUFFIXES: tuple[str, ...] = tuple(sorted(_folded_set("stem_suffixes"), key=len, reverse=True))
