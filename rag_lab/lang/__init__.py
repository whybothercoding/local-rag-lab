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
STOPWORDS_BY_LANG: dict[str, frozenset[str]] = {
    lang: frozenset(fold(w) for key in ("stopwords", "negations", "chitchat_words") for w in _load(lang)[key])
    for lang in LANGUAGES
}
_GREEK = re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]")


def detect_language(text: str) -> str:
    """Best-guess language of a short question: Greek by script, otherwise by function-word votes.

    Cheap and deterministic, for keeping cached answers in the language they were asked in. Returns
    "und" when there is no evidence either way.
    """
    if _GREEK.search(text or ""):
        return "el"
    tokens = words(text)
    votes = {lang: sum(1 for t in tokens if t in STOPWORDS_BY_LANG[lang]) for lang in LANGUAGES if lang != "el"}
    best = max(votes, key=lambda lang: votes[lang])
    tied = [lang for lang, n in votes.items() if n == votes[best]]
    return best if votes[best] > 0 and len(tied) == 1 else "und"


def same_word(a: str, b: str) -> bool:
    """Inflection-tolerant match of folded words: "γαμο" ~ "γαμου", "reporter" ~ "reportes", "flight" ~ "flights"."""
    if a == b:
        return True
    shorter, longer = sorted((a, b), key=len)
    if len(shorter) >= 4 and longer.startswith(shorter):
        return True
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    return common >= 5


NO_INFO_MARKERS: tuple[str, ...] = tuple(sorted({fold(m) for lang in LANGUAGES for m in _load(lang)["no_info_markers"]}))


_ATTRIBUTION = re.compile(
    "^(?:" + "|".join(re.escape(fold(x)) for lang in LANGUAGES for x in _load(lang)["attribution_prefixes"]) + r")\s*,?\s*",
)


def strip_attribution(sentence: str) -> str:
    """Drop a leading "According to the documents," / "Selon les documents," so only the claim is judged."""
    match = _ATTRIBUTION.match(fold(sentence))
    if not match:
        return sentence
    rest = sentence[len(match.group(0)):] if len(fold(sentence)) == len(sentence) else sentence.split(",", 1)[-1]
    rest = rest.strip()
    return rest[:1].upper() + rest[1:] if rest else sentence


def is_no_info_sentence(sentence: str) -> bool:
    """True for a sentence that only says the documents lack something ("The documents do not mention the
    dress code."). It makes no claim to check. A sentence that also states a figure is not skipped."""
    folded = fold(sentence)
    return any(marker in folded for marker in NO_INFO_MARKERS) and not re.search(r"\d", sentence)


CHITCHAT_EXAMPLES: tuple[str, ...] = tuple(t for lang in LANGUAGES for t in _load(lang)["chitchat_examples"])


def chitchat_reply(lang: str) -> str:
    return str(_load(lang if lang in LANGUAGES else "en")["chitchat_reply"])


# Longest first, so "ungen" is tried before "en" and "n".
STEM_SUFFIXES: tuple[str, ...] = tuple(sorted(_folded_set("stem_suffixes"), key=len, reverse=True))
