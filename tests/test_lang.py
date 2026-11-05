"""Multilingual text layer: folding, Unicode words, language data, and the heuristics that use them."""
import json
from pathlib import Path

import pytest

from rag_lab.lang import (
    ARTICLE_NUMBERS,
    LANGUAGES,
    NEGATIONS,
    NUMBER_WORDS,
    STOPWORDS,
    fold,
    words,
)
from rag_lab.pipeline.entry import router_node
from rag_lab.pipeline.grading import grade_hallucination_node
from rag_lab.pipeline.text import _content_terms, _coverage, _keywords, _numbers, _stem_variants, _unsupported_numbers
from rag_lab.retrieval.semantic_cache import _signature

DATA = Path(__file__).resolve().parent.parent / "rag_lab" / "lang" / "data"


def test_fold_is_case_and_accent_insensitive_in_every_script():
    assert fold("Τετάρτη") == fold("ΤΕΤΑΡΤΗ") == fold("τεταρτη")
    assert fold("Kündigung") == "kundigung"
    assert fold("Straße") == "strasse"
    assert fold("Télétravail") == "teletravail"
    assert fold("ς") == fold("σ"), "final sigma folds like any sigma"


def test_words_keep_every_script_and_drop_underscores():
    assert words("Η τηλεργασία ΔΕΝ επιτρέπεται; Kündigung für_alle 2. Klasse") == [
        "η", "τηλεργασια", "δεν", "επιτρεπεται", "kundigung", "fur", "alle", "2", "klasse"]


def test_every_language_ships_complete_data():
    keys = {"stopwords", "negations", "number_words", "article_numbers", "stem_suffixes", "chitchat_words", "chitchat_phrases"}
    for lang in LANGUAGES:
        data = json.loads((DATA / f"{lang}.json").read_text(encoding="utf-8"))
        assert keys <= set(data), lang
        assert data["negations"] and data["number_words"] and data["stopwords"], lang


@pytest.mark.parametrize("word", ["not", "nicht", "keine", "pas", "jamais", "δεν", "μην", "ποτέ"])
def test_negations_cover_every_language(word):
    assert fold(word) in NEGATIONS


def test_negated_questions_have_different_cache_signatures_in_every_language():
    pairs = [
        ("Are deployments allowed on Tuesday?", "Are deployments not allowed on Tuesday?"),
        ("Ist Homeoffice am Montag erlaubt?", "Ist Homeoffice am Montag nicht erlaubt?"),
        ("Le télétravail depuis l'étranger est-il autorisé ?", "Le télétravail depuis l'étranger n'est-il pas autorisé ?"),
        ("Επιτρέπεται η τηλεργασία την Τετάρτη;", "Δεν επιτρέπεται η τηλεργασία την Τετάρτη;"),
    ]
    for positive, negative in pairs:
        assert _signature(positive) != _signature(negative), negative
    assert _signature("Ist Homeoffice erlaubt?") == _signature("IST HOMEOFFICE ERLAUBT?")


def test_number_words_fold_to_digits_but_articles_do_not_count_as_stated_numbers():
    assert NUMBER_WORDS["funf"] == "5" and NUMBER_WORDS["cinq"] == "5" and NUMBER_WORDS[fold("πέντε")] == "5"
    assert {"un", "une", "μια"} <= ARTICLE_NUMBERS
    assert "5" in _numbers("Die Frist beträgt fünf Tage.")
    # "une demande" and "μια ιατρική βεβαίωση" are articles, not a claim about the number one.
    assert _unsupported_numbers("Il faut une demande écrite.", ["Une demande est nécessaire."], "") == set()
    assert _unsupported_numbers("Χρειάζεται μια ιατρική βεβαίωση.", ["ιατρική βεβαίωση"], "") == set()
    assert _unsupported_numbers("Es sind sieben Tage.", ["Die Frist beträgt 7 Tage."], "") == set()
    assert _unsupported_numbers("Es sind acht Tage.", ["Die Frist beträgt 7 Tage."], "") == {"8"}


def test_content_terms_and_coverage_work_for_greek_and_german():
    question = "Πόσες ημέρες τηλεργασίας επιτρέπονται την εβδομάδα;"
    terms = _content_terms(question)
    assert {fold(w) for w in ("ημέρες", "τηλεργασίας", "επιτρέπονται", "εβδομάδα")} <= terms
    assert not terms & STOPWORDS
    assert _coverage(terms, "Η τηλεργασία επιτρέπεται έως 2 ημέρες την εβδομάδα.") >= 0.5
    assert "kundigungsfrist" in _content_terms("Wie lange ist die Kündigungsfrist?")


def test_keywords_and_stems_are_language_aware():
    keywords = _keywords("Wie viele Tage Homeoffice sind erlaubt?")
    assert "homeoffice" in keywords and "wie" not in keywords
    assert "tag" in _stem_variants(["tagen"])
    assert "jour" in _stem_variants(["jours"])
    assert "flight" in _stem_variants(["flights"])


@pytest.mark.parametrize("greeting", ["Hallo!", "Bonjour", "Γεια σας", "merci", "Wer bist du?", "Qui es-tu ?"])
def test_router_recognises_chitchat_in_every_language(greeting):
    assert router_node({"sanitized_question": greeting})["intent"] == "chitchat"


@pytest.mark.parametrize("question", ["Hallo, wie lange ist die Kündigungsfrist?", "Bonjour, combien de jours de congés ?"])
def test_router_sends_greetings_with_a_question_to_retrieval(question):
    assert router_node({"sanitized_question": question})["intent"] == "rag"


def test_greek_answers_get_a_real_groundedness_check():
    """Before the Unicode text layer, Greek answers had no words to check and always passed."""
    docs = ["Η άδεια γάμου είναι 5 εργάσιμες ημέρες."]
    grounded = grade_hallucination_node({"documents": docs, "generation": "Η άδεια γάμου είναι 5 εργάσιμες ημέρες.",
                                         "telemetry": {"generator_deterministic": True}})
    assert grounded["telemetry"]["groundedness_ratio"] == 1.0
    invented = grade_hallucination_node({"documents": docs,
                                         "generation": "Οι υπάλληλοι λαμβάνουν δωρεάν μεταφορά και γυμναστήριο.",
                                         "telemetry": {"generator_deterministic": True}})
    assert invented["telemetry"]["groundedness_ratio"] is not None
    assert invented["telemetry"]["groundedness_ratio"] < 0.25


@pytest.mark.parametrize(("text", "lang"), [
    ("How many days per week may employees work from home?", "en"),
    ("Wie viele Tage pro Woche darf man im Homeoffice arbeiten?", "de"),
    ("Combien de jours de télétravail sont autorisés par semaine ?", "fr"),
    ("Πόσες ημέρες τηλεργασίας επιτρέπονται;", "el"),
    ("Rollback SLA?", "und"),
])
def test_detect_language_on_short_questions(text, lang):
    from rag_lab.lang import detect_language

    assert detect_language(text) == lang


# --- Swiss and Greek personal data --------------------------------------------------------

@pytest.mark.parametrize(("text", "placeholder"), [
    ("AHV-Nr. 756.1234.5678.97 bitte angeben", "[REDACTED_AHV]"),
    ("AVS 7561234567897", "[REDACTED_AHV]"),
    ("ΑΜΚΑ 15038512347", "[REDACTED_AMKA]"),
    ("Rufen Sie 044 123 45 67 an", "[REDACTED_PHONE]"),
    ("Natel 079.123.45.67", "[REDACTED_PHONE]"),
    ("Τηλέφωνο 210 1234567", "[REDACTED_PHONE]"),
    ("Κινητό 694 123 4567", "[REDACTED_PHONE]"),
    ("+41 44 123 45 67", "[REDACTED_PHONE]"),
    ("+30 694 123 4567", "[REDACTED_PHONE]"),
])
def test_swiss_and_greek_personal_data_is_redacted(text, placeholder):
    from rag_lab.security.redaction import redact_pii

    assert placeholder in redact_pii(text), redact_pii(text)


@pytest.mark.parametrize("text", [
    "AHV 756.1234.5678.98",                # wrong check digit
    "Rechnung 15038512348",                # 11 digits, Luhn fails
    "Referenz 45138512347",                # 11 digits, day 45 is not a date
    "Das Lohnband liegt zwischen 128000 und 152000 CHF.",
    "Bis zum 31.03.2026 einreichen.",
    "Die Frist beträgt 7 Tage, nach 3 Monaten 2 Monate.",
    "Οι αποδείξεις υποβάλλονται εντός 30 ημερών, επίδομα 40 ευρώ.",
    "Bestellnummer 0441234567",            # bare digits are never a phone number
])
def test_ordinary_figures_survive_redaction(text):
    from rag_lab.security.redaction import redact_pii

    assert redact_pii(text) == text
