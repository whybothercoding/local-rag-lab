"""Groundedness by entailment: each answer sentence must be stated by a retrieved passage."""
import pytest

from rag_lab.config import settings
from rag_lab.pipeline.grading import grade_hallucination_node
from rag_lab.pipeline.text import split_sentences
from rag_lab.retrieval.nli import NLIUnavailableError, _premises, sentence_support

TRAVEL = open("eval/corpus/finance/travel.md", encoding="utf-8").read()
LEAVE_EL = open("eval/corpus/athens/adeies.md", encoding="utf-8").read()
NOTICE_DE = open("eval/corpus/zurich/kuendigung.md", encoding="utf-8").read()


def _grade(answer: str, docs: list[str]) -> dict:
    return grade_hallucination_node({"documents": docs, "generation": answer, "original_question": "",
                                     "telemetry": {"generator_deterministic": False}})


def test_the_false_business_class_answer_is_rejected():
    """Phase 3 found the real model claiming this; word overlap alone passed it."""
    out = _grade("Business class is the default for flights under six hours.", [TRAVEL])
    assert out["hallucination_status"] == "retry"
    assert out["telemetry"]["groundedness_grader"] == "nli_rejected"
    assert "Business class is the default" in out["grader_feedback"]


@pytest.mark.parametrize("answer", [
    "Economy class is the default for flights under six hours.",
    "Flights shorter than six hours are booked in economy.",
    "Für Flüge unter sechs Stunden ist Economy der Standard.",          # cross-lingual
    "Economy is the default for short flights; business class needs the director's prior approval.",
])
def test_supported_answers_pass_in_any_language(answer):
    out = _grade(answer, [TRAVEL])
    assert out["hallucination_status"] == "grounded", out["telemetry"]
    assert out["telemetry"]["nli_supported"] is True


def test_an_invented_condition_is_rejected():
    out = _grade("Business class requires prior approval from the department director and the CEO's signature.", [TRAVEL])
    assert out["hallucination_status"] == "retry"


def test_a_sentence_about_another_topic_is_not_entailed_by_a_long_unrelated_passage():
    """A Greek leave passage once 'entailed' a sentence about expense receipts as a two-sentence premise."""
    assert sentence_support("Οι αποδείξεις εξόδων υποβάλλονται εντός 30 ημερών.", [LEAVE_EL]) < settings.groundedness_min_entailment


def test_single_sentence_premises_judge_a_claim_among_competing_figures():
    """'2 months after probation' next to '4 months from year 10' made an earlier model call the true claim
    contradicted when the whole passage was the premise. Single sentences must work whatever the model does."""
    claim = "Nach der Probezeit gilt eine Kündigungsfrist von 2 Monaten auf das Monatsende."
    assert sentence_support(claim, [NOTICE_DE]) > 0.9
    assert all(not p.startswith("#") for p in _premises([NOTICE_DE], claim))


def test_no_information_sentences_are_not_claims():
    answer = "The documents do not mention a dress code. The rollback SLA is under 5 minutes."
    out = _grade(answer, ["# Rollback Standards\nAll tier-1 and tier-2 microservices must maintain an automated rollback SLA of under 5 minutes."])
    assert out["hallucination_status"] == "grounded"
    assert out["telemetry"]["nli_claims"] == 1


def test_sentence_splitting_respects_numbers_times_and_greek_questions():
    assert split_sentences("Bahnreisen in der 2. Klasse; die 1. Klasse ab 3 Stunden.") == [
        "Bahnreisen in der 2. Klasse; die 1. Klasse ab 3 Stunden."]
    assert split_sentences("Window 02:00 to 04:00 UTC. Πότε; Τώρα.") == ["Window 02:00 to 04:00 UTC.", "Πότε;", "Τώρα."]
    assert split_sentences("- one point\n- two points") == ["one point", "two points"]


def test_unavailable_nli_falls_back_to_the_lexical_check(monkeypatch):
    def broken(*args, **kwargs):
        raise NLIUnavailableError("model missing")

    monkeypatch.setattr("rag_lab.retrieval.nli.sentence_support", broken)
    out = _grade("Economy class is the default for flights under six hours.", [TRAVEL])
    assert out["hallucination_status"] == "grounded"
    assert out["telemetry"]["groundedness_fallback"].startswith("NLI model unavailable")


def test_extractive_prefix_is_not_judged_as_a_claim():
    out = _grade("Based on available documentation: Economy class is the default for flights under six hours.", [TRAVEL])
    assert out["hallucination_status"] == "grounded"


@pytest.mark.parametrize("sentence", [
    "The context does not provide specific information about the travel policy for longer flights.",
    "The documents do not provide information about the office dress code.",
    "Die Dokumente geben keine Auskunft über Parkplätze.",
    "Les documents ne précisent pas la politique de stationnement.",
    "Τα έγγραφα δεν παρέχουν πληροφορίες για στάθμευση.",
])
def test_no_information_phrasings_in_every_language(sentence):
    from rag_lab.lang import is_no_info_sentence

    assert is_no_info_sentence(sentence)


def test_attribution_phrases_are_not_part_of_the_claim():
    teleworking = open("eval/corpus/geneve/teletravail.md", encoding="utf-8").read()
    out = _grade("Selon les documents, le télétravail est autorisé jusqu'à 2 jours par semaine.", [teleworking])
    assert out["hallucination_status"] == "grounded", out["telemetry"]


def test_an_added_condition_from_a_real_model_answer_is_rejected():
    """qwen3:8b added 'before reimbursement'; the source states no such condition."""
    expenses = open("eval/corpus/finance/expenses.md", encoding="utf-8").read()
    out = _grade("Receipts for business expenses above $50 require manager sign-off before reimbursement.", [expenses])
    assert out["hallucination_status"] == "retry"


# --- the model itself: verdicts must not depend on the machine -----------------------------------------------------

def _doc(path: str) -> str:
    return open(f"eval/corpus/{path}", encoding="utf-8").read()


# (supported?, claim, source). Scores on Apple silicon and on x86 Linux were identical to four decimals for the pinned
# fp32 model; the int8 file of the previous model returned 0.12 for the first claim on x86 and 0.93 on arm64.
LABELLED = [
    (True, "Flights shorter than six hours are booked in economy.", "finance/travel.md"),
    (True, "Hotel rates above 180 EUR per night need finance review before booking.", "finance/travel.md"),
    (True, "The rollback SLA is under 5 minutes.", "engineering/rollback.md"),
    (True, "All business expense receipts above $50 require manager sign-off.", "finance/expenses.md"),
    (True, "Nach der Probezeit gilt eine Kündigungsfrist von 2 Monaten auf das Monatsende.", "zurich/kuendigung.md"),
    (True, "Am Montag ist Homeoffice nicht erlaubt.", "zurich/homeoffice.md"),
    (True, "Le télétravail est autorisé jusqu'à 2 jours par semaine.", "geneve/teletravail.md"),
    (True, "Η άδεια γάμου είναι 5 εργάσιμες ημέρες.", "athens/adeies.md"),
    (True, "Οι πτήσεις κάτω από 4 ώρες κλείνονται σε οικονομική θέση.", "athens/exoda.md"),
    (False, "Business class is the default for flights under six hours.", "finance/travel.md"),
    (False, "Expense reports must be submitted within 90 days of purchase.", "finance/expenses.md"),
    (False, "Standard production deployments happen on Monday and Friday.", "engineering/deployments.md"),
    (False, "Während der Probezeit beträgt die Kündigungsfrist 3 Monate.", "zurich/kuendigung.md"),
    (False, "Am Freitag ist Homeoffice nicht erlaubt.", "zurich/homeoffice.md"),
    (False, "Le télétravail depuis l'étranger est autorisé sans accord.", "geneve/teletravail.md"),
    (False, "Οι αποδείξεις εξόδων υποβάλλονται εντός 30 ημερών.", "athens/adeies.md"),   # right fact, wrong document
    (False, "Η άδεια γάμου είναι 10 εργάσιμες ημέρες.", "athens/adeies.md"),
]


@pytest.mark.parametrize("supported, claim, path", LABELLED, ids=[c[1][:44] for c in LABELLED])
def test_labelled_claims_are_judged_with_a_wide_margin(supported, claim, path):
    """Far from the 0.7 gate on both sides, in every language: a model swap or a numerics change shows up here."""
    score = sentence_support(claim, [_doc(path)])
    assert (score >= 0.8) if supported else (score <= 0.5), score


def test_the_pinned_nli_weights_are_full_precision():
    """Dynamic int8 quantization of this model family made scores depend on the CPU (a true sentence: 0.93 on arm64,
    0.12 on x86), and a verdict must not depend on the machine. Do not pin quantized NLI weights."""
    from rag_lab.security.model_integrity import PINNED_MODELS

    weights = [f for f in PINNED_MODELS[settings.nli_model]["files"] if f.endswith(".onnx")]
    assert weights, "the NLI pin must list its weights file"
    for name in weights:
        assert not any(marker in name for marker in ("quant", "int8", "uint8", "q4", "bnb4", "fp16")), name
