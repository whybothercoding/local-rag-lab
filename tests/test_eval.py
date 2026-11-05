import json
import subprocess
import sys
from pathlib import Path

from rag_lab.ops.eval import SUITE_GATES
from rag_lab.ops.eval_metrics import (
    check_thresholds,
    cited_sources,
    context_precision,
    context_recall,
    keyword_recall,
    summarize,
)

ROOT = Path(__file__).resolve().parent.parent


def test_cited_sources_strips_chunk_ids():
    citations = ["[engineering_0_0_ab12] deployments.md", "[engineering_1_0_cd34] rollback.md"]
    assert cited_sources(citations) == ["deployments.md", "rollback.md"]


def test_context_recall_and_precision():
    cited = ["deployments.md", "rollback.md", "incidents.md"]
    assert context_recall(cited, ["deployments.md"]) == 1.0
    assert context_recall(cited, ["vat.md"]) == 0.0
    assert context_precision(cited, ["deployments.md"]) == 1 / 3
    assert context_precision([], ["deployments.md"]) == 0.0


def test_keyword_recall_is_case_insensitive():
    answer = "Deployments run on TUESDAY and Thursday."
    assert keyword_recall(answer, ["tuesday", "thursday"]) == 1.0
    assert keyword_recall(answer, ["tuesday", "friday"]) == 0.5


def test_summarize_separates_answer_security_and_abstention_cases():
    results = [
        {"expect": "answer", "context_recall": 1.0, "context_precision": 0.5, "keyword_recall": 1.0,
         "groundedness": 0.8, "passed": True, "abstained": False, "latency_ms": 10,
         "judge_supported": True, "degraded": []},
        {"expect": "denied", "passed": True, "abstained": False, "latency_ms": 2, "degraded": []},
        {"expect": "abstain", "passed": False, "abstained": False, "latency_ms": 30, "degraded": ["not_indexed"]},
    ]
    summary = summarize(results)
    assert summary["retrieval_recall"] == 1.0
    assert summary["answer_rate"] == 1.0
    assert summary["security_accuracy"] == 1.0
    assert summary["abstention_accuracy"] == 0.0
    assert summary["judge_support"] == 1.0
    assert summary["degraded_cases"] == 1
    assert summary["median_latency_ms"] == 10


def test_answer_rate_drops_when_answerable_questions_are_abstained():
    results = [
        {"expect": "answer", "context_recall": 1.0, "keyword_recall": 1.0, "groundedness": 1.0,
         "passed": True, "abstained": False, "latency_ms": 1, "judge_supported": None, "degraded": []},
        {"expect": "answer", "context_recall": 0.0, "keyword_recall": 0.0, "groundedness": None,
         "passed": False, "abstained": True, "latency_ms": 1, "judge_supported": None, "degraded": []},
    ]
    assert summarize(results)["answer_rate"] == 0.5


def test_threshold_gate_reports_each_failure():
    summary = {"retrieval_recall": 0.5, "security_accuracy": 1.0}
    failures = check_thresholds(summary, {"retrieval_recall": 0.9, "security_accuracy": 1.0, "groundedness": 0.5})
    assert len(failures) == 2
    assert any("retrieval_recall" in f for f in failures)
    assert any("groundedness" in f for f in failures)


def _run_eval(tmp_path, *args: str, **env: str) -> subprocess.CompletedProcess:
    """eval runs as its own process (it redirects every store before importing the project)."""
    import os

    environment = {k: v for k, v in os.environ.items() if not k.endswith(("_PATH", "_DIR", "DB_URI"))}
    environment.update({"EVAL_OUT_DIR": str(tmp_path / "eval-out"), **env})
    return subprocess.run([sys.executable, "-m", "rag_lab.ops.eval", *args], cwd=ROOT, capture_output=True,
                          text=True, timeout=900, env=environment)


def test_evaluation_runs_end_to_end_and_meets_its_gates(tmp_path):
    completed = _run_eval(tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((tmp_path / "eval-out" / "report.json").read_text(encoding="utf-8"))
    assert report["mode"] == "deterministic"
    assert set(report["suites"]) == set(SUITE_GATES)
    assert report["suites"]["en"]["failures"] == []
    assert "All gated suites met their thresholds" in completed.stdout


def test_importing_eval_has_no_side_effects(tmp_path, monkeypatch):
    """The packaging sweep and these tests import the module; only main() may redirect or wipe stores."""
    import importlib
    import os

    monkeypatch.setenv("EVAL_OUT_DIR", str(tmp_path / "keep"))
    (tmp_path / "keep").mkdir()
    (tmp_path / "keep" / "marker").write_text("x")
    before = dict(os.environ)
    import rag_lab.ops.eval as eval_module

    importlib.reload(eval_module)
    assert (tmp_path / "keep" / "marker").exists()
    assert dict(os.environ) == before


def test_llm_mode_refuses_to_run_without_a_model(tmp_path):
    completed = _run_eval(tmp_path, "--llm", "--suite", "en", OLLAMA_BASE_URL="http://127.0.0.1:9")
    assert completed.returncode == 2
    assert "ollama serve" in completed.stdout


# --- golden set ----------------------------------------------------------------------------

GOLDEN = json.loads((ROOT / "eval" / "golden.json").read_text(encoding="utf-8"))
EXPECTS = {"answer", "abstain", "blocked", "denied", "fresh", "cached", "chitchat"}
LANGS = {"en", "de", "fr", "el"}


def test_golden_cases_are_well_formed():
    ids = [c["id"] for c in GOLDEN]
    assert len(ids) == len(set(ids)), "case ids must be unique"
    for case in GOLDEN:
        assert case["suite"] in SUITE_GATES, case["id"]
        assert case["lang"] in LANGS, case["id"]
        assert case["split"] in ("calibration", "test"), case["id"]
        assert case.get("expect", "answer") in EXPECTS, case["id"]
        assert (ROOT / "eval" / "corpus" / case["workspace"]).is_dir(), case["id"]
        assert case["question"].strip(), case["id"]
        for field in ("history", "setup"):
            assert isinstance(case.get(field, []), list), case["id"]
        if case.get("expect", "answer") == "answer":
            assert case["expected_sources"] and case["expected_keywords"], case["id"]
            for source in case["expected_sources"]:
                assert (ROOT / "eval" / "corpus" / case["workspace"] / source).is_file(), (case["id"], source)
            for keyword in case["expected_keywords"]:
                assert isinstance(keyword, str) or (isinstance(keyword, list) and all(isinstance(k, str) for k in keyword))
        if case.get("expect") in ("fresh", "cached"):
            assert case.get("setup"), f"{case['id']}: a cache case needs an earlier question"
        if case["suite"] == "conversation":
            assert case.get("history"), f"{case['id']}: a conversation case needs earlier turns"


def test_golden_set_covers_every_language_suite_and_risk():
    """Guards the set from shrinking back: every language, both splits, and each risk type stay covered."""
    assert len(GOLDEN) >= 140
    for lang in LANGS:
        rows = [c for c in GOLDEN if c["lang"] == lang]
        assert len(rows) >= 20, lang
        assert {c["split"] for c in rows} == {"calibration", "test"}, lang
    for lang in LANGS - {"en"}:
        expects = {c.get("expect", "answer") for c in GOLDEN if c["lang"] == lang}
        assert {"answer", "abstain", "blocked", "denied"} <= expects, lang
        assert any(c.get("expect") == "fresh" for c in GOLDEN if c["lang"] == lang), f"{lang}: negation cache case"
        assert any(c["suite"] == "conversation" for c in GOLDEN if c["lang"] == lang), f"{lang}: follow-up case"
    cross = [c for c in GOLDEN if c["workspace"] in ("zurich", "geneve", "athens") and c["lang"] == "en"]
    assert cross, "English questions over non-English documents"
    assert any(c.get("principal") == "eval-secret" for c in GOLDEN if c["lang"] != "en"), "clearance outside English"


# --- gates -----------------------------------------------------------------------------------

def test_keyword_alternatives_accept_either_language():
    assert keyword_recall("Am Montag nicht erlaubt.", [["montag", "monday"]]) == 1.0
    assert keyword_recall("Not on Monday.", [["montag", "monday"], "45"]) == 0.5


def test_cache_accuracy_is_measured_and_skipped_when_absent():
    rows = [
        {"expect": "fresh", "passed": False, "abstained": False, "latency_ms": 1, "degraded": []},
        {"expect": "cached", "passed": True, "abstained": False, "latency_ms": 1, "degraded": []},
    ]
    assert summarize(rows)["cache_accuracy"] == 0.5
    assert check_thresholds({"answer_rate": 1.0}, {"answer_rate": 0.9, "cache_accuracy": 1.0}) == []


def test_suite_gates_follow_the_phase_plan():
    from rag_lab.ops.eval import LLM_JUDGE_MIN, suite_gates

    assert suite_gates("en", {}, llm_mode=False)["security_accuracy"] == 1.0
    assert suite_gates("multilingual", {}, llm_mode=False)["security_accuracy"] == 1.0
    assert suite_gates("multilingual", {"answer_rate": 0.1}, llm_mode=False)["answer_rate"] == 0.9, "overrides are en-only"
    assert suite_gates("conversation", {}, llm_mode=True)["answer_rate"] == 0.9
    assert suite_gates("en", {}, llm_mode=True)["judge_support"] == LLM_JUDGE_MIN
    assert suite_gates("en", {"answer_rate": 0.5}, llm_mode=False)["answer_rate"] == 0.5


def test_inconclusive_cache_cases_are_reported_not_counted():
    rows = [
        {"expect": "fresh", "passed": True, "abstained": False, "latency_ms": 1, "degraded": []},
        {"expect": "fresh", "passed": None, "abstained": False, "latency_ms": 1, "degraded": []},
    ]
    summary = summarize(rows)
    assert summary["cache_accuracy"] == 1.0
    assert summary["cache_conclusive_rate"] == 0.5


def test_deterministic_runs_require_every_cache_case_to_be_conclusive():
    from rag_lab.ops.eval import suite_gates

    assert suite_gates("en", {}, llm_mode=False)["cache_conclusive_rate"] == 1.0
    assert "cache_conclusive_rate" not in suite_gates("en", {}, llm_mode=True)


def test_judge_only_needs_an_independent_judge_model(tmp_path, monkeypatch, capsys):
    from rag_lab.ops.eval import judge_report

    monkeypatch.delenv("EVAL_JUDGE_MODEL", raising=False)
    assert judge_report(tmp_path / "report.json") == 2
    assert "another family" in capsys.readouterr().out


def test_judge_only_adds_verdicts_and_rescores(tmp_path, monkeypatch):
    from langchain_core.messages import AIMessage

    from rag_lab.ops.eval import judge_report

    class Judge:
        def __init__(self, **kwargs):
            pass

        def invoke(self, prompt):
            return AIMessage(content="yes" if "45 CHF" in prompt else "no")

    monkeypatch.setenv("EVAL_JUDGE_MODEL", "other-family:7b")
    monkeypatch.setattr("langchain_ollama.ChatOllama", Judge)
    row = {"expect": "answer", "split": "test", "suite": "en", "lang": "en", "passed": True, "abstained": False,
           "latency_ms": 1, "degraded": [], "judge_supported": None, "documents": ["Pauschal 45 CHF."]}
    report = {"suites": {"en": {}}, "results": [{**row, "answer": "45 CHF per month."},
                                                 {**row, "answer": "100 CHF per month.", "documents": ["Other."]}]}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert judge_report(path) == 0
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["judge_model"] == "other-family:7b"
    assert saved["suites"]["en"]["test"]["judge_support"] == 0.5


def test_ollama_models_answer_without_a_thinking_pass(monkeypatch):
    from rag_lab.pipeline import llm_provider

    assert llm_provider._ollama_chat().reasoning is False
