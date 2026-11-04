import subprocess
import sys
from pathlib import Path

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


def test_evaluation_runs_end_to_end_and_meets_its_gates():
    """Runs eval.py as a separate process so its isolated stores never touch the test session's environment."""
    completed = subprocess.run(
        [sys.executable, "-m", "rag_lab.ops.eval"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (ROOT / "eval" / "out" / "report.json").exists()
    assert "All thresholds met" in completed.stdout
