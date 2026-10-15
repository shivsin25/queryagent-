import json

import pytest

from query_tool.config.settings import Settings
from query_tool.evaluation import metrics
from query_tool.evaluation.dataset import EvalCase, load_cases
from query_tool.evaluation.runner import EvalReport, EvalRunner, latest_report
from query_tool.retrieval.extraction import extract_document
from tests.conftest import FakeLLMProvider

CASES = [
    EvalCase(id="refund", question="Can I get a refund on annual plans?",
             evidence=["refunded within 30 days"], facts=["30 days"]),
    EvalCase(id="offtopic", question="What is the capital of France?", answerable=False),
]


# ---- metrics ------------------------------------------------------------


def test_evidence_ranks_ignore_case_whitespace_and_emphasis():
    chunks = ["Nothing here.", "Annual plans can be **refunded**\nwithin  30 days."]
    ranks = metrics.evidence_ranks(["refunded within 30 days", "monthly plans"], chunks)
    assert ranks == [2, None]
    assert metrics.recall(ranks) == 0.5
    assert metrics.reciprocal_rank(ranks) == 0.5
    assert metrics.recall([]) is None


@pytest.mark.parametrize(
    ("answer", "nodes", "abstained"),
    [
        ("Refunds within 30 days [1].", ["writer"], False),
        ("The sources do not mention a Tokyo office.", ["writer"], True),
        ("Nimbus's CEO isn't named in the documents.", ["writer"], True),
        ("Nimbus is not HIPAA compliant [1].", ["writer"], False),
        ("A CEO is not mentioned in the sources.", ["writer"], True),
        ("anything", ["preprocess", "retrieve", "no_context"], True),
    ],
)
def test_is_abstention(answer, nodes, abstained):
    assert metrics.is_abstention(answer, nodes) is abstained


def test_percentile_and_rates():
    assert metrics.percentile([5, 1, 3, 2, 4], 95) == 5
    assert metrics.percentile([], 95) is None
    assert metrics.rate([True, False, None]) == 0.5


# ---- dataset ------------------------------------------------------------


def test_load_cases_reports_bad_lines(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"id": "a", "question": "q?", "facts": ["x"]}\n\n{"id": "b"}\n')
    with pytest.raises(ValueError, match="cases.jsonl:3"):
        load_cases(path)

    path.write_text('{"id": "a", "question": "q?", "facts": ["x"]}\n' * 2)
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(path)

    path.write_text('{"id": "a", "question": "q?"}\n')
    with pytest.raises(ValueError, match="needs evidence or facts"):
        load_cases(path)


def test_bundled_dataset_evidence_exists_in_the_sample_docs():
    """Guards the dataset against drifting away from the documents it quotes."""
    settings = Settings(_env_file=None)
    texts = []
    for path in sorted(settings.sample_docs_dir.iterdir()):
        document = extract_document(path.name, path.read_bytes())
        texts += [line for section in document.sections for _, line in section.lines]
    corpus = metrics.normalize(" ".join(texts))
    cases = load_cases(settings.eval_dataset)
    assert len(cases) >= 20
    missing = [(c.id, e) for c in cases for e in c.evidence if metrics.normalize(e) not in corpus]
    assert missing == []


# ---- runner -------------------------------------------------------------


def test_retrieval_mode_needs_no_llm(make_service):
    service = make_service(FakeLLMProvider(configured=False), rerank=True)
    report = EvalRunner(service).run(CASES, "retrieval", "inline")

    refund, offtopic = report.cases
    assert refund.passed and refund.recall == 1.0 and refund.evidence_ranks == [1]
    assert offtopic.passed is None and offtopic.retrieved == 0
    assert report.summary["retrieval_recall"] == 1.0
    assert report.summary["offtopic_rejection_rate"] == 1.0
    assert "answer_accuracy" not in report.summary


def test_full_mode_scores_answers_and_abstentions(make_service):
    report = EvalRunner(make_service(rerank=True)).run(CASES, "full", "inline")

    refund, offtopic = report.cases
    assert refund.passed and refund.fact_hits == [True] and refund.approved
    assert refund.llm_calls == 4  # rewrite, write, review, polish
    assert offtopic.passed and offtopic.abstained and offtopic.llm_calls == 1
    assert report.summary["pass_rate"] == 1.0
    assert report.summary["answer_accuracy"] == 1.0
    assert report.summary["claim_support"] == 1.0
    assert report.by_category["general"]["pass_rate"] == 1.0


def test_failing_case_is_recorded_not_raised(make_service):
    llm = FakeLLMProvider(writer=[RuntimeError("rate limited")])
    report = EvalRunner(make_service(llm)).run(CASES[:1], "full", "inline")
    assert report.cases[0].error == "RuntimeError: rate limited"
    assert report.cases[0].passed is False
    assert report.summary["error_rate"] == 1.0


def test_report_round_trip_and_latest(make_service, tmp_path):
    runner = EvalRunner(make_service())
    first = runner.run(CASES, "retrieval", "inline")
    first_path = first.save(tmp_path)
    assert latest_report(tmp_path, "retrieval") == first_path
    assert latest_report(tmp_path, "full") is None

    loaded = EvalReport.load(first_path)
    assert loaded.summary == first.summary
    assert json.loads(first_path.read_text())["cases"][0]["passed"] is True
    assert "groq_api_key" not in loaded.config
