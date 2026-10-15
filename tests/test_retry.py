"""Answering again when the user isn't satisfied: limit, escalation and honest reporting."""

import pytest
from fastapi.testclient import TestClient

from query_tool.api.main import create_app
from query_tool.config.settings import Settings
from query_tool.core.exceptions import NotFoundError, RetryLimitError
from query_tool.core.policy import RerankPolicy
from query_tool.core.schemas import QueryRequest, QueryResponse, RetrievedChunk
from query_tool.services.query_log import QueryLog
from query_tool.services.retry import Attempt, assess, escalate, same_question
from query_tool.storage.database import Database
from tests.conftest import FakeLLMProvider, review_json

FIRST = "Annual plans can be refunded within 30 days [1]."
BETTER = ("Annual plans can be refunded within 30 days of purchase [1]. Monthly plans are not "
          "refundable, and Enterprise customers get a 1 hour response for critical incidents "
          "[2].")


def _prompt(llm: FakeLLMProvider, role: str, call: int) -> str:
    return "\n".join(m.content for m in llm.models[role].calls[call])


# ---- the limit ----------------------------------------------------------


def test_three_retries_then_the_limit(make_service):
    service = make_service()
    first = service.run(QueryRequest(question="Refund policy?"))
    assert (first.attempt, first.retries_left, first.root_request_id) == (1, 3, first.request_id)

    previous = first
    for attempt, left in [(2, 2), (3, 1), (4, 0)]:
        previous = service.run(QueryRequest(question="Refund policy?",
                                            retry_of=previous.request_id))
        assert (previous.attempt, previous.retries_left) == (attempt, left)
        assert previous.root_request_id == first.request_id

    with pytest.raises(RetryLimitError, match="Re-query limit reached"):
        service.run(QueryRequest(question="Refund policy?", retry_of=previous.request_id))
    # Retrying an older attempt doesn't reopen the question either.
    with pytest.raises(RetryLimitError):
        service.run(QueryRequest(question="Refund policy?", retry_of=first.request_id))


def test_asking_the_same_question_again_in_a_session_is_a_retry(make_service):
    service = make_service()
    first = service.run(QueryRequest(question="What is the refund policy?", session_id="s1"))
    again = service.run(QueryRequest(question="what is the refund policy", session_id="s1"))
    assert again.attempt == 2 and again.root_request_id == first.request_id

    new = service.run(QueryRequest(question="What is the enterprise response time?",
                                   session_id="s1"))
    assert new.attempt == 1
    elsewhere = service.run(QueryRequest(question="What is the refund policy?", session_id="s2"))
    assert elsewhere.attempt == 1


def test_unknown_retry_is_rejected(make_service):
    with pytest.raises(NotFoundError):
        make_service().run(QueryRequest(question="Refund policy?", retry_of="0" * 12))


def test_api_reports_the_limit_and_bad_references(make_service, settings):
    settings.max_retries_per_question = 1
    with TestClient(create_app(make_service())) as client:
        first = client.post("/api/v1/query", json={"question": "Refund policy?"}).json()
        body = {"question": "Refund policy?", "retry_of": first["request_id"]}
        second = client.post("/api/v1/query", json=body).json()
        assert second["attempt"] == 2 and second["retries_left"] == 0

        body["retry_of"] = second["request_id"]
        for path in ("/api/v1/query", "/api/v1/query/stream"):
            response = client.post(path, json=body)
            assert response.status_code == 429
            assert "Re-query limit reached" in response.json()["detail"]

        body["retry_of"] = "0" * 12
        assert client.post("/api/v1/query", json=body).status_code == 404
        body["retry_of"] = "not-an-id"
        assert client.post("/api/v1/query", json=body).status_code == 422


# ---- a retry tries harder -------------------------------------------------


def test_retry_skips_the_cache_and_sees_the_earlier_answer(make_service):
    llm = FakeLLMProvider(writer=[FIRST, BETTER])
    service = make_service(llm)
    first = service.run(QueryRequest(question="Refund policy?"))
    retry = service.run(QueryRequest(question="Refund policy?", retry_of=first.request_id,
                                     retry_feedback="It ignores monthly plans"))

    assert not retry.cached and retry.answer == BETTER
    for role in ("planner", "writer"):
        prompt = _prompt(llm, role, 1)
        assert "asked this 1 time(s) before" in prompt
        assert "Answer 1: Annual plans can be refunded within 30 days" in prompt
        assert "It ignores monthly plans" in prompt
    assert "Write a better answer" in _prompt(llm, "writer", 1)
    assert "Search differently this time" in _prompt(llm, "planner", 1)


def test_retry_gets_an_extra_revision(make_service):
    llm = FakeLLMProvider(reviewer=[review_json("revise", 0.3, ["Too vague"])])
    service = make_service(llm)
    first = service.run(QueryRequest(question="Refund policy?", max_revisions=1))
    writes_before = len(llm.models["writer"].calls)
    assert writes_before == 2  # the first answer and 1 revision

    service.run(QueryRequest(question="Refund policy?", max_revisions=1,
                             retry_of=first.request_id))
    assert len(llm.models["writer"].calls) - writes_before == 3  # and now 2 revisions


def test_escalation_widens_the_search_each_time():
    settings = Settings(_env_file=None, max_research_rounds=1)
    base = RerankPolicy(rerank_k=12, min_score=-5, min_chunk_score=-7, margin=3)
    first = escalate(settings, base, Attempt(1, "", []), top_k=5, max_revisions=2)
    assert first.policy is None and first.top_k == 5

    levels = [escalate(settings, base, Attempt(n, "r", []), top_k=5, max_revisions=2)
              for n in (2, 3, 4)]
    assert [e.top_k for e in levels] == [7, 9, 11]
    assert [e.policy.margin for e in levels] == [5, 7, 9]
    assert [e.policy.min_chunk_score for e in levels] == [-8, -9, -10]
    assert all(e.policy.rerank_k == 20 and e.policy.min_score == -5 for e in levels)
    assert all(e.max_revisions == 3 and e.max_research_rounds == 2 for e in levels)


# ---- honest reporting ---------------------------------------------------


def _response(answer: str, sources: list[str], queries: list[str]) -> QueryResponse:
    chunks = [RetrievedChunk(chunk_id=s, doc_id="d", source="f", text=s, parent_id=s)
              for s in sources]
    return QueryResponse(request_id=answer[:12].ljust(12, "x"), question="q",
                         standalone_question="q", search_queries=queries, answer=answer,
                         approved=True, revisions=0, sources=chunks, latency_ms=1)


def test_a_retry_that_says_the_same_thing_is_not_called_an_improvement():
    before = _response("Refunds within 30 days [1].", ["a"], ["refund"])
    same = _response("Refunds within 30 days [1].", ["a", "b"], ["refund", "refund window"])
    report = assess(before, same, None)
    assert not report.improved and "don't contain more" in report.note
    assert any("Read 2 passages (before: 1), 1 of them new" in c for c in report.changes)


def test_a_retry_with_new_sources_reports_what_changed():
    before = _response("Refunds within 30 days [1].", ["a"], ["refund"])
    better = _response("Annual plans: 30 days [1]. Monthly plans: no refunds [2].",
                       ["a", "b"], ["refund", "monthly plan refund"])
    report = assess(before, better, "Missing monthly plans")
    assert report.improved and report.note is None
    assert report.changes[0] == 'Used your feedback: "Missing monthly plans"'
    assert any("Cites 1 source(s) the earlier answer didn't use" in c for c in report.changes)
    assert any('"monthly plan refund"' in c for c in report.changes)


def test_an_improved_retry_replaces_the_cached_answer(make_service):
    llm = FakeLLMProvider(writer=[FIRST, BETTER])
    service = make_service(llm)
    first = service.run(QueryRequest(question="Refund policy?"))
    retry = service.run(QueryRequest(question="Refund policy?", retry_of=first.request_id))
    assert retry.retry.improved

    later = service.run(QueryRequest(question="Refund policy?"))  # someone else asks
    assert later.cached and later.answer == BETTER
    assert (later.attempt, later.retry) == (1, None)


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        ("Refund policy?", "refund   policy", True),
        ("What is the refund policy for annual plans?",
         "What's the refund policy for the annual plans", True),
        ("What is the refund policy for annual plans?", "What is the enterprise SLA?", False),
        ("Refunds?", "Refund policy?", False),
    ],
)
def test_same_question(a, b, same):
    assert same_question(a, b) is same


def test_query_log_upgrades_an_older_database(tmp_path):
    db = Database(tmp_path / "app.sqlite3")
    with db.connect() as conn:
        conn.execute(
            "CREATE TABLE queries (request_id TEXT PRIMARY KEY, session_id TEXT, question TEXT "
            "NOT NULL, standalone_question TEXT NOT NULL, answer TEXT NOT NULL, approved "
            "INTEGER NOT NULL, cached INTEGER NOT NULL, latency_ms REAL NOT NULL, cost_usd "
            "REAL NOT NULL, created_at TEXT NOT NULL)"
        )
        conn.execute("INSERT INTO queries VALUES ('abc', 's', 'q', 'q', 'a', 1, 0, 1, 0, 't')")
    old = QueryLog(db).get("abc")
    assert (old.root_id, old.attempt, old.response) == ("abc", 1, None)
