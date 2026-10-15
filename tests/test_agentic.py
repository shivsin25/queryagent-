"""Conversation memory, query decomposition, corrective retrieval and per-role providers."""

import pytest
from fastapi.testclient import TestClient

from query_tool.api.main import create_app
from query_tool.config.settings import Settings
from query_tool.core.schemas import QueryRequest, Turn
from query_tool.evaluation.dataset import EvalCase
from query_tool.evaluation.runner import EvalRunner
from query_tool.graph.routing import after_research, should_revise
from query_tool.retrieval.ingestion import IngestionService
from query_tool.retrieval.retriever import HybridRetriever, RerankPolicy
from query_tool.services.conversations import ConversationStore
from query_tool.services.llm import ChatModelFactory
from query_tool.storage.database import Database
from tests.conftest import (
    FakeLLMProvider,
    HashEmbedder,
    KeywordReranker,
    plan_json,
    review_json,
)

FOLLOW_UP_PLAN = plan_json(
    "enterprise response time critical incidents",
    standalone="What is the response time for critical incidents for Enterprise customers?",
)


def _nodes(response) -> list[str]:
    return [step.node for step in response.trace]


def _prompt(llm: FakeLLMProvider, role: str, call: int = 0) -> str:
    """Everything sent to a role's model in one call (the JSON fallback appends a message)."""
    return "\n".join(m.content for m in llm.models[role].calls[call])


# ---- conversation store -------------------------------------------------


def test_conversation_store_orders_limits_trims_and_persists(tmp_path):
    path = tmp_path / "app.sqlite3"
    store = ConversationStore(Database(path), max_turns_per_session=3)
    for i in range(5):
        store.append("s1", Turn(question=f"q{i}", answer=f"a{i}"))
    store.append("s2", Turn(question="other", answer="x"))

    assert [t.question for t in store.history("s1")] == ["q2", "q3", "q4"]  # trimmed to 3
    assert [t.question for t in store.history("s1", limit=2)] == ["q3", "q4"]
    assert store.history("s1")[0].standalone_question == "q2"  # defaults to the question

    reopened = ConversationStore(Database(path))
    assert len(reopened.history("s1")) == 3
    assert reopened.delete("s1") == 3
    assert reopened.history("s1") == [] and len(reopened.history("s2")) == 1


# ---- follow-up questions ------------------------------------------------


def test_follow_up_is_resolved_with_history_and_remembered(make_service):
    llm = FakeLLMProvider(planner=[FOLLOW_UP_PLAN])
    service = make_service(llm)
    service.conversations.append(
        "chat1", Turn(question="Can I get a refund on annual plans?",
                      answer="Annual plans can be refunded within 30 days [1].")
    )
    response = service.run(QueryRequest(question="And what about enterprise support?",
                                        session_id="chat1"))

    assert response.session_id == "chat1"
    assert response.standalone_question.startswith("What is the response time")
    planner_prompt = _prompt(llm, "planner")
    assert "User: Can I get a refund on annual plans?" in planner_prompt
    assert "[1]" not in planner_prompt  # old citation numbers would mislead
    writer_prompt = llm.models["writer"].calls[0][-1].content
    assert "Question: What is the response time" in writer_prompt

    history = service.conversations.history("chat1")
    assert [t.question for t in history] == [
        "Can I get a refund on annual plans?", "And what about enterprise support?",
    ]
    assert history[-1].standalone_question == response.standalone_question


def test_follow_up_the_planner_failed_to_rewrite_is_still_retrieved(make_service):
    """Seen live: the planner copied "And for Business customers?" unchanged but still
    wrote a good search query. Retrieval must not depend on the reworded question alone."""
    llm = FakeLLMProvider(planner=[plan_json("enterprise response time critical incidents",
                                             standalone="And what about them?")])
    service = make_service(llm, rerank=True)
    service.conversations.append("chat1", Turn(question="Refund policy?", answer="30 days."))
    response = service.run(QueryRequest(question="And what about them?", session_id="chat1",
                                        top_k=1))
    assert [s.text.split("\n")[0] for s in response.sources] == ["Support"]


def test_without_session_nothing_is_remembered_and_history_is_ignored(make_service):
    llm = FakeLLMProvider(planner=[FOLLOW_UP_PLAN])
    service = make_service(llm)
    response = service.run(QueryRequest(question="Refund policy?"))
    assert response.standalone_question == "Refund policy?"
    assert "Conversation so far:\n(none)" in _prompt(llm, "planner")


def test_planner_failure_with_history_warns_that_context_was_lost(make_service):
    service = make_service(FakeLLMProvider(planner=[RuntimeError("boom")]))
    service.conversations.append("chat1", Turn(question="Earlier?", answer="Yes."))
    response = service.run(QueryRequest(question="And then?", session_id="chat1"))
    assert response.standalone_question == "And then?"
    assert any("earlier turns" in w for w in response.warnings)


# ---- query decomposition ------------------------------------------------


@pytest.fixture
def three_chunk_retriever(store):
    ingestion = IngestionService(store, HashEmbedder(), 500, 0)
    ingestion.ingest_bytes("a1.txt", b"Annual refund rules for annual plans.")
    ingestion.ingest_bytes("a2.txt", b"Annual refund requests for annual billing.")
    ingestion.ingest_bytes("b.txt", b"Enterprise response targets.")
    return HybridRetriever(store, HashEmbedder(), KeywordReranker(),
                           policy=RerankPolicy(min_score=-5))


def test_every_part_of_a_question_gets_its_best_chunk(three_chunk_retriever):
    queries = ["annual refund", "enterprise response time phone"]
    single = three_chunk_retriever.retrieve(queries, top_k=2, rerank_queries=queries[:1])
    assert {d.source for d in single} == {"a1.txt", "a2.txt"}

    covered = three_chunk_retriever.retrieve(queries, top_k=2, rerank_queries=queries)
    assert {d.source for d in covered} == {"a1.txt", "b.txt"}
    assert covered[0].score >= covered[1].score  # still ordered by relevance


def test_irrelevant_part_is_not_forced_in(three_chunk_retriever):
    queries = ["annual refund", "capital of france"]
    found = three_chunk_retriever.retrieve(queries, top_k=2, rerank_queries=queries)
    assert {d.source for d in found} == {"a1.txt", "a2.txt"}


def test_exclude_skips_chunks_already_in_use(three_chunk_retriever):
    first = three_chunk_retriever.retrieve(["annual refund"], top_k=1)
    again = three_chunk_retriever.retrieve(["annual refund"], top_k=1,
                                           exclude=frozenset({first[0].chunk_id}))
    assert again and again[0].chunk_id != first[0].chunk_id


def test_planner_sub_queries_reach_the_retriever(make_service):
    llm = FakeLLMProvider(planner=[plan_json("annual refund", "enterprise response time")])
    response = make_service(llm, rerank=True).run(
        QueryRequest(question="Refunds on annual plans and enterprise response times?",
                     top_k=2)
    )
    assert response.search_queries == ["annual refund", "enterprise response time"]
    assert {s.text.split("\n")[0] for s in response.sources} == {"Billing", "Support"}


# ---- corrective retrieval -----------------------------------------------


def test_missing_information_triggers_a_second_search_and_rewrite(make_service):
    llm = FakeLLMProvider(
        writer=[
            "Annual plans can be refunded within 30 days [1].",
            "Annual plans can be refunded within 30 days [1]. Enterprise customers get a "
            "1 hour response time for critical incidents [2].",
        ],
        reviewer=[
            review_json("revise", 0.9, missing=["enterprise response time critical incidents"]),
            review_json(),
        ],
    )
    response = make_service(llm, rerank=True).run(
        QueryRequest(question="Can I get a refund on annual plans?", top_k=1)
    )

    assert _nodes(response) == [
        "preprocess", "retrieve", "writer", "reviewer", "research", "writer", "reviewer",
        "enhancer",
    ]
    assert [s.text.split("\n")[0] for s in response.sources] == ["Billing", "Support"]
    assert response.approved and response.revisions == 1
    assert llm.models["feedback"].calls == []  # research writes its own instructions
    revision_prompt = llm.models["writer"].calls[1][-1].content
    assert "New sources [2] were found" in revision_prompt
    assert [c.index for c in response.citations] == [1, 2]


def test_approved_answer_with_nothing_new_found_is_kept(make_service):
    llm = FakeLLMProvider(reviewer=[review_json(missing=["capital of france"])])
    response = make_service(llm, rerank=True).run(
        QueryRequest(question="Can I get a refund on annual plans?", top_k=1)
    )
    assert _nodes(response) == [
        "preprocess", "retrieve", "writer", "reviewer", "research", "enhancer",
    ]
    assert response.approved and response.revisions == 0
    assert len(llm.models["writer"].calls) == 1


def test_research_runs_at_most_max_rounds(make_service):
    llm = FakeLLMProvider(
        reviewer=[review_json("revise", 0.5, missing=["capital of france"])]
    )
    response = make_service(llm, rerank=True).run(
        QueryRequest(question="Can I get a refund on annual plans?", max_revisions=2)
    )
    assert _nodes(response).count("research") == 1
    assert _nodes(response).count("feedback") == 2
    assert not response.approved


def _review(**kwargs):
    from query_tool.core.schemas import ReviewResult

    base = dict(grounding_score=1, accuracy_score=1, completeness_score=1, verdict="approve")
    return ReviewResult(**{**base, **kwargs})


@pytest.mark.parametrize(
    ("state", "route"),
    [
        ({"review": _review(approved=True, missing_information=["x"]), "revision_count": 0,
          "max_revisions": 2, "research_rounds": 0, "max_research_rounds": 1}, "research"),
        ({"review": _review(approved=True, missing_information=["x"]), "revision_count": 0,
          "max_revisions": 2, "research_rounds": 1, "max_research_rounds": 1}, "enhancer"),
        ({"review": _review(missing_information=["x"]), "revision_count": 2,
          "max_revisions": 2, "research_rounds": 0, "max_research_rounds": 1}, "enhancer"),
        ({"review": _review(missing_information=["x"]), "revision_count": 0,
          "max_revisions": 2, "research_rounds": 1, "max_research_rounds": 1}, "feedback"),
    ],
)
def test_should_revise_with_research(state, route):
    assert should_revise(state) == route


def test_after_research():
    assert after_research({"new_sources": 2, "review": _review()}) == "writer"
    assert after_research({"new_sources": 0, "review": _review(approved=True)}) == "enhancer"
    assert after_research({"new_sources": 0, "review": _review()}) == "feedback"


# ---- per-role providers -------------------------------------------------


def test_reviewer_can_use_another_provider():
    settings = Settings(_env_file=None, llm_provider="gemini", google_api_key="g",
                        reviewer_provider="groq")
    factory = ChatModelFactory(settings)
    assert factory.route("writer") == ("gemini", settings.resolved_primary_model)
    assert factory.route("planner") == ("gemini", settings.resolved_fast_model)
    assert factory.route("reviewer") == ("groq", "llama-3.3-70b-versatile")
    assert not factory.is_configured()
    assert "GROQ_API_KEY" in factory.missing_key_message()
    assert "GOOGLE_API_KEY" not in factory.missing_key_message()


def test_reviewer_model_override_on_same_provider():
    settings = Settings(_env_file=None, llm_provider="groq", groq_api_key="k",
                        reviewer_model="some-other-model")
    factory = ChatModelFactory(settings)
    assert factory.route("reviewer") == ("groq", "some-other-model")
    assert factory.is_configured() and factory.missing_key_message() == ""


def test_openai_writer_with_gemini_reviewer():
    settings = Settings(_env_file=None, llm_provider="openai", openai_api_key="sk-test",
                        google_api_key="g", reviewer_provider="gemini", llm_max_retries=1)
    factory = ChatModelFactory(settings)
    assert factory.route("writer") == ("openai", "gpt-4.1-mini")
    assert factory.route("planner") == ("openai", "gpt-4.1-nano")
    assert factory.route("reviewer") == ("gemini", "gemini-3.8-flash")
    assert factory.is_configured()

    writer = factory.get("writer")  # built offline; no request is made
    assert type(writer).__name__ == "ChatOpenAI"
    assert writer.model_name == "gpt-4.1-mini" and writer.max_retries == 1


def test_missing_openai_key_names_the_right_variable():
    settings = Settings(_env_file=None, llm_provider="openai")
    message = ChatModelFactory(settings).missing_key_message()
    assert "OPENAI_API_KEY" in message and "platform.openai.com" in message


def test_list_chat_models_hides_non_chat_models(monkeypatch):
    import openai

    from query_tool.services.llm import list_chat_models

    class FakeModels:
        def list(self):
            ids = ["gpt-4.1-mini", "o4-mini", "gpt-4o-transcribe", "text-embedding-3-small",
                   "dall-e-3", "omni-moderation-latest", "gpt-realtime", "whisper-1"]
            return [type("M", (), {"id": i}) for i in ids]

    monkeypatch.setattr(openai, "OpenAI", lambda api_key: type("C", (), {"models": FakeModels()}))
    settings = Settings(_env_file=None, llm_provider="openai", openai_api_key="sk-test")
    assert list_chat_models(settings, "openai") == ["gpt-4.1-mini", "o4-mini"]


# ---- API and eval -------------------------------------------------------


def test_session_endpoints(make_service):
    with TestClient(create_app(make_service())) as client:
        body = {"question": "Refund policy?", "session_id": "web-1"}
        assert client.post("/api/v1/query", json=body).json()["session_id"] == "web-1"
        turns = client.get("/api/v1/sessions/web-1").json()
        assert [t["question"] for t in turns] == ["Refund policy?"]

        assert client.delete("/api/v1/sessions/web-1").status_code == 204
        assert client.get("/api/v1/sessions/web-1").json() == []
        assert client.get("/api/v1/sessions/bad id!").status_code == 422
        bad = {"question": "Refund policy?", "session_id": "../etc"}
        assert client.post("/api/v1/query", json=bad).status_code == 422


def test_eval_replays_history_and_cleans_up(make_service):
    case = EvalCase(
        id="follow-up", question="And for enterprise?", facts=["1 hour"],
        history=[Turn(question="Refund policy?", answer="30 days [1].")],
    )
    llm = FakeLLMProvider(
        planner=[FOLLOW_UP_PLAN],
        writer=["Enterprise customers get a 1 hour response time for critical incidents [1]."],
    )
    service = make_service(llm)
    report = EvalRunner(service).run([case], "full", "inline")

    result = report.cases[0]
    assert result.passed and result.standalone_question.startswith("What is the response")
    assert "User: Refund policy?" in _prompt(llm, "planner")
    with Database(service.settings.app_db).connect() as db:  # the throwaway session is gone
        assert db.execute("SELECT COUNT(*) FROM turns").fetchone()[0] == 0

    retrieval = EvalRunner(service).run([case], "retrieval", "inline")
    assert retrieval.cases == [] and retrieval.skipped == ["follow-up"]
