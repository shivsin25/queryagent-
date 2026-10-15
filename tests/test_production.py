"""Cost tracking, the daily budget, the answer cache, index versioning, feedback, streaming."""

import pytest
from fastapi.testclient import TestClient

from query_tool.api.main import create_app
from query_tool.config.settings import Settings
from query_tool.core.exceptions import BudgetExceededError
from query_tool.core.schemas import CallUsage, QueryRequest, Turn
from query_tool.retrieval.ingestion import IngestionService
from query_tool.retrieval.retriever import HybridRetriever
from query_tool.retrieval.store import VectorStore
from query_tool.services.cache import normalize_question
from query_tool.services.llm import ChatModelFactory
from query_tool.services.usage import UNKNOWN_PAID_PRICE, PriceTable
from tests.conftest import FakeLLMProvider, HashEmbedder, review_json

GPT41_MINI_CALL = (1000 * 0.40 + 200 * 1.60) / 1_000_000  # what one paid fake call costs


def _calls(llm: FakeLLMProvider) -> int:
    return sum(len(model.calls) for model in llm.models.values())


# ---- prices -------------------------------------------------------------


def test_price_table():
    prices = PriceTable({"my-model": (1.0, 2.0)})
    assert prices.price("openai", "gpt-4.1-mini-2025-04-14") == (0.40, 1.60)  # dated snapshot
    assert prices.price("openai", "gpt-4.1") == (2.00, 8.00)  # not confused with -mini
    assert prices.price("gemini", "gemini-3.8-flash") == (0.0, 0.0)  # free tier
    assert prices.price("openai", "gpt-99") == UNKNOWN_PAID_PRICE  # unknown: assume dear
    assert prices.price("openai", "my-model") == (1.0, 2.0)  # QT_MODEL_PRICES
    assert prices.cost("openai", "gpt-4.1-mini", 1000, 200) == pytest.approx(GPT41_MINI_CALL)


# ---- usage tracking -----------------------------------------------------


def test_every_llm_call_is_costed_by_node_and_recorded(make_service):
    llm = FakeLLMProvider(paid=True)
    service = make_service(llm)
    response = service.run(QueryRequest(question="Refund policy?"))

    nodes = [c.node for c in response.usage.calls]
    assert nodes == ["preprocess", "writer", "reviewer", "enhancer"]
    assert response.usage.input_tokens == 4000 and response.usage.output_tokens == 800
    assert response.usage.cost_usd == pytest.approx(4 * GPT41_MINI_CALL)
    assert service.usage.spent_today() == pytest.approx(4 * GPT41_MINI_CALL)


def test_free_provider_calls_are_counted_at_zero_cost(make_service):
    response = make_service().run(QueryRequest(question="Refund policy?"))
    assert len(response.usage.calls) == 4
    assert response.usage.cost_usd == 0
    assert all(c.estimated for c in response.usage.calls)  # the fake reports no token counts


# ---- daily budget -------------------------------------------------------


@pytest.fixture
def over_budget(make_service, settings):
    settings.daily_budget_usd = 0.01
    service = make_service(FakeLLMProvider(paid=True))
    service.usage.record("earlier", CallUsage(node="writer", provider="openai",
                                              model="gpt-4.1", input_tokens=0,
                                              output_tokens=0, cost_usd=0.02))
    return service


def test_budget_blocks_new_llm_work(over_budget):
    with pytest.raises(BudgetExceededError, match="daily budget"):
        over_budget.run(QueryRequest(question="Refund policy?"))
    assert _calls(over_budget.llm) == 0


def test_budget_errors_reach_api_clients(over_budget):
    with TestClient(create_app(over_budget)) as client:
        response = client.post("/api/v1/query", json={"question": "Refund policy?"})
        assert response.status_code == 429
        assert "QT_DAILY_BUDGET_USD" in response.json()["detail"]
        assert client.get("/api/v1/health").json()["spent_today_usd"] == 0.02
    events = list(over_budget.stream(QueryRequest(question="Refund policy?")))
    assert events[-1].event == "error" and events[-1].data["type"] == "BudgetExceededError"


def test_cached_answers_are_served_over_budget(make_service, settings):
    service = make_service(FakeLLMProvider(paid=True))
    service.run(QueryRequest(question="Refund policy?"))
    settings.daily_budget_usd = 0.0
    assert service.run(QueryRequest(question="refund policy")).cached


# ---- answer cache -------------------------------------------------------


def test_repeated_question_is_answered_from_cache_for_free(make_service):
    llm = FakeLLMProvider(paid=True)
    service = make_service(llm)
    first = service.run(QueryRequest(question="Refund policy?"))
    calls = _calls(llm)

    second = service.run(QueryRequest(question="  REFUND   policy "))
    assert second.cached and not first.cached
    assert _calls(llm) == calls  # no LLM call
    assert second.answer == first.answer and second.usage.cost_usd == 0
    assert second.request_id != first.request_id
    assert second.question == "  REFUND   policy "

    third = service.run(QueryRequest(question="Refund policy?", use_cache=False))
    assert not third.cached and _calls(llm) > calls


def test_cache_misses_when_documents_or_options_change(make_service):
    llm = FakeLLMProvider()
    service = make_service(llm)
    service.run(QueryRequest(question="Refund policy?"))
    assert not service.run(QueryRequest(question="Refund policy?", top_k=2)).cached

    service.ingestion.ingest_bytes("new.txt", b"Refunds are paid within 5 business days.")
    assert not service.run(QueryRequest(question="Refund policy?")).cached


def test_unapproved_answers_and_follow_ups_are_not_cached(make_service):
    llm = FakeLLMProvider(reviewer=[review_json("revise", 0.2)])
    service = make_service(llm)
    service.run(QueryRequest(question="Refund policy?", max_revisions=0))
    assert not service.run(QueryRequest(question="Refund policy?", max_revisions=0)).cached

    service = make_service(FakeLLMProvider(), seed=False)
    service.conversations.append("chat", Turn(question="Earlier?", answer="Yes."))
    service.run(QueryRequest(question="And then?", session_id="chat"))
    assert not service.run(QueryRequest(question="And then?", session_id="chat")).cached


def test_off_topic_refusals_are_cached(make_service):
    service = make_service(rerank=True)
    service.run(QueryRequest(question="What is the capital of France?"))
    assert service.run(QueryRequest(question="what is the capital of france")).cached


def test_normalize_question():
    assert normalize_question("  Refund   Policy?! ") == "refund policy"


def test_eval_runs_bypass_the_cache(make_service):
    from query_tool.evaluation.dataset import EvalCase
    from query_tool.evaluation.runner import EvalRunner

    llm = FakeLLMProvider(paid=True)
    service = make_service(llm)
    service.run(QueryRequest(question="Refund policy?"))
    case = EvalCase(id="c", question="Refund policy?", facts=["30 days"])
    report = EvalRunner(service).run([case], "full", "inline")
    assert report.cases[0].cost_usd == pytest.approx(4 * GPT41_MINI_CALL)
    assert report.summary["total_cost_usd"] == pytest.approx(4 * GPT41_MINI_CALL)


def test_eval_stops_when_the_budget_runs_out(over_budget):
    from query_tool.evaluation.dataset import EvalCase
    from query_tool.evaluation.runner import EvalRunner

    case = EvalCase(id="c", question="Refund policy?", facts=["30 days"])
    with pytest.raises(BudgetExceededError):
        EvalRunner(over_budget).run([case, case], "full", "inline")


# ---- index version across processes ------------------------------------


def test_index_changes_are_seen_by_other_store_instances(settings):
    writer = VectorStore(settings.chroma_dir, settings.collection_name)
    reader = VectorStore(settings.chroma_dir, settings.collection_name)  # e.g. another worker
    retriever = HybridRetriever(reader, HashEmbedder())
    before = reader.version

    IngestionService(writer, HashEmbedder(), 500, 0).ingest_bytes(
        "notes.txt", b"The office closes at 6pm on Fridays."
    )
    assert reader.version != before
    found = retriever.retrieve(["office closes fridays"], top_k=1)
    assert found and found[0].source == "notes.txt"  # BM25 rebuilt from the new version


# ---- query log and feedback ---------------------------------------------


def test_feedback_is_stored_against_logged_queries(make_service):
    service = make_service()
    response = service.run(QueryRequest(question="Refund policy?"))
    assert service.query_log.add_feedback(response.request_id, "down", "Too vague")
    assert service.query_log.add_feedback(response.request_id, "up", None)  # replaces it
    assert not service.query_log.add_feedback("unknown", "up", None)

    (item,) = service.query_log.feedback()
    assert item.rating == "up" and item.question == "Refund policy?"
    assert service.query_log.feedback(rating="down") == []


def test_feedback_and_usage_endpoints(make_service):
    with TestClient(create_app(make_service(FakeLLMProvider(paid=True)))) as client:
        request_id = client.post("/api/v1/query", json={"question": "Refund?"}).json()[
            "request_id"]
        ok = client.post("/api/v1/feedback", json={"request_id": request_id, "rating": "down"})
        assert ok.status_code == 204
        missing = client.post("/api/v1/feedback", json={"request_id": "nope", "rating": "up"})
        assert missing.status_code == 404
        bad = client.post("/api/v1/feedback", json={"request_id": request_id, "rating": "meh"})
        assert bad.status_code == 422

        usage = client.get("/api/v1/usage").json()
        assert usage["spent_today_usd"] == pytest.approx(4 * GPT41_MINI_CALL)
        assert usage["by_model"][0]["model"] == "gpt-4.1-mini"


# ---- streaming ----------------------------------------------------------


def test_stream_sends_only_the_writers_text_as_tokens(make_service):
    llm = FakeLLMProvider(
        writer=["Refunds are always available [1].", "Annual plans: refund within 30 days [1]."],
        reviewer=[review_json("revise", 0.3), review_json()],
    )
    events = list(make_service(llm).stream(QueryRequest(question="Refund policy?")))

    tokens = [e for e in events if e.event == "token"]
    assert {e.node for e in tokens} == {"writer"}
    text = "".join(e.data["text"] for e in tokens)
    assert "Refunds are always available" in text and "refund within 30 days" in text
    assert events[-1].event == "result" and events[-1].data["usage"]["calls"]


def test_cache_hit_streams_a_cache_step(make_service):
    service = make_service()
    service.run(QueryRequest(question="Refund policy?"))
    events = list(service.stream(QueryRequest(question="Refund policy?")))
    assert [e.event for e in events] == ["start", "step", "result"]
    assert events[1].node == "cache" and events[-1].data["cached"] is True


# ---- model construction ---------------------------------------------------


def test_only_the_writer_streams_and_output_is_capped():
    settings = Settings(_env_file=None, llm_provider="openai", openai_api_key="sk-test",
                        max_output_tokens=777)
    factory = ChatModelFactory(settings)
    writer, planner = factory.get("writer"), factory.get("planner")
    assert writer.disable_streaming is False and planner.disable_streaming is True
    assert writer.max_tokens == 777 and writer.stream_usage is True
    assert factory.get("writer") is writer  # cached
