from query_tool.core.schemas import QueryRequest
from query_tool.graph.builder import NO_CONTEXT_ANSWER
from tests.conftest import SAMPLE_DOC, FakeLLMProvider, review_json


def _nodes(response) -> list[str]:
    return [step.node for step in response.trace]


def test_ingestion_is_idempotent_and_replaces_changed_files(make_service, store):
    service = make_service()
    first_count = store.count()
    assert service.ingestion.ingest_bytes("billing.md", SAMPLE_DOC) is None
    assert store.count() == first_count

    updated = SAMPLE_DOC + b"\nRefunds are paid within 5 business days.\n"
    info = service.ingestion.ingest_bytes("billing.md", updated)
    docs = store.list_documents()
    assert len(docs) == 1 and docs[0].doc_id == info.doc_id


def test_retriever_ranks_relevant_chunk_first(make_service):
    service = make_service()
    response = service.run(QueryRequest(question="How long do I have to get a refund?", top_k=1))
    assert "refunded within 30 days" in response.sources[0].text


def test_approved_on_first_pass(make_service):
    response = make_service().run(QueryRequest(question="Can I get a refund on annual plans?"))
    assert response.approved
    assert response.revisions == 0
    assert _nodes(response) == ["preprocess", "retrieve", "writer", "reviewer", "enhancer"]
    assert response.answer == "Annual plans can be refunded within 30 days [1]."
    assert [c.index for c in response.citations] == [1]
    assert response.search_queries == ["refund policy annual plan"]
    assert response.standalone_question == response.question  # no history: never reworded


def test_revision_loop_until_approved(make_service):
    llm = FakeLLMProvider(
        writer=["Refunds are always available [1].", "Annual plans: refund within 30 days [1]."],
        reviewer=[review_json("revise", 0.3, ["Overstates refund policy"]), review_json()],
    )
    response = make_service(llm).run(QueryRequest(question="Refund policy?"))

    assert response.approved and response.revisions == 1
    assert _nodes(response) == [
        "preprocess", "retrieve", "writer", "reviewer", "feedback", "writer", "reviewer",
        "enhancer",
    ]
    assert [d.text for d in response.drafts] == [
        "Refunds are always available [1].",
        "Annual plans: refund within 30 days [1].",
    ]
    # The second writer call must include the reviewer-driven instructions.
    revision_prompt = llm.models["writer"].calls[1][-1].content
    assert "1. Fix the citation." in revision_prompt


def test_stops_at_max_revisions_with_warning(make_service):
    llm = FakeLLMProvider(reviewer=[review_json("revise", 0.2, ["Wrong"])])
    response = make_service(llm).run(QueryRequest(question="Refund policy?", max_revisions=2))

    assert not response.approved
    assert response.revisions == 2
    assert _nodes(response).count("writer") == 3
    assert any("revision limit" in w for w in response.warnings)


def test_low_scores_block_approval_even_if_llm_approves(make_service):
    llm = FakeLLMProvider(reviewer=[review_json("approve", 0.5)])
    response = make_service(llm).run(QueryRequest(question="Refund policy?", max_revisions=0))
    assert not response.approved


def test_invalid_citation_blocks_approval(make_service):
    llm = FakeLLMProvider(writer=["Refunds within 30 days [9]."])
    response = make_service(llm).run(QueryRequest(question="Refund policy?", max_revisions=0))
    assert not response.approved
    assert response.review.citation_issues


def test_enhancer_output_that_drops_citations_is_discarded(make_service):
    llm = FakeLLMProvider(enhancer=["Annual plans can be refunded within 30 days."])
    response = make_service(llm).run(QueryRequest(question="Refund policy?"))
    assert response.answer == "Annual plans can be refunded within 30 days [1]."


def test_reviewer_failure_returns_unverified_answer(make_service):
    llm = FakeLLMProvider(reviewer=[RuntimeError("boom")])
    response = make_service(llm).run(QueryRequest(question="Refund policy?"))
    assert not response.approved and response.revisions == 0
    assert response.review.status == "failed"
    assert any("could not verify" in w for w in response.warnings)


def test_planner_failure_falls_back_to_question(make_service):
    llm = FakeLLMProvider(planner=[RuntimeError("rate limited")])
    response = make_service(llm).run(QueryRequest(question="Refund   policy?"))
    assert response.search_queries == ["Refund policy?"]
    assert response.answer
    assert any("planning failed" in w for w in response.warnings)


def test_empty_index_short_circuits(make_service):
    response = make_service(seed=False).run(QueryRequest(question="Refund policy?"))
    assert response.answer == NO_CONTEXT_ANSWER
    assert _nodes(response) == ["preprocess", "retrieve", "no_context"]


def test_stream_emits_steps_then_result(make_service):
    events = list(make_service().stream(QueryRequest(question="Refund policy?")))
    assert events[0].event == "start"
    assert [e.node for e in events if e.event == "step"][:2] == ["preprocess", "retrieve"]
    assert events[-1].event == "result"
    assert events[-1].data["approved"] is True


def test_stream_reports_errors_as_events(make_service):
    llm = FakeLLMProvider(writer=[RuntimeError("provider down")])
    events = list(make_service(llm).stream(QueryRequest(question="Refund policy?")))
    assert events[-1].event == "error"
    assert "provider down" in events[-1].data["message"]
