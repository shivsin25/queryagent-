"""Relevance gate, section-aware chunking and claim-level citation checks."""

from query_tool.agents.claims import ClaimChecker, claim_issues, claim_support, split_claims
from query_tool.core.schemas import QueryRequest, RetrievedChunk
from query_tool.graph.builder import NO_CONTEXT_ANSWER
from query_tool.retrieval.ingestion import IngestionService, markdown_sections, tidy_text
from query_tool.retrieval.retriever import HybridRetriever
from query_tool.services.pipeline import QueryService
from tests.conftest import (
    SAMPLE_DOC,
    FakeLLMProvider,
    HashEmbedder,
    KeywordReranker,
    review_json,
)


def _chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=text[:8], doc_id="d", source="s.md", text=text)


# ---- chunking -----------------------------------------------------------


def test_tidy_text_collapses_table_padding():
    table = "| Plan       | Price |\n|------------|-------|\n| Starter    | Free  |\n\n\n\nEnd"
    assert tidy_text(table) == "| Plan | Price |\n|---|---|\n| Starter | Free |\n\nEnd"


def test_markdown_sections_carry_heading_path_and_ignore_code_fences():
    text = (
        "# Guide\nIntro.\n## Trials\nTrial text.\n```\n# not a heading\n```\n"
        "### Ending\nEnd text.\n## Refunds\nRefund text.\n## Empty\n"
    )
    assert markdown_sections(text) == [
        ("Guide", "Intro."),
        ("Guide > Trials", "Trial text.\n```\n# not a heading\n```"),
        ("Guide > Trials > Ending", "End text."),
        ("Guide > Refunds", "Refund text."),
    ]


def test_markdown_is_chunked_per_section_with_header():
    service = IngestionService(None, None, chunk_size=500, chunk_overlap=0)
    pieces = service.split("guide.md", b"# Guide\n## Trials\nA 14-day trial.\n## Refunds\nNone.")
    assert [p.text for p in pieces] == [
        "Guide > Trials\n\nA 14-day trial.",
        "Guide > Refunds\n\nNone.",
    ]


def test_changing_chunk_settings_reindexes_unchanged_files(store):
    data = b"The office closes at 6pm on Fridays."
    ingestion = IngestionService(store, HashEmbedder(), 120, 0)
    assert ingestion.ingest_bytes("notes.txt", data) is not None
    assert ingestion.ingest_bytes("notes.txt", data) is None

    rechunked = IngestionService(store, HashEmbedder(), 300, 0)
    assert rechunked.ingest_bytes("notes.txt", data) is not None
    assert [d.source for d in store.list_documents()] == ["notes.txt"]


# ---- relevance gate -----------------------------------------------------


def test_off_topic_question_skips_the_writer(make_service):
    llm = FakeLLMProvider()
    response = make_service(llm, rerank=True).run(
        QueryRequest(question="What is the capital of France?")
    )
    assert response.answer == NO_CONTEXT_ANSWER
    assert not response.approved and not response.sources
    assert llm.models["writer"].calls == []


def test_on_topic_question_passes_the_gate(make_service):
    response = make_service(rerank=True).run(
        QueryRequest(question="Can I get a refund on annual plans?")
    )
    assert response.approved
    assert "refunded within 30 days" in response.sources[0].text


# ---- claim checks -------------------------------------------------------


def test_split_claims_joins_wrapped_lines_and_skips_fragments():
    text = (
        "Here is what the sources say:\n\n"
        "- **Refunds** are possible within\n  30 days [1]. Monthly plans are not refundable. [2]\n"
        "- Enterprise gets a 1 hour response [2, 3].\n"
        "Uptime is 99.9% for Business [3]"
    )
    assert split_claims(text) == [
        ("Refunds are possible within 30 days.", [1]),
        ("Monthly plans are not refundable.", [2]),
        ("Enterprise gets a 1 hour response.", [2, 3]),
        ("Uptime is 99.9% for Business", [3]),
    ]


def test_claim_checker_flags_sentences_their_sources_do_not_support():
    documents = [
        _chunk("Annual plans can be refunded within 30 days."),
        _chunk("Enterprise customers get a 1 hour response time."),
    ]
    draft = (
        "Annual plans are refundable within 30 days [2]. "
        "Enterprise customers get a 1 hour response time [1][2]. "
        "The sources do not mention phone support. "
        "Refunds take five business days [7]."
    )
    claims = ClaimChecker(KeywordReranker(), threshold=-4).check(draft, documents)

    assert [c.supported for c in claims] == [False, True, None, None]
    assert claim_support(claims) == 0.5
    issues = claim_issues(claims)
    assert len(issues) == 1 and "cites [2]" in issues[0]


def test_miscited_claim_blocks_approval_and_reaches_the_writer(make_service):
    llm = FakeLLMProvider(
        writer=[
            "Annual plans can be refunded within 30 days of purchase [2].",
            "Annual plans can be refunded within 30 days of purchase [1].",
        ],
    )
    response = make_service(llm, rerank=True).run(
        QueryRequest(question="Can I get a refund on annual plans?")
    )

    first, second = response.reviews
    assert not first.approved and first.claims[0].supported is False
    assert first.claim_support == 0.0
    assert second.approved and second.claim_support == 1.0
    assert response.revisions == 1
    feedback_prompt = llm.models["feedback"].calls[0][-1].content
    assert "does not appear to support it" in feedback_prompt


def test_claim_checker_failure_falls_back_to_llm_review(settings, store):
    class BrokenReranker(KeywordReranker):
        def score_pairs(self, pairs):
            raise RuntimeError("onnx crashed")

    embedder = HashEmbedder()
    service = QueryService(
        settings, FakeLLMProvider(), store, embedder, HybridRetriever(store, embedder),
        ClaimChecker(BrokenReranker(), threshold=-4),
    )
    service.ingestion.ingest_bytes("billing.md", SAMPLE_DOC)
    response = service.run(QueryRequest(question="Can I get a refund on annual plans?"))
    assert response.approved and response.review.claims == []


def test_llm_review_is_accepted_without_claims_to_check(make_service):
    llm = FakeLLMProvider(writer=["I could not find this in the sources."],
                          reviewer=[review_json()])
    response = make_service(llm, rerank=True).run(
        QueryRequest(question="Can I get a refund on annual plans?")
    )
    assert response.approved and response.review.claim_support is None
