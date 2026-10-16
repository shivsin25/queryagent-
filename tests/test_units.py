import pytest

from query_tool.agents.common import check_citations, extract_citations
from query_tool.agents.enhancer import polish_rejection_reason
from query_tool.agents.preprocessor import clean_query, sanitize_rewrite
from query_tool.core.schemas import ReviewAssessment, ReviewResult
from query_tool.graph.routing import has_context, should_revise
from query_tool.retrieval.retriever import reciprocal_rank_fusion
from query_tool.services.llm import extract_json, invoke_structured
from tests.conftest import ScriptedChatModel, review_json


def test_pasted_keys_lose_stray_whitespace_and_quotes():
    from query_tool.config.settings import Settings

    settings = Settings(_env_file=None, llm_provider=" openai\n",
                        openai_api_key="sk-proj-abc123 ", google_api_key="  ")
    assert settings.llm_provider == "openai"
    assert settings.openai_api_key.get_secret_value() == "sk-proj-abc123"
    assert settings.google_api_key is None
    quoted = Settings(_env_file=None, groq_api_key='"gsk_xyz"\r\n')
    assert quoted.groq_api_key.get_secret_value() == "gsk_xyz"


def test_logs_never_contain_api_keys():
    import logging

    from query_tool.core.logging import RedactingFormatter, redact

    line = ("Illegal header value b'Bearer sk-proj-GIQvfIabcdefghijklmnop '; google "
            "AIzaSyA1234567890abcdefghijk; groq gsk_abcdefghijklmnop1234")
    cleaned = redact(line)
    assert "sk-proj" not in cleaned and "AIza" not in cleaned and "gsk_" not in cleaned
    assert "Bearer [redacted]" in cleaned

    try:
        raise RuntimeError("Bearer sk-proj-0123456789abcdefghij")
    except RuntimeError:
        record = logging.LogRecord("t", logging.ERROR, __file__, 1, "failed", None,
                                   exc_info=__import__("sys").exc_info())
    formatted = RedactingFormatter("%(message)s").format(record)
    assert "0123456789abcdefghij" not in formatted and "[redacted]" in formatted


def test_clean_query_normalises_whitespace_and_control_chars():
    assert clean_query("  what\tis\x00 the   SLA?\n") == "what is the SLA?"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('"refund policy annual"', "refund policy annual"),
        ("Search query: refund policy\nextra line", "refund policy"),
        ("", "fallback"),
        ("x" * 1000, "fallback"),
    ],
)
def test_sanitize_rewrite(raw, expected):
    assert sanitize_rewrite(raw, "fallback") == expected


def test_extract_citations_handles_all_formats():
    assert extract_citations("A [1]. B [2][3]. C [4, 5]. Not [x].") == {1, 2, 3, 4, 5}


def test_check_citations_flags_out_of_range_sources():
    assert check_citations("Fine [1] [2].", 2) == []
    assert "[0, 3]" in check_citations("Bad [3] and [0].", 2)[0]


def test_review_scores_on_other_scales_are_normalised():
    review = ReviewAssessment(
        grounding_score=9, accuracy_score=80, completeness_score=0.5, verdict="Approve"
    )
    assert (review.grounding_score, review.accuracy_score) == (0.9, 0.8)
    assert review.verdict == "approve"


def _review(**kwargs) -> ReviewResult:
    base = dict(grounding_score=0.5, accuracy_score=0.5, completeness_score=0.5, verdict="revise")
    return ReviewResult(**{**base, **kwargs})


@pytest.mark.parametrize(
    ("state", "route"),
    [
        ({"review": _review(approved=True), "revision_count": 0, "max_revisions": 2}, "enhancer"),
        ({"review": _review(), "revision_count": 0, "max_revisions": 2}, "feedback"),
        ({"review": _review(), "revision_count": 2, "max_revisions": 2}, "enhancer"),
        ({"review": _review(status="failed"), "revision_count": 0, "max_revisions": 2},
         "enhancer"),
        ({"review": _review(), "revision_count": 0, "max_revisions": 0}, "enhancer"),
    ],
)
def test_should_revise(state, route):
    assert should_revise(state) == route


def test_has_context():
    assert has_context({"documents": []}) == "no_context"
    assert has_context({"documents": ["x"]}) == "writer"


def test_rrf_rewards_items_ranked_by_several_retrievers():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c"], ["b"]])
    assert [item for item, _ in fused] == ["b", "c", "a"]


def test_extract_json_from_fenced_block_and_prose():
    assert extract_json('Here:\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": {"b": 2}} done') == {"a": {"b": 2}}


def test_invoke_structured_falls_back_to_json_parsing():
    # The scripted model has no tool calling, so this exercises the JSON fallback path.
    llm = ScriptedChatModel(replies=[f"```json\n{review_json('revise', 0.4)}\n```"], calls=[])
    result = invoke_structured(llm, [], ReviewAssessment)
    assert result.verdict == "revise" and result.grounding_score == 0.4


def test_think_blocks_are_stripped():
    from langchain_core.messages import AIMessage

    from query_tool.services.llm import message_text

    assert message_text(AIMessage(content="<think>hmm</think>Answer [1]")) == "Answer [1]"


@pytest.mark.parametrize(
    ("polished", "reason"),
    [
        ("Refunds within **30 days** [1].", None),
        ("Refunds within 30 days.", "citations changed"),
        ("", "empty output"),
        ("Refunds [1]. " + "Filler. " * 50, "answer grew too much"),
    ],
)
def test_polish_guard(polished, reason):
    assert polish_rejection_reason("Refunds within 30 days [1].", polished) == reason
