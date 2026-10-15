import operator
from typing import Annotated, TypedDict

from query_tool.core.policy import RerankPolicy
from query_tool.core.schemas import DraftRecord, RetrievedChunk, ReviewResult, TraceEvent, Turn


class GraphState(TypedDict, total=False):
    # Inputs
    question: str
    history: list[Turn]
    top_k: int
    max_revisions: int
    max_research_rounds: int
    doc_ids: list[str] | None  # restrict every search to these documents
    # Retries: the earlier answers to this question, what the user said was wrong with them,
    # and the wider search to use this time.
    attempt: int
    previous_answers: list[str]
    retry_feedback: str | None
    rerank_policy: RerankPolicy | None

    # Produced by the pipeline
    blocked: bool  # stopped by the input guardrail
    standalone_question: str
    search_queries: list[str]
    documents: list[RetrievedChunk]
    draft: str
    review: ReviewResult | None
    feedback: str | None
    revision_count: int
    research_rounds: int
    new_sources: int
    final_answer: str
    approved: bool

    # Append-only history (merged across nodes by LangGraph reducers)
    drafts: Annotated[list[DraftRecord], operator.add]
    reviews: Annotated[list[ReviewResult], operator.add]
    trace: Annotated[list[TraceEvent], operator.add]
    warnings: Annotated[list[str], operator.add]


def retry_context(state: GraphState, answer_chars: int = 700) -> str:
    """The earlier answers and the user's complaint, for prompts. Empty for a first answer."""
    previous = state.get("previous_answers") or []
    if not previous:
        return ""
    answers = "\n".join(
        f"Answer {i}: {' '.join(a.split())[:answer_chars]}" for i, a in enumerate(previous, 1)
    )
    complaint = state.get("retry_feedback") or "(no reason given)"
    return (f"The user has asked this {len(previous)} time(s) before and was not satisfied "
            f"with the answers:\n{answers}\nWhat the user said was wrong: {complaint}")


def search_scope(state: GraphState) -> frozenset[str] | None:
    """The documents every search is limited to, or None for all of them."""
    return frozenset(state["doc_ids"]) if state.get("doc_ids") else None
