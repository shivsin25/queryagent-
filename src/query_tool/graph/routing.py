from typing import Literal

from langgraph.graph import END

from query_tool.core.state import GraphState


def after_preprocess(state: GraphState) -> Literal["retrieve", "__end__"]:
    return END if state.get("blocked") else "retrieve"


def has_context(state: GraphState) -> Literal["writer", "no_context"]:
    return "writer" if state.get("documents") else "no_context"


def should_revise(state: GraphState) -> Literal["research", "feedback", "enhancer"]:
    """Loop until approved, unverifiable, or out of revisions.

    Missing information is searched for first, even in an approved answer: the search is
    local and cheap, and the writer is only called again if it finds something new.
    """
    review = state.get("review")
    if review is None or review.status == "failed":
        return "enhancer"
    if state.get("revision_count", 0) >= state.get("max_revisions", 0):
        return "enhancer"
    if review.missing_information and (
        state.get("research_rounds", 0) < state.get("max_research_rounds", 0)
    ):
        return "research"
    return "enhancer" if review.approved else "feedback"


def after_research(state: GraphState) -> Literal["writer", "feedback", "enhancer"]:
    if state.get("new_sources"):
        return "writer"
    return "enhancer" if state["review"].approved else "feedback"
