"""Corrective retrieval: search again for what the reviewer says the sources are missing.

Runs locally (no LLM call). New chunks are appended after the existing ones, so citation
numbers in the current draft stay valid, and the writer is only called again if something
new was found.
"""

import logging

from query_tool.agents.feedback import numbered, review_findings
from query_tool.agents.preprocessor import clean_query
from query_tool.core.schemas import RetrievedChunk, ReviewResult
from query_tool.core.state import GraphState, search_scope
from query_tool.core.tracing import NodeFn, traced
from query_tool.retrieval.retriever import HybridRetriever

logger = logging.getLogger(__name__)

MAX_GAP_QUERIES = 3


def research_instructions(gaps: list[str], first: int, last: int, review: ReviewResult) -> str:
    added = f"[{first}]" if first == last else f"[{first}] to [{last}]"
    steps = [
        f"New sources {added} were found for what the answer was missing "
        f"({'; '.join(gaps)}). Use them to answer those parts and cite them. Keep the "
        "existing citation numbers as they are."
    ]
    if not review.approved:
        steps += review_findings(review)
    return numbered(steps)


def make_research_node(retriever: HybridRetriever, research_k: int) -> NodeFn:
    @traced("research")
    def research(state: GraphState) -> dict:
        review = state["review"]
        documents = state["documents"]
        gaps = [g for g in (clean_query(m) for m in review.missing_information) if g]
        gaps = list(dict.fromkeys(gaps))[:MAX_GAP_QUERIES]

        found: list[RetrievedChunk] = []
        if gaps:
            try:
                found = retriever.retrieve(
                    gaps, top_k=research_k, rerank_queries=gaps,
                    exclude=frozenset(
                        key for d in documents for key in (d.chunk_id, d.parent_id) if key
                    ),
                    doc_ids=search_scope(state),
                )
            except Exception:
                logger.exception("Research retrieval failed; continuing with current sources")

        update: dict = {
            "research_rounds": state.get("research_rounds", 0) + 1,
            "new_sources": len(found),
            "_detail": {
                "queries": gaps,
                "new_chunks": len(found),
                "sources": sorted({d.source for d in found}),
            },
        }
        if found:
            update |= {
                "documents": [*documents, *found],
                "revision_count": state.get("revision_count", 0) + 1,
                "feedback": research_instructions(
                    gaps, len(documents) + 1, len(documents) + len(found), review
                ),
            }
        return update

    return research
