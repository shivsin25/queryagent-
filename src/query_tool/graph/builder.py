from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from query_tool.agents import (
    make_enhancer,
    make_feedback,
    make_preprocessor,
    make_research_node,
    make_reviewer,
    make_writer,
)
from query_tool.agents.claims import ClaimChecker
from query_tool.core.state import GraphState, search_scope
from query_tool.core.tracing import NodeFn, traced
from query_tool.graph.routing import (
    after_preprocess,
    after_research,
    has_context,
    should_revise,
)
from query_tool.retrieval.retriever import HybridRetriever
from query_tool.services.llm import LLMProvider

NO_CONTEXT_ANSWER = (
    "I couldn't find anything relevant in the indexed documents to answer this question. "
    "Upload documents that cover the topic, or try rephrasing."
)
NO_CONTEXT_IN_SCOPE_ANSWER = (
    "I couldn't find anything relevant in the selected documents to answer this question. "
    "Try another question, or search all documents."
)


def make_retrieve_node(retriever: HybridRetriever) -> NodeFn:
    @traced("retrieve")
    def retrieve(state: GraphState) -> dict:
        question = state["standalone_question"]
        search_queries = state["search_queries"]
        # The reranker (and the relevance gate) judge chunks against the user's own question.
        # Search queries are added when they carry something the question lacks: the parts
        # of a multi-part question, each guaranteed its best chunk, or the context of a
        # follow-up whose rewritten question may still be vague (seen live with a small
        # planner model).
        rerank = [question]
        if len(search_queries) > 1 or state.get("history"):
            rerank = list(dict.fromkeys([question, *search_queries]))
        documents = retriever.retrieve(
            list(dict.fromkeys([*search_queries, question])),
            top_k=state["top_k"],
            rerank_queries=rerank,
            doc_ids=search_scope(state),
            policy=state.get("rerank_policy"),  # a retry searches wider
        )
        return {
            "documents": documents,
            "_detail": {
                "count": len(documents),
                "sources": sorted({d.source for d in documents}),
                "top_score": documents[0].score if documents else None,
                "attempt": state.get("attempt", 1),
            },
        }

    return retrieve


@traced("no_context")
def no_context(state: GraphState) -> dict:
    answer = NO_CONTEXT_IN_SCOPE_ANSWER if search_scope(state) else NO_CONTEXT_ANSWER
    return {"final_answer": answer, "approved": False}


def build_graph(
    llm: LLMProvider,
    retriever: HybridRetriever,
    approval_threshold: float,
    claim_checker: ClaimChecker | None = None,
    research_k: int = 3,
) -> CompiledStateGraph:
    graph = StateGraph(GraphState)
    graph.add_node("preprocess", make_preprocessor(llm))
    graph.add_node("retrieve", make_retrieve_node(retriever))
    graph.add_node("no_context", no_context)
    graph.add_node("writer", make_writer(llm))
    graph.add_node("reviewer", make_reviewer(llm, approval_threshold, claim_checker))
    graph.add_node("research", make_research_node(retriever, research_k))
    graph.add_node("feedback", make_feedback(llm))
    graph.add_node("enhancer", make_enhancer(llm))

    graph.add_edge(START, "preprocess")
    graph.add_conditional_edges("preprocess", after_preprocess, ["retrieve", END])
    graph.add_conditional_edges("retrieve", has_context, ["writer", "no_context"])
    graph.add_edge("writer", "reviewer")
    graph.add_conditional_edges("reviewer", should_revise, ["research", "feedback", "enhancer"])
    graph.add_conditional_edges("research", after_research, ["writer", "feedback", "enhancer"])
    graph.add_edge("feedback", "writer")
    graph.add_edge("enhancer", END)
    graph.add_edge("no_context", END)
    return graph.compile()
