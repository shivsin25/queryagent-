"""Application service: owns the graph and its dependencies, used by the API and CLI."""

import logging
import time
import uuid
from collections.abc import Iterator
from typing import Any

from query_tool import __version__
from query_tool.agents.claims import ClaimChecker
from query_tool.agents.common import extract_citations
from query_tool.config.settings import Settings
from query_tool.core.exceptions import (
    BudgetExceededError,
    DocumentError,
    LLMNotConfiguredError,
    NotFoundError,
    RetryLimitError,
)
from query_tool.core.schemas import (
    Citation,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    StreamEvent,
    Turn,
    UsageSummary,
)
from query_tool.core.state import GraphState
from query_tool.graph.builder import build_graph
from query_tool.retrieval.embeddings import Embedder, FastEmbedEmbedder, FastEmbedReranker
from query_tool.retrieval.ingestion import IngestionService
from query_tool.retrieval.retriever import HybridRetriever, RerankPolicy
from query_tool.retrieval.store import VectorStore
from query_tool.services.cache import AnswerCache, pipeline_fingerprint
from query_tool.services.conversations import ConversationStore
from query_tool.services.llm import ChatModelFactory, LLMProvider, content_text
from query_tool.services.query_log import QueryLog
from query_tool.services.retry import (
    Attempt,
    assess,
    escalate,
    improved_answer_for_cache,
    same_question,
)
from query_tool.services.usage import PriceTable, UsageLedger, UsageTracker
from query_tool.storage.database import Database

logger = logging.getLogger(__name__)

SNIPPET_CHARS = 320
# The node whose text is streamed to clients as it is generated.
STREAMED_NODE = "writer"
# Earlier answers shown to the planner and writer on a retry.
MAX_PREVIOUS_ANSWERS = 3


class QueryService:
    def __init__(
        self,
        settings: Settings,
        llm: LLMProvider,
        store: VectorStore,
        embedder: Embedder,
        retriever: HybridRetriever,
        claim_checker: ClaimChecker | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.store = store
        self.retriever = retriever
        self.ingestion = IngestionService(
            store, embedder, settings.chunk_size, settings.chunk_overlap,
            settings.parent_chunk_size,
        )
        db = Database(settings.app_db)
        self.conversations = ConversationStore(db)
        self.usage = UsageLedger(db)
        self.prices = PriceTable(settings.model_prices)
        self.cache = AnswerCache(db, settings.cache_ttl_hours)
        self.query_log = QueryLog(db)
        self._fingerprint = pipeline_fingerprint(settings)
        self._graph = build_graph(
            llm, retriever, settings.approval_threshold, claim_checker, settings.research_k
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> "QueryService":
        store = VectorStore(settings.chroma_dir, settings.collection_name)
        embedder = FastEmbedEmbedder(settings.embedding_model, settings.model_cache_dir)
        reranker = (
            FastEmbedReranker(settings.rerank_model, settings.model_cache_dir)
            if settings.rerank_enabled
            else None
        )
        policy = RerankPolicy(
            rerank_k=settings.rerank_k,
            min_score=settings.min_relevance,
            min_chunk_score=settings.min_chunk_score,
            margin=settings.rerank_margin,
        )
        retriever = HybridRetriever(store, embedder, reranker, settings.candidate_k, policy)
        claim_checker = (
            ClaimChecker(reranker, settings.claim_support_threshold)
            if reranker and settings.claim_check_enabled
            else None
        )
        return cls(
            settings, ChatModelFactory(settings), store, embedder, retriever, claim_checker
        )

    # ---- documents -------------------------------------------------------

    def seed_if_empty(self) -> IngestResponse | None:
        if self.store.count() or not self.settings.sample_docs_dir.is_dir():
            return None
        logger.info("Vector store is empty; indexing sample documents")
        return self.ingestion.ingest_paths([self.settings.sample_docs_dir])

    def health(self) -> HealthResponse:
        docs = self.store.list_documents()
        s = self.settings
        reviewer_provider, reviewer_model = s.resolved_reviewer
        return HealthResponse(
            version=__version__,
            llm_provider=s.llm_provider,
            llm_configured=self.llm.is_configured(),
            primary_model=s.resolved_primary_model,
            fast_model=s.resolved_fast_model,
            reviewer_model=f"{reviewer_provider}:{reviewer_model}",
            embedding_model=s.embedding_model,
            rerank_enabled=s.rerank_enabled,
            documents=len(docs),
            chunks=sum(d.chunks for d in docs),
            spent_today_usd=round(self.usage.spent_today(), 6),
            daily_budget_usd=s.daily_budget_usd,
        )

    # ---- queries ---------------------------------------------------------

    def ensure_ready(self) -> None:
        if not self.llm.is_configured():
            raise LLMNotConfiguredError(self.llm.missing_key_message())

    def resolve_documents(self, refs: list[str]) -> list[str]:
        """Doc ids for a mix of doc ids and file names, e.g. from the command line."""
        docs = self.store.list_documents()
        by_name = {d.source.lower(): d.doc_id for d in docs}
        ids = {d.doc_id for d in docs}
        resolved = []
        for ref in refs:
            doc_id = ref if ref in ids else by_name.get(ref.lower())
            if doc_id is None:
                raise DocumentError(f"Document {ref} not found.")
            resolved.append(doc_id)
        return resolved

    def check_scope(self, request: QueryRequest) -> None:
        if request.doc_ids:
            known = {d.doc_id for d in self.store.list_documents()}
            if missing := sorted(set(request.doc_ids) - known):
                raise DocumentError(f"Document {', '.join(missing)} not found.")

    def ensure_budget(self) -> None:
        """Refuse new LLM work once today's recorded spend reaches the daily budget."""
        budget = self.settings.daily_budget_usd
        if budget is None:
            return
        spent = self.usage.spent_today()
        if spent >= budget:
            raise BudgetExceededError(
                f"Today's LLM spend (${spent:.4f}) has reached the daily budget of "
                f"${budget:.2f} (QT_DAILY_BUDGET_USD). Raise it in .env, or wait until "
                "midnight UTC. Cached answers are still served."
            )

    def resolve_attempt(self, request: QueryRequest) -> Attempt:
        """Is this a retry, and of which question? Raises once the retries are used up.

        A retry is either explicit (`retry_of`) or the same question asked again straight
        after in the same session.
        """
        parent = None
        if request.retry_of:
            parent = self.query_log.get(request.retry_of)
            if parent is None:
                raise NotFoundError(f"Request {request.retry_of} not found.")
        elif request.session_id:
            last = self.query_log.latest_in_session(request.session_id)
            if last and same_question(last.question, request.question):
                parent = last
        if parent is None:
            return Attempt(number=1, root_id="", previous=[])

        chain = self.query_log.attempts(parent.root_id)
        answered = max((q.attempt for q in chain), default=parent.attempt)
        limit = self.settings.max_retries_per_question
        if answered >= limit + 1:
            raise RetryLimitError(
                f"Re-query limit reached: this question has already been answered {answered} "
                f"times (the first answer and {limit} retries), so it can't be asked again. "
                "Ask a new or more specific question instead."
            )
        previous = [q.response for q in chain if q.response is not None]
        return Attempt(answered + 1, parent.root_id, previous, request.retry_feedback)

    def check_request(self, request: QueryRequest) -> Attempt:
        """Everything that can refuse a request before any work is done."""
        self.ensure_ready()
        self.check_scope(request)
        return self.resolve_attempt(request)

    def run(self, request: QueryRequest) -> QueryResponse:
        attempt = self.check_request(request)
        request_id, start = uuid.uuid4().hex[:12], time.perf_counter()
        state = self._initial_state(request, attempt)
        cache_key = self._cache_key(request, state, attempt)
        if cached := self._from_cache(cache_key, request, request_id, start):
            return cached

        self.ensure_budget()
        tracker = UsageTracker(self.usage, self.prices, request_id)
        final = self._graph.invoke(state, config=self._config(state, tracker))
        response = self._build_response(final, request, request_id, start, tracker, attempt)
        self._finish(request, response, cache_key, attempt)
        return response

    def stream(self, request: QueryRequest) -> Iterator[StreamEvent]:
        """Yield `start`, then a `step` per finished node with `token`s while the writer
        writes, then the full `result` (or an `error`)."""
        attempt = self.check_request(request)
        request_id, start = uuid.uuid4().hex[:12], time.perf_counter()
        state = self._initial_state(request, attempt)
        yield StreamEvent(event="start", data={"request_id": request_id,
                                               "attempt": attempt.number})
        try:
            cache_key = self._cache_key(request, state, attempt)
            if cached := self._from_cache(cache_key, request, request_id, start):
                yield StreamEvent(event="step", node="cache",
                                  data={"node": "cache", "detail": {"cached": True}})
                yield StreamEvent(event="result", data=cached.model_dump(mode="json"))
                return

            self.ensure_budget()
            tracker = UsageTracker(self.usage, self.prices, request_id)
            final: GraphState | None = None
            for mode, chunk in self._graph.stream(
                state,
                config=self._config(state, tracker),
                stream_mode=["updates", "values", "messages"],
            ):
                if mode == "values":
                    final = chunk
                elif mode == "messages":
                    message, metadata = chunk
                    text = content_text(message.content)
                    if text and metadata.get("langgraph_node") == STREAMED_NODE:
                        yield StreamEvent(event="token", node=STREAMED_NODE, data={"text": text})
                else:
                    for node, update in chunk.items():
                        trace = (update or {}).get("trace") or []
                        yield StreamEvent(
                            event="step",
                            node=node,
                            data=trace[-1].model_dump(mode="json") if trace else {},
                        )
            if final is None:
                raise RuntimeError("Pipeline produced no state")
            response = self._build_response(final, request, request_id, start, tracker,
                                            attempt)
            self._finish(request, response, cache_key, attempt)
            yield StreamEvent(event="result", data=response.model_dump(mode="json"))
        except Exception as exc:
            if not isinstance(exc, BudgetExceededError):
                logger.exception("Pipeline failed")
            yield StreamEvent(event="error", data={"message": str(exc), "type": type(exc).__name__})

    # ---- internals -------------------------------------------------------

    def _initial_state(self, request: QueryRequest, attempt: Attempt) -> GraphState:
        s = self.settings
        history = (
            self.conversations.history(request.session_id, s.history_turns)
            if request.session_id and s.history_turns
            else []
        )
        effort = escalate(
            s, self.retriever.policy, attempt,
            top_k=request.top_k or s.top_k,
            max_revisions=(request.max_revisions if request.max_revisions is not None
                           else s.max_revisions),
        )
        return {
            "question": request.question,
            "history": history,
            "top_k": effort.top_k,
            "max_revisions": effort.max_revisions,
            "max_research_rounds": effort.max_research_rounds,
            "rerank_policy": effort.policy,
            "doc_ids": sorted(set(request.doc_ids)) if request.doc_ids else None,
            "attempt": attempt.number,
            "previous_answers": [r.answer for r in attempt.previous][-MAX_PREVIOUS_ANSWERS:],
            "retry_feedback": attempt.feedback,
            "revision_count": 0,
        }

    def _config(self, state: GraphState, tracker: UsageTracker) -> dict[str, Any]:
        # 5 nodes on the direct path, 3 per revision, and 1 per research round that finds
        # nothing; the rest is slack. The tracker sees every LLM call made inside the graph.
        limit = 10 + 3 * state["max_revisions"] + 2 * state["max_research_rounds"]
        return {"recursion_limit": limit, "callbacks": [tracker]}

    def _cache_key(
        self, request: QueryRequest, state: GraphState, attempt: Attempt
    ) -> str | None:
        # A follow-up depends on its conversation, so only standalone questions are cached.
        # A retry exists because the earlier answer wasn't good enough: never serve it again.
        if (not (request.use_cache and self.cache.enabled) or state["history"]
                or attempt.is_retry):
            return None
        return AnswerCache.key(request.question, state["top_k"], state["max_revisions"],
                               state["doc_ids"], self._fingerprint, self.store.version)

    def _from_cache(
        self, key: str | None, request: QueryRequest, request_id: str, start: float
    ) -> QueryResponse | None:
        hit = self.cache.get(key) if key else None
        if hit is None:
            return None
        response = hit.model_copy(update={
            "request_id": request_id,
            "session_id": request.session_id,
            "question": request.question,
            "cached": True,
            "usage": UsageSummary(),
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            # A cached answer starts a new chain of attempts for this asker.
            "root_request_id": request_id,
            "attempt": 1,
            "retries_left": self.settings.max_retries_per_question,
            "retry": None,
        })
        self._finish(request, response, cache_key=None, attempt=None)
        return response

    def _finish(
        self,
        request: QueryRequest,
        response: QueryResponse,
        cache_key: str | None,
        attempt: Attempt | None,
    ) -> None:
        """Bookkeeping after an answer. A failure here never costs the user the answer."""
        try:
            self.query_log.record(response, cache_key)
            # An improved, approved retry replaces the first answer in the cache, so the
            # next person who asks gets the better one straight away.
            if attempt and attempt.is_retry and improved_answer_for_cache(response):
                root = self.query_log.get(attempt.root_id)
                if root and root.cache_key:
                    self.cache.put(root.cache_key, response)
            if request.session_id and response.answer:
                self.conversations.append(request.session_id, Turn(
                    question=response.question,
                    standalone_question=response.standalone_question,
                    answer=response.answer,
                    approved=response.approved,
                ))
            # Approved answers, and "not in the documents", are safe to reuse; an answer
            # that missed approval might do better on another try.
            declined = any(step.node == "no_context" for step in response.trace)
            if cache_key and (response.approved or declined):
                self.cache.put(cache_key, response)
        except Exception:
            logger.exception("Could not record the query")

    def _build_response(
        self,
        state: GraphState,
        request: QueryRequest,
        request_id: str,
        start: float,
        tracker: UsageTracker,
        attempt: Attempt,
    ) -> QueryResponse:
        documents = state.get("documents", [])
        answer = state.get("final_answer", "")
        citations = [
            Citation(
                index=i,
                source=documents[i - 1].source,
                page=documents[i - 1].page,
                chunk_id=documents[i - 1].chunk_id,
                snippet=(documents[i - 1].passage or documents[i - 1].text)[:SNIPPET_CHARS],
                score=documents[i - 1].score,
            )
            for i in sorted(extract_citations(answer))
            if 1 <= i <= len(documents)
        ]
        standalone = state.get("standalone_question", state["question"])
        response = QueryResponse(
            request_id=request_id,
            session_id=request.session_id,
            root_request_id=attempt.root_id or request_id,
            attempt=attempt.number,
            retries_left=max(0, self.settings.max_retries_per_question - (attempt.number - 1)),
            question=state["question"],
            standalone_question=standalone,
            search_queries=state.get("search_queries", [standalone]),
            answer=answer,
            approved=state.get("approved", False),
            revisions=state.get("revision_count", 0),
            review=state.get("review"),
            reviews=state.get("reviews", []),
            drafts=state.get("drafts", []),
            citations=citations,
            sources=documents,
            trace=state.get("trace", []),
            warnings=state.get("warnings", []),
            usage=UsageSummary(calls=tracker.calls),
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )
        if attempt.is_retry and attempt.previous:
            response.retry = assess(attempt.previous[-1], response, attempt.feedback)
        return response
