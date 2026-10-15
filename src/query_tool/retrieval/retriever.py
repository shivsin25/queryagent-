"""Hybrid retrieval.

1. Candidates: for every query, dense search (bi-encoder embeddings in Chroma's HNSW
   approximate-nearest-neighbour index) and keyword search (BM25 on an inverted index).
2. Fusion: Reciprocal Rank Fusion of all those rankings.
3. Rerank: the cross-encoder, the slow and accurate stage, scores only the best `rerank_k`
   fused candidates.
"""

import logging
import threading
from collections import defaultdict

import numpy as np

from query_tool.core.policy import RerankPolicy
from query_tool.core.schemas import RetrievedChunk
from query_tool.retrieval.bm25 import Bm25, tokenize
from query_tool.retrieval.embeddings import Embedder, Reranker
from query_tool.retrieval.store import Chunk, VectorStore

logger = logging.getLogger(__name__)

__all__ = ["HybridRetriever", "RerankPolicy", "reciprocal_rank_fusion", "tokenize"]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class _Bm25Index:
    """BM25 over every chunk in the store, rebuilt when the store changes (in any process)."""

    def __init__(self, store: VectorStore) -> None:
        self._store = store
        self._lock = threading.Lock()
        self._key: tuple[str, int] | None = None
        self._chunks: list[Chunk] = []
        self._by_id: dict[str, Chunk] = {}
        self._doc_ids = np.array([], dtype=object)
        self._bm25 = Bm25([])

    def _refresh(self) -> None:
        key = (self._store.version, self._store.count())
        if key == self._key:
            return
        with self._lock:
            if key == self._key:
                return
            chunks = self._store.all_chunks()
            self._bm25 = Bm25([c.text for c in chunks])
            self._chunks = chunks
            self._by_id = {c.chunk_id: c for c in chunks}
            self._doc_ids = np.array([c.doc_id for c in chunks], dtype=object)
            self._key = key
            logger.debug("BM25 index rebuilt over %d chunks", len(chunks))

    def chunks_by_id(self) -> dict[str, Chunk]:
        self._refresh()
        return self._by_id

    def search(self, query: str, k: int, doc_ids: frozenset[str] | None = None) -> list[str]:
        self._refresh()
        # Filter before ranking, so chunks of other documents can't crowd out the scope.
        allowed = np.isin(self._doc_ids, list(doc_ids)) if doc_ids else None
        return [self._chunks[i].chunk_id for i in self._bm25.top(query, k, allowed)]


def _expand_to_parents(selected: list[tuple[Chunk, float]]) -> list[RetrievedChunk]:
    """Give the writer each matched child's parent, once per parent, at its best score."""
    results: list[RetrievedChunk] = []
    seen: set[str] = set()
    for chunk, score in selected:  # best first
        parent = chunk.parent_id or chunk.chunk_id
        if parent in seen:
            continue
        seen.add(parent)
        results.append(RetrievedChunk(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            source=chunk.source,
            text=chunk.parent_text or chunk.text,
            passage=chunk.text,
            parent_id=chunk.parent_id,
            page=chunk.page,
            chunk_index=chunk.chunk_index,
            score=round(float(score), 4),
        ))
    return results


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        reranker: Reranker | None = None,
        candidate_k: int = 20,
        policy: RerankPolicy | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._candidate_k = candidate_k
        self._policy = policy or RerankPolicy()
        self._bm25 = _Bm25Index(store)

    def retrieve(
        self,
        queries: list[str],
        top_k: int,
        rerank_queries: list[str] | None = None,
        exclude: frozenset[str] = frozenset(),
        doc_ids: frozenset[str] | None = None,
        policy: RerankPolicy | None = None,
    ) -> list[RetrievedChunk]:
        """Search with every query and fuse the results.

        rerank_queries are what the cross-encoder scores chunks against (default: the first
        query). With several, each one is guaranteed its best chunk, so every part of a
        multi-part question is represented. Chunks in `exclude` are never returned; with
        `doc_ids`, only chunks of those documents are. `policy` overrides the retriever's own
        for this call (a retry searches wider).
        """
        if self._store.count() == 0:
            return []

        rankings: list[list[str]] = []
        for query in queries:
            embedding = self._embedder.embed_query(query)
            dense = self._store.query(embedding, self._candidate_k, doc_ids)
            rankings.append([chunk.chunk_id for chunk, _ in dense])
            rankings.append(self._bm25.search(query, self._candidate_k, doc_ids))

        by_id = self._bm25.chunks_by_id()
        fused = [
            (chunk, score) for cid, score in reciprocal_rank_fusion(rankings)
            if (chunk := by_id.get(cid)) is not None
            and cid not in exclude and chunk.parent_id not in exclude
        ][: self._candidate_k]
        if not fused:
            return []

        selected = fused[:top_k]
        if self._reranker is not None:
            try:
                selected = self._rerank([c for c, _ in fused], rerank_queries or queries[:1],
                                        top_k, policy or self._policy)
            except Exception:
                logger.exception("Reranker failed; using fused ranking")
        return _expand_to_parents(selected)

    @property
    def policy(self) -> RerankPolicy:
        return self._policy

    def _rerank(
        self, chunks: list[Chunk], queries: list[str], top_k: int, policy: RerankPolicy
    ) -> list[tuple[Chunk, float]]:
        chunks = chunks[: policy.rerank_k]  # the cross-encoder is the expensive stage
        n = len(chunks)
        flat = self._reranker.score_pairs([(q, c.text) for q in queries for c in chunks])
        per_query = [flat[i * n:(i + 1) * n] for i in range(len(queries))]
        query_best = [max(scores) for scores in per_query]
        best = [max(scores) for scores in zip(*per_query, strict=True)]

        # Gate on the best chunk only: in a multi-part question a chunk that answers one part
        # can score low against the whole question and still be needed.
        if policy.min_score is not None and max(best) < policy.min_score:
            logger.info("Best chunk scored %.2f, below %.2f; treating as out of scope",
                        max(best), policy.min_score)
            return []

        def worth_sending(i: int) -> bool:
            if policy.min_chunk_score is not None and best[i] < policy.min_chunk_score:
                return False
            return policy.margin is None or any(
                scores[i] >= top - policy.margin
                for scores, top in zip(per_query, query_best, strict=True)
            )

        picked: list[int] = []
        if len(queries) > 1:  # every part of the question gets its best chunk
            for scores in per_query:
                top = max(range(n), key=scores.__getitem__)
                relevant = policy.min_score is None or scores[top] >= policy.min_score
                if relevant and top not in picked:
                    picked.append(top)
        for i in sorted(range(n), key=best.__getitem__, reverse=True):
            if not picked or (i not in picked and worth_sending(i)):
                picked.append(i)
        picked = sorted(picked[:top_k], key=best.__getitem__, reverse=True)
        return [(chunks[i], best[i]) for i in picked]
