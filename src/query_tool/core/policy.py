from dataclasses import dataclass


@dataclass(frozen=True)
class RerankPolicy:
    """How the cross-encoder's scores decide what reaches the writer.

    Cross-encoder scores are logits, comparable across queries, so all cutoffs need one.
    The best chunk for every query is always kept (if the question is in scope at all);
    the cutoffs only decide which further chunks are worth sending.
    """

    rerank_k: int = 20  # fused candidates the cross-encoder scores; the rest are dropped
    min_score: float | None = None  # the best chunk must reach this, or it's out of scope
    min_chunk_score: float | None = None  # absolute floor for every other chunk
    # Keep a chunk only if it scores within this of the best chunk for the same query. The
    # chunk that answers a question is nearly always that query's best; the rest is noise.
    margin: float | None = None
