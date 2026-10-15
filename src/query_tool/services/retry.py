"""Answering a question again when the user wasn't satisfied.

A retry must not just regenerate the same answer, so each one:
- skips the answer cache;
- searches wider: more chunks, a looser relevance cutoff, more candidates for the
  cross-encoder, one more follow-up search and one more revision;
- asks the planner for different phrasings, and shows the planner and writer the earlier
  answers and the user's complaint;
- reports what it did differently, and says honestly when the documents gave nothing more.

After `max_retries_per_question` retries the question is refused, so a user can't loop on it.
"""

from dataclasses import dataclass, replace

from query_tool.agents.common import extract_citations
from query_tool.config.settings import Settings
from query_tool.core.policy import RerankPolicy
from query_tool.core.schemas import QueryResponse, RetryReport
from query_tool.retrieval.bm25 import tokenize
from query_tool.services.cache import normalize_question

# Two questions with this much word overlap (after stemming) are the same question asked again.
SAME_QUESTION_OVERLAP = 0.8
# Answers this similar say the same thing.
SAME_ANSWER_OVERLAP = 0.85
MAX_TOP_K = 12
MAX_REVISIONS = 5


def _overlap(a: str, b: str) -> float:
    left, right = set(tokenize(a)), set(tokenize(b))
    return len(left & right) / len(left | right) if left | right else 1.0


def same_question(a: str, b: str) -> bool:
    if normalize_question(a) == normalize_question(b):
        return True
    return min(len(tokenize(a)), len(tokenize(b))) >= 3 and _overlap(a, b) >= SAME_QUESTION_OVERLAP


@dataclass(frozen=True)
class Attempt:
    """Where a request sits in the chain of answers to one question."""

    number: int  # 1 for the first answer
    root_id: str  # request_id of the first answer
    previous: list[QueryResponse]  # earlier answers, first first
    feedback: str | None = None

    @property
    def is_retry(self) -> bool:
        return self.number > 1


@dataclass(frozen=True)
class Escalation:
    """How much harder a retry tries than the first attempt."""

    top_k: int
    max_revisions: int
    max_research_rounds: int
    policy: RerankPolicy | None  # None: the retriever's own policy


def escalate(settings: Settings, base_policy: RerankPolicy, attempt: Attempt, top_k: int,
             max_revisions: int) -> Escalation:
    level = attempt.number - 1  # 0 for a first answer, 1-3 for retries
    if level == 0:
        return Escalation(top_k, max_revisions, settings.max_research_rounds, None)
    return Escalation(
        top_k=min(top_k + 2 * level, MAX_TOP_K),
        max_revisions=min(max_revisions + 1, MAX_REVISIONS),
        max_research_rounds=settings.max_research_rounds + 1,
        policy=replace(
            base_policy,
            rerank_k=max(base_policy.rerank_k, 20),
            margin=None if base_policy.margin is None else base_policy.margin + 2 * level,
            min_chunk_score=(None if base_policy.min_chunk_score is None
                             else base_policy.min_chunk_score - level),
        ),
    )


def assess(previous: QueryResponse, current: QueryResponse, feedback: str | None) -> RetryReport:
    """Compare a retry with the attempt before it, in terms the user can check."""
    changes: list[str] = []
    if feedback:
        changes.append(f"Used your feedback: \"{feedback}\"")
    new_queries = [q for q in current.search_queries if q not in previous.search_queries]
    if new_queries:
        changes.append(f"Searched with {len(current.search_queries)} queries "
                       f"(before: {len(previous.search_queries)}), including "
                       + ", ".join(f"\"{q}\"" for q in new_queries[:3]))
    seen = {c.parent_id or c.chunk_id for c in previous.sources}
    new_sections = [c for c in current.sources if (c.parent_id or c.chunk_id) not in seen]
    changes.append(f"Read {len(current.sources)} passages (before: {len(previous.sources)})"
                   + (f", {len(new_sections)} of them new" if new_sections else ""))

    cited_before = {previous.sources[i - 1].parent_id or previous.sources[i - 1].chunk_id
                    for i in extract_citations(previous.answer)
                    if 1 <= i <= len(previous.sources)}
    cited_now = {current.sources[i - 1].parent_id or current.sources[i - 1].chunk_id
                 for i in extract_citations(current.answer) if 1 <= i <= len(current.sources)}
    new_citations = cited_now - cited_before
    if new_citations:
        changes.append(f"Cites {len(new_citations)} source(s) the earlier answer didn't use")

    before, now = previous.review, current.review
    if before and now and before.status == now.status == "ok":
        if now.completeness_score != before.completeness_score:
            changes.append(f"Reviewer's completeness score {now.completeness_score:.0%} "
                           f"(before: {before.completeness_score:.0%})")
        complete_enough = now.completeness_score >= before.completeness_score
    else:
        complete_enough = current.approved or not previous.approved

    different = _overlap(previous.answer, current.answer) < SAME_ANSWER_OVERLAP
    improved = different and (bool(new_citations) or complete_enough)
    note = None
    if not improved:
        note = ("I searched more widely this time but found nothing beyond what the earlier "
                "answer said, so the documents probably don't contain more on this. Try asking "
                "about a specific part, or add a document that covers it.")
    return RetryReport(previous_request_id=previous.request_id, changes=changes,
                       improved=improved, note=note)


def improved_answer_for_cache(response: QueryResponse) -> bool:
    """An improved, approved retry replaces the cached first answer."""
    return bool(response.retry and response.retry.improved and response.approved)
