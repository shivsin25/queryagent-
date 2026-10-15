"""Run an evaluation dataset through the retriever or the full pipeline and score it."""

import logging
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field

from query_tool.core.exceptions import BudgetExceededError
from query_tool.core.schemas import QueryRequest, QueryResponse
from query_tool.evaluation import metrics
from query_tool.evaluation.dataset import EvalCase
from query_tool.services.pipeline import QueryService

logger = logging.getLogger(__name__)

Mode = Literal["retrieval", "full"]

# Settings recorded with every report so two runs can be told apart. Never includes secrets.
CONFIG_KEYS = frozenset({
    "llm_provider", "embedding_model", "rerank_enabled", "rerank_model", "chunk_size",
    "chunk_overlap", "top_k", "candidate_k", "min_relevance", "claim_check_enabled",
    "claim_support_threshold", "max_revisions", "approval_threshold", "temperature",
    "max_research_rounds", "research_k", "history_turns", "reviewer_provider",
})


class Metric(BaseModel):
    label: str
    higher_is_better: bool = True
    percent: bool = True
    money: bool = False


# Summary keys in display order. Retrieval metrics come from the chunks the pipeline used.
METRICS: dict[str, Metric] = {
    "pass_rate": Metric(label="Cases passed"),
    "retrieval_hit_rate": Metric(label="Hit rate (any evidence found)"),
    "retrieval_recall": Metric(label="Evidence recall@k"),
    "retrieval_mrr": Metric(label="Evidence MRR"),
    "retrieval_precision": Metric(label="Precision (chunks with evidence)"),
    "mean_chunks": Metric(label="Chunks sent to the writer", higher_is_better=False,
                          percent=False),
    "false_rejection_rate": Metric(label="Answerable but no chunks", higher_is_better=False),
    "offtopic_rejection_rate": Metric(label="Unanswerable and no chunks"),
    "answer_accuracy": Metric(label="Answers with every fact"),
    "fact_recall": Metric(label="Fact recall"),
    "abstention_rate": Metric(label="Correct abstentions"),
    "false_refusal_rate": Metric(label="False refusals", higher_is_better=False),
    "approval_rate": Metric(label="Reviewer approval"),
    "claim_support": Metric(label="Cited claims supported"),
    "research_rate": Metric(label="Answers that searched again", higher_is_better=False),
    "mean_revisions": Metric(label="Revisions per answer", higher_is_better=False, percent=False),
    "mean_llm_calls": Metric(label="LLM calls per answer", higher_is_better=False, percent=False),
    "mean_cost_usd": Metric(label="Cost per answer", higher_is_better=False, percent=False,
                            money=True),
    "total_cost_usd": Metric(label="Total cost of the run", higher_is_better=False,
                             percent=False, money=True),
    "mean_latency_ms": Metric(label="Mean latency (ms)", higher_is_better=False, percent=False),
    "p95_latency_ms": Metric(label="p95 latency (ms)", higher_is_better=False, percent=False),
    "error_rate": Metric(label="Errors", higher_is_better=False),
}


class CaseResult(BaseModel):
    id: str
    category: str
    answerable: bool
    retrieved: int = 0
    top_score: float | None = None
    evidence_ranks: list[int | None] = Field(default_factory=list)
    recall: float | None = None
    reciprocal_rank: float | None = None
    precision: float | None = None
    latency_ms: float = 0.0
    error: str | None = None

    # Full mode only
    answer: str | None = None
    standalone_question: str | None = None
    search_queries: list[str] = Field(default_factory=list)
    researched: bool | None = None
    fact_hits: list[bool] = Field(default_factory=list)
    abstained: bool | None = None
    approved: bool | None = None
    revisions: int | None = None
    llm_calls: int | None = None
    claim_support: float | None = None
    cost_usd: float | None = None

    @computed_field
    @property
    def fact_recall(self) -> float | None:
        return metrics.rate(self.fact_hits) if self.fact_hits else None

    @computed_field
    @property
    def passed(self) -> bool | None:
        """None when the mode can't judge the case (unanswerable, retrieval only)."""
        if self.error:
            return False
        if self.answer is None:
            if not self.answerable:
                return None
            return self.retrieved > 0 and self.recall in (None, 1.0)
        if not self.answerable:
            return bool(self.abstained)
        return all(self.fact_hits) and not self.abstained


class EvalReport(BaseModel):
    created_at: datetime
    mode: Mode
    dataset: str
    config: dict[str, Any]
    summary: dict[str, float | None]
    by_category: dict[str, dict[str, float | None]]
    cases: list[CaseResult]
    skipped: list[str] = Field(
        default_factory=list, description="Case ids this mode can't run (follow-ups need the LLM)."
    )

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.created_at:%Y%m%d-%H%M%S}-{self.mode}.json"
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "EvalReport":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def latest_report(directory: Path, mode: Mode) -> Path | None:
    reports = sorted(directory.glob(f"*-{mode}.json")) if directory.is_dir() else []
    return reports[-1] if reports else None


def summarize(results: Sequence[CaseResult]) -> dict[str, float | None]:
    answerable = [r for r in results if r.answerable and not r.error]
    unanswerable = [r for r in results if not r.answerable and not r.error]
    answered = [r for r in results if r.answer is not None]
    latencies = [r.latency_ms for r in results if not r.error]

    summary: dict[str, float | None] = {
        "pass_rate": metrics.rate(r.passed for r in results),
        "retrieval_hit_rate": metrics.rate(
            None if r.recall is None else r.recall > 0 for r in answerable
        ),
        "retrieval_recall": metrics.mean(r.recall for r in answerable),
        "retrieval_mrr": metrics.mean(r.reciprocal_rank for r in answerable),
        "retrieval_precision": metrics.mean(r.precision for r in answerable),
        "mean_chunks": metrics.mean(r.retrieved for r in answerable),
        "false_rejection_rate": metrics.rate(r.retrieved == 0 for r in answerable),
        "offtopic_rejection_rate": metrics.rate(r.retrieved == 0 for r in unanswerable),
        "mean_latency_ms": metrics.mean(latencies),
        "p95_latency_ms": metrics.percentile(latencies, 95),
        "error_rate": metrics.rate(r.error is not None for r in results),
    }
    if answered:
        answered_ok = [r for r in answered if r.answerable]
        summary |= {
            "answer_accuracy": metrics.rate(all(r.fact_hits) and not r.abstained
                                            for r in answered_ok),
            "fact_recall": metrics.mean(r.fact_recall for r in answered_ok),
            "abstention_rate": metrics.rate(r.abstained for r in answered if not r.answerable),
            "false_refusal_rate": metrics.rate(r.abstained for r in answered_ok),
            "approval_rate": metrics.rate(r.approved for r in answered_ok),
            "claim_support": metrics.mean(r.claim_support for r in answered),
            "research_rate": metrics.rate(r.researched for r in answered),
            "mean_revisions": metrics.mean(r.revisions for r in answered),
            "mean_llm_calls": metrics.mean(r.llm_calls for r in answered),
            "mean_cost_usd": metrics.mean(r.cost_usd for r in answered),
            "total_cost_usd": sum(r.cost_usd or 0 for r in answered),
        }
    return {key: summary[key] for key in METRICS if key in summary}


class EvalRunner:
    def __init__(self, service: QueryService) -> None:
        self._service = service

    def run(
        self,
        cases: Sequence[EvalCase],
        mode: Mode,
        dataset: str,
        top_k: int | None = None,
        delay_s: float = 0.0,
        on_result: Callable[[CaseResult], None] | None = None,
    ) -> EvalReport:
        if mode == "full":
            self._service.ensure_ready()
            self._service.ensure_budget()
        top_k = top_k or self._service.settings.top_k
        # Without the LLM a follow-up can't be resolved into a standalone question.
        skipped = [c.id for c in cases if mode == "retrieval" and c.history]
        cases = [c for c in cases if c.id not in skipped]

        results: list[CaseResult] = []
        for i, case in enumerate(cases):
            if delay_s and mode == "full" and i:
                time.sleep(delay_s)  # stay under free-tier rate limits
            result = self._run_case(case, mode, top_k)
            results.append(result)
            if on_result:
                on_result(result)

        by_category: dict[str, list[CaseResult]] = defaultdict(list)
        for result in results:
            by_category[result.category].append(result)

        config = self._service.settings.model_dump(include=CONFIG_KEYS)
        config |= {"top_k": top_k, "primary_model": self._service.settings.resolved_primary_model,
                   "fast_model": self._service.settings.resolved_fast_model}
        return EvalReport(
            created_at=datetime.now(UTC),
            mode=mode,
            dataset=dataset,
            config=config,
            summary=summarize(results),
            by_category={name: summarize(group) for name, group in sorted(by_category.items())},
            cases=results,
            skipped=skipped,
        )

    def _run_case(self, case: EvalCase, mode: Mode, top_k: int) -> CaseResult:
        result = CaseResult(id=case.id, category=case.category, answerable=case.answerable)
        start = time.perf_counter()
        try:
            doc_ids = self._service.resolve_documents(case.documents) or None
            if mode == "retrieval":
                # The same queries the retrieve node would use, with the expected sub-queries
                # standing in for the planner.
                queries = list(dict.fromkeys([case.question, *case.search_queries]))
                chunks = self._service.retriever.retrieve(
                    queries, top_k=top_k, rerank_queries=queries,
                    doc_ids=frozenset(doc_ids) if doc_ids else None,
                )
            else:
                response = self._ask(case, top_k, doc_ids)
                chunks = response.sources
                nodes = [step.node for step in response.trace]
                result.answer = response.answer
                result.standalone_question = response.standalone_question
                result.search_queries = response.search_queries
                result.researched = "research" in nodes
                result.fact_hits = metrics.fact_hits(case.facts, response.answer)
                result.abstained = metrics.is_abstention(response.answer, nodes)
                result.approved = response.approved
                result.revisions = response.revisions
                result.llm_calls = metrics.llm_calls(nodes)
                result.claim_support = response.review.claim_support if response.review else None
                result.cost_usd = response.usage.cost_usd
        except BudgetExceededError:
            raise  # every later case would fail the same way
        except Exception as exc:  # one failing case (e.g. a rate limit) must not end the run
            logger.warning("Eval case %s failed: %s", case.id, exc)
            result.error = f"{type(exc).__name__}: {exc}"
            return result
        finally:
            result.latency_ms = round((time.perf_counter() - start) * 1000, 1)

        result.retrieved = len(chunks)
        result.top_score = chunks[0].score if chunks else None
        result.evidence_ranks = metrics.evidence_ranks(case.evidence, [c.text for c in chunks])
        result.recall = metrics.recall(result.evidence_ranks)
        result.reciprocal_rank = metrics.reciprocal_rank(result.evidence_ranks)
        result.precision = metrics.chunk_precision(case.evidence, [c.text for c in chunks])
        return result

    def _ask(self, case: EvalCase, top_k: int, doc_ids: list[str] | None) -> QueryResponse:
        """Run a case through the pipeline, replaying its history into a throwaway session."""
        # The cache is bypassed: a run must measure the pipeline, not earlier answers.
        request = QueryRequest(question=case.question, top_k=top_k, doc_ids=doc_ids,
                               use_cache=False)
        if not case.history:
            return self._service.run(request)
        conversations = self._service.conversations
        session_id = f"eval-{uuid.uuid4().hex[:12]}"
        try:
            for turn in case.history:
                conversations.append(session_id, turn)
            return self._service.run(request.model_copy(update={"session_id": session_id}))
        finally:
            conversations.delete(session_id)
