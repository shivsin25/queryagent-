from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, computed_field, field_validator

SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
DOC_ID_PATTERN = r"^[0-9a-f]{16}$"
REQUEST_ID_PATTERN = r"^[0-9a-f]{12}$"


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    max_revisions: int | None = Field(default=None, ge=0, le=5)
    session_id: str | None = Field(
        default=None,
        pattern=SESSION_ID_PATTERN,
        description="Conversation id. Follow-up questions are read in the context of the "
        "earlier turns of the same session.",
    )
    use_cache: bool = Field(
        default=True, description="Reuse a cached answer to the same question if there is one."
    )
    doc_ids: list[Annotated[str, Field(pattern=DOC_ID_PATTERN)]] | None = Field(
        default=None,
        max_length=50,
        description="Answer only from these documents. Omit to search all of them.",
    )
    retry_of: str | None = Field(
        default=None,
        pattern=REQUEST_ID_PATTERN,
        description="request_id of an answer the user was not satisfied with. The question is "
        "answered again with a wider search, and the earlier answers as guidance. Asking the "
        "same question again in the same session counts as a retry too.",
    )
    retry_feedback: str | None = Field(
        default=None, max_length=500, description="What was wrong with the earlier answer."
    )


class CallUsage(BaseModel):
    """One LLM call."""

    node: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    estimated: bool = Field(default=False, description="Provider reported no token counts.")


class UsageSummary(BaseModel):
    calls: list[CallUsage] = Field(default_factory=list)

    @computed_field
    @property
    def cost_usd(self) -> float:
        return round(sum(c.cost_usd for c in self.calls), 6)

    @computed_field
    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @computed_field
    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)


class Turn(BaseModel):
    """One question and answer in a conversation."""

    question: str
    answer: str
    standalone_question: str = ""
    approved: bool = False
    created_at: datetime | None = None


class QueryPlan(BaseModel):
    """What the planner LLM returns."""

    standalone_question: str = Field(
        description="The latest question rewritten so it can be understood without the "
        "conversation. Copy it unchanged if it already can."
    )
    search_queries: list[str] = Field(
        description="One keyword search query, or one per part (at most 3) if the question "
        "asks about several separate things."
    )

    @field_validator("search_queries")
    @classmethod
    def _clean_queries(cls, queries: list[str]) -> list[str]:
        return list(dict.fromkeys(q.strip() for q in queries if q.strip()))


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    source: str
    text: str = Field(description="What the writer reads: the matched passage's section.")
    page: int | None = None
    chunk_index: int = 0
    score: float = 0.0
    passage: str = Field(default="", description="The passage that matched the question.")
    parent_id: str | None = None


class ReviewAssessment(BaseModel):
    """What the reviewer LLM returns."""

    grounding_score: float = Field(
        ge=0, le=1, description="Share of factual claims supported by the source they cite (0-1)."
    )
    accuracy_score: float = Field(
        ge=0, le=1, description="How faithfully claims reflect the sources, 0-1."
    )
    completeness_score: float = Field(
        ge=0, le=1, description="How fully the question is answered given the sources, 0-1."
    )
    verdict: Literal["approve", "revise"]
    issues: list[str] = Field(default_factory=list, description="Specific, fixable problems.")
    unsupported_claims: list[str] = Field(
        default_factory=list, description="Claims not supported by the sources, quoted verbatim."
    )
    missing_information: list[str] = Field(
        default_factory=list,
        description="Facts the question asks for that none of the sources contain, each "
        "phrased as a short search query.",
    )

    @field_validator("grounding_score", "accuracy_score", "completeness_score", mode="before")
    @classmethod
    def _normalise_scale(cls, value: Any) -> Any:
        # Models sometimes answer on a 0-10 or 0-100 scale.
        try:
            number = float(value)
        except (TypeError, ValueError):
            return value
        if 1 < number <= 10:
            return number / 10
        if 10 < number <= 100:
            return number / 100
        return number

    @field_validator("verdict", mode="before")
    @classmethod
    def _normalise_verdict(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value


class ClaimCheck(BaseModel):
    """One sentence of a draft, scored against the sources it cites."""

    text: str
    citations: list[int] = Field(default_factory=list)
    support: float | None = Field(
        default=None, description="Best cross-encoder score against the cited sources."
    )
    supported: bool | None = Field(default=None, description="None when the sentence is uncited.")


class ReviewResult(ReviewAssessment):
    """Reviewer output plus the deterministic checks and approval gate."""

    iteration: int = 0
    status: Literal["ok", "failed"] = "ok"
    citation_issues: list[str] = Field(default_factory=list)
    claims: list[ClaimCheck] = Field(default_factory=list)
    claim_support: float | None = Field(
        default=None, description="Share of cited sentences their sources support."
    )
    approved: bool = False


class DraftRecord(BaseModel):
    iteration: int
    text: str


class TraceEvent(BaseModel):
    node: str
    started_at: datetime
    duration_ms: float
    detail: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    index: int
    source: str
    page: int | None = None
    chunk_id: str
    snippet: str
    score: float


class RetryReport(BaseModel):
    """How a retry differed from the attempt before it."""

    previous_request_id: str
    changes: list[str] = Field(description="What was done differently this time.")
    improved: bool = Field(
        description="The answer changed in substance: it is clearly different, and cites new "
        "sources or covers the question at least as completely."
    )
    note: str | None = Field(default=None, description="Shown when it did not improve.")


class QueryResponse(BaseModel):
    request_id: str
    session_id: str | None = None
    root_request_id: str | None = Field(
        default=None, description="request_id of the first attempt at this question."
    )
    attempt: int = 1
    retries_left: int = 0
    retry: RetryReport | None = None
    question: str
    standalone_question: str
    search_queries: list[str]
    answer: str
    approved: bool
    revisions: int
    review: ReviewResult | None = None
    reviews: list[ReviewResult] = Field(default_factory=list)
    drafts: list[DraftRecord] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    sources: list[RetrievedChunk] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    usage: UsageSummary = Field(default_factory=UsageSummary)
    cached: bool = Field(default=False, description="Served from the answer cache, at no cost.")
    latency_ms: float


class StreamEvent(BaseModel):
    """`token` events carry the writer's text as it is generated (`data.text`); a revision
    starts a new draft, and the `result` event holds the final answer."""

    event: Literal["start", "step", "token", "result", "error"]
    node: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class DocumentInfo(BaseModel):
    doc_id: str
    source: str
    chunks: int
    ingested_at: str
    warnings: list[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    indexed: list[DocumentInfo] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    llm_provider: str
    llm_configured: bool
    primary_model: str
    fast_model: str
    reviewer_model: str
    embedding_model: str
    rerank_enabled: bool
    documents: int
    chunks: int
    spent_today_usd: float
    daily_budget_usd: float | None


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=64)
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=1000)


class FeedbackItem(BaseModel):
    request_id: str
    rating: Literal["up", "down"]
    comment: str | None
    question: str
    answer: str
    created_at: datetime
