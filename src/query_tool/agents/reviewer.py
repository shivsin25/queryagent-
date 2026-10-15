import logging

from query_tool.agents.claims import ClaimChecker, claim_issues, claim_support
from query_tool.agents.common import (
    check_citations,
    extract_citations,
    format_context,
    looks_like_refusal,
)
from query_tool.core.schemas import ClaimCheck, RetrievedChunk, ReviewAssessment, ReviewResult
from query_tool.core.state import GraphState
from query_tool.core.tracing import NodeFn, traced
from query_tool.prompts import REVIEWER_PROMPT
from query_tool.services.llm import LLMProvider, invoke_structured

logger = logging.getLogger(__name__)


def make_reviewer(
    llm: LLMProvider, approval_threshold: float, claim_checker: ClaimChecker | None = None
) -> NodeFn:
    def check_claims(draft: str, documents: list[RetrievedChunk]) -> list[ClaimCheck]:
        if claim_checker is None:
            return []
        try:
            return claim_checker.check(draft, documents)
        except Exception:
            logger.exception("Claim check failed; relying on the LLM reviewer alone")
            return []

    @traced("reviewer")
    def review(state: GraphState) -> dict:
        documents = state["documents"]
        draft = state["draft"]
        iteration = state.get("revision_count", 0)
        claims = check_claims(draft, documents)
        citation_issues = check_citations(draft, len(documents)) + claim_issues(claims)
        if not extract_citations(draft) and not looks_like_refusal(draft):
            citation_issues.append(
                "The answer cites no sources. Cite the source of every fact, or say that the "
                "documents don't cover the question."
            )
        checks = {
            "iteration": iteration,
            "citation_issues": citation_issues,
            "claims": claims,
            "claim_support": claim_support(claims),
        }

        messages = REVIEWER_PROMPT.format_messages(
            question=state["standalone_question"], context=format_context(documents), draft=draft
        )
        try:
            assessment = invoke_structured(llm.get("reviewer"), messages, ReviewAssessment)
        except Exception:
            logger.exception("Reviewer failed; the answer will be returned unverified")
            result = ReviewResult(
                grounding_score=0,
                accuracy_score=0,
                completeness_score=0,
                verdict="revise",
                status="failed",
                **checks,
            )
        else:
            approved = (
                assessment.verdict == "approve"
                and min(assessment.grounding_score, assessment.accuracy_score)
                >= approval_threshold
                and not citation_issues
            )
            result = ReviewResult(**assessment.model_dump(), **checks, approved=approved)

        return {
            "review": result,
            "reviews": [result],
            "_detail": {
                "iteration": iteration,
                "status": result.status,
                "grounding": result.grounding_score,
                "accuracy": result.accuracy_score,
                "completeness": result.completeness_score,
                "claim_support": result.claim_support,
                "missing": result.missing_information,
                "approved": result.approved,
                "issues": len(result.issues) + len(citation_issues),
            },
        }

    return review
