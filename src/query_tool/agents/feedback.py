import logging

from query_tool.agents.common import format_context
from query_tool.core.schemas import ReviewResult
from query_tool.core.state import GraphState
from query_tool.core.tracing import NodeFn, traced
from query_tool.prompts import FEEDBACK_PROMPT
from query_tool.services.llm import LLMProvider, invoke_text

logger = logging.getLogger(__name__)


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) or "- (none)"


def numbered(steps: list[str]) -> str:
    return "\n".join(f"{i}. {step}" for i, step in enumerate(steps, start=1))


def review_findings(review: ReviewResult) -> list[str]:
    steps = [*review.citation_issues, *review.issues]
    steps += [f"Remove or source this unsupported claim: {c}" for c in review.unsupported_claims]
    return steps


def fallback_instructions(review: ReviewResult) -> str:
    """Used when the feedback LLM call fails: pass the findings through verbatim."""
    return numbered(
        review_findings(review)
        or ["Make sure every factual sentence cites the source that supports it."]
    )


def make_feedback(llm: LLMProvider) -> NodeFn:
    @traced("feedback")
    def feedback(state: GraphState) -> dict:
        review = state["review"]
        revision = state.get("revision_count", 0) + 1
        messages = FEEDBACK_PROMPT.format_messages(
            question=state["standalone_question"],
            context=format_context(state["documents"]),
            draft=state["draft"],
            grounding=f"{review.grounding_score:.2f}",
            accuracy=f"{review.accuracy_score:.2f}",
            completeness=f"{review.completeness_score:.2f}",
            issues=_bullets([*review.citation_issues, *review.issues]),
            unsupported=_bullets(review.unsupported_claims),
        )
        try:
            instructions = invoke_text(llm.get("feedback"), messages)
            instructions = instructions or fallback_instructions(review)
        except Exception:
            logger.exception("Feedback agent failed; forwarding reviewer findings directly")
            instructions = fallback_instructions(review)

        return {
            "feedback": instructions,
            "revision_count": revision,
            "_detail": {"revision": revision, "instructions": instructions},
        }

    return feedback
