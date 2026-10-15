import logging

from query_tool.agents.common import extract_citations
from query_tool.core.state import GraphState
from query_tool.core.tracing import NodeFn, traced
from query_tool.prompts import ENHANCER_PROMPT
from query_tool.services.llm import LLMProvider, invoke_text

logger = logging.getLogger(__name__)

# A polish pass that grows the answer this much is probably adding content.
MAX_GROWTH_RATIO = 1.5


def polish_rejection_reason(draft: str, polished: str) -> str | None:
    if not polished:
        return "empty output"
    if extract_citations(polished) != extract_citations(draft):
        return "citations changed"
    if len(polished) > MAX_GROWTH_RATIO * len(draft):
        return "answer grew too much"
    return None


def make_enhancer(llm: LLMProvider) -> NodeFn:
    @traced("enhancer")
    def enhance(state: GraphState) -> dict:
        draft = state["draft"]
        review = state.get("review")
        approved = bool(review and review.approved)

        warnings: list[str] = []
        if review is None or review.status == "failed":
            warnings.append("The reviewer could not verify this answer; check the citations.")
        elif not approved:
            warnings.append(
                f"Reached the revision limit ({state.get('max_revisions', 0)}) before the "
                "reviewer approved the answer."
            )

        try:
            polished = invoke_text(
                llm.get("enhancer"),
                ENHANCER_PROMPT.format_messages(
                    question=state["standalone_question"], draft=draft
                ),
            )
            rejection = polish_rejection_reason(draft, polished)
        except Exception:
            logger.exception("Enhancer failed; returning the reviewed draft")
            polished, rejection = "", "enhancer error"

        if rejection:
            logger.info("Discarding polished answer: %s", rejection)
        final = draft if rejection else polished

        return {
            "final_answer": final,
            "approved": approved,
            "warnings": warnings,
            "_detail": {"polished": rejection is None, "rejected_because": rejection},
        }

    return enhance
