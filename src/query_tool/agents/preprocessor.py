import logging
import re
import unicodedata

from query_tool.agents.common import strip_citations
from query_tool.core.guardrails import BLOCKED_ANSWER, injection_signals
from query_tool.core.schemas import QueryPlan, Turn
from query_tool.core.state import GraphState, retry_context
from query_tool.core.tracing import NodeFn, traced
from query_tool.prompts import PLANNER_PROMPT, PLANNER_RETRY_BLOCK
from query_tool.services.llm import LLMProvider, invoke_structured

logger = logging.getLogger(__name__)

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_LABEL_RE = re.compile(r"^(rewritten\s+)?(search\s+)?query\s*:\s*", re.IGNORECASE)

MAX_SEARCH_QUERIES = 3
# Earlier answers are context for resolving references, not sources, so keep them short.
HISTORY_ANSWER_CHARS = 600


def clean_query(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def sanitize_rewrite(raw: str, fallback: str) -> str:
    """Keep the first non-empty line, strip quotes/labels, reject runaway output."""
    line = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
    line = _LABEL_RE.sub("", line).strip().strip("\"'`").strip()
    if not line or len(line) > 3 * max(len(fallback), 60):
        return fallback
    return line


def format_history(history: list[Turn]) -> str:
    if not history:
        return "(none)"
    lines: list[str] = []
    for turn in history:
        answer = " ".join(strip_citations(turn.answer).split())
        if len(answer) > HISTORY_ANSWER_CHARS:
            answer = answer[:HISTORY_ANSWER_CHARS].rsplit(" ", 1)[0] + " ..."
        lines += [f"User: {turn.standalone_question or turn.question}", f"Assistant: {answer}"]
    return "\n".join(lines)


def make_preprocessor(llm: LLMProvider) -> NodeFn:
    @traced("preprocess")
    def preprocess(state: GraphState) -> dict:
        cleaned = clean_query(state["question"])
        # Guardrail first: an attempt to override the instructions never reaches an LLM.
        if signals := injection_signals(cleaned):
            logger.warning("Blocked a question that tries to %s", ", ".join(signals))
            return {
                "question": cleaned,
                "standalone_question": cleaned,
                "search_queries": [],
                "blocked": True,
                "final_answer": BLOCKED_ANSWER,
                "approved": False,
                "_detail": {"blocked": True, "reasons": signals},
            }

        history = state.get("history") or []
        warnings: list[str] = []
        try:
            context = retry_context(state)
            plan = invoke_structured(
                llm.get("planner"),
                PLANNER_PROMPT.format_messages(
                    history=format_history(history),
                    question=cleaned,
                    retry_block=PLANNER_RETRY_BLOCK.format(retry_context=context)
                    if context else "",
                ),
                QueryPlan,
            )
        except Exception:
            logger.exception("Query planning failed; using the cleaned question")
            plan = None
            warnings.append(
                "Query planning failed, so retrieval used the question as typed"
                + (" and earlier turns of the conversation were ignored." if history else ".")
            )

        # Without history the question already stands alone; don't let the LLM reword it.
        standalone = cleaned
        if plan and history:
            standalone = sanitize_rewrite(plan.standalone_question, cleaned)
        queries = [sanitize_rewrite(q, standalone) for q in plan.search_queries] if plan else []
        queries = list(dict.fromkeys(queries))[:MAX_SEARCH_QUERIES] or [standalone]

        return {
            "question": cleaned,
            "standalone_question": standalone,
            "search_queries": queries,
            "warnings": warnings,
            "_detail": {
                "standalone_question": standalone,
                "search_queries": queries,
                "history_turns": len(history),
            },
        }

    return preprocess
