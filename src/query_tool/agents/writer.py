from query_tool.agents.common import extract_citations, format_context
from query_tool.core.exceptions import PipelineError
from query_tool.core.schemas import DraftRecord
from query_tool.core.state import GraphState, retry_context
from query_tool.core.tracing import NodeFn, traced
from query_tool.prompts import RETRY_BLOCK, REVISION_BLOCK, WRITER_PROMPT
from query_tool.services.llm import LLMProvider, invoke_text


def make_writer(llm: LLMProvider) -> NodeFn:
    @traced("writer")
    def write(state: GraphState) -> dict:
        iteration = state.get("revision_count", 0)
        feedback = state.get("feedback")
        revision_block = ""
        if feedback and state.get("draft"):
            revision_block = REVISION_BLOCK.format(draft=state["draft"], feedback=feedback)

        retry = retry_context(state)
        messages = WRITER_PROMPT.format_messages(
            context=format_context(state["documents"]),
            question=state["standalone_question"],
            revision_block=(RETRY_BLOCK.format(retry_context=retry) if retry else "")
            + revision_block,
        )
        draft = invoke_text(llm.get("writer"), messages)
        if not draft:
            raise PipelineError("The writer returned an empty answer.")

        return {
            "draft": draft,
            "drafts": [DraftRecord(iteration=iteration, text=draft)],
            "_detail": {
                "iteration": iteration,
                "characters": len(draft),
                "citations": sorted(extract_citations(draft)),
            },
        }

    return write
