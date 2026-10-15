from langchain_core.prompts import ChatPromptTemplate

FEEDBACK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You turn a reviewer's findings into precise revision instructions for a writer.\n"
            "Rules:\n"
            "- Write a short numbered list of concrete edits (remove, correct, add a citation, "
            "add a missing detail from source [n]).\n"
            "- Point to the source number that holds the correct information.\n"
            "- If a claim has no support in any source, tell the writer to remove it.\n"
            "- Do not write the revised answer yourself.",
        ),
        (
            "human",
            "Question: {question}\n\nSources:\n{context}\n\nDraft:\n{draft}\n\n"
            "Reviewer scores: grounding={grounding}, accuracy={accuracy}, "
            "completeness={completeness}\n"
            "Issues:\n{issues}\n\nUnsupported claims:\n{unsupported}",
        ),
    ]
)
