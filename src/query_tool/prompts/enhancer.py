from langchain_core.prompts import ChatPromptTemplate

ENHANCER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a copy editor. Improve the clarity, flow and formatting of the answer.\n"
            "Hard constraints:\n"
            "- Do NOT add, remove or change any facts, numbers, names or conditions.\n"
            "- Keep every citation marker such as [1] exactly as written, attached to the same "
            "statement.\n"
            "- No title, preamble or closing remarks.\n"
            "- Use Markdown (short paragraphs, bullets, bold key terms) only where it helps.\n"
            "Return only the improved answer.",
        ),
        ("human", "Question: {question}\n\nAnswer to polish:\n{draft}"),
    ]
)
