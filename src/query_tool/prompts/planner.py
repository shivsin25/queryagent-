from langchain_core.prompts import ChatPromptTemplate

PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You prepare the latest user question for a document search system.\n\n"
            "standalone_question: rewrite the latest question so it can be understood without "
            "the conversation. A question that starts with 'and', 'what about' or 'how about', "
            "or that uses words like 'it', 'that', 'they' or 'that plan', depends on the "
            "conversation and MUST be rewritten. Example: after 'What is the P1 response time "
            "for Enterprise customers?', the question 'And for Business customers?' becomes "
            "'What is the P1 response time for Business customers?'. Copy the question "
            "unchanged only if it already makes complete sense on its own.\n\n"
            "search_queries: keyword search queries for the standalone question.\n"
            "- Use ONE query unless the question asks about several separate things; then "
            "write one query per thing, at most 3.\n"
            "- Keep every named entity, number and technical term.\n"
            "- Remove greetings, filler and politeness.\n\n"
            "Never answer the question and never add facts.",
        ),
        ("human", "Conversation so far:\n{history}\n\nLatest question: {question}{retry_block}"),
    ]
)

# Appended to the question when the user is asking again because earlier answers fell short.
PLANNER_RETRY_BLOCK = (
    "\n\n{retry_context}\n"
    "Search differently this time: write 2 or 3 search queries that use other phrasings, "
    "synonyms and related terms, aimed at what those answers were missing."
)
