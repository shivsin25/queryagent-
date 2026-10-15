import uuid

import streamlit as st

from api_client import ApiError
from story import flow_html, journey, promises_html
from ui import (
    escape_dollars,
    fetch_documents,
    fetch_health,
    get_client,
    hero,
    html,
    render_result,
    step_label,
)

EXAMPLES = [
    "What is the P1 response time for Enterprise customers?",
    "Can I get a refund on an annual plan after 6 weeks?",
    "What is the liability cap in the data processing agreement?",
    "Is Nimbus HIPAA compliant, and which data regions are available?",
]

messages: list[dict] = st.session_state.setdefault("messages", [])
session_id: str = st.session_state.setdefault("session_id", uuid.uuid4().hex[:12])


def landing() -> None:
    """Who we are, how an answer is made, and why it can be trusted."""
    hero(
        "Grounded question answering",
        "Answers you can trust, from your own documents",
        "Ask anything about the files in the knowledge base. A team of AI agents finds the "
        "relevant passages, drafts an answer that cites each one, and has it fact-checked "
        "before you see it. If the documents don't say, it tells you so.",
    )
    html('<div class="qt-section-label">How every answer is made</div>' + flow_html())
    html('<div class="qt-section-label">What you can count on</div>' + promises_html())
    html('<div class="qt-section-label">Try a question</div>')


def scope_picker() -> list[str]:
    """Which documents to answer from. "doc_scope" outlives the widget, which Streamlit
    forgets whenever another page is shown; the Documents page sets it via "Ask about"."""
    documents = {d["doc_id"]: d["source"] for d in fetch_documents()}
    if "_scope" not in st.session_state:
        st.session_state["_scope"] = [i for i in st.session_state.get("doc_scope", [])
                                      if i in documents]
    scope: list[str] = st.multiselect(
        "Answer from", options=list(documents), format_func=lambda i: documents.get(i, i),
        key="_scope", placeholder="All documents in the knowledge base",
    )
    st.session_state["doc_scope"] = scope
    return scope


def ask(question: str, scope: list[str], retry_of: str | None = None,
        why: str | None = None) -> None:
    """Show the question, then the agent at work, then its answer."""
    user_text = question if not retry_of else (
        f"↻ **Try again:** {question}" + (f"  \n*What was missing:* {why}" if why else "")
    )
    messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        flow_area, now_area, draft_area = st.empty(), st.empty(), st.empty()
        steps: list[tuple[str, dict]] = []
        flow_area.markdown(flow_html(journey(steps, finished=False), compact=True),
                           unsafe_allow_html=True)
        now_area.markdown('<div class="qt-now">Screening the question…</div>',
                          unsafe_allow_html=True)
        result, error, draft, new_draft = None, None, "", False
        try:
            for event in get_client().query_stream(
                question, st.session_state.get("top_k", 5),
                st.session_state.get("max_revisions", 2), session_id, scope,
                retry_of=retry_of, retry_feedback=why,
            ):
                if event["event"] == "step":
                    detail = event["data"].get("detail", {})
                    steps.append((event["node"], detail))
                    flow_area.markdown(flow_html(journey(steps, finished=False), compact=True),
                                       unsafe_allow_html=True)
                    now_area.markdown(f'<div class="qt-now">{step_label(event["node"], detail)}'
                                      "</div>", unsafe_allow_html=True)
                    # The writer's next text is a new draft, not a continuation.
                    new_draft = new_draft or event["node"] == "writer"
                elif event["event"] == "token":
                    draft = event["data"]["text"] if new_draft else draft + event["data"]["text"]
                    new_draft = False
                    draft_area.markdown(f"{draft} ▌")
                elif event["event"] == "result":
                    result = event["data"]
                elif event["event"] == "error":
                    error = event["data"]["message"]
        except ApiError as exc:
            error = str(exc)
        flow_area.empty()
        now_area.empty()
        draft_area.empty()

        if result:
            render_result(result, latest=True)
            messages.append({"role": "assistant", "result": result})
        else:
            show_error(error or "No answer was returned.")
            messages.append({"role": "assistant", "error": error or "No answer was returned."})
    fetch_health.clear()


def show_error(message: str) -> None:
    message = escape_dollars(message)  # budget messages hold amounts
    if "Re-query limit" in message:
        st.warning(message, icon=":material/block:")
    elif "budget" in message:
        st.warning(message, icon=":material/savings:")
    else:
        st.error(message, icon=":material/error:")


# ---- page ---------------------------------------------------------------------------

pending: dict | None = st.session_state.pop("pending", None)  # set by "Try again"
question = st.chat_input("Ask a question about your documents…")
if question:
    pending = {"question": question}

if not messages and not pending:
    landing()
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if cols[i % 2].button(example, key=f"ex-{i}", width="stretch",
                              icon=":material/arrow_outward:"):
            pending = {"question": example}

if messages or pending:
    html('<div class="qt-eyebrow">Ask your documents</div>')
scope = scope_picker()

last_answer = max((i for i, m in enumerate(messages) if "result" in m), default=None)
for i, message in enumerate(messages):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        elif "result" in message:
            # "Try again" belongs to the latest answer only, unless a new one is coming.
            render_result(message["result"], latest=i == last_answer and not pending)
        else:
            show_error(message.get("error") or message.get("content", ""))

if pending:
    ask(pending["question"], scope, pending.get("retry_of"), pending.get("why"))
