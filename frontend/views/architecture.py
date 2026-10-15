import streamlit as st

from api_client import ApiError
from story import flow_html
from ui import get_client, hero, html

hero(
    "How it works",
    "A team of agents, each with one job",
    "One model writing an answer on its own will sometimes guess. Here, separate agents plan, "
    "search, write and check, and deterministic safeguards sit between them, so an answer "
    "only reaches you once it is grounded in your documents.",
)
html(flow_html())


def chapter(number: int, title: str, lead: str) -> None:
    html(f'<div class="qt-eyebrow" style="margin-top:2.2rem">Chapter {number}</div>')
    st.subheader(title, anchor=False)
    st.caption(lead)


chapter(1, "The agent graph",
        "A LangGraph state machine. Arrows that loop back are the self-correction.")
graph_col, legend_col = st.columns([1.5, 1], gap="large")
with legend_col:
    st.markdown(
        "**Reading the graph**\n\n"
        "- Blue boxes are the AI agents that write and judge.\n"
        "- White boxes are deterministic steps; *no AI call* means they cost nothing.\n"
        "- From **Review** an answer can go back to **Write** twice over: with revision "
        "notes, or after a second search finds what was missing.\n"
        "- Two exits end early and honestly: **Refused** and **Not in the documents**."
    )
graph_col.graphviz_chart(
    """
    digraph {
      rankdir=TB; bgcolor="transparent"; nodesep=0.5; ranksep=0.42;
      node [shape=box, style="rounded,filled", fontname="Segoe UI", fontsize=12,
            color="#e4e2dc", fillcolor="#ffffff", fontcolor="#1d1d1f", margin="0.22,0.1"];
      edge [color="#a19e96", fontname="Segoe UI", fontsize=10.5, fontcolor="#6b6a66"];

      q [label="Question", shape=oval, fillcolor="#eef0ff", color="#c9cdfa"];
      guard [label="Guard\\nno AI call"];
      plan [label="Plan\\nfollow-ups · sub-questions"];
      ret [label="Retrieve\\nhybrid · rerank · parents"];
      none [label="Not in the documents", fillcolor="#f6f5f1"];
      write [label="Write\\ncited draft", fillcolor="#eef0ff", color="#c9cdfa"];
      review [label="Review\\nscores + citation check", fillcolor="#eef0ff", color="#c9cdfa"];
      research [label="Search again\\nno AI call"];
      notes [label="Revision notes"];
      polish [label="Polish\\nfacts locked"];
      out [label="Answer", shape=oval, fillcolor="#ecf8ef", color="#bfe5c9"];
      blocked [label="Refused", fillcolor="#fdeeee", color="#f1c4c4"];

      q -> guard -> plan -> ret;
      guard -> blocked [label=" injection"];
      ret -> write [label=" relevant"];
      ret -> none [label=" nothing relevant"];
      write -> review;
      review -> research [label=" facts missing"];
      research -> write [label=" found more"];
      review -> notes [label=" needs fixes"];
      notes -> write;
      review -> polish [label=" approved"];
      polish -> out;
    }
    """,
    width="stretch",
)

chapter(2, "Finding the right passages",
        "Fast methods narrow the field; the accurate, slower one only ranks the shortlist.")
left, right = st.columns([1.2, 1], gap="large")
with left:
    st.markdown(
        "1. **Two searches at once.** Meaning-based search (bi-encoder embeddings in an HNSW "
        "index) and keyword search (BM25 with stemming, so *refund* matches *refunded*).\n"
        "2. **Fusion.** Reciprocal Rank Fusion merges both rankings.\n"
        "3. **Reranking, only where it pays.** A cross-encoder reads the top 12 candidates "
        "against the question, the most accurate and most expensive step.\n"
        "4. **Only what's relevant.** A passage is kept only if it scores close to the best "
        "one for its question, or for one part of a multi-part question.\n"
        "5. **Context.** Small passages are matched; the writer reads the whole section each "
        "one belongs to."
    )
with right:
    a, b = st.columns(2)
    a.metric("Precision", "89%", "+67 pts", border=True,
             help="Share of passages sent to the writer that hold the answer (was 22%).")
    b.metric("Recall", "100%", "+2 pts", border=True,
             help="Passages that hold the answer and were found, on the evaluation set.")
    a.metric("Passages per answer", "1.3", "-3.7", delta_color="inverse", border=True)
    b.metric("Search time", "471 ms", "-260 ms", delta_color="inverse", border=True)
    st.caption("Measured on 50 labelled questions with `qt eval --retrieval-only`.")

chapter(3, "Checking before you see it",
        "Approval is never left to a single model's opinion.")
st.markdown(
    "- **Reviewer agent** scores grounding, accuracy and completeness, and lists what is "
    "missing.\n"
    "- **Citation check**: every sentence is scored against the passage it cites by a local "
    "model, so a fact pinned to the wrong source blocks approval.\n"
    "- **Self-correction**: the draft goes back with precise fixes, or the agent searches "
    "again for what was missing (without spending an AI call on the search).\n"
    "- **Hard rules**: an answer that cites nothing, or cites a source that doesn't exist, "
    "is never approved. The final polish can't change a fact or a citation."
)

chapter(4, "Staying safe", "Guardrails that don't rely on an AI behaving well.")
st.markdown(
    "- Questions that try to override the assistant (*ignore your instructions*, *reveal your "
    "prompt*) are refused before any AI sees them.\n"
    "- Text inside documents is treated as data, never as instructions; suspicious uploads "
    "are flagged.\n"
    "- Off-topic questions stop at retrieval with \"not in the documents\" instead of a guess."
)

chapter(5, "When you're not satisfied",
        "Asking again should get a better answer, not the same one.")
st.markdown(
    "Press **Try again** (or ask the same question again) and the next attempt skips the "
    "cache, searches wider, reads more passages, gets one more revision, and is shown the "
    "earlier answers and what you said was wrong. It then reports exactly what changed, or "
    "says honestly that the documents hold nothing more. Each question allows three retries; "
    "an improved answer replaces the cached one for everyone after you."
)

chapter(6, "Keeping costs low", "Every AI call is measured, and most questions need few.")
try:
    usage = get_client().usage()
except ApiError:
    usage = None
if usage:
    budget = usage["daily_budget_usd"]
    spent = usage["spent_today_usd"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Spent today", f"${spent:.4f}", border=True)
    c2.metric("Daily budget", f"${budget:.2f}" if budget else "None", border=True)
    c3.metric("AI calls today", sum(m["calls"] for m in usage["by_model"]), border=True)
st.markdown(
    "Repeated questions come from a cache at no cost, off-topic questions never reach an AI, "
    "replies are length-capped, and new questions stop once the daily budget is spent."
)

chapter(7, "Built with", "Free and local where possible.")
st.markdown(
    "| Part | Technology |\n|---|---|\n"
    "| Orchestration | LangGraph state graph with conditional routing |\n"
    "| Language models | Groq, Gemini, OpenAI or Ollama; the reviewer can use a different "
    "provider from the writer |\n"
    "| Search | fastembed `bge-small-en-v1.5` + Chroma HNSW, inverted-index BM25, "
    "`ms-marco-MiniLM-L-6-v2` cross-encoder, all on CPU |\n"
    "| Documents | PDF repair, Word headings and tables, parent-child chunks |\n"
    "| Service | FastAPI with streamed steps and tokens; SQLite for conversations, usage, "
    "cache and feedback |\n"
    "| Quality | 55-question evaluation set, deterministic metrics, offline test suite |"
)
