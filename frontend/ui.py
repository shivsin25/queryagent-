"""Shared layout pieces: the design system, sidebar, and rendering of an answer."""

import contextlib
import hmac
import os

import altair as alt
import pandas as pd
import streamlit as st

from api_client import ApiClient, ApiError
from story import flow_html, journey_from_result

ACCENT = "#4338ca"
NODE_LABELS = {
    "preprocess": "Guard & plan",
    "retrieve": "Retrieve",
    "writer": "Write",
    "reviewer": "Review",
    "research": "Search again",
    "feedback": "Revision notes",
    "enhancer": "Polish",
    "no_context": "Nothing relevant",
    "cache": "Answer cache",
}

_CSS = """
<style>
:root {
  --qt-accent: #4338ca; --qt-accent-soft: #eef0ff; --qt-ink: #1d1d1f; --qt-muted: #6b6a66;
  --qt-line: #e4e2dc; --qt-card: #ffffff; --qt-good: #15803d; --qt-good-soft: #ecf8ef;
  --qt-warn: #b45309; --qt-warn-soft: #fff6e6; --qt-bad: #b91c1c; --qt-bad-soft: #fdeeee;
}
.block-container { padding-top: 4.4rem; padding-bottom: 6rem; max-width: 1120px; }
h1, h2, h3 { letter-spacing: -0.015em; }

/* Brand */
.qt-brand { display: flex; align-items: center; gap: .7rem; margin: .2rem 0 1.1rem; }
.qt-logo { width: 2.3rem; height: 2.3rem; border-radius: .7rem; display: grid;
  place-items: center; color: #fff; font-weight: 800; font-size: 1.15rem;
  background: linear-gradient(135deg, #4338ca 0%, #6d5ce8 55%, #0f766e 100%);
  box-shadow: 0 6px 16px -6px rgba(67, 56, 202, .55); }
.qt-brand-name { font-weight: 750; font-size: 1.08rem; color: var(--qt-ink); line-height: 1.1; }
.qt-brand-tag { font-size: .78rem; color: var(--qt-muted); }

/* Hero */
.qt-eyebrow { text-transform: uppercase; letter-spacing: .09em; font-size: .72rem;
  font-weight: 700; color: var(--qt-accent); margin-bottom: .35rem; }
.qt-hero h1 { font-size: 2.15rem; line-height: 1.15; margin: 0 0 .5rem; padding: 0; }
.qt-hero p { color: var(--qt-muted); font-size: 1.03rem; max-width: 46rem; margin: 0; }
.qt-section-label { font-weight: 650; font-size: .95rem; margin: 1.5rem 0 .55rem;
  color: var(--qt-ink); }

/* The six-stage flow */
.qt-flow { display: flex; align-items: stretch; gap: .35rem; margin: .2rem 0 .4rem; }
.qt-stage { flex: 1 1 0; min-width: 0; background: var(--qt-card); border: 1px solid var(--qt-line);
  border-radius: .75rem; padding: .7rem .75rem .75rem; transition: all .25s ease; }
.qt-stage-head { display: flex; align-items: center; gap: .45rem; }
.qt-stage-mark { flex: none; width: 1.45rem; height: 1.45rem; border-radius: 999px;
  display: grid; place-items: center; font-size: .75rem; font-weight: 700;
  background: var(--qt-accent-soft); color: var(--qt-accent); }
.qt-stage-title { font-weight: 650; font-size: .9rem; color: var(--qt-ink); }
.qt-stage-note { font-size: .77rem; color: var(--qt-muted); margin-top: .35rem; line-height: 1.35; }
.qt-loop { margin-left: auto; font-size: .7rem; font-weight: 700; color: var(--qt-accent);
  background: var(--qt-accent-soft); border-radius: 999px; padding: .05rem .4rem; }
.qt-arrow { align-self: center; color: #b5b2aa; font-size: .95rem; }
.qt-done .qt-stage-mark { background: var(--qt-good-soft); color: var(--qt-good); }
.qt-warn { border-color: #f0d8ad; }
.qt-warn .qt-stage-mark { background: var(--qt-warn-soft); color: var(--qt-warn); }
.qt-blocked { border-color: #f1c4c4; background: #fffafa; }
.qt-blocked .qt-stage-mark { background: var(--qt-bad-soft); color: var(--qt-bad); }
.qt-skipped { opacity: .45; }
.qt-pending .qt-stage-mark { background: #f1f0ec; color: #a19e96; }
.qt-pending { opacity: .7; }
.qt-active { border-color: var(--qt-accent); box-shadow: 0 0 0 3px rgba(67, 56, 202, .12);
  animation: qt-pulse 1.4s ease-in-out infinite; }
.qt-active .qt-stage-mark { background: var(--qt-accent); color: #fff; }
.qt-active .qt-stage-note::after { content: "working…"; }
@keyframes qt-pulse { 50% { box-shadow: 0 0 0 6px rgba(67, 56, 202, .06); } }
.qt-compact .qt-stage { padding: .45rem .55rem .5rem; border-radius: .6rem; }
.qt-compact .qt-stage-title { font-size: .8rem; }
.qt-compact .qt-stage-mark { width: 1.2rem; height: 1.2rem; font-size: .66rem; }
.qt-compact .qt-stage-note { font-size: .7rem; margin-top: .2rem; }
.qt-compact .qt-arrow { font-size: .8rem; }
.qt-flow-banner { font-size: .8rem; color: var(--qt-accent); background: var(--qt-accent-soft);
  border-radius: .55rem; padding: .35rem .65rem; margin-top: .35rem; display: inline-block; }
@media (max-width: 900px) {
  .qt-flow { flex-wrap: wrap; } .qt-arrow { display: none; }
  .qt-stage { flex: 1 1 calc(33% - .4rem); }
}

/* Promises */
.qt-promises { display: grid; grid-template-columns: repeat(4, 1fr); gap: .6rem; }
.qt-promise { border-left: 3px solid var(--qt-accent); padding: .1rem 0 .1rem .7rem; }
.qt-promise-title { font-weight: 650; font-size: .88rem; color: var(--qt-ink); }
.qt-promise-text { font-size: .8rem; color: var(--qt-muted); line-height: 1.4; margin-top: .1rem; }
@media (max-width: 900px) { .qt-promises { grid-template-columns: repeat(2, 1fr); } }

/* Example questions */
[class*="st-key-ex-"] button { justify-content: flex-start; text-align: left; min-height: 3.1rem;
  background: var(--qt-card); font-size: .88rem; }
[class*="st-key-ex-"] button:hover { border-color: var(--qt-accent); color: var(--qt-accent); }

/* Answer card */
.qt-answer-head { display: flex; flex-wrap: wrap; align-items: center; gap: .4rem;
  margin-bottom: .35rem; }
.qt-badge { display: inline-flex; align-items: center; gap: .3rem; padding: .16rem .6rem;
  border-radius: 999px; font-size: .78rem; font-weight: 650; border: 1px solid transparent; }
.qt-badge.good { color: var(--qt-good); background: var(--qt-good-soft); border-color: #bfe5c9; }
.qt-badge.warn { color: var(--qt-warn); background: var(--qt-warn-soft); border-color: #f3dcb2; }
.qt-badge.bad { color: var(--qt-bad); background: var(--qt-bad-soft); border-color: #f1c4c4; }
.qt-badge.info { color: var(--qt-accent); background: var(--qt-accent-soft);
  border-color: #d9dcfb; }
.qt-chip { font-size: .76rem; color: var(--qt-muted); background: #f3f2ee; border-radius: 999px;
  padding: .14rem .55rem; }
.qt-journey-label { font-size: .72rem; text-transform: uppercase; letter-spacing: .08em;
  color: var(--qt-muted); font-weight: 650; margin: .6rem 0 .3rem; }
.qt-now { font-size: .82rem; color: var(--qt-muted); margin-top: .25rem; }

/* Sidebar */
.qt-status { display: flex; align-items: center; gap: .5rem; font-size: .85rem; font-weight: 600; }
.qt-dot { width: .55rem; height: .55rem; border-radius: 999px; background: var(--qt-good);
  box-shadow: 0 0 0 3px var(--qt-good-soft); }
.qt-dot.off { background: var(--qt-bad); box-shadow: 0 0 0 3px var(--qt-bad-soft); }
.qt-dot.warn { background: var(--qt-warn); box-shadow: 0 0 0 3px var(--qt-warn-soft); }
.qt-side-meta { font-size: .78rem; color: var(--qt-muted); line-height: 1.55;
  margin: .45rem 0 .2rem; }
.qt-side-meta b { color: var(--qt-ink); font-weight: 600; }
</style>
"""

BRAND_HTML = (
    '<div class="qt-brand"><div class="qt-logo">Q</div><div>'
    '<div class="qt-brand-name">Query Tool</div>'
    '<div class="qt-brand-tag">Answers from your documents</div></div></div>'
)


def money(amount: float, digits: int = 4) -> str:
    """A dollar amount that Streamlit's Markdown won't read as the start of a formula."""
    return f"\\${amount:.{digits}f}"


def escape_dollars(text: str) -> str:
    return text.replace("$", "\\$")


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def require_password() -> None:
    """Ask for QT_UI_PASSWORD before showing the app, if one is set (e.g. on a public host,
    so strangers can't spend the LLM budget). Does nothing otherwise."""
    expected = os.getenv("QT_UI_PASSWORD")
    if not expected or st.session_state.get("_authenticated"):
        return
    html(BRAND_HTML)
    with st.form("login"):
        password = st.text_input("Password", type="password")
        entered = st.form_submit_button("Enter", type="primary")
    if entered:
        if hmac.compare_digest(password.encode(), expected.encode()):
            st.session_state["_authenticated"] = True
            st.rerun()
        st.error("Wrong password.")
    st.stop()


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def hero(eyebrow: str, title: str, subtitle: str) -> None:
    html(f'<div class="qt-hero"><div class="qt-eyebrow">{eyebrow}</div>'
         f"<h1>{title}</h1><p>{subtitle}</p></div>")


@st.cache_resource
def get_client() -> ApiClient:
    return ApiClient()


@st.cache_data(ttl=10, show_spinner=False)
def fetch_documents() -> list[dict]:
    try:
        return get_client().documents()
    except ApiError:
        return []


@st.cache_data(ttl=10, show_spinner=False)
def fetch_health() -> dict | None:
    try:
        return get_client().health()
    except ApiError:
        return None


def new_conversation() -> None:
    if session_id := st.session_state.pop("session_id", None):
        with contextlib.suppress(ApiError):  # a new session id is used either way
            get_client().delete_session(session_id)
    st.session_state["messages"] = []
    st.session_state.pop("pending", None)


def sidebar() -> None:
    with st.sidebar:
        html(BRAND_HTML)
        health = fetch_health()
        if health is None:
            html('<div class="qt-status"><span class="qt-dot off"></span>API offline</div>')
            st.caption(f"Start it with `uv run qt serve` ({get_client().base_url}).")
        else:
            ready = health["llm_configured"]
            html(f'<div class="qt-status"><span class="qt-dot{"" if ready else " warn"}"></span>'
                 f'{"Online and ready" if ready else "Online, no LLM key in .env"}</div>')
            html(
                '<div class="qt-side-meta">'
                f'<b>Writer</b> {health["llm_provider"]} · {health["primary_model"]}<br>'
                f'<b>Reviewer</b> {health["reviewer_model"].replace(":", " · ")}<br>'
                f'<b>Search</b> hybrid · {"reranked" if health["rerank_enabled"] else "fused"}'
                f'<br><b>Knowledge</b> {health["documents"]} documents · '
                f'{health["chunks"]} passages</div>'
            )
            spent, budget = health["spent_today_usd"], health["daily_budget_usd"]
            if budget:
                st.progress(min(spent / budget, 1.0),
                            text=f"Spent today {money(spent)} of {money(budget, 2)}")
            else:
                st.caption(f"Spent today {money(spent)}")

        st.button("New conversation", icon=":material/add_comment:", width="stretch",
                  on_click=new_conversation)
        with st.expander("Answer settings", icon=":material/tune:"):
            st.slider("Passages to read (top-k)", 1, 10, 5, key="top_k")
            st.slider("Revisions allowed", 0, 4, 2, key="max_revisions")


def step_label(node: str, detail: dict) -> str:
    """One line on what a pipeline step just did, shown under the live flow."""
    if node == "cache":
        return "Answered from the cache: asked before, no AI calls needed"
    if node == "preprocess":
        if detail.get("blocked"):
            return "Guard stopped the question before any AI call"
        queries = ", ".join(f"“{q}”" for q in detail.get("search_queries", []))
        if detail.get("history_turns"):
            return (f"Read the follow-up as “{detail.get('standalone_question', '')}”; "
                    f"searching for {queries}")
        return f"Planned the search: {queries}"
    if node == "research":
        found = detail.get("new_chunks", 0)
        return (f"Searched again for what was missing: {found} new passage(s)" if found
                else "Searched again for what was missing: nothing new")
    if node == "retrieve":
        if not detail.get("count"):
            return "Found nothing relevant in the documents"
        return f"Found {detail['count']} relevant passage(s) in {', '.join(detail['sources'])}"
    if node == "writer":
        it = detail.get("iteration", 0)
        return "Wrote a revised draft" if it else "Wrote the first draft, citing every fact"
    if node == "reviewer":
        if detail.get("status") == "failed":
            return "The reviewer was unavailable; the answer will be marked unverified"
        if detail.get("approved"):
            return "The reviewer approved the answer"
        return "The reviewer sent the draft back with fixes"
    if node == "feedback":
        return "Turned the review into revision notes"
    if node == "enhancer":
        return "Polished the wording" if detail.get("polished") else "Kept the reviewed wording"
    return NODE_LABELS.get(node, node)


# ---- answer card ----------------------------------------------------------------


def _answer_kind(result: dict) -> tuple[str, str]:
    nodes = {t["node"]: t.get("detail", {}) for t in result.get("trace", [])}
    if nodes.get("preprocess", {}).get("blocked"):
        return "bad", "Blocked by the guard"
    if "no_context" in nodes:
        return "info", "Not covered by the documents"
    if result["approved"]:
        return "good", "✓ Verified answer"
    return "warn", "⚠ Not fully verified"


def _answer_head(result: dict) -> None:
    kind, label = _answer_kind(result)
    chips = []
    if result.get("attempt", 1) > 1:
        chips.append(f"Attempt {result['attempt']}")
    if result["citations"]:
        chips.append(f"{len(result['citations'])} source(s) cited")
    chips.append(f"{result['latency_ms'] / 1000:.1f}s")
    # &#36; is "$": a literal one could pair with another and become a Markdown formula.
    chips.append("cached · &#36;0" if result.get("cached")
                 else f"&#36;{result.get('usage', {}).get('cost_usd', 0):.4f}")
    html(f'<div class="qt-answer-head"><span class="qt-badge {kind}">{label}</span>'
         + "".join(f'<span class="qt-chip">{c}</span>' for c in chips) + "</div>")


def _retry_report(result: dict) -> None:
    retry = result.get("retry")
    if not retry:
        return
    changes = "\n".join(f"- {c}" for c in retry["changes"])
    if retry["improved"]:
        st.success(f"**Attempt {result['attempt']} improved on the last answer.** "
                   f"What changed:\n{changes}", icon=":material/trending_up:")
    else:
        st.info(f"**Attempt {result['attempt']} found nothing new.** {retry['note']}\n\n"
                f"What it tried:\n{changes}", icon=":material/search_off:")


def _retry_control(result: dict) -> None:
    left = result.get("retries_left", 0)
    request_id = result["request_id"]
    if left <= 0:
        st.caption(":material/block: Re-query limit reached for this question")
        return
    with st.popover(f"Try again · {left} left", icon=":material/refresh:"):
        st.markdown("**Not satisfied?** The next attempt searches wider, reads more passages "
                    "and is told what the earlier answer missed.")
        why = st.text_input("What was missing or wrong? (optional)", key=f"why-{request_id}",
                            placeholder="e.g. it doesn't mention monthly plans")
        if st.button("Search wider and answer again", key=f"retry-{request_id}",
                     type="primary", width="stretch"):
            st.session_state["pending"] = {"question": result["question"],
                                           "retry_of": request_id, "why": why or None}
            st.rerun()


def _feedback_widget(request_id: str) -> None:
    key = f"fb-{request_id}"

    def send() -> None:
        choice = st.session_state.get(key)
        if choice is not None:
            with contextlib.suppress(ApiError):  # a lost rating isn't worth an error message
                get_client().feedback(request_id, "up" if choice == 1 else "down")

    st.feedback("thumbs", key=key, on_change=send)


def render_result(result: dict, latest: bool = False) -> None:
    """An answer, how it was made, and what can be done with it."""
    _answer_head(result)
    _retry_report(result)
    st.markdown(result["answer"])
    for warning in result.get("warnings", []):
        st.warning(warning, icon=":material/warning:")

    html('<div class="qt-journey-label">How this answer was made</div>'
         + flow_html(journey_from_result(result), compact=True))

    feedback_col, retry_col, download_col, _ = st.columns([1.1, 1.6, 1.4, 2.5],
                                                           vertical_alignment="center")
    with feedback_col:
        _feedback_widget(result["request_id"])
    with retry_col:
        if latest:
            _retry_control(result)
    with download_col:
        st.download_button(
            "Download", data=_to_markdown(result), file_name=f"answer-{result['request_id']}.md",
            mime="text/markdown", icon=":material/download:", key=f"dl-{result['request_id']}",
            type="tertiary",
        )
    _details(result)


def _details(result: dict) -> None:
    # A toggle rather than an expander: the tabs hold expanders, which can't be nested.
    if not st.toggle("Show sources, review and trace", key=f"details-{result['request_id']}"):
        return
    sources_tab, review_tab, drafts_tab, trace_tab = st.tabs(
        ["Sources", "Review", "Revision history", "Trace"]
    )
    with sources_tab:
        _sources(result)
    with review_tab:
        _review(result)
    with drafts_tab:
        _drafts(result)
    with trace_tab:
        _trace(result)


def _sources(result: dict) -> None:
    if not result["citations"]:
        st.caption("The answer cites no sources.")
    for c in result["citations"]:
        page = f" · page {c['page']}" if c.get("page") else ""
        with st.expander(f"[{c['index']}] {c['source']}{page}"):
            st.markdown(f"> {c['snippet']}…")
            st.caption(f"Relevance {c['score']:.2f} · passage `{c['chunk_id']}`")
    uncited = len(result["sources"]) - len(result["citations"])
    if uncited > 0:
        st.caption(f"{uncited} more passage(s) were read but not cited.")
    queries = " · ".join(f"`{q}`" for q in result["search_queries"])
    if queries:
        st.caption(f"Searched for: {queries}")
    if result["standalone_question"] != result["question"]:
        st.caption(f"Follow-up understood as: “{result['standalone_question']}”")


def _review(result: dict) -> None:
    review = result.get("review")
    if not review:
        st.caption("No review ran: there was nothing in the documents to check against.")
        return
    if review["status"] == "failed":
        st.caption("The reviewer could not be reached for this answer.")
        return
    metrics = [("Grounding", review["grounding_score"]), ("Accuracy", review["accuracy_score"]),
               ("Completeness", review["completeness_score"])]
    if review.get("claim_support") is not None:
        metrics.append(("Citations check out", review["claim_support"]))
    for col, (label, value) in zip(st.columns(len(metrics)), metrics, strict=True):
        col.metric(label, f"{value:.0%}", border=True)
    _claim_table(review.get("claims", []))
    issues = review["citation_issues"] + review["issues"]
    if issues:
        st.markdown("**Reviewer notes**")
        st.markdown("\n".join(f"- {i}" for i in issues))


def _drafts(result: dict) -> None:
    reviews = {r["iteration"]: r for r in result.get("reviews", [])}
    if not result["drafts"]:
        st.caption("No drafts were written.")
    for draft in result["drafts"]:
        r = reviews.get(draft["iteration"])
        verdict = ""
        if r and r["status"] == "ok":
            verdict = (" · approved" if r["approved"] else " · sent back") + (
                f" (grounding {r['grounding_score']:.0%})"
            )
        with st.expander(f"Draft {draft['iteration'] + 1}{verdict}"):
            st.markdown(draft["text"])
            if r and (r["issues"] or r["unsupported_claims"]):
                st.markdown("**Reviewer findings**")
                st.markdown("\n".join(f"- {i}" for i in r["issues"] + r["unsupported_claims"]))


def _trace(result: dict) -> None:
    if result["trace"]:
        st.altair_chart(_timing_chart(result["trace"]), width="stretch")
    calls = result.get("usage", {}).get("calls", [])
    if calls:
        st.dataframe(
            pd.DataFrame([{"step": c["node"], "model": c["model"], "tokens in": c["input_tokens"],
                           "tokens out": c["output_tokens"], "cost": c["cost_usd"]}
                          for c in calls]),
            hide_index=True, width="stretch",
            column_config={"cost": st.column_config.NumberColumn(format="$%.5f")},
        )
    st.caption(f"Request `{result['request_id']}`")


def _timing_chart(trace: list[dict]) -> alt.Chart:
    df = pd.DataFrame(
        {
            "step": [f"{i + 1}. {NODE_LABELS.get(t['node'], t['node'])}"
                     for i, t in enumerate(trace)],
            "seconds": [t["duration_ms"] / 1000 for t in trace],
        }
    )
    return (
        alt.Chart(df, title="Time per step")
        .mark_bar(color=ACCENT, cornerRadiusEnd=4, size=14)
        .encode(
            y=alt.Y("step:N", sort=None, title=None),
            x=alt.X("seconds:Q", title="seconds", axis=alt.Axis(grid=True, gridOpacity=0.4)),
            tooltip=[alt.Tooltip("step:N"), alt.Tooltip("seconds:Q", format=".2f")],
        )
        .properties(height=max(120, 28 * len(df)))
        .configure_view(strokeWidth=0)
    )


def _claim_table(claims: list[dict]) -> None:
    """Each sentence of the reviewed draft against the sources it cites."""
    if not claims:
        return
    verdicts = {True: "✓ supported", False: "✗ wrong source", None: "– not cited"}
    st.markdown("**Citation check**: every sentence scored against the passage it cites")
    st.dataframe(
        pd.DataFrame(
            [{"check": verdicts[c["supported"]], "sentence": c["text"],
              "cites": ", ".join(map(str, c["citations"])), "score": c["support"]}
             for c in claims]
        ),
        hide_index=True,
        width="stretch",
        column_config={"score": st.column_config.NumberColumn(format="%.1f")},
    )


def _to_markdown(result: dict) -> str:
    lines = [f"# {result['question']}", "", result["answer"], "", "## Sources", ""]
    for c in result["citations"]:
        page = f", page {c['page']}" if c.get("page") else ""
        lines.append(f"[{c['index']}] {c['source']}{page}")
    return "\n".join(lines) + "\n"
