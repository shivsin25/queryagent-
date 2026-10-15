from pathlib import PurePath

import pandas as pd
import streamlit as st

from api_client import ApiError
from ui import fetch_documents, fetch_health, get_client, hero, html

INGESTION_STEPS = [
    ("Read", "Text from PDF, Word, Markdown or plain text. Scanned PDFs are flagged."),
    ("Repair", "PDF headers, footers and page numbers removed; split sentences rejoined."),
    ("Structure", "Headings and clause numbers become sections; tables kept."),
    ("Chunk", "Small passages for precise search, each tied to its full section."),
    ("Index", "Embedded for meaning, indexed for keywords. Ready to answer."),
]

hero(
    "Knowledge base",
    "What the agent knows",
    "Every answer comes from these files, and only these. Upload a document and it can be "
    "asked about within seconds, on its own or alongside the rest.",
)
client = get_client()

try:
    documents = client.documents()
except ApiError as exc:
    st.error(str(exc), icon=":material/cloud_off:")
    st.stop()

kinds = pd.Series([PurePath(d["source"]).suffix.lstrip(".").upper() or "TXT" for d in documents])
c1, c2, c3 = st.columns(3)
c1.metric("Documents", len(documents), border=True)
c2.metric("Searchable passages", sum(d["chunks"] for d in documents), border=True)
c3.metric("Formats", " · ".join(sorted(kinds.unique())) if documents else "—", border=True)

upload_col, how_col = st.columns([1.15, 1], gap="large")
with upload_col, st.form("upload", clear_on_submit=True, border=True):
    st.markdown("**Add documents**")
    files = st.file_uploader(
        "PDF, DOCX, Markdown or text", type=["pdf", "docx", "txt", "md", "markdown"],
        accept_multiple_files=True,
    )
    submitted = st.form_submit_button("Add to the knowledge base", icon=":material/upload:",
                                      type="primary", width="stretch")
with how_col:
    html('<div class="qt-section-label" style="margin-top:0">What happens to a file</div>'
         + "".join(f'<div class="qt-promise" style="margin-bottom:.55rem">'
                   f'<div class="qt-promise-title">{i}. {title}</div>'
                   f'<div class="qt-promise-text">{text}</div></div>'
                   for i, (title, text) in enumerate(INGESTION_STEPS, 1)))

if submitted and files:
    with st.spinner(f"Reading and indexing {len(files)} file(s)…"):
        try:
            result = client.upload([(f.name, f.getvalue()) for f in files])
        except ApiError as exc:
            st.error(str(exc))
        else:
            for doc in result["indexed"]:
                st.success(f"**{doc['source']}** is ready: {doc['chunks']} passages. Select it "
                           "below and press **Ask about selected** to question just this file.",
                           icon=":material/task_alt:")
                for warning in doc.get("warnings", []):
                    st.warning(f"**{doc['source']}**: {warning}", icon=":material/shield:")
            for reason in result["skipped"]:
                st.warning(f"Skipped {reason}")
            fetch_health.clear()
            fetch_documents.clear()
            documents = client.documents()

html('<div class="qt-section-label">Documents</div>')
if not documents:
    st.info("No documents yet. Add some above.", icon=":material/upload_file:")
    st.stop()

table = pd.DataFrame(documents)
table["type"] = [PurePath(s).suffix.lstrip(".").upper() or "TXT" for s in table["source"]]
table["ingested_at"] = pd.to_datetime(table["ingested_at"], errors="coerce")
table = table[["source", "type", "chunks", "ingested_at", "doc_id"]]
selection = st.dataframe(
    table,
    hide_index=True,
    width="stretch",
    on_select="rerun",
    selection_mode="multi-row",
    column_config={
        "source": st.column_config.TextColumn("File", width="large"),
        "type": st.column_config.TextColumn("Type", width="small"),
        "chunks": st.column_config.NumberColumn("Passages", width="small"),
        "ingested_at": st.column_config.DatetimeColumn("Added (UTC)", format="D MMM YYYY, HH:mm"),
        "doc_id": "Document ID",
    },
)

selected = [table.iloc[i] for i in selection.selection.rows]
ask_col, delete_col, _ = st.columns([1.3, 1, 2])
if ask_col.button(
    f"Ask about {len(selected)} selected" if selected else "Select files to ask about",
    disabled=not selected,
    icon=":material/forum:",
    type="primary",
    width="stretch",
):
    st.session_state["doc_scope"] = [row["doc_id"] for row in selected]
    st.session_state.pop("_scope", None)  # the Ask page rebuilds its selector from doc_scope
    st.switch_page("views/chat.py")
if delete_col.button(
    f"Delete {len(selected)} selected" if selected else "Select files to delete",
    disabled=not selected,
    icon=":material/delete:",
    width="stretch",
):
    for row in selected:
        try:
            client.delete(row["doc_id"])
        except ApiError as exc:
            st.error(f"{row['source']}: {exc}")
    fetch_health.clear()
    fetch_documents.clear()
    st.rerun()
