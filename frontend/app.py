import streamlit as st

st.set_page_config(
    page_title="Query Tool · Grounded answers from your documents",
    page_icon=":material/verified:",
    layout="wide",
)

from ui import inject_css, require_password, sidebar  # noqa: E402  (after set_page_config)

inject_css()
require_password()
navigation = st.navigation(
    [
        st.Page("views/chat.py", title="Ask", icon=":material/forum:", default=True),
        st.Page("views/documents.py", title="Knowledge base", icon=":material/library_books:"),
        st.Page("views/architecture.py", title="How it works", icon=":material/account_tree:"),
    ],
    position="top",
)
sidebar()
navigation.run()
