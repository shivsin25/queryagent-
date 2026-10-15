"""Run the API inside the Streamlit process, for hosts that run only one app.

Streamlit Community Cloud starts nothing but `streamlit run`. With QT_EMBEDDED_API set, the
UI starts the FastAPI app itself, on a free localhost port in a background thread, once per
process. Everything else is unchanged: the UI still talks to the API over HTTP.
"""

import os
import socket
import sys
import threading
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
STARTUP_TIMEOUT_S = 600  # the first start downloads the search models


def enabled() -> bool:
    return os.getenv("QT_EMBEDDED_API", "").lower() in {"1", "true", "yes"}


def serve(app, host: str = "127.0.0.1") -> str:
    """Start an ASGI app with uvicorn in a daemon thread; return its URL once it's ready."""
    import uvicorn

    with socket.socket() as probe:
        probe.bind((host, 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    threading.Thread(target=server.run, name="query-tool-api", daemon=True).start()
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("The embedded API did not start in time.")
        time.sleep(0.2)
    return f"http://{host}:{port}"


def _secrets_to_env() -> None:
    """Copy top-level Streamlit secrets into the environment, where the API's settings
    look for keys. Variables already set win."""
    try:
        items = dict(st.secrets)
    except Exception:  # no secrets file: running locally with .env instead
        return
    for key, value in items.items():
        if isinstance(value, str | int | float | bool):
            os.environ.setdefault(key, str(value))


@st.cache_resource(show_spinner="Starting the answer engine. The first start downloads the "
                                "search models, which takes a minute or two…")
def start_api() -> str:
    _secrets_to_env()
    if str(ROOT / "src") not in sys.path:  # the backend isn't pip-installed on such hosts
        sys.path.insert(0, str(ROOT / "src"))
    from query_tool.api.main import create_app

    url = serve(create_app())
    os.environ["QT_API_URL"] = url
    return url
