"""The Streamlit pages, driven headlessly against a real API server (with a scripted LLM)."""

import socket
import sys
import threading
import time
from pathlib import Path

import pytest
import streamlit as st
import uvicorn
from streamlit.testing.v1 import AppTest

from query_tool.api.main import create_app
from tests.conftest import FakeLLMProvider

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
sys.path.insert(0, str(FRONTEND))  # the pages import ui, story and api_client

from story import journey  # noqa: E402

# ---- the story model -----------------------------------------------------------


def _status(steps, finished=False):
    j = journey(steps, finished)
    return {key: (s.status, s.note) for key, s in j.stages.items()}, j


def test_journey_of_a_normal_answer():
    steps = [
        ("preprocess", {"search_queries": ["refund policy"]}),
        ("retrieve", {"count": 2, "sources": ["a.md", "b.pdf"]}),
        ("writer", {"iteration": 0}),
        ("reviewer", {"approved": True, "claim_support": 1.0}),
        ("enhancer", {"polished": True}),
    ]
    states, j = _status(steps, finished=True)
    assert all(status == "done" for status, _ in states.values())
    assert states["retrieve"][1] == "2 passages from 2 documents"
    assert states["review"][1] == "Approved · 100% of citations check out"


def test_journey_while_running_marks_the_next_stage_active():
    states, _ = _status([("preprocess", {"search_queries": ["a", "b"]})])
    assert states["plan"] == ("done", "Split into 2 searches")
    assert states["retrieve"][0] == "active" and states["write"][0] == "pending"
    states, _ = _status([])
    assert states["guard"][0] == "active"


def test_journey_with_a_revision_and_a_second_search():
    steps = [
        ("preprocess", {"search_queries": ["q"]}),
        ("retrieve", {"count": 1, "sources": ["a.md"]}),
        ("writer", {"iteration": 0}),
        ("reviewer", {"approved": False}),
        ("research", {"new_chunks": 2}),
    ]
    states, j = _status(steps)
    assert states["review"] == ("warn", "Sent back for revision · searched again, 2 new passages")
    assert states["write"][0] == "active"
    steps += [("writer", {"iteration": 1}), ("reviewer", {"approved": True})]
    _, j = _status(steps)
    assert j.stages["write"].loops == 1 and j.stages["polish"].status == "active"


def test_journey_of_blocked_declined_and_cached_questions():
    states, j = _status([("preprocess", {"blocked": True, "reasons": ["role change"]})], True)
    assert j.blocked and states["guard"] == ("blocked", "Stopped: role change")
    assert states["retrieve"][0] == "skipped"

    states, _ = _status([("preprocess", {"search_queries": ["x"]}),
                         ("retrieve", {"count": 0}), ("no_context", {})], True)
    assert states["retrieve"] == ("warn", "Nothing relevant found")
    assert states["write"] == ("skipped", "Not needed")

    _, j = _status([("cache", {})], True)
    assert j.cached and all(s.status == "skipped" for s in j.stages.values())


# ---- the pages, end to end ---------------------------------------------------------


@pytest.fixture
def api(make_service, monkeypatch):
    service = make_service(FakeLLMProvider())
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(service), host="127.0.0.1", port=port,
                                           log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    monkeypatch.setenv("QT_API_URL", f"http://127.0.0.1:{port}")
    st.cache_resource.clear()
    st.cache_data.clear()
    yield service
    server.should_exit = True
    thread.join(timeout=5)
    st.cache_resource.clear()
    st.cache_data.clear()


def _page(name: str) -> AppTest:
    return AppTest.from_file(str(FRONTEND / "views" / name), default_timeout=60).run()


def _text(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def _button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def test_landing_tells_the_story(api):
    at = _page("chat.py")
    assert not at.exception
    text = _text(at)
    for line in ("Answers you can trust", "How every answer is made", "What you can count on",
                 "Guard", "Retrieve", "Review", "Polish", "Every claim is cited"):
        assert line in text
    assert len([b for b in at.button if b.key and b.key.startswith("ex-")]) == 4


def test_ask_retry_and_reach_the_limit(api, settings):
    settings.max_retries_per_question = 1
    at = _page("chat.py")
    at.chat_input[0].set_value("Refund policy?").run()
    assert not at.exception
    text = _text(at)
    assert "Annual plans can be refunded within 30 days" in text
    assert "Verified answer" in text and "How this answer was made" in text

    first = at.session_state["messages"][-1]["result"]
    at.text_input(key=f"why-{first['request_id']}").input("It skips monthly plans").run()
    _button(at, "Search wider and answer again").click().run()
    assert not at.exception
    retry = at.session_state["messages"][-1]["result"]
    assert retry["attempt"] == 2 and retry["root_request_id"] == first["request_id"]
    assert "Try again:" in _text(at) and "It skips monthly plans" in _text(at)
    assert any("Re-query limit reached" in c.value for c in at.caption)


def test_an_example_question_can_be_asked(api):
    at = _page("chat.py")
    _button(at, "What is the P1 response time for Enterprise customers?").click().run()
    assert not at.exception
    assert at.session_state["messages"][-1]["result"]["answer"]


def test_offline_api_is_reported_not_raised(api, monkeypatch):
    monkeypatch.setenv("QT_API_URL", "http://127.0.0.1:9")  # nothing listens here
    st.cache_resource.clear()
    st.cache_data.clear()
    at = _page("chat.py")
    at.chat_input[0].set_value("Refund policy?").run()
    assert not at.exception
    assert any("Cannot reach the API" in e.value or "Connection" in e.value for e in at.error)


def test_knowledge_base_and_how_it_works_pages(api):
    docs = _page("documents.py")
    assert not docs.exception
    assert "What happens to a file" in _text(docs)
    assert [m.value for m in docs.metric][:2] == ["1", "2"]

    how = _page("architecture.py")
    assert not how.exception
    assert "A team of agents, each with one job" in _text(how)
    assert len(how.subheader) == 7
