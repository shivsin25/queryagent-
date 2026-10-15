"""The story the UI tells: who the agent is, how an answer is made, and how this one was.

The same six stages appear on the landing page (what each one does), live in the chat while
an answer is being made, and on every answer card (what each one did for that answer).
"""

from dataclasses import dataclass, field
from html import escape
from typing import Literal

Status = Literal["intro", "pending", "active", "done", "warn", "blocked", "skipped"]


@dataclass(frozen=True)
class Stage:
    key: str
    title: str
    blurb: str  # what the stage does, in one line


STAGES: list[Stage] = [
    Stage("guard", "Guard", "Refuses attempts to hijack the assistant, before any AI runs"),
    Stage("plan", "Plan", "Understands follow-ups and splits multi-part questions"),
    Stage("retrieve", "Retrieve", "Hybrid search, reranked, keeping only relevant passages"),
    Stage("write", "Write", "Drafts an answer that cites a source for every fact"),
    Stage("review", "Review", "Fact-checks each claim, then revises or searches again"),
    Stage("polish", "Polish", "Tidies the wording, never the facts or citations"),
]
_ORDER = [s.key for s in STAGES]

PROMISES: list[tuple[str, str]] = [
    ("Every claim is cited", "Each sentence points to the passage, and page, it came from."),
    ("Checked before you see it", "A reviewer and a citation checker approve every answer."),
    ("Says when it doesn't know", "Questions the documents don't cover are declined, not guessed."),
    ("Spends carefully", "Every call is costed; a daily budget and a cache keep it cheap."),
]


@dataclass
class StageState:
    status: Status = "pending"
    note: str = ""
    loops: int = 0  # times the writer went round again


@dataclass
class Journey:
    stages: dict[str, StageState] = field(
        default_factory=lambda: {key: StageState() for key in _ORDER}
    )
    cached: bool = False
    blocked: bool = False


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def journey(steps: list[tuple[str, dict]], finished: bool) -> Journey:
    """What each stage did, from the pipeline's steps so far (node name, trace detail)."""
    j = Journey()
    s = j.stages
    next_up = "guard"
    for node, detail in steps:
        if node == "cache":
            j.cached = True
            for state in s.values():
                state.status, state.note = "skipped", ""
            s["retrieve"].note = "Asked before: answered from the cache"
            return j
        if node == "preprocess":
            if detail.get("blocked"):
                j.blocked = True
                reasons = ", ".join(detail.get("reasons", [])) or "unsafe request"
                s["guard"].status, s["guard"].note = "blocked", f"Stopped: {reasons}"
                next_up = ""
                continue
            s["guard"].status, s["guard"].note = "done", "Safe to answer"
            queries = detail.get("search_queries", [])
            if detail.get("history_turns"):
                note = f"Follow-up read as: {detail.get('standalone_question', '')}"
            elif len(queries) > 1:
                note = f"Split into {len(queries)} searches"
            else:
                note = "One focused search"
            s["plan"].status, s["plan"].note = "done", note
            next_up = "retrieve"
        elif node == "retrieve":
            count = detail.get("count", 0)
            if count:
                docs = len(detail.get("sources", []))
                s["retrieve"].status = "done"
                s["retrieve"].note = (f"{_plural(count, 'passage')} from "
                                      f"{_plural(docs, 'document')}")
                next_up = "write"
            else:
                s["retrieve"].status, s["retrieve"].note = "warn", "Nothing relevant found"
                next_up = ""
        elif node == "no_context":
            for key in ("write", "review", "polish"):
                s[key].status, s[key].note = "skipped", "Not needed"
        elif node == "writer":
            iteration = detail.get("iteration", 0)
            s["write"].status = "done"
            s["write"].loops = iteration
            s["write"].note = "First draft" if not iteration else f"Draft {iteration + 1}"
            next_up = "review"
        elif node == "reviewer":
            if detail.get("status") == "failed":
                s["review"].status, s["review"].note = "warn", "Reviewer unavailable"
                next_up = "polish"
            elif detail.get("approved"):
                support = detail.get("claim_support")
                cited = f" · {support:.0%} of citations check out" if support is not None else ""
                s["review"].status, s["review"].note = "done", f"Approved{cited}"
                next_up = "polish"
            else:
                s["review"].status, s["review"].note = "warn", "Sent back for revision"
                next_up = "write"
        elif node == "research":
            found = detail.get("new_chunks", 0)
            s["review"].note += (f" · searched again, {_plural(found, 'new passage')}" if found
                                 else " · searched again, nothing new")
            next_up = "write" if found else "polish"
        elif node == "feedback":
            next_up = "write"
        elif node == "enhancer":
            s["polish"].status = "done"
            s["polish"].note = "Wording polished" if detail.get("polished") else "Kept as reviewed"
            next_up = ""

    if not finished and next_up:
        s[next_up].status = "active"
    if finished:
        for state in s.values():
            if state.status in ("pending", "active"):
                state.status = "skipped"
    return j


def journey_from_result(result: dict) -> Journey:
    steps = [(t["node"], t.get("detail", {})) for t in result.get("trace", [])]
    if result.get("cached"):
        steps = [("cache", {})]
    return journey(steps, finished=True)


_STATUS_MARK = {"done": "✓", "warn": "!", "blocked": "✕", "skipped": "–", "active": "",
                "pending": "", "intro": ""}


def flow_html(j: Journey | None = None, compact: bool = False) -> str:
    """The six stages as a row of connected cards. Without a journey: the introduction."""
    cards = []
    for number, stage in enumerate(STAGES, 1):
        state = j.stages[stage.key] if j else StageState(status="intro", note=stage.blurb)
        mark = _STATUS_MARK[state.status] or str(number)
        loops = (f'<span class="qt-loop" title="Revised">↺ {state.loops}</span>'
                 if state.loops else "")
        cards.append(
            f'<div class="qt-stage qt-{state.status}">'
            f'<div class="qt-stage-head"><span class="qt-stage-mark">{mark}</span>'
            f'<span class="qt-stage-title">{escape(stage.title)}</span>{loops}</div>'
            f'<div class="qt-stage-note">{escape(state.note)}</div></div>'
        )
    joined = '<div class="qt-arrow" aria-hidden="true">→</div>'.join(cards)
    banner = ""
    if j and j.cached:
        banner = ('<div class="qt-flow-banner">⚡ Answered instantly from the cache: '
                  "asked before, no AI calls needed</div>")
    return (f'<div class="qt-flow{" qt-compact" if compact else ""}" role="list" '
            f'aria-label="How this answer was made">{joined}</div>{banner}')


def promises_html() -> str:
    items = "".join(
        f'<div class="qt-promise"><div class="qt-promise-title">{escape(title)}</div>'
        f'<div class="qt-promise-text">{escape(text)}</div></div>'
        for title, text in PROMISES
    )
    return f'<div class="qt-promises">{items}</div>'
