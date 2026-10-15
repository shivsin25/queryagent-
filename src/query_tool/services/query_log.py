"""Every answered query, plus the users' thumbs up/down on them.

The log also links retries: each answer records the first attempt at its question
(`root_id`) and its attempt number, which is how the re-query limit is enforced and how a
retry knows what the earlier answers said. Down-voted answers are the best source of new
evaluation cases: `qt feedback --export` turns them into eval stubs to fill in.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from query_tool.core.schemas import FeedbackItem, QueryResponse
from query_tool.storage.database import Database

_SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    request_id TEXT PRIMARY KEY,
    session_id TEXT,
    question TEXT NOT NULL,
    standalone_question TEXT NOT NULL,
    answer TEXT NOT NULL,
    approved INTEGER NOT NULL,
    cached INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    cost_usd REAL NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    request_id TEXT PRIMARY KEY REFERENCES queries (request_id),
    rating INTEGER NOT NULL,
    comment TEXT,
    created_at TEXT NOT NULL
);
"""
# Added after the first release; migrate() adds them to existing databases.
_COLUMNS = {"queries": {"root_id": "TEXT", "attempt": "INTEGER NOT NULL DEFAULT 1",
                        "cache_key": "TEXT", "response": "TEXT"}}
_INDEXES = """
CREATE INDEX IF NOT EXISTS queries_by_session ON queries (session_id, created_at);
CREATE INDEX IF NOT EXISTS queries_by_root ON queries (root_id, attempt);
"""

Rating = Literal["up", "down"]


@dataclass(frozen=True)
class LoggedQuery:
    request_id: str
    question: str
    root_id: str
    attempt: int
    cache_key: str | None
    response: QueryResponse | None


class QueryLog:
    def __init__(self, db: Database) -> None:
        self._db = db
        db.migrate(_SCHEMA, _COLUMNS)
        db.migrate(_INDEXES)

    def record(self, response: QueryResponse, cache_key: str | None = None) -> None:
        with self._db.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO queries (request_id, session_id, question, "
                "standalone_question, answer, approved, cached, latency_ms, cost_usd, "
                "created_at, root_id, attempt, cache_key, response) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (response.request_id, response.session_id, response.question,
                 response.standalone_question, response.answer, int(response.approved),
                 int(response.cached), response.latency_ms, response.usage.cost_usd,
                 datetime.now(UTC).isoformat(timespec="microseconds"),
                 response.root_request_id or response.request_id, response.attempt,
                 cache_key, response.model_dump_json()),
            )

    def _select(self, where: str, params: tuple) -> list[LoggedQuery]:
        with self._db.connect() as db:
            rows = db.execute(
                "SELECT request_id, question, COALESCE(root_id, request_id), "
                f"COALESCE(attempt, 1), cache_key, response FROM queries {where}",
                params,
            ).fetchall()
        return [
            LoggedQuery(rid, question, root, attempt, key,
                        QueryResponse.model_validate_json(raw) if raw else None)
            for rid, question, root, attempt, key, raw in rows
        ]

    def get(self, request_id: str) -> LoggedQuery | None:
        found = self._select("WHERE request_id = ?", (request_id,))
        return found[0] if found else None

    def latest_in_session(self, session_id: str) -> LoggedQuery | None:
        found = self._select(
            "WHERE session_id = ? ORDER BY created_at DESC LIMIT 1", (session_id,)
        )
        return found[0] if found else None

    def attempts(self, root_id: str) -> list[LoggedQuery]:
        """Every answer to one question, first attempt first."""
        return self._select(
            "WHERE COALESCE(root_id, request_id) = ? ORDER BY COALESCE(attempt, 1)", (root_id,)
        )

    def add_feedback(self, request_id: str, rating: Rating, comment: str | None) -> bool:
        """Store (or replace) a rating. False if the request is unknown."""
        with self._db.connect() as db:
            if not db.execute("SELECT 1 FROM queries WHERE request_id = ?",
                              (request_id,)).fetchone():
                return False
            db.execute(
                "INSERT OR REPLACE INTO feedback (request_id, rating, comment, created_at) "
                "VALUES (?,?,?,?)",
                (request_id, 1 if rating == "up" else -1, comment,
                 datetime.now(UTC).isoformat(timespec="seconds")),
            )
        return True

    def feedback(self, rating: Rating | None = None, limit: int = 100) -> list[FeedbackItem]:
        where = "" if rating is None else "WHERE f.rating = ?"
        params: tuple = () if rating is None else (1 if rating == "up" else -1,)
        with self._db.connect() as db:
            rows = db.execute(
                "SELECT f.request_id, f.rating, f.comment, q.question, q.answer, f.created_at "
                f"FROM feedback f JOIN queries q USING (request_id) {where} "
                "ORDER BY f.created_at DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [
            FeedbackItem(request_id=rid, rating="up" if r > 0 else "down", comment=comment,
                         question=q, answer=a, created_at=datetime.fromisoformat(at))
            for rid, r, comment, q, a, at in rows
        ]
