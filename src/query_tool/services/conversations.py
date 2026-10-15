"""Conversation history, kept in the application database."""

from datetime import UTC, datetime

from query_tool.core.schemas import Turn
from query_tool.storage.database import Database

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    question TEXT NOT NULL,
    standalone_question TEXT NOT NULL,
    answer TEXT NOT NULL,
    approved INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS turns_by_session ON turns (session_id, id);
"""


class ConversationStore:
    def __init__(self, db: Database, max_turns_per_session: int = 100) -> None:
        self._db = db
        self._max_turns = max_turns_per_session
        db.migrate(_SCHEMA)

    def history(self, session_id: str, limit: int | None = None) -> list[Turn]:
        """The most recent turns of a session, oldest first."""
        with self._db.connect() as db:
            rows = db.execute(
                "SELECT question, standalone_question, answer, approved, created_at FROM turns "
                "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit if limit is not None else self._max_turns),
            ).fetchall()
        return [
            Turn(question=q, standalone_question=s, answer=a, approved=bool(ok),
                 created_at=datetime.fromisoformat(at))
            for q, s, a, ok, at in reversed(rows)
        ]

    def append(self, session_id: str, turn: Turn) -> None:
        created_at = (turn.created_at or datetime.now(UTC)).isoformat(timespec="seconds")
        with self._db.connect() as db:
            db.execute(
                "INSERT INTO turns (session_id, question, standalone_question, answer, approved, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, turn.question, turn.standalone_question or turn.question,
                 turn.answer, int(turn.approved), created_at),
            )
            # Keep each session bounded; only the latest turns are ever read back.
            db.execute(
                "DELETE FROM turns WHERE session_id = ? AND id NOT IN "
                "(SELECT id FROM turns WHERE session_id = ? ORDER BY id DESC LIMIT ?)",
                (session_id, session_id, self._max_turns),
            )

    def delete(self, session_id: str) -> int:
        with self._db.connect() as db:
            return db.execute("DELETE FROM turns WHERE session_id = ?", (session_id,)).rowcount
