"""The application's SQLite file: conversations, LLM usage, answer cache and query log.

SQLite (in WAL mode) is shared safely by the API, its workers and the CLI on one machine.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class Database:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")  # readers don't block the writer

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        # One short-lived connection per call keeps every store safe to use from any thread.
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:  # commits on success, rolls back on error
                yield db
        finally:
            db.close()

    def migrate(self, schema: str, columns: dict[str, dict[str, str]] | None = None) -> None:
        """Create the tables, then add any of `columns` ({table: {name: type}}) an older
        database lacks, so existing data survives schema changes."""
        with self.connect() as db:
            db.executescript(schema)
            for table, wanted in (columns or {}).items():
                present = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
                for name, kind in wanted.items():
                    if name not in present:
                        db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")
