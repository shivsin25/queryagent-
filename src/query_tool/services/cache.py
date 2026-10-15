"""Reuse approved answers to repeated questions, so asking twice costs nothing.

The key covers everything that could change the answer: the question (normalised), the
request options, the document index version, the settings that shape the pipeline and the
text of every prompt. Changing any of them misses the cache; nothing is ever served stale.
"""

import hashlib
import json
import logging
from datetime import UTC, datetime, timedelta

from query_tool.config.settings import Settings
from query_tool.core.schemas import QueryResponse
from query_tool.prompts import (
    ENHANCER_PROMPT,
    FEEDBACK_PROMPT,
    PLANNER_PROMPT,
    PLANNER_RETRY_BLOCK,
    RETRY_BLOCK,
    REVIEWER_PROMPT,
    REVISION_BLOCK,
    WRITER_PROMPT,
)
from query_tool.storage.database import Database

logger = logging.getLogger(__name__)

# Bump when pipeline code changes answers in a way the settings and prompts don't capture.
CACHE_VERSION = 1

_SETTINGS_IN_KEY = frozenset({
    "llm_provider", "primary_model", "fast_model", "reviewer_provider", "reviewer_model",
    "temperature", "max_output_tokens", "embedding_model", "rerank_enabled", "rerank_model",
    "candidate_k", "min_relevance", "claim_check_enabled", "claim_support_threshold",
    "approval_threshold", "max_research_rounds", "research_k", "chunk_size",
    "chunk_overlap", "parent_chunk_size", "min_chunk_score", "rerank_margin", "rerank_k",
})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS answer_cache (
    key TEXT PRIMARY KEY,
    response TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def pipeline_fingerprint(settings: Settings) -> str:
    prompts = [
        message.prompt.template
        for template in (PLANNER_PROMPT, WRITER_PROMPT, REVIEWER_PROMPT, FEEDBACK_PROMPT,
                         ENHANCER_PROMPT)
        for message in template.messages
    ]
    payload = {
        "version": CACHE_VERSION,
        "settings": settings.model_dump(mode="json", include=_SETTINGS_IN_KEY),
        "prompts": [*prompts, REVISION_BLOCK, RETRY_BLOCK, PLANNER_RETRY_BLOCK],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def normalize_question(question: str) -> str:
    """'Refund policy?' and 'refund  policy' are the same question."""
    return " ".join(question.lower().split()).rstrip("?.! ")


class AnswerCache:
    def __init__(self, db: Database, ttl_hours: float) -> None:
        self._db = db
        self._ttl = timedelta(hours=ttl_hours)
        db.migrate(_SCHEMA)

    @property
    def enabled(self) -> bool:
        return self._ttl > timedelta(0)

    @staticmethod
    def key(question: str, top_k: int, max_revisions: int, doc_ids: list[str] | None,
            fingerprint: str, index_version: str) -> str:
        raw = json.dumps([normalize_question(question), top_k, max_revisions,
                          sorted(doc_ids or []), fingerprint, index_version])
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> QueryResponse | None:
        if not self.enabled:
            return None
        cutoff = (datetime.now(UTC) - self._ttl).isoformat(timespec="seconds")
        with self._db.connect() as db:
            row = db.execute(
                "SELECT response FROM answer_cache WHERE key = ? AND created_at >= ?",
                (key, cutoff),
            ).fetchone()
        if row is None:
            return None
        try:
            return QueryResponse.model_validate_json(row[0])
        except ValueError:  # written by an older, incompatible version
            logger.warning("Ignoring unreadable cache entry")
            return None

    def put(self, key: str, response: QueryResponse) -> None:
        if not self.enabled:
            return
        now = datetime.now(UTC)
        with self._db.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO answer_cache (key, response, created_at) VALUES (?,?,?)",
                (key, response.model_dump_json(), now.isoformat(timespec="seconds")),
            )
            db.execute("DELETE FROM answer_cache WHERE created_at < ?",
                       ((now - self._ttl).isoformat(timespec="seconds"),))

    def clear(self) -> int:
        with self._db.connect() as db:
            return db.execute("DELETE FROM answer_cache").rowcount
